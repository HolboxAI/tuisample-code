mod app;
mod config;
mod danger;
mod dateutil;
mod device;
mod freetier;
mod llm;
mod notice;
mod providers;
mod quota;
mod telemetry;
mod tools;
mod theme;
mod ui;
mod upgrade;
mod usage;
mod workspace;

use app::{App, AppState};
use config::Config;
use workspace::Workspace;
use crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyCode, KeyEventKind, KeyModifiers,
    KeyboardEnhancementFlags, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, supports_keyboard_enhancement, EnterAlternateScreen,
    LeaveAlternateScreen,
};
use llm::StreamEvent;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use std::error::Error;
use std::io;
use std::time::Duration;
use tokio::sync::mpsc;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Handle flags before touching the terminal, so `--version` works when
    // piped and `--upgrade` still runs even if the config file is broken.
    let mut upgrade = false;
    let mut force = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "-V" | "--version" => {
                println!("tuisample-code {VERSION}");
                return Ok(());
            }
            "-h" | "--help" => {
                print_help();
                return Ok(());
            }
            "-u" | "--upgrade" => upgrade = true,
            "-f" | "--force" => force = true,
            other => {
                eprintln!("Unknown argument: {other}\n");
                print_help();
                std::process::exit(2);
            }
        }
    }

    if upgrade {
        // Handled here rather than returned: the default runtime handler prints
        // Err via Debug, which turns a connection failure into a wall of
        // struct-dump instead of a sentence.
        if let Err(e) = upgrade::run(force).await {
            eprintln!("❌ Upgrade failed: {e}");
            std::process::exit(1);
        }
        return Ok(());
    }
    if force {
        eprintln!("--force only means something alongside --upgrade.\n");
        print_help();
        std::process::exit(2);
    }

    let mut config = Config::load()?;
    // Before anything is drawn: the colours depend on the terminal's
    // background, and asking for that needs the terminal to itself, with
    // no alternate screen up and nothing else reading stdin.
    theme::init(theme::resolve_mode(&config.ui.theme));

    // Enrol before the terminal is touched, so a slow or unreachable gateway is
    // an ordinary line on stdout rather than a frozen alternate screen. This
    // only blocks for a brand-new install (it needs a device token before the
    // app is usable at all) -- an already-enrolled device's budget is fetched
    // after the terminal comes up instead, see `refresh_budget_on_start`.
    let (free_tier_status, refresh_budget_on_start) = enrol_free_tier(&mut config).await;
    let (workspace, workspace_status) = open_workspace(&config);

    // Detached, not awaited: a slow or unreachable telemetry endpoint must
    // never delay the terminal coming up. See telemetry.rs -- this is a
    // no-op until a real endpoint is configured, and every failure inside it
    // is already silent.
    tokio::spawn(telemetry::ping_active_if_new_day(VERSION));

    let enhanced = setup_terminal()?;
    install_panic_hook(enhanced);

    // Discard any keystrokes the terminal buffered while we were blocked above,
    // typed before raw mode existed to consume them. Left alone, enabling raw
    // mode releases them straight into the event loop, where they land as
    // "typed" characters in the input box the instant the UI appears.
    while event::poll(Duration::from_millis(0))? {
        let _ = event::read()?;
    }

    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new(config);
    app.free_tier_status = free_tier_status;
    app.refresh_budget = refresh_budget_on_start;
    // Loaded before the first prompt so a limit already spent today is in force
    // from the start, not after the first request slips through.
    if app.config.quota.enabled {
        app.quota = quota::DailyQuota::load(&quota::today());
    }
    app.workspace_status = workspace_status;
    app.workspace_root = workspace
        .as_ref()
        .map(|ws| ws.root().display().to_string())
        .unwrap_or_default();
    let (tx, mut rx) = mpsc::channel::<(u64, StreamEvent)>(256);

    let result = run_app(&mut terminal, &mut app, workspace.as_ref(), tx, &mut rx).await;

    restore_terminal(enhanced)?;
    if let Err(e) = &result {
        eprintln!("Error: {e}");
    }
    println!("Goodbye!");
    result
}

/// Resolve the root the model is confined to, and a line describing the outcome.
///
/// A workspace that cannot be opened must not stop the app: it still works as a
/// plain chat client, just without file access, so the failure degrades to a
/// notice on the welcome screen instead of a startup error.
fn open_workspace(config: &Config) -> (Option<Workspace>, String) {
    if !config.tools.enabled {
        return (None, "off (enabled = false in config.toml)".to_string());
    }
    match Workspace::new(&config.tools.workspace) {
        Ok(workspace) => {
            let root = workspace.root().display().to_string();
            // Every one of these is worth seeing before typing the first prompt:
            // a shell tool can change anything, and unattended mode means it can
            // do so without asking.
            let mut status = format!("commands run in {root}");
            if !config.tools.require_approval {
                status.push_str(" — UNATTENDED, no approval prompt");
            }
            if workspace.is_broad() {
                status.push_str(" — this is a very broad directory");
            }
            (Some(workspace), status)
        }
        Err(e) => (None, format!("off — {e}")),
    }
}

async fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    app: &mut App,
    workspace: Option<&Workspace>,
    tx: mpsc::Sender<(u64, StreamEvent)>,
    rx: &mut mpsc::Receiver<(u64, StreamEvent)>,
) -> Result<(), Box<dyn Error>> {
    loop {
        terminal.draw(|f| ui::render(f, app))?;

        // Keyboard / paste input.
        if event::poll(Duration::from_millis(16))? {
            match event::read()? {
                Event::Key(key) if key.kind != KeyEventKind::Release => {
                    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
                    match key.code {
                        KeyCode::Char('c') | KeyCode::Char('d') if ctrl => break,
                        _ => app.handle_key(key),
                    }
                }
                Event::Paste(text) => app.handle_paste(text),
                _ => {}
            }
        }

        // Drain every token that has arrived; doing only one per frame caps
        // throughput at ~60 tokens/sec and looks like the app is stalling.
        while let Ok((id, event)) = rx.try_recv() {
            // The budget lookup belongs to no request, so the stale-id guard
            // must not apply to it: a user who submits a prompt while the
            // lookup is in flight would otherwise silently lose the answer.
            match event {
                // None of these belong to a request, so the stale-id guard
                // below must not apply: a user who submits a prompt while
                // enrolment is in flight would otherwise lose the result.
                StreamEvent::FreeTierBudget(line) => {
                    app.free_tier_status = line;
                    continue;
                }
                StreamEvent::FreeTierEnrolled(e) => {
                    app.free_tier_enrolled(*e);
                    continue;
                }
                StreamEvent::FreeTierFailed(reason) => {
                    app.free_tier_failed(reason);
                    continue;
                }
                _ => {}
            }
            if id != app.request_id {
                continue; // stale: belongs to a cancelled request
            }
            match event {
                StreamEvent::Token(token) => app.append_token(&token),
                StreamEvent::ToolCalls(calls) => app.request_tools(calls),
                StreamEvent::ToolsFinished(outcomes) => app.finish_tools(outcomes),
                StreamEvent::Usage(u) => app.record_exact_usage(u),
                StreamEvent::FreeTierBudget(_)
                | StreamEvent::FreeTierEnrolled(_)
                | StreamEvent::FreeTierFailed(_) => {
                    unreachable!("handled before the stale-id guard")
                }
                StreamEvent::Done => app.finish_stream(),
                StreamEvent::Notice(note) => app.note(note),
                StreamEvent::Error(err) => app.fail_stream(err),
            }
        }

        // The only place `finish_stream`/`fail_stream`/`cancel`'s queued
        // usage actually reaches disk -- see `App::pending_usage`'s doc
        // comment on why `app.rs` itself never writes this directly. Catches
        // both this loop's own draining above and anything `handle_key`
        // queued earlier this same iteration (a cancel via Esc).
        for (tokens, model) in app.pending_usage.drain(..) {
            usage::record_turn(tokens, &model);
        }
        // Same reasoning, same place: `App` marks the quota dirty, this loop is
        // the only thing that writes it.
        if app.quota_dirty {
            app.quota.save();
            app.quota_dirty = false;
        }

        // Free-tier budget refresh. Spawned rather than awaited: this runs on
        // the render loop, and a gateway that takes ten seconds to answer must
        // not freeze the UI for ten seconds. The answer comes back as an event.
        if app.enrol_requested {
            app.enrol_requested = false;
            // Runs on a clone: `App` owns the real config, so the answer comes
            // back as an event rather than being written from another task.
            let mut cfg = app.config.clone();
            cfg.free_tier.enabled = true;
            cfg.llm.api_key.clear(); // enrol regardless of the key in use now
            cfg.free_tier.device_token.clear();
            let tx_enrol = tx.clone();
            tokio::spawn(async move {
                let event = match freetier::register(&mut cfg).await {
                    Ok(enrolment) => StreamEvent::FreeTierEnrolled(Box::new(freetier::Enrolled {
                        endpoint: cfg.llm.endpoint.clone(),
                        device_token: cfg.free_tier.device_token.clone(),
                        model: cfg.llm.model.clone(),
                        fallback_id: cfg.free_tier.fallback_id.clone(),
                        daily_limit_usd: enrolment.daily_limit_usd,
                    })),
                    Err(e) => StreamEvent::FreeTierFailed(e),
                };
                let _ = tx_enrol.send((0, event)).await;
            });
        }

        if app.refresh_budget {
            app.refresh_budget = false;
            if freetier::is_free_tier(&app.config) {
                let config = app.config.clone();
                let id = app.request_id;
                let tx_budget = tx.clone();
                tokio::spawn(async move {
                    if let Ok(budget) = freetier::fetch_budget(&config).await {
                        let _ = tx_budget
                            .send((id, StreamEvent::FreeTierBudget(budget.summary())))
                            .await;
                    }
                });
            }
        }

        // Fire a pending request.
        if app.state == AppState::Sending {
            app.request_id += 1;
            let id = app.request_id;
            let endpoint = app.config.llm.endpoint.clone();
            let model = app.config.llm.model.clone();
            let api_key = app.config.llm.api_key.clone();
            let max_tokens = app.config.llm.max_tokens;

            // Withholding the schemas once the budget is spent is what actually
            // stops a runaway loop: the model has nothing left to call, so it
            // answers. Saying "stop" in the prompt alone would only be a request.
            let budget_left = app.tool_steps < app.config.tools.max_steps;
            // Exact counts make the quota real; without them it falls back to the
            // same character estimate `usage.rs` uses.
            let include_usage = app.config.quota.enabled && app.config.quota.include_usage;
            let (schemas, system) = match workspace {
                Some(ws) => (
                    if budget_left { tools::schemas() } else { Vec::new() },
                    Some(tools::system_prompt(ws, &app.config.tools, budget_left)),
                ),
                None => (Vec::new(), None),
            };
            let history = app.history(system.as_deref());
            let tx_clone = tx.clone();

            let handle = tokio::spawn(async move {
                llm::stream_chat(
                    llm::Target { endpoint: &endpoint, model: &model, api_key: &api_key, max_tokens, include_usage },
                    history,
                    schemas,
                    id,
                    tx_clone,
                )
                .await;
            });

            app.abort = Some(handle.abort_handle());
            app.state = AppState::Streaming;
        }

        // Run the commands the user allowed.
        //
        // Spawned rather than run inline: a command may take a minute, and doing
        // it on the event loop would freeze the whole UI -- no redraw, no Esc, no
        // way to tell a slow build from a hang. Results come back on the same
        // channel as tokens, so the stale-request-id guard covers them too.
        if app.state == AppState::ExecutingTools && !app.approved_tools.is_empty() {
            let calls = std::mem::take(&mut app.approved_tools);
            let tools_config = app.config.tools.clone();
            match workspace {
                Some(ws) => {
                    let ws = ws.clone();
                    let id = app.request_id;
                    let tx_clone = tx.clone();
                    let handle = tokio::spawn(async move {
                        let mut outcomes = Vec::with_capacity(calls.len());
                        for call in &calls {
                            outcomes.push(tools::execute(call, &ws, &tools_config).await);
                        }
                        let _ = tx_clone
                            .send((id, StreamEvent::ToolsFinished(outcomes)))
                            .await;
                    });
                    app.abort = Some(handle.abort_handle());
                }
                // Only reachable if a model invents tool calls for a schema it
                // was never sent. Answer them anyway, or the history is left
                // invalid and the next prompt fails instead of this one.
                None => app.fail_stream(
                    "The model asked to run a command, but the command tool is not enabled."
                        .to_string(),
                ),
            }
        }

        if app.should_exit {
            break;
        }
    }

    Ok(())
}

/// Enrol in the free tier if this install needs it, returning a line for the
/// welcome screen and whether the event loop still owes it a budget fetch.
///
/// Every failure degrades to a notice rather than an error: offline, blocked by
/// a proxy, or gateway down -- the app still starts, and still works for anyone
/// with their own key. An empty string means there is nothing worth saying.
async fn enrol_free_tier(config: &mut Config) -> (String, bool) {
    if freetier::is_free_tier(config) {
        // Deferred to `run_app`'s existing `refresh_budget` handling (the same
        // path a mid-session refresh uses) instead of awaited here: this is
        // the common case on every ordinary launch, and a slow or unreachable
        // gateway must not hold the terminal back just to fetch a number the
        // welcome screen can equally well pick up a moment later.
        return (format!("Free tier — {}", config.llm.model), true);
    }
    if !freetier::should_register(config) {
        return (String::new(), false);
    }

    println!("Setting up the free tier (no sign-in needed)…");
    let status = match freetier::register(config).await {
        Ok(enrolment) => {
            // Persist so the next launch reuses this device rather than enrolling
            // again -- and so the budget is not reset by a restart.
            if let Err(e) = config.save() {
                eprintln!("Warning: could not save the device token ({e}). It will enrol again next launch.");
            }
            format!(
                "Free tier — {} · ${:.2}/day, resets at UTC midnight",
                enrolment.model, enrolment.daily_limit_usd
            )
        }
        Err(e) => {
            eprintln!("Free tier unavailable: {e}");
            format!("unavailable — {e}")
        }
    };
    (status, false)
}

fn print_help() {
    println!(
        "tuisample-code {VERSION}
Terminal UI for an OpenAI-compatible LLM endpoint.

USAGE:
    tuisample-code [FLAGS]

FLAGS:
    -V, --version    Print version and exit
    -h, --help       Print this help and exit
    -u, --upgrade    Update to the latest release
    -f, --force      With --upgrade: reinstall even if already up to date

CONFIG (environment overrides ~/.tuisample-code/config.toml):
    TUISAMPLE_ENDPOINT    Base URL, e.g. https://llm.internal:8443
    TUISAMPLE_MODEL       Model name
    TUISAMPLE_API_KEY     Bearer token

TOOLS (read_file, write_file, run_command; writes and commands need your
       approval each time -- see the [tools] table in config.toml):
    TUISAMPLE_WORKSPACE       Directory these operate in (default: cwd)
    TUISAMPLE_TOOLS_ENABLED   Set to 0 to send no tool schema at all
    TUISAMPLE_TOOLS_APPROVAL  Set to 0 to stop asking before each write/command.
                              For scripted testing only -- it hands the model
                              unattended file and shell access.
                              See the [tools] table in config.toml for
                              auto_approve_read_only, command_timeout_secs,
                              max_output_bytes, max_steps.

UPGRADE:
    TUISAMPLE_UPGRADE_URL_BASE
                          Fetch updates from a fork or internal mirror
                          instead of github.com

COMMANDS (type in the input box, press Enter):
    /provider             Pick a provider + model + API key, saved to config.toml
    /model                Pick a model for the currently configured provider
    /new                  Forget the conversation and start fresh
    /usage                Show this machine's local token history
    /quota                Show today's limits and what's been spent
    /quota set <what> <n> Set your own limit: requests, tokens or usd
    /quota clear          Remove your own limits
    /quota override       Keep working past today's limit
    /quota reset          Cancel an override

FREE TIER (a fresh install with no API key enrols anonymously and gets a small
           daily budget on one model -- no sign-in. Only a hash of a hardware
           id is sent; prompts are never logged. Configuring your own key opts
           out entirely and sends traffic straight to your provider):
    TUISAMPLE_FREE_TIER       Set to 0 to never contact the gateway
    TUISAMPLE_GATEWAY         Point at a different gateway (e.g. staging)

DAILY LIMITS (optional, off by default -- every limit is 0 = no limit, so this
              only counts until you set one. See [quota] in config.toml):
    TUISAMPLE_QUOTA_ENABLED         Set to 0 to disable counting entirely
    TUISAMPLE_MAX_REQUESTS_PER_DAY  Requests before prompts are refused
    TUISAMPLE_MAX_TOKENS_PER_DAY    Prompt + completion tokens per UTC day
    TUISAMPLE_MAX_USD_PER_DAY       Spend per UTC day; needs [quota.pricing]
                                    entries for the models you use, or cost
                                    cannot be computed and reads as unpriced

KEYS:
    Enter                 Send prompt
    Alt/Shift-Enter       New line
    Esc                   Cancel request
    Ctrl-C                Exit"
    );
}

/// Returns true if the kitty keyboard protocol was enabled (so it can be popped later).
fn setup_terminal() -> Result<bool, Box<dyn Error>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    crossterm::execute!(stdout, EnterAlternateScreen, EnableBracketedPaste)?;

    // Optional: lets terminals that support it distinguish Shift/Ctrl-Enter.
    let enhanced = supports_keyboard_enhancement().unwrap_or(false);
    if enhanced {
        crossterm::execute!(
            stdout,
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
        )?;
    }
    Ok(enhanced)
}

fn restore_terminal(enhanced: bool) -> Result<(), Box<dyn Error>> {
    let mut stdout = io::stdout();
    if enhanced {
        let _ = crossterm::execute!(stdout, PopKeyboardEnhancementFlags);
    }
    crossterm::execute!(stdout, DisableBracketedPaste, LeaveAlternateScreen)?;
    disable_raw_mode()?;
    Ok(())
}

/// Without this a panic leaves the terminal in raw mode on the alternate screen,
/// with the backtrace invisible.
fn install_panic_hook(enhanced: bool) {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = restore_terminal(enhanced);
        default_hook(info);
    }));
}
