/// Use a trained N-tuple agent to play 2048 on a live website via browser automation.
///
/// Usage:
///     cargo run --release --bin play_web -- checkpoints/checkpoint_ep0850000.bin
///     cargo run --release --bin play_web -- checkpoints/checkpoint_ep0850000.bin --depth adaptive

use std::env;
use std::io::Write;
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};

use clap::Parser;
use headless_chrome::browser::tab::Tab;
use headless_chrome::{Browser, LaunchOptions};

use rust_2048::board::{Action, Board, apply_action, max_tile, set_tile};
use rust_2048::expectimax::{self, Depth, expectimax_action};
use rust_2048::ntuple::NTupleNetwork;
use rust_2048::train;

struct GameConfig {
    storage_key: String,
    container_selector: String,
    new_game_selector: String,
}

fn require_env(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| panic!("{name} environment variable not set"))
}

#[derive(Parser)]
#[command(name = "play_web", about = "Play 2048 on a live website with a trained agent")]
struct Cli {
    /// Path to checkpoint file (.bin)
    checkpoint: String,

    /// Search depth: integer, "adaptive", or custom schedule like "10:1,6:2,0:3"
    #[arg(long, default_value = "0")]
    depth: String,

    /// Seconds between moves (default: 0.0)
    #[arg(long, default_value_t = 0.0)]
    move_delay: f64,

    /// Game URL (overrides env)
    #[arg(long)]
    url: Option<String>,
}

const ACTION_KEYS: [&str; 4] = ["ArrowUp", "ArrowRight", "ArrowDown", "ArrowLeft"];

fn parse_board_json(json_str: &str) -> Option<(Board, u32, bool)> {
    let parsed: serde_json::Value = serde_json::from_str(json_str).ok()?;

    let board_arr = parsed["board"].as_array()?;
    let score = parsed["score"].as_u64()? as u32;
    let over = parsed["over"].as_bool().unwrap_or(false);

    let mut board: Board = 0;
    for (pos, val) in board_arr.iter().enumerate() {
        let tile_val = val.as_u64().unwrap_or(0) as u32;
        if tile_val > 0 {
            let exp = tile_val.trailing_zeros() as u8;
            set_tile(&mut board, pos, exp);
        }
    }

    Some((board, score, over))
}

/// Build JS snippet that reads the board from localStorage (reused across calls).
fn read_board_js(storage_key: &str) -> String {
    format!(
        r#"(() => {{
            const raw = localStorage.getItem("{storage_key}");
            if (!raw) return null;
            const state = JSON.parse(raw);
            const board = [];
            for (let r = 0; r < 4; r++)
                for (let c = 0; c < 4; c++) {{
                    const cell = state.grid[r][c];
                    board.push(cell ? cell.value : 0);
                }}
            return JSON.stringify({{ board, score: state.score, over: state.over }});
        }})()"#,
    )
}

/// Read the board state from the browser's localStorage.
fn read_board(tab: &Tab, js: &str) -> Option<(Board, u32, bool)> {
    let result = tab.evaluate(js, false).ok()?;
    let json_str = result.value?.as_str()?.to_string();
    parse_board_json(&json_str)
}

fn send_action(tab: &Tab, action: Action) {
    let key = ACTION_KEYS[action as usize];
    let _ = tab.press_key(key);
}

/// Send key via CDP, then poll for board change (no sleep between polls).
fn wait_for_board_change(
    tab: &Tab,
    read_js: &str,
    old_board: Board,
    timeout: Duration,
) -> Option<(Board, u32, bool)> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Some((board, score, over)) = read_board(tab, read_js) {
            if board != old_board || over {
                return Some((board, score, over));
            }
        }
    }
    read_board(tab, read_js)
}

fn select_action(
    board: Board,
    network: &NTupleNetwork,
    depth: &Option<Depth>,
) -> Option<Action> {
    match depth {
        Some(d) => expectimax_action(board, network, d),
        None => train::select_action(network, board).map(|(a, _)| a),
    }
}

fn game_loop(
    tab: &Tab,
    config: &GameConfig,
    network: &NTupleNetwork,
    depth: &Option<Depth>,
    move_delay: Duration,
) {
    let read_js = read_board_js(&config.storage_key);

    loop {
        println!("Press Enter to start playing (set up the game in the browser first)...");
        let mut buf = String::new();
        std::io::stdin().read_line(&mut buf).unwrap();

        let mut move_count = 0u32;
        loop {
            let state = read_board(tab, &read_js);
            if state.is_none() {
                println!("No game state found in localStorage. Start a game first.");
                break;
            }
            let (board, score, game_over) = state.unwrap();

            if game_over {
                println!(
                    "\nGAME OVER — Score: {}  Max tile: {}  Moves: {}",
                    score,
                    max_tile(board),
                    move_count
                );
                break;
            }

            if Action::ALL.iter().all(|&a| apply_action(board, a).0 == board) {
                println!(
                    "\nNo valid moves — Score: {}  Max tile: {}  Moves: {}",
                    score,
                    max_tile(board),
                    move_count
                );
                break;
            }

            let action = match select_action(board, network, depth) {
                Some(a) => a,
                None => break,
            };
            send_action(tab, action);

            let new_state = wait_for_board_change(tab, &read_js, board, Duration::from_secs(2));
            match new_state {
                Some((new_board, _, _)) if new_board == board => continue,
                None => continue,
                _ => {}
            }

            let (new_board, new_score, _) = new_state.unwrap();
            move_count += 1;

            print!(
                "\rMove {:<5} {:<5?}  Score: {:<8} Max: {}",
                move_count,
                action,
                new_score,
                max_tile(new_board)
            );
            std::io::stdout().flush().ok();

            if !move_delay.is_zero() {
                thread::sleep(move_delay);
            }
        }

        println!("\nPress Enter to play again, or q to quit:");
        let mut buf = String::new();
        std::io::stdin().read_line(&mut buf).unwrap();
        if buf.trim().eq_ignore_ascii_case("q") {
            break;
        }

        if let Ok(el) = tab.find_element(&config.new_game_selector) {
            let _ = el.click();
        }
        thread::sleep(Duration::from_millis(500));
    }
}

fn main() {
    // Load .env from the project root (one level up from rust_2048/)
    let _ = dotenvy::from_filename("../.env");
    let _ = dotenvy::dotenv();

    let config = GameConfig {
        storage_key: require_env("GAME_STORAGE_KEY"),
        container_selector: require_env("GAME_CONTAINER_SELECTOR"),
        new_game_selector: require_env("NEW_GAME_BUTTON_SELECTOR"),
    };
    let game_url = require_env("GAME_URL");

    let cli = Cli::parse();

    let network = NTupleNetwork::load(Path::new(&cli.checkpoint))
        .expect("Failed to load checkpoint");

    let depth = match expectimax::parse_depth(&cli.depth) {
        Ok(Depth::Fixed(0)) => None,
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("Invalid --depth: {e}");
            std::process::exit(1);
        }
    };

    let url = cli.url.as_deref().unwrap_or(&game_url);
    let move_delay = Duration::from_secs_f64(cli.move_delay);

    let browser = Browser::new(
        LaunchOptions::default_builder()
            .headless(false)
            .build()
            .expect("Failed to build launch options"),
    )
    .expect("Failed to launch browser");

    let tab = browser.new_tab().expect("Failed to open tab");
    tab.navigate_to(url).expect("Failed to navigate");
    tab.wait_until_navigated().expect("Navigation timeout");

    println!("Opened {url}");

    // Disable CSS animations
    let _ = tab.evaluate(
        r#"(() => {
            const style = document.createElement('style');
            style.textContent = '* { transition: none !important; animation: none !important; }';
            document.head.appendChild(style);
        })()"#,
        false,
    );

    // Click on the game container so keyboard events are captured
    if let Ok(el) = tab.find_element(&config.container_selector) {
        let _ = el.click();
    }

    game_loop(&tab, &config, &network, &depth, move_delay);

    println!("Press Enter to close the browser...");
    let mut buf = String::new();
    std::io::stdin().read_line(&mut buf).unwrap();
}
