/// Watch a trained agent play 2048 in the terminal.
///
/// Usage:
///     cargo run --release --bin watch -- checkpoints/checkpoint_ep0350000.bin
///     cargo run --release --bin watch -- checkpoints/checkpoint_ep0350000.bin --depth adaptive

use std::io::{self, Write};
use std::path::Path;
use std::time::Duration;

use clap::Parser;
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEvent},
    execute,
    style::{Attribute, Color, SetForegroundColor},
    terminal::{self, ClearType},
};

use rust_2048::board::{Action, Game, get_tile, max_tile};
use rust_2048::expectimax::{self, Depth};
use rust_2048::ntuple::NTupleNetwork;
use rust_2048::train;

const CELL_W: usize = 7;
const DELAY_STEPS: &[f64] = &[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0];

#[derive(Parser)]
#[command(name = "watch", about = "Watch a trained agent play 2048")]
struct Cli {
    /// Path to checkpoint file (.bin)
    checkpoint: String,

    /// Seconds between moves
    #[arg(long, default_value_t = 0.0)]
    delay: f64,

    /// Search depth: integer, "adaptive", or custom schedule like "10:1,6:2,0:3"
    #[arg(long, default_value = "0")]
    depth: String,
}

fn tile_color(val: u32) -> Color {
    match val {
        2 => Color::White,
        4 => Color::Cyan,
        8 => Color::Green,
        16 => Color::Yellow,
        32 => Color::Red,
        64 => Color::Magenta,
        128 => Color::Blue,
        256 => Color::Cyan,
        512 => Color::Green,
        1024 => Color::Yellow,
        2048 => Color::Red,
        4096 => Color::Magenta,
        8192 => Color::Blue,
        16384 => Color::White,
        32768 => Color::Cyan,
        _ => Color::DarkGrey,
    }
}

fn tile_bold(val: u32) -> bool {
    val >= 256
}

fn draw(
    stdout: &mut io::Stdout,
    game: &Game,
    action: Option<Action>,
    game_over: bool,
    depth_label: &str,
) -> io::Result<()> {
    execute!(stdout, cursor::MoveTo(0, 0), terminal::Clear(ClearType::All))?;

    // Header
    write!(
        stdout,
        "2048 N-TUPLE{} — q: quit  r: new game  \u{2190}/\u{2192}: speed\r\n",
        depth_label,
    )?;
    write!(
        stdout,
        "Score: {:<10}  Max tile: {}\r\n",
        game.score,
        max_tile(game.board)
    )?;

    if let Some(a) = action {
        write!(stdout, "Action: {:?}\r\n", a)?;
    } else {
        write!(stdout, "\r\n")?;
    }

    // Board
    let separator = format!("+{}", format!("{:-<w$}+", "", w = CELL_W).repeat(4));
    write!(stdout, "\r\n{separator}\r\n")?;

    for r in 0..4 {
        write!(stdout, "|")?;
        for c in 0..4 {
            let exp = get_tile(game.board, r * 4 + c);
            let val = if exp == 0 { 0 } else { 1u32 << exp };
            let cell = if val == 0 {
                ".".to_string()
            } else {
                val.to_string()
            };
            let color = tile_color(val);
            let bold = tile_bold(val);

            execute!(stdout, SetForegroundColor(color))?;
            if bold {
                write!(stdout, "{}", Attribute::Bold)?;
            }
            write!(stdout, "{:^w$}", cell, w = CELL_W)?;
            if bold {
                write!(stdout, "{}", Attribute::Reset)?;
            }
            execute!(stdout, SetForegroundColor(Color::Reset))?;
            write!(stdout, "|")?;
        }
        write!(stdout, "\r\n{separator}\r\n")?;
    }

    if game_over {
        write!(
            stdout,
            "\r\nGAME OVER — Score: {}  Max tile: {}\r\n",
            game.score,
            max_tile(game.board)
        )?;
        write!(stdout, "Press r for new game, q to quit\r\n")?;
    }

    stdout.flush()
}

fn select_action_for_watch(
    game: &Game,
    network: &NTupleNetwork,
    depth: &Option<Depth>,
) -> Option<Action> {
    match depth {
        Some(d) => expectimax::expectimax_action(game.board, network, d),
        None => train::select_action(network, game.board).map(|(a, _)| a),
    }
}

fn run(
    stdout: &mut io::Stdout,
    game: &mut Game,
    network: &NTupleNetwork,
    mut delay_idx: usize,
    depth: &Option<Depth>,
    depth_label: &str,
) -> io::Result<()> {
    let mut action: Option<Action> = None;
    let mut game_over = false;

    loop {
        draw(stdout, game, action, game_over, depth_label)?;

        if !game_over {
            let delay_ms = (DELAY_STEPS[delay_idx] * 1000.0) as u64;
            if delay_ms > 0 {
                std::thread::sleep(Duration::from_millis(delay_ms));
            }

            if let Some(a) = select_action_for_watch(game, network, depth) {
                action = Some(a);
                game.step(a);
                game_over = select_action_for_watch(game, network, depth).is_none();
            } else {
                game_over = true;
            }
        }

        // Poll for key input (non-blocking when playing, blocking when game over)
        let timeout = if game_over {
            Duration::from_secs(60)
        } else {
            Duration::from_millis(1)
        };

        if event::poll(timeout)? {
            if let Event::Key(KeyEvent { code, .. }) = event::read()? {
                match code {
                    KeyCode::Char('q') => break,
                    KeyCode::Char('r') => {
                        game.reset();
                        action = None;
                        game_over = false;
                    }
                    KeyCode::Right => {
                        delay_idx = (delay_idx + 1).min(DELAY_STEPS.len() - 1);
                    }
                    KeyCode::Left => {
                        delay_idx = delay_idx.saturating_sub(1);
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(())
}

fn main() -> io::Result<()> {
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

    let depth_label = match &depth {
        None => String::new(),
        Some(Depth::Fixed(d)) => format!("  depth={d}"),
        Some(Depth::Adaptive(_)) => "  depth=adaptive".to_string(),
    };

    // Find closest delay step
    let delay_idx = DELAY_STEPS
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (*a - cli.delay).abs().partial_cmp(&(*b - cli.delay).abs()).unwrap()
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    let mut stdout = io::stdout();
    terminal::enable_raw_mode()?;
    execute!(stdout, cursor::Hide)?;

    let mut game = Game::new();
    game.reset();

    let result = run(&mut stdout, &mut game, &network, delay_idx, &depth, &depth_label);

    // Cleanup terminal
    execute!(stdout, cursor::Show)?;
    terminal::disable_raw_mode()?;
    println!();

    result
}
