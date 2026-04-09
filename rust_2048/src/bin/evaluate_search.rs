use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use clap::Parser;
use rust_2048::board::{Action, Game, apply_action, max_tile};
use rust_2048::expectimax::{Depth, expectimax_action, parse_depth};
use rust_2048::ntuple::NTupleNetwork;

#[derive(Parser)]
#[command(
    name = "evaluate_search",
    about = "Evaluate N-tuple network performance across expectimax search depths"
)]
struct Cli {
    /// Path to checkpoint file
    checkpoint: String,

    /// Search depths to evaluate (integers, "adaptive", or custom schedules like "10:1,6:2,0:3")
    #[arg(long, num_args = 1.., default_values_t = vec!["0".to_string(), "1".to_string(), "2".to_string()])]
    depths: Vec<String>,

    /// Number of games per depth
    #[arg(long, default_value_t = 50)]
    games: u32,
}

/// Play one game. Returns (score, max_tile, avg_ms_per_move).
fn run_game(network: &NTupleNetwork, depth: &Depth) -> (u32, u32, f64) {
    let mut game = Game::new();
    game.reset();
    let mut total_time = 0.0_f64;
    let mut moves = 0u32;

    loop {
        // Check if any action changes the board
        let has_valid = Action::ALL
            .iter()
            .any(|&a| apply_action(game.board, a).0 != game.board);
        if !has_valid {
            break;
        }

        let t0 = Instant::now();
        let action = if matches!(depth, Depth::Fixed(0)) {
            greedy_action(network, game.board)
        } else {
            expectimax_action(game.board, network, depth)
        };
        total_time += t0.elapsed().as_secs_f64();

        if let Some(a) = action {
            game.step(a);
            moves += 1;
        } else {
            break;
        }
    }

    let avg_ms = (total_time / moves.max(1) as f64) * 1000.0;
    (game.score, max_tile(game.board), avg_ms)
}

/// Greedy action: pick the action with the highest reward + V(afterstate).
fn greedy_action(network: &NTupleNetwork, board: u64) -> Option<Action> {
    let mut best_action: Option<Action> = None;
    let mut best_value = f64::NEG_INFINITY;

    for &action in &Action::ALL {
        let (afterstate, reward) = apply_action(board, action);
        if afterstate == board {
            continue;
        }
        let value = reward as f64 + network.evaluate(afterstate) as f64;
        if value > best_value {
            best_value = value;
            best_action = Some(action);
        }
    }

    best_action
}

fn main() {
    let cli = Cli::parse();

    let network = NTupleNetwork::load(Path::new(&cli.checkpoint))
        .unwrap_or_else(|e| panic!("Failed to load checkpoint: {e}"));

    let depths: Vec<(String, Depth)> = cli
        .depths
        .iter()
        .map(|s| {
            let d = parse_depth(s).unwrap_or_else(|e| panic!("Invalid depth '{s}': {e}"));
            (s.clone(), d)
        })
        .collect();

    let mut results_summary: Vec<(String, f64, f64, f64)> = Vec::new();

    for (label, depth) in &depths {
        let mut scores = Vec::with_capacity(cli.games as usize);
        let mut max_tiles = Vec::with_capacity(cli.games as usize);
        let mut times_ms = Vec::with_capacity(cli.games as usize);

        let width = cli.games.to_string().len();
        println!("\n--- Depth {label} ({} games) ---", cli.games);

        for i in 0..cli.games {
            let (score, tile, avg_ms) = run_game(&network, depth);
            scores.push(score);
            max_tiles.push(tile);
            times_ms.push(avg_ms);
            println!(
                "  game {:>width$}/{}: score={:>7}  max={:>5}  avg={:>8.1}ms/move",
                i + 1,
                cli.games,
                score,
                tile,
                avg_ms,
            );
        }

        let mean_score = scores.iter().sum::<u32>() as f64 / scores.len() as f64;
        let median_score = {
            let mut sorted = scores.clone();
            sorted.sort();
            let n = sorted.len();
            if n % 2 == 0 {
                (sorted[n / 2 - 1] + sorted[n / 2]) as f64 / 2.0
            } else {
                sorted[n / 2] as f64
            }
        };
        let max_score = *scores.iter().max().unwrap();
        let avg_time = times_ms.iter().sum::<f64>() / times_ms.len() as f64;

        let mut tile_counts: HashMap<u32, u32> = HashMap::new();
        for &t in &max_tiles {
            *tile_counts.entry(t).or_insert(0) += 1;
        }

        println!("\nDepth {label} summary:");
        println!(
            "  score: mean={mean_score:.0}  median={median_score:.0}  max={max_score}",
        );
        println!("  avg time/move: {avg_time:.1}ms");
        println!("  max tile distribution:");

        let mut tiles_sorted: Vec<u32> = tile_counts.keys().copied().collect();
        tiles_sorted.sort_by(|a, b| b.cmp(a));
        for tile in tiles_sorted {
            let count = tile_counts[&tile];
            let pct = count as f64 / max_tiles.len() as f64 * 100.0;
            println!(
                "    {tile:>5}: {pct:>5.1}% ({count}/{})",
                max_tiles.len()
            );
        }

        results_summary.push((label.clone(), mean_score, median_score, avg_time));
    }

    if results_summary.len() > 1 {
        println!("\n=== Comparison ===");
        println!("{:>8} {:>12} {:>14} {:>12}", "Depth", "Mean Score", "Median Score", "Avg ms/move");
        for (label, mean_s, med_s, avg_t) in &results_summary {
            println!("{label:>8} {mean_s:>12.0} {med_s:>14.0} {avg_t:>12.1}");
        }
    }
}
