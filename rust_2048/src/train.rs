/// Training loop for N-tuple network with TD(0) afterstate learning.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use crate::board::{Action, Board, Game, apply_action, max_tile};
use crate::ntuple::NTupleNetwork;

pub struct TrainConfig {
    pub lr: f32,
    pub max_episodes: u32,
    pub eval_interval: u32,
    pub eval_episodes: u32,
    pub v_init: f32,
    pub checkpoint_dir: String,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            lr: 0.0025,
            max_episodes: 100_000,
            eval_interval: 1_000,
            eval_episodes: 25,
            v_init: 0.0,
            checkpoint_dir: "checkpoints".into(),
        }
    }
}

/// Select the best action: argmax_a [reward(a) + V(afterstate(a))].
/// Returns (action, afterstate) or None if terminal.
#[inline]
pub fn select_action(network: &NTupleNetwork, board: Board) -> Option<(Action, Board)> {
    let mut best_action: Option<Action> = None;
    let mut best_afterstate: Board = 0;
    let mut best_value = f64::NEG_INFINITY;

    for &action in &Action::ALL {
        let (afterstate, reward) = apply_action(board, action);
        if afterstate == board {
            continue;
        }
        let value = reward as f64 + network.evaluate(afterstate) as f64;
        if value > best_value {
            best_action = Some(action);
            best_afterstate = afterstate;
            best_value = value;
        }
    }

    best_action.map(|a| (a, best_afterstate))
}

/// Compute max_a [reward(a) + V(afterstate(a))] from a board state.
/// Returns 0 if terminal (no valid actions).
#[inline]
fn best_afterstate_value(network: &NTupleNetwork, board: Board) -> f32 {
    let mut best = 0.0f64;
    let mut found = false;

    for &action in &Action::ALL {
        let (afterstate, reward) = apply_action(board, action);
        if afterstate == board {
            continue;
        }
        let value = reward as f64 + network.evaluate(afterstate) as f64;
        if !found || value > best {
            best = value;
            found = true;
        }
    }

    if found { best as f32 } else { 0.0 }
}

/// Run one episode with online TD(0) updates.
fn run_episode(game: &mut Game, network: &mut NTupleNetwork, lr: f32) {
    game.reset();

    loop {
        let Some((action, afterstate)) = select_action(network, game.board) else {
            break;
        };

        let current_value = network.evaluate(afterstate);

        game.step(action);

        let target = best_afterstate_value(network, game.board);
        let td_error = target - current_value;
        network.update(afterstate, lr * td_error);
    }
}

/// Evaluate the network over multiple episodes with greedy policy (no updates).
fn evaluate(game: &mut Game, network: &NTupleNetwork, num_episodes: u32) -> EvalResult {
    let mut scores = Vec::with_capacity(num_episodes as usize);
    let mut max_tiles = Vec::with_capacity(num_episodes as usize);

    for _ in 0..num_episodes {
        game.reset();
        loop {
            let Some((action, _)) = select_action(network, game.board) else {
                break;
            };
            game.step(action);
        }
        scores.push(game.score);
        max_tiles.push(max_tile(game.board));
    }

    let mean_score = scores.iter().map(|&s| s as f64).sum::<f64>() / scores.len() as f64;
    let max_score = *scores.iter().max().unwrap_or(&0);

    let mut tile_distribution: HashMap<u32, u32> = HashMap::new();
    for &t in &max_tiles {
        *tile_distribution.entry(t).or_insert(0) += 1;
    }

    EvalResult {
        mean_score,
        max_score,
        tile_distribution,
    }
}

struct EvalResult {
    mean_score: f64,
    max_score: u32,
    tile_distribution: HashMap<u32, u32>,
}

impl std::fmt::Display for EvalResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut tiles: Vec<_> = self.tile_distribution.iter().collect();
        tiles.sort_by_key(|(k, _)| *k);
        let tile_str: Vec<String> = tiles.iter().map(|(k, v)| format!("{k}: {v}")).collect();
        write!(
            f,
            "mean={:.0}  max={}  tiles={{{}}}",
            self.mean_score,
            self.max_score,
            tile_str.join(", ")
        )
    }
}

/// Extract the episode number from a checkpoint filename like "checkpoint_ep0100000.bin".
fn parse_checkpoint_episode(path: &str) -> Option<u32> {
    let filename = Path::new(path).file_stem()?.to_str()?;
    let ep_str = filename.strip_prefix("checkpoint_ep")?;
    ep_str.parse().ok()
}

/// Main training entry point.
pub fn train(config: &TrainConfig, resume: Option<&str>) {
    let mut game = Game::new();

    let start_episode = resume
        .and_then(parse_checkpoint_episode)
        .unwrap_or(0);

    let mut network = if let Some(path) = resume {
        eprintln!("Resuming from {path} (episode {start_episode})");
        NTupleNetwork::load(Path::new(path)).expect("Failed to load checkpoint")
    } else {
        NTupleNetwork::new(crate::ntuple::DEFAULT_PATTERNS, config.v_init)
    };

    let checkpoint_dir = Path::new(&config.checkpoint_dir);
    std::fs::create_dir_all(checkpoint_dir).ok();

    let end_episode = start_episode + config.max_episodes;
    let width = (end_episode as f64).log10().ceil() as usize;
    let start_time = Instant::now();

    eprintln!(
        "Training episodes {}..{}  lr={}  eval_interval={}  eval_episodes={}  v_init={}",
        start_episode + 1, end_episode, config.lr, config.eval_interval, config.eval_episodes, config.v_init
    );

    for episode in (start_episode + 1)..=end_episode {
        run_episode(&mut game, &mut network, config.lr);

        if episode % config.eval_interval == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let trained = episode - start_episode;
            let eps = trained as f64 / elapsed;
            let result = evaluate(&mut game, &network, config.eval_episodes);
            eprintln!(
                "EVAL ep {:>width$}  {}  [{:.1}s, {:.0} ep/s]",
                episode, result, elapsed, eps,
            );

            let filename = format!("checkpoint_ep{:0>width$}.bin", episode);
            let path = checkpoint_dir.join(filename);
            if let Err(e) = network.save(&path) {
                eprintln!("  Warning: failed to save checkpoint: {e}");
            }
        }
    }

    let final_path = checkpoint_dir.join("final.bin");
    network.save(&final_path).expect("Failed to save final checkpoint");

    let total_time = start_time.elapsed().as_secs_f64();
    eprintln!(
        "Training complete. {} episodes in {:.1}s ({:.0} ep/s). Saved to {:?}",
        config.max_episodes,
        total_time,
        config.max_episodes as f64 / total_time,
        final_path,
    );
}
