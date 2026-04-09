/// Expectimax search for 2048 with N-tuple network evaluation.
///
/// Builds a depth-limited game tree of alternating max (player move) and chance
/// (random tile spawn) nodes. Since N-tuple evaluation is just LUT lookups,
/// we evaluate inline during traversal (no batching needed).

use crate::board::{Action, Board, apply_action, count_empty, get_tile, set_tile};
use crate::ntuple::NTupleNetwork;

const PROB_SPAWN_2: f64 = 0.9;
const PROB_SPAWN_4: f64 = 0.1;

/// Adaptive depth schedule: maps board emptiness to search depth.
///
/// Thresholds are `(min_empty_cells, depth)` pairs checked in order.
/// Must be sorted descending by `min_empty_cells` and end with a 0 entry.
#[derive(Clone, Debug)]
pub struct DepthSchedule {
    thresholds: Vec<(u32, u32)>,
}

impl DepthSchedule {
    pub fn new(thresholds: Vec<(u32, u32)>) -> Self {
        Self { thresholds }
    }

    pub fn default_adaptive() -> Self {
        Self {
            thresholds: vec![(6, 1), (2, 2), (0, 3)],
        }
    }

    pub fn get_depth(&self, board: Board) -> u32 {
        let empty = count_empty(board);
        for &(min_empty, depth) in &self.thresholds {
            if empty >= min_empty {
                return depth;
            }
        }
        1
    }
}

/// Depth specification: either a fixed depth or an adaptive schedule.
#[derive(Clone, Debug)]
pub enum Depth {
    Fixed(u32),
    Adaptive(DepthSchedule),
}

impl Depth {
    pub fn resolve(&self, board: Board) -> u32 {
        match self {
            Depth::Fixed(d) => *d,
            Depth::Adaptive(schedule) => schedule.get_depth(board),
        }
    }
}

/// Parse a depth CLI argument.
///
/// Valid formats:
/// - An integer (e.g. "2") for fixed depth
/// - "adaptive" for the default schedule
/// - A custom schedule like "10:1,6:2,0:3"
pub fn parse_depth(value: &str) -> Result<Depth, String> {
    if value == "adaptive" {
        return Ok(Depth::Adaptive(DepthSchedule::default_adaptive()));
    }
    if value.contains(':') {
        let mut thresholds = Vec::new();
        for token in value.split(',') {
            let parts: Vec<&str> = token.split(':').collect();
            if parts.len() != 2 {
                return Err(format!("expected 'min_empty:depth', got '{token}'"));
            }
            let min_empty: u32 = parts[0].parse().map_err(|e| format!("{e}"))?;
            let depth: u32 = parts[1].parse().map_err(|e| format!("{e}"))?;
            thresholds.push((min_empty, depth));
        }
        return Ok(Depth::Adaptive(DepthSchedule::new(thresholds)));
    }
    let d: u32 = value.parse().map_err(|e| format!("{e}"))?;
    Ok(Depth::Fixed(d))
}

/// Select the best action using expectimax search.
pub fn expectimax_action(board: Board, network: &NTupleNetwork, depth: &Depth) -> Option<Action> {
    let search_depth = depth.resolve(board);
    let mut best_action: Option<Action> = None;
    let mut best_value = f64::NEG_INFINITY;

    for &action in &Action::ALL {
        let (afterstate, reward) = apply_action(board, action);
        if afterstate == board {
            continue;
        }
        let value = reward as f64 + chance_value(afterstate, network, search_depth);
        if value > best_value {
            best_value = value;
            best_action = Some(action);
        }
    }

    best_action
}

/// Expected value of a chance node (random tile spawn on afterstate).
fn chance_value(afterstate: Board, network: &NTupleNetwork, depth: u32) -> f64 {
    let empty = count_empty(afterstate);
    if empty == 0 {
        return 0.0;
    }
    let cell_prob = 1.0 / empty as f64;
    let mut expected = 0.0;

    for pos in 0..16 {
        if get_tile(afterstate, pos) != 0 {
            continue;
        }
        for (tile_exp, tile_prob) in [(1u8, PROB_SPAWN_2), (2u8, PROB_SPAWN_4)] {
            let mut board = afterstate;
            set_tile(&mut board, pos, tile_exp);
            let value = max_value(board, network, depth - 1);
            expected += cell_prob * tile_prob * value;
        }
    }

    expected
}

/// Value of a max node: best action's reward + chance value.
fn max_value(board: Board, network: &NTupleNetwork, depth: u32) -> f64 {
    if depth == 0 {
        return leaf_value(board, network);
    }

    let mut best = f64::NEG_INFINITY;
    let mut found = false;

    for &action in &Action::ALL {
        let (afterstate, reward) = apply_action(board, action);
        if afterstate == board {
            continue;
        }
        let value = reward as f64 + chance_value(afterstate, network, depth);
        if value > best {
            best = value;
            found = true;
        }
    }

    if found { best } else { 0.0 }
}

/// Evaluate a leaf board: max_a [reward(a) + V(afterstate(a))].
fn leaf_value(board: Board, network: &NTupleNetwork) -> f64 {
    let mut best = f64::NEG_INFINITY;
    let mut found = false;

    for &action in &Action::ALL {
        let (afterstate, reward) = apply_action(board, action);
        if afterstate == board {
            continue;
        }
        let value = reward as f64 + network.evaluate(afterstate) as f64;
        if value > best {
            best = value;
            found = true;
        }
    }

    if found { best } else { 0.0 }
}
