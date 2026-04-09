/// Bitboard game engine for 2048.
///
/// Board is a u64 with 4 bits per tile storing the exponent:
///   0 = empty, 1 = 2, 2 = 4, ..., 15 = 32768
///
/// Layout (row-major, position 0 = top-left):
///   bits [0:3] = pos 0, bits [4:7] = pos 1, ..., bits [60:63] = pos 15
///
/// Rows are stored in 16-bit chunks: row 0 = bits [0:15], row 3 = bits [48:63].

use rand::{Rng, SeedableRng};
use std::sync::LazyLock;

pub type Board = u64;

/// Precomputed tables for all 65536 possible 16-bit rows.
struct MoveTables {
    left: Vec<u16>,
    right: Vec<u16>,
    score: Vec<u32>,
}

static TABLES: LazyLock<MoveTables> = LazyLock::new(build_tables);

fn build_tables() -> MoveTables {
    let mut tables = MoveTables {
        left: vec![0; 65536],
        right: vec![0; 65536],
        score: vec![0; 65536],
    };

    for row in 0u32..65536 {
        let tiles = [
            (row & 0xF) as u8,
            ((row >> 4) & 0xF) as u8,
            ((row >> 8) & 0xF) as u8,
            ((row >> 12) & 0xF) as u8,
        ];

        let (merged, score) = slide_left(tiles);
        let result =
            merged[0] as u16 | (merged[1] as u16) << 4 | (merged[2] as u16) << 8 | (merged[3] as u16) << 12;

        tables.left[row as usize] = result;
        tables.score[row as usize] = score;

        // Right = reverse, slide left, reverse
        let rev_row = reverse_row(row as u16);
        // We'll fill right table from the reversed perspective
        let rev_tiles = [
            (rev_row & 0xF) as u8,
            ((rev_row >> 4) & 0xF) as u8,
            ((rev_row >> 8) & 0xF) as u8,
            ((rev_row >> 12) & 0xF) as u8,
        ];
        let (rev_merged, _) = slide_left(rev_tiles);
        let rev_result = rev_merged[0] as u16
            | (rev_merged[1] as u16) << 4
            | (rev_merged[2] as u16) << 8
            | (rev_merged[3] as u16) << 12;
        tables.right[row as usize] = reverse_row(rev_result);
    }

    tables
}

#[inline]
fn reverse_row(row: u16) -> u16 {
    (row >> 12) | ((row >> 4) & 0x00F0) | ((row << 4) & 0x0F00) | (row << 12)
}

fn slide_left(tiles: [u8; 4]) -> ([u8; 4], u32) {
    // Remove zeros
    let mut compact = [0u8; 4];
    let mut ci = 0;
    for &t in &tiles {
        if t != 0 {
            compact[ci] = t;
            ci += 1;
        }
    }

    // Merge adjacent equal tiles
    let mut result = [0u8; 4];
    let mut score = 0u32;
    let mut ri = 0;
    let mut i = 0;
    while i < ci {
        if i + 1 < ci && compact[i] == compact[i + 1] {
            let merged = compact[i] + 1; // exponent increments by 1
            result[ri] = merged;
            score += 1 << merged; // actual tile value = 2^merged
            ri += 1;
            i += 2;
        } else {
            result[ri] = compact[i];
            ri += 1;
            i += 1;
        }
    }

    (result, score)
}

/// Transpose the board (swap rows and columns).
#[inline]
fn transpose(board: Board) -> Board {
    // Swap individual 4-bit nibbles to transpose the 4x4 grid.
    // Position (r,c) -> (c,r), i.e., bit offset r*16+c*4 -> c*16+r*4
    let mut result: Board = 0;
    for r in 0..4 {
        for c in 0..4 {
            let tile = (board >> (r * 16 + c * 4)) & 0xF;
            result |= tile << (c * 16 + r * 4);
        }
    }
    result
}

#[inline]
fn extract_row(board: Board, row: usize) -> u16 {
    ((board >> (row * 16)) & 0xFFFF) as u16
}

#[inline]
fn set_row(board: &mut Board, row: usize, val: u16) {
    let shift = row * 16;
    *board &= !(0xFFFF << shift);
    *board |= (val as Board) << shift;
}

/// Extract the tile exponent at a given position (0-15).
#[inline]
pub fn get_tile(board: Board, pos: usize) -> u8 {
    ((board >> (pos * 4)) & 0xF) as u8
}

/// Set the tile exponent at a given position (0-15).
#[inline]
pub fn set_tile(board: &mut Board, pos: usize, val: u8) {
    let shift = pos * 4;
    *board &= !(0xF << shift);
    *board |= (val as Board) << shift;
}

/// Actions: Up=0, Right=1, Down=2, Left=3
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Action {
    Up = 0,
    Right = 1,
    Down = 2,
    Left = 3,
}

impl Action {
    pub const ALL: [Action; 4] = [Action::Up, Action::Right, Action::Down, Action::Left];
}

/// Apply an action to the board. Returns (new_board, merge_score).
/// Does NOT spawn a new tile.
#[inline]
pub fn apply_action(board: Board, action: Action) -> (Board, f32) {
    let tables = &*TABLES;
    match action {
        Action::Left => apply_left(board, tables),
        Action::Right => apply_right(board, tables),
        Action::Up => {
            let t = transpose(board);
            let (result, score) = apply_left(t, tables);
            (transpose(result), score)
        }
        Action::Down => {
            let t = transpose(board);
            let (result, score) = apply_right(t, tables);
            (transpose(result), score)
        }
    }
}

#[inline]
fn apply_left(board: Board, tables: &MoveTables) -> (Board, f32) {
    let mut result: Board = 0;
    let mut score = 0u32;
    for r in 0..4 {
        let row = extract_row(board, r);
        let new_row = tables.left[row as usize];
        set_row(&mut result, r, new_row);
        score += tables.score[row as usize];
    }
    (result, score as f32)
}

#[inline]
fn apply_right(board: Board, tables: &MoveTables) -> (Board, f32) {
    let mut result: Board = 0;
    let mut score = 0u32;
    for r in 0..4 {
        let row = extract_row(board, r);
        let new_row = tables.right[row as usize];
        set_row(&mut result, r, new_row);
        // Score for right = score of reversed row slid left
        score += tables.score[reverse_row(row) as usize];
    }
    (result, score as f32)
}

/// Count empty cells on the board.
#[inline]
pub fn count_empty(board: Board) -> u32 {
    let mut count = 0;
    let mut b = board;
    for _ in 0..16 {
        if b & 0xF == 0 {
            count += 1;
        }
        b >>= 4;
    }
    count
}

/// Get the maximum tile value on the board.
pub fn max_tile(board: Board) -> u32 {
    let mut max_exp = 0u8;
    for pos in 0..16 {
        let exp = get_tile(board, pos);
        if exp > max_exp {
            max_exp = exp;
        }
    }
    if max_exp == 0 { 0 } else { 1 << max_exp }
}

/// Game state.
pub struct Game {
    pub board: Board,
    pub score: u32,
    rng: rand::rngs::SmallRng,
}

impl Game {
    pub fn new() -> Self {
        Self {
            board: 0,
            score: 0,
            rng: rand::rngs::SmallRng::from_os_rng(),
        }
    }

    pub fn reset(&mut self) {
        self.board = 0;
        self.score = 0;
        self.spawn_tile();
        self.spawn_tile();
    }

    /// Apply action. Returns reward. Spawns a tile if the board changed.
    pub fn step(&mut self, action: Action) -> f32 {
        let (new_board, reward) = apply_action(self.board, action);
        if new_board == self.board {
            return 0.0;
        }
        self.board = new_board;
        self.score += reward as u32;
        self.spawn_tile();
        reward
    }

    /// Spawn a 2 (90%) or 4 (10%) on a random empty cell.
    fn spawn_tile(&mut self) {
        let empty: u32 = count_empty(self.board);
        if empty == 0 {
            return;
        }
        let idx = self.rng.random_range(0..empty);
        let val: u8 = if self.rng.random::<f32>() < 0.9 { 1 } else { 2 }; // exponent: 1=2, 2=4

        // Find the idx-th empty cell
        let mut count = 0u32;
        for pos in 0..16 {
            if get_tile(self.board, pos) == 0 {
                if count == idx {
                    set_tile(&mut self.board, pos, val);
                    return;
                }
                count += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slide_left_basic() {
        // [2, 2, 0, 0] -> [4, 0, 0, 0], score = 4
        assert_eq!(slide_left([1, 1, 0, 0]), ([2, 0, 0, 0], 4));
    }

    #[test]
    fn test_slide_left_no_cascade() {
        // [2, 2, 2, 0] -> [4, 2, 0, 0], score = 4
        assert_eq!(slide_left([1, 1, 1, 0]), ([2, 1, 0, 0], 4));
    }

    #[test]
    fn test_slide_left_double_merge() {
        // [2, 2, 4, 4] -> [4, 8, 0, 0], score = 12
        assert_eq!(slide_left([1, 1, 2, 2]), ([2, 3, 0, 0], 12));
    }

    #[test]
    fn test_apply_action_left() {
        // Row 0: [2, 2, 0, 0] = exponents [1, 1, 0, 0]
        let board: Board = 0x0011; // row 0 = 0x0011
        let (new_board, score) = apply_action(board, Action::Left);
        // Expected: [4, 0, 0, 0] = exponent [2, 0, 0, 0] = 0x0002
        assert_eq!(extract_row(new_board, 0), 0x0002);
        assert_eq!(score, 4.0);
    }

    #[test]
    fn test_game_reset_spawns_two_tiles() {
        let mut game = Game::new();
        game.reset();
        let empty = count_empty(game.board);
        assert_eq!(empty, 14);
    }

    #[test]
    fn test_max_tile() {
        // board with a single tile of value 2048 (exponent 11) at position 0
        let board: Board = 11;
        assert_eq!(max_tile(board), 2048);
    }
}
