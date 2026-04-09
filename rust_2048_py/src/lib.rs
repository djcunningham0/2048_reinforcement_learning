use std::path::Path;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use rust_2048::board::{self, Board};
use rust_2048::expectimax;
use rust_2048::ntuple::NTupleNetwork;
use rust_2048::train;

/// Convert a Python board (list of 16 tile values: 0, 2, 4, ..., 32768)
/// to a Rust bitboard (u64 with 4-bit exponents per cell).
#[pyfunction]
fn board_from_python(tiles: Vec<u32>) -> PyResult<u64> {
    if tiles.len() != 16 {
        return Err(PyValueError::new_err("board must have 16 tiles"));
    }
    let mut board: Board = 0;
    for (pos, &val) in tiles.iter().enumerate() {
        if val > 0 {
            let exp = val.trailing_zeros() as u8;
            board::set_tile(&mut board, pos, exp);
        }
    }
    Ok(board)
}

/// Convert a Rust bitboard back to a list of 16 tile values.
#[pyfunction]
fn board_to_python(board: u64) -> Vec<u32> {
    (0..16)
        .map(|pos| {
            let exp = board::get_tile(board, pos);
            if exp == 0 { 0 } else { 1u32 << exp }
        })
        .collect()
}

#[pyclass]
struct RustNTupleNetwork {
    inner: NTupleNetwork,
}

#[pymethods]
impl RustNTupleNetwork {
    /// Load a network from a .bin checkpoint file.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let net = NTupleNetwork::load(Path::new(path))
            .map_err(|e| PyValueError::new_err(format!("failed to load checkpoint: {e}")))?;
        Ok(Self { inner: net })
    }

    /// Evaluate a bitboard, returning the network's value estimate.
    fn evaluate(&self, board: u64) -> f32 {
        self.inner.evaluate(board)
    }

    /// Greedy action selection: argmax_a [reward(a) + V(afterstate(a))].
    /// Returns the action as an int (0=Up, 1=Right, 2=Down, 3=Left), or None if terminal.
    fn select_action(&self, board: u64) -> Option<u8> {
        train::select_action(&self.inner, board).map(|(a, _)| a as u8)
    }

    /// Expectimax action selection with the given depth string.
    /// Depth formats: "0" (greedy), "2" (fixed), "adaptive", "10:1,6:2,0:3" (custom).
    /// Returns the action as an int, or None if terminal.
    fn expectimax_action(&self, py: Python<'_>, board: u64, depth: &str) -> PyResult<Option<u8>> {
        let parsed = expectimax::parse_depth(depth)
            .map_err(|e| PyValueError::new_err(format!("invalid depth: {e}")))?;
        let result = py.allow_threads(|| expectimax::expectimax_action(board, &self.inner, &parsed));
        Ok(result.map(|a| a as u8))
    }
}

#[pymodule]
fn rust_2048_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(board_from_python, m)?)?;
    m.add_function(wrap_pyfunction!(board_to_python, m)?)?;
    m.add_class::<RustNTupleNetwork>()?;
    Ok(())
}
