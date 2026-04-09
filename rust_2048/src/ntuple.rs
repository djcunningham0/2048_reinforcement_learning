/// N-tuple network with lookup tables and 8-fold symmetry.
///
/// Each pattern is a tuple of board positions. For each pattern, we maintain a
/// LUT of size 16^tuple_size. The board value is the sum of all LUT lookups
/// across all patterns and all 8 symmetries.

use crate::board::{Board, get_tile};
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Default 6-tuple patterns from Szubert & Jaskowski 2014.
pub const DEFAULT_PATTERNS: &[&[usize]] = &[
    &[0, 1, 2, 3, 4, 5],
    &[4, 5, 6, 7, 8, 9],
    &[4, 5, 6, 8, 9, 10],
    &[8, 9, 10, 12, 13, 14],
];

/// 8-fold symmetry permutations of a 4x4 grid (4 rotations x 2 flip states).
/// Each permutation maps original position -> transformed position.
const SYMMETRIES: [[usize; 16]; 8] = build_symmetries();

const fn build_symmetries() -> [[usize; 16]; 8] {
    let mut syms = [[0usize; 16]; 8];
    let mut si = 0;
    let mut rotate = 0;
    while rotate < 4 {
        let mut flip = 0;
        while flip < 2 {
            let mut pos = 0;
            while pos < 16 {
                let mut r = pos / 4;
                let mut c = pos % 4;
                // Apply rotation (CW)
                let mut rot = 0;
                while rot < rotate {
                    let tmp = r;
                    r = c;
                    c = 3 - tmp;
                    rot += 1;
                }
                // Apply horizontal flip
                if flip == 1 {
                    c = 3 - c;
                }
                syms[si][pos] = r * 4 + c;
                pos += 1;
            }
            si += 1;
            flip += 1;
        }
        rotate += 1;
    }
    syms
}

pub struct NTupleNetwork {
    /// Flat LUT array (all patterns concatenated).
    luts: Vec<f32>,
    /// Offset into `luts` for each pattern.
    offsets: Vec<usize>,
    /// Precomputed symmetry-transformed pattern positions.
    /// Shape: [num_patterns][8][tuple_size]
    sym_patterns: Vec<Vec<[usize; 8]>>,
    /// Powers of 16 for index computation: [16^(n-1), ..., 1]
    powers: Vec<Vec<usize>>,
    /// Original patterns (for save/load).
    patterns: Vec<Vec<usize>>,
    num_patterns: usize,
}

impl NTupleNetwork {
    pub fn new(patterns: &[&[usize]], v_init: f32) -> Self {
        let num_patterns = patterns.len();
        let tuple_size = patterns[0].len();

        // Compute LUT sizes and offsets
        let lut_size: usize = 16usize.pow(tuple_size as u32);
        let total_size = num_patterns * lut_size;
        let offsets: Vec<usize> = (0..num_patterns).map(|i| i * lut_size).collect();

        // Initialize LUTs
        let per_weight = if v_init != 0.0 {
            v_init / (num_patterns as f32 * 8.0)
        } else {
            0.0
        };
        let luts = vec![per_weight; total_size];

        // Precompute symmetry-transformed patterns
        // sym_patterns[pattern_idx][tuple_pos] = [sym0_pos, sym1_pos, ..., sym7_pos]
        let mut sym_patterns = Vec::with_capacity(num_patterns);
        for pattern in patterns {
            let mut positions = Vec::with_capacity(tuple_size);
            for &p in *pattern {
                let mut sym_pos = [0usize; 8];
                for (s, sym) in SYMMETRIES.iter().enumerate() {
                    sym_pos[s] = sym[p];
                }
                positions.push(sym_pos);
            }
            sym_patterns.push(positions);
        }

        // Powers of 16
        let mut powers = Vec::with_capacity(num_patterns);
        for pattern in patterns {
            let n = pattern.len();
            let p: Vec<usize> = (0..n).map(|k| 16usize.pow((n - 1 - k) as u32)).collect();
            powers.push(p);
        }

        let owned_patterns: Vec<Vec<usize>> = patterns.iter().map(|p| p.to_vec()).collect();

        Self {
            luts,
            offsets,
            sym_patterns,
            powers,
            patterns: owned_patterns,
            num_patterns,
        }
    }

    /// Evaluate a board by summing LUT lookups across all patterns and symmetries.
    #[inline]
    pub fn evaluate(&self, board: Board) -> f32 {
        // Extract all 16 tile exponents once
        let indices = extract_indices(board);
        let mut total = 0.0f32;

        for i in 0..self.num_patterns {
            let offset = self.offsets[i];
            let pattern_positions = &self.sym_patterns[i];
            let pattern_powers = &self.powers[i];

            for s in 0..8 {
                let mut lut_index = 0usize;
                for (k, (positions, &power)) in
                    pattern_positions.iter().zip(pattern_powers.iter()).enumerate()
                {
                    let _ = k;
                    lut_index += indices[positions[s]] as usize * power;
                }
                total += unsafe { *self.luts.get_unchecked(offset + lut_index) };
            }
        }

        total
    }

    /// Add delta to every LUT entry accessed when evaluating this board.
    #[inline]
    pub fn update(&mut self, board: Board, delta: f32) {
        let indices = extract_indices(board);

        for i in 0..self.num_patterns {
            let offset = self.offsets[i];
            let pattern_positions = &self.sym_patterns[i];
            let pattern_powers = &self.powers[i];

            for s in 0..8 {
                let mut lut_index = 0usize;
                for (positions, &power) in pattern_positions.iter().zip(pattern_powers.iter()) {
                    lut_index += indices[positions[s]] as usize * power;
                }
                unsafe {
                    *self.luts.get_unchecked_mut(offset + lut_index) += delta;
                }
            }
        }
    }

    /// Save network to a gzip-compressed binary file.
    ///
    /// Format (after decompression):
    ///   [num_patterns: u32] [tuple_sizes: u32 x N] [patterns: u32 x sum(sizes)]
    ///   [lut_data: f32 x total_entries]
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = File::create(path)?;
        let mut w = GzEncoder::new(BufWriter::new(file), Compression::fast());

        // Header
        w.write_all(&(self.num_patterns as u32).to_le_bytes())?;
        for pattern in &self.patterns {
            w.write_all(&(pattern.len() as u32).to_le_bytes())?;
            for &pos in pattern {
                w.write_all(&(pos as u32).to_le_bytes())?;
            }
        }

        // LUT data
        for &val in &self.luts {
            w.write_all(&val.to_le_bytes())?;
        }

        w.finish()?;
        Ok(())
    }

    /// Load network from a binary file (auto-detects gzip vs raw).
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mut buf_reader = BufReader::new(file);

        // Peek at first two bytes to detect gzip magic number (0x1f 0x8b)
        let mut magic = [0u8; 2];
        buf_reader.read_exact(&mut magic)?;

        let mut data = Vec::new();
        if magic == [0x1f, 0x8b] {
            // Gzip: chain the magic bytes back and decompress
            let chained = std::io::Cursor::new(magic).chain(buf_reader);
            GzDecoder::new(chained).read_to_end(&mut data)?;
        } else {
            // Raw: chain the magic bytes back and read as-is
            data.extend_from_slice(&magic);
            buf_reader.read_to_end(&mut data)?;
        }

        let mut r = &data[..];
        let mut buf4 = [0u8; 4];

        r.read_exact(&mut buf4)?;
        let num_patterns = u32::from_le_bytes(buf4) as usize;

        let mut patterns: Vec<Vec<usize>> = Vec::with_capacity(num_patterns);
        for _ in 0..num_patterns {
            r.read_exact(&mut buf4)?;
            let tuple_size = u32::from_le_bytes(buf4) as usize;
            let mut pattern = Vec::with_capacity(tuple_size);
            for _ in 0..tuple_size {
                r.read_exact(&mut buf4)?;
                pattern.push(u32::from_le_bytes(buf4) as usize);
            }
            patterns.push(pattern);
        }

        let pattern_refs: Vec<&[usize]> = patterns.iter().map(|p| p.as_slice()).collect();
        let mut network = Self::new(&pattern_refs, 0.0);

        // Read LUT data
        let total_size = network.luts.len();
        let mut lut_bytes = vec![0u8; total_size * 4];
        r.read_exact(&mut lut_bytes)?;
        for (i, chunk) in lut_bytes.chunks_exact(4).enumerate() {
            network.luts[i] = f32::from_le_bytes(chunk.try_into().unwrap());
        }

        Ok(network)
    }
}

/// Extract all 16 tile exponents from a board into an array.
#[inline]
fn extract_indices(board: Board) -> [u8; 16] {
    let mut indices = [0u8; 16];
    for pos in 0..16 {
        indices[pos] = get_tile(board, pos);
    }
    indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetries_identity() {
        // First symmetry should be identity
        assert_eq!(
            SYMMETRIES[0],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        );
    }

    #[test]
    fn test_symmetries_count() {
        assert_eq!(SYMMETRIES.len(), 8);
    }

    #[test]
    fn test_evaluate_zero_board() {
        let network = NTupleNetwork::new(DEFAULT_PATTERNS, 0.0);
        assert_eq!(network.evaluate(0), 0.0);
    }

    #[test]
    fn test_update_changes_value() {
        let mut network = NTupleNetwork::new(DEFAULT_PATTERNS, 0.0);
        let board: Board = 0x1234_0000_0000_0000; // some non-zero board
        assert_eq!(network.evaluate(board), 0.0);
        network.update(board, 1.0);
        assert!(network.evaluate(board) > 0.0);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let mut network = NTupleNetwork::new(DEFAULT_PATTERNS, 0.0);
        // Set some values
        let board: Board = 0x1111_2222_3333_4444;
        network.update(board, 5.0);
        let val_before = network.evaluate(board);

        let path = Path::new("/tmp/test_ntuple.bin");
        network.save(path).unwrap();
        let loaded = NTupleNetwork::load(path).unwrap();
        let val_after = loaded.evaluate(board);

        assert!((val_before - val_after).abs() < 1e-4);
        std::fs::remove_file(path).ok();
    }
}
