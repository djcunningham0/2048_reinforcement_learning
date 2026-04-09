# Rust implementation of 2048 reinforcement learning

This directory contains a Rust implementation of the 2048 game engine and N-tuple network training and inference process.
The code was adapted from the Python code in `../rl_2048/` and `../scripts/`, and all Rust code was written by Claude Code.

## Example

Train an N-tuple network agent (run from this directory):

```
cargo run --release --bin train -- --max-episodes 100000
```

Checkpoints are written to `checkpoints/` by default. See `--help` for all options.
