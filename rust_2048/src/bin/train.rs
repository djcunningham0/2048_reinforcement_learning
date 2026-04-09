use clap::Parser;
use rust_2048::train::TrainConfig;

#[derive(Parser)]
#[command(name = "rust_2048", about = "N-tuple network training for 2048")]
struct Cli {
    /// Learning rate for TD updates
    #[arg(long, default_value_t = 0.0025)]
    lr: f32,

    /// Maximum number of training episodes
    #[arg(long, default_value_t = 1_000_000)]
    max_episodes: u32,

    /// Episodes between evaluations
    #[arg(long, default_value_t = 100_000)]
    eval_interval: u32,

    /// Number of episodes per evaluation
    #[arg(long, default_value_t = 50)]
    eval_episodes: u32,

    /// Optimistic initialization value (0 = disabled)
    #[arg(long, default_value_t = 0.0)]
    v_init: f32,

    /// Directory for saving checkpoints
    #[arg(long, default_value = "checkpoints")]
    checkpoint_dir: String,

    /// Path to a checkpoint file to resume training from
    #[arg(long)]
    resume: Option<String>,
}

fn main() {
    let cli = Cli::parse();

    let config = TrainConfig {
        lr: cli.lr,
        max_episodes: cli.max_episodes,
        eval_interval: cli.eval_interval,
        eval_episodes: cli.eval_episodes,
        v_init: cli.v_init,
        checkpoint_dir: cli.checkpoint_dir,
    };

    rust_2048::train::train(&config, cli.resume.as_deref());
}
