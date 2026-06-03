use tracing_subscriber::{EnvFilter, fmt};

pub fn init() {
    let filter = std::env::var("FERRIX_LOG")
        .or_else(|_| std::env::var("RUST_LOG"))
        .unwrap_or_else(|_| "warn".to_string());

    let env_filter = EnvFilter::try_new(filter).unwrap_or_else(|_| EnvFilter::new("warn"));

    let _ = fmt()
        .with_env_filter(env_filter)
        .with_writer(std::io::stderr)
        .try_init();
}
