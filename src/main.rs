mod bbox;
mod request_handler;

use anyhow::Result;
use clap::Parser;
use hyper::{server::conn::http1, service::service_fn};
use hyper_util::rt::TokioIo;
use request_handler::handle_request;
use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
    thread,
};
use tokio::net::TcpListener;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Address to listen on. Default 127.0.0.1:3003
    #[arg(short, long, default_value_t = SocketAddr::from(([127, 0, 0, 1], 3003)))]
    listen_address: SocketAddr,

    /// Raster file
    #[arg(short, long, required = true)]
    source: Vec<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Create a dedicated Tokio runtime for Dataset tasks.
    let dataset_runtime = Arc::new(
        tokio::runtime::Builder::new_current_thread()
            // .worker_threads(1)
            .max_blocking_threads(thread::available_parallelism()?.into())
            .enable_all()
            .on_thread_stop(|| {
                println!("thread stopping");
            })
            .on_thread_start(|| {
                println!("thread starting");
            })
            .build()?,
    );

    let sources: &'static [&Path] = Box::leak(
        Box::leak(args.source.into_boxed_slice())
            .iter()
            .map(|pb| pb.as_path())
            .collect::<Vec<&Path>>()
            .into_boxed_slice(),
    );

    let listener = TcpListener::bind(args.listen_address).await?;

    loop {
        let (stream, _) = listener.accept().await?;

        let io = TokioIo::new(stream);

        let pool = dataset_runtime.clone();

        let sfn = service_fn(move |req| {
            let pool = pool.clone();

            async move { handle_request(pool, req, sources).await }
        });

        tokio::spawn(async move {
            if let Err(err) = http1::Builder::new().serve_connection(io, sfn).await {
                eprintln!("Error serving connection: {err:?}");
            }
        });
    }
}
