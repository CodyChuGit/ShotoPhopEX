//! Hold `--gb` GB of the card until killed: makes a 32 GB card behave like a smaller one for a
//! memory check (`vram_hold --gb 20` leaves a 5090 about 12 GB).
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    gb: f64,
}

fn main() -> Result<()> {
    let a = Args::parse();
    let dev = Device::new_cuda(0)?;
    let n = (a.gb * (1u64 << 30) as f64 / 4.0) as usize;
    let t = Tensor::ones(n, DType::F32, &dev)?;
    dev.synchronize()?;
    println!("holding {:.1} GB; free/total {:?}", a.gb, qwen_image_rs::device::free_vram(&dev).map(|(f, t)| (f >> 20, t >> 20)));
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
        let _ = &t;
    }
}
