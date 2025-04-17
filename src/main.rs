use tensorize_rs::{
    CpuTensorizer, GpuTensorizer, IMAGENET_DEFAULT_CONFIG_NO_CROP, ImageResizer, Tensorizer,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let resizer = ImageResizer::new(224, 224).await?;
    let img = image::open("image.png")?;
    let img2 = image::open("image.jpg")?;
    resizer.rescale(&img, "rescaled.png").await?;

    let tensorizer = GpuTensorizer::new(IMAGENET_DEFAULT_CONFIG_NO_CROP).await?;
    let _ = tensorizer.tensorize(&img2).await?;
    let gpu_tensor = tensorizer.tensorize(&img).await?;
    println!("{gpu_tensor:?}");

    let cpu = CpuTensorizer::default_no_crop().await?;
    let _ = cpu.tensorize(&img2).await?;
    let cpu_tensor = cpu.tensorize(&img).await?;
    println!("{cpu_tensor:?}");

    let diff = &gpu_tensor - &cpu_tensor;
    let squared_sum: f32 = diff.iter().map(|x| x.powi(2)).sum();
    let distance = squared_sum.sqrt();

    let max_norm: f32 = diff.iter().map(|x| x.abs()).fold(0.0, |acc, x| acc.max(x));

    let sum_abs_diff: f32 = diff.iter().map(|x| x.abs()).sum();
    let count = diff.len();
    let mean_abs_diff = sum_abs_diff / count as f32;

    println!(
        "GPU-CPU diagnostics\nAvgError: {mean_abs_diff}\nMaxNorm: {max_norm}\nL2 Distance: {distance}"
    );
    Ok(())
}
