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
    let _ = tensorizer.tensorize(&img).await?;
    let tensor = tensorizer.tensorize(&img2).await?;
    println!("{tensor:?}");

    let cpu = CpuTensorizer::default_no_crop().await?;
    let _ = cpu.tensorize(&img).await?;
    let cpu_tensor = cpu.tensorize(&img2).await?;
    println!("{cpu_tensor:?}");
    Ok(())
}
