use tensorize_rs::{
    CpuTensorizer, GpuTensorizer, IMAGENET_DEFAULT_CONFIG_NO_CROP, ImageResizer, Tensorizer,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let resizer = ImageResizer::new(224, 224).await?;
    let img = image::open("image.png")?;
    resizer.rescale(&img, "rescaled.png").await?;

    let tensorizer = GpuTensorizer::new(IMAGENET_DEFAULT_CONFIG_NO_CROP).await?;
    let tensor = tensorizer.tensorize(&img).await?;
    println!("{tensor:?}");

    let cpu = CpuTensorizer::default_no_crop().await?;
    let cpu_tensor = cpu.tensorize(&img).await?;
    println!("{cpu_tensor:?}");
    Ok(())
}
