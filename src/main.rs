use cpu_tensor::{CpuTensorizer, IMAGENET_DEFAULT_CONFIG_NO_CROP};
use gpu_tensor::GpuTensorizer;
use image_resizer::ImageResizer;
use tensorizer_trait::Tensorizer;
pub mod cpu_tensor;
pub mod gpu_tensor;
pub mod image_resizer;
pub mod tensorizer_trait;

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
