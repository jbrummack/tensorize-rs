use cpu_tensor::IMAGENET_DEFAULT_CONFIG_NO_CROP;
use image_resizer::ImageResizer;
use image_tensor::Tensorizer;

pub mod cpu_tensor;
pub mod image_resizer;
pub mod image_tensor;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let resizer = ImageResizer::new(224, 224).await?;
    let img = image::open("image.png")?;
    resizer.rescale(&img, "rescaled.png").await?;

    let tensorizer =
        Tensorizer::new(224, 224, None, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).await?;
    let tensor = tensorizer.tensorize(&img).await?;
    println!("{tensor:?}");

    let cpu = IMAGENET_DEFAULT_CONFIG_NO_CROP;
    let cpu_tensor = cpu.ort_value3(&img)?;
    println!("{cpu_tensor:?}");
    Ok(())
}
