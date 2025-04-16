use image_resizer::ImageResizer;

pub mod image_resizer;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let resizer = ImageResizer::new(224, 224).await?;
    let img = image::open("image.png")?;
    resizer.rescale(&img, "rescaled.png").await?;
    Ok(())
}
