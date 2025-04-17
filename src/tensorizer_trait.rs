use image::DynamicImage;
use ndarray::{Array3, Array4};

use crate::cpu_tensor::{IMAGENET_DEFAULT_CONFIG, IMAGENET_DEFAULT_CONFIG_NO_CROP, ImageConvert};

pub trait Tensorizer {
    type BuildType;
    fn new(
        config: ImageConvert,
    ) -> impl std::future::Future<Output = anyhow::Result<Self::BuildType>>;
    fn default() -> impl std::future::Future<Output = anyhow::Result<Self::BuildType>> {
        Self::new(IMAGENET_DEFAULT_CONFIG)
    }
    fn default_no_crop() -> impl std::future::Future<Output = anyhow::Result<Self::BuildType>> {
        Self::new(IMAGENET_DEFAULT_CONFIG_NO_CROP)
    }
    fn tensorize(
        &self,
        image: &DynamicImage,
    ) -> impl std::future::Future<Output = anyhow::Result<Array3<f32>>>;
    fn tensorize_batch(
        &self,
        image: &DynamicImage,
    ) -> impl std::future::Future<Output = anyhow::Result<Array4<f32>>>;
}
