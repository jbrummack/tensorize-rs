use image::DynamicImage;
use ndarray::{Array3, Array4, Dim};

use crate::tensorizer_trait::Tensorizer;

pub const IMAGENET_DEFAULT_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_DEFAULT_STD: [f32; 3] = [0.229, 0.224, 0.225];
pub const IMAGENET_DEFAULT_CONFIG: ImageConvert = ImageConvert {
    channels: 3,
    width: 256,
    height: 256,
    crop: 224,
    mean: IMAGENET_DEFAULT_MEAN,
    std: IMAGENET_DEFAULT_STD,
    interpolation: image::imageops::FilterType::CatmullRom,
};

pub const IMAGENET_DEFAULT_CONFIG_NO_CROP: ImageConvert = ImageConvert {
    channels: 3,
    width: 224,
    height: 224,
    crop: 224,
    mean: IMAGENET_DEFAULT_MEAN,
    std: IMAGENET_DEFAULT_STD,
    interpolation: image::imageops::FilterType::CatmullRom,
};

pub struct ImageConvert {
    //pub batches: u16,
    pub channels: u8,
    pub width: u32,
    pub height: u32,
    pub crop: u16, //central crop of the image
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub interpolation: image::imageops::FilterType,
}

pub struct CpuTensorizer {
    conv: ImageConvert,
}

impl Tensorizer for CpuTensorizer {
    type BuildType = CpuTensorizer;

    async fn new(config: crate::cpu_tensor::ImageConvert) -> anyhow::Result<Self::BuildType> {
        Ok(CpuTensorizer { conv: config })
    }

    async fn tensorize(&self, image: &DynamicImage) -> anyhow::Result<ndarray::Array3<f32>> {
        self.conv.ort_value3(image)
    }

    async fn tensorize_batch(&self, image: &DynamicImage) -> anyhow::Result<ndarray::Array4<f32>> {
        self.conv.ort_value(image)
    }
}

impl ImageConvert {
    //#[cfg(feature = "ort")]
    fn ort_value(&self, image: &DynamicImage) -> anyhow::Result<Array4<f32>> {
        let normalized_data = self.create_data(image);
        let tensor_shape: [usize; 4] = [
            1,
            self.channels as usize,
            self.crop as usize,
            self.crop as usize,
        ];
        // let tensor_args = (tensor_shape, normalized_data);
        let input_array =
            ndarray::Array4::<f32>::from_shape_vec(Dim(tensor_shape), normalized_data)?; //ort::value::Tensor::from_array(tensor_args)?;
        Ok(input_array)
    }

    fn ort_value3(&self, image: &DynamicImage) -> anyhow::Result<Array3<f32>> {
        let normalized_data = self.create_data(image);
        let tensor_shape: [usize; 3] = [
            self.channels as usize,
            self.crop as usize,
            self.crop as usize,
        ];
        // let tensor_args = (tensor_shape, normalized_data);
        let input_array =
            ndarray::Array3::<f32>::from_shape_vec(Dim(tensor_shape), normalized_data)?; //ort::value::Tensor::from_array(tensor_args)?;
        Ok(input_array)
    }

    fn create_data(&self, image: &DynamicImage) -> Vec<f32> {
        let resized = image.resize_exact(self.width, self.height, self.interpolation);

        // Central crop to 224 x 224
        let (width, height) = (resized.width(), resized.height());
        let crop_x = (width - 224) / 2;
        let crop_y = (height - 224) / 2;
        let cropped = resized.crop_imm(crop_x, crop_y, 224, 224);

        // Convert to f32 array and rescale to [0.0, 1.0]
        let cropped_rgb = cropped.to_rgb8();
        let (width, height) = (cropped.width(), cropped.height());

        let mut data = Vec::new();

        for pixel in cropped_rgb.pixels() {
            data.push(pixel[0] as f32 / 255.0); // R
            data.push(pixel[1] as f32 / 255.0); // G
            data.push(pixel[2] as f32 / 255.0); // B
        }

        // Reshape into [3, 224, 224] and normalize
        let mean = self.mean;
        let std = self.std;

        let mut normalized_data = vec![0.0; data.len()];
        for c in 0..3 {
            for i in 0..(width * height) as usize {
                normalized_data[c * (width * height) as usize + i] =
                    (data[i * 3 + c] - mean[c]) / std[c];
            }
        }
        normalized_data
    }
}
