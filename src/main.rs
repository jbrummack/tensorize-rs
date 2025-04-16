// Dependencies for Cargo.toml:
// [dependencies]
// wgpu = "0.17"
// image = "0.24"
// pollster = "0.3"
// bytemuck = { version = "1.14", features = ["derive"] }
// futures-intrusive = "0.5"

use image::{DynamicImage, GenericImageView, ImageBuffer, Rgba};
use std::num::NonZeroU32;
use std::path::Path;
use wgpu::{ShaderModule, include_wgsl, util::DeviceExt};

pub mod image_resizer;

// Define the parameters for resizing
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ResizeParams {
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load the image
    let img_path = "image.png"; // Replace with your image path
    let img = image::open(img_path).expect("Failed to open image");
    let (input_width, input_height) = img.dimensions();

    // Target dimensions
    let output_width = 224u32;
    let output_height = 224u32;

    println!(
        "Resizing image from {}x{} to {}x{}",
        input_width, input_height, output_width, output_height
    );

    // Convert image to RGBA format
    let rgba_img = img.to_rgba8();
    let img_data = rgba_img.into_raw();

    // Initialize wgpu
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::default(),
        })
        .await?;

    // Create textures for input and output
    let texture_size = wgpu::Extent3d {
        width: input_width,
        height: input_height,
        depth_or_array_layers: 1,
    };

    let input_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Input Texture"),
        size: texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    // Create output texture
    let output_texture_size = wgpu::Extent3d {
        width: output_width,
        height: output_height,
        depth_or_array_layers: 1,
    };

    let output_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Output Texture"),
        size: output_texture_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    // Upload image data to the input texture
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &input_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &img_data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(NonZeroU32::new(4 * input_width).unwrap().into()),
            rows_per_image: Some(NonZeroU32::new(input_height).unwrap().into()),
        },
        texture_size,
    );

    // Create the resize parameters buffer
    let resize_params = ResizeParams {
        input_width,
        input_height,
        output_width,
        output_height,
    };
    let resize_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Resize Parameters Buffer"),
        contents: bytemuck::cast_slice(&[resize_params]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Create the compute shader
    let shader = create_catmull_rom_shader(&device);

    // Create bind group layout and bind group
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Resize Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Resize Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &input_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(
                    &output_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: resize_params_buffer.as_entire_binding(),
            },
        ],
    });

    // Create pipeline layout and compute pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Resize Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Resize Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Execute the compute shader
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Resize Command Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Resize Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups((output_width + 15) / 16, (output_height + 15) / 16, 1);
    }

    // Calculate bytes_per_row with proper alignment (256 bytes)
    // wgpu requires bytes_per_row to be aligned to COPY_BYTES_PER_ROW_ALIGNMENT (256)
    let align = 256;
    let bytes_per_pixel = 4; // RGBA8 = 4 bytes per pixel
    let unpadded_bytes_per_row = output_width * bytes_per_pixel;
    let padding = (align - unpadded_bytes_per_row % align) % align;
    let padded_bytes_per_row = unpadded_bytes_per_row + padding;

    // Create output buffer to retrieve the resized image data
    let output_buffer_size = padded_bytes_per_row as u64 * output_height as u64;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Copy the output texture to the output buffer
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &output_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &output_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(NonZeroU32::new(padded_bytes_per_row).unwrap().into()),
                rows_per_image: Some(NonZeroU32::new(output_height).unwrap().into()),
            },
        },
        output_texture_size,
    );

    queue.submit(std::iter::once(encoder.finish()));

    // Read back the output buffer
    let buffer_slice = output_buffer.slice(..);

    buffer_slice.map_async(wgpu::MapMode::Read, move |_| {});

    device.poll(wgpu::PollType::Wait)?;

    let data = buffer_slice.get_mapped_range();

    // Create a new vector to unpad the rows
    let mut result_rgba = Vec::with_capacity((output_width * output_height * 4) as usize);

    // Copy each row, removing padding
    for y in 0..output_height {
        let row_start = (y as usize * padded_bytes_per_row as usize);
        let row_end = row_start + (output_width as usize * 4);
        result_rgba.extend_from_slice(&data.as_ref()[row_start..row_end]);
    }

    drop(data);
    output_buffer.unmap();

    // Save the output image
    let output_image =
        ImageBuffer::<Rgba<u8>, _>::from_raw(output_width, output_height, result_rgba)
            .expect("Failed to create output image");

    let output_path = "output_224x224.png";
    output_image
        .save(output_path)
        .expect("Failed to save output image");
    println!("Image resized and saved to {}", output_path);

    Ok(())
}

fn create_catmull_rom_shader(device: &wgpu::Device) -> ShaderModule {
    device.create_shader_module(include_wgsl!("tensor.wgsl"))
}
