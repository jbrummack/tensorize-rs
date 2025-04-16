use anyhow::Ok;
use bytemuck::{Pod, Zeroable};
use image::RgbaImage;
use std::num::NonZeroU32;
use wgpu::include_wgsl;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScaleUniform([f32; 2]);

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Hello, world!");
    run().await?;
    Ok(())
}

async fn run() -> anyhow::Result<()> {
    //SETUP CTX
    let img = image::open("image.png")?.to_rgba8();
    let (src_width, src_height) = img.dimensions();

    let dst_width = src_width / 2; //224;
    let dst_height = src_height / 2; //224;

    let instance = wgpu::Instance::default();
    let adapter = instance.request_adapter(&Default::default()).await?;
    let (device, queue) = adapter.request_device(&Default::default()).await?;

    let shader_src = include_wgsl!("resize.wgsl");
    let shader = device.create_shader_module(shader_src);

    //LOAD TEXTURE

    let src_texture = device.create_texture_with_data(
        &queue,
        &wgpu::TextureDescriptor {
            label: Some("Source texture"),
            size: wgpu::Extent3d {
                width: src_width,
                height: src_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::wgt::TextureDataOrder::LayerMajor,
        &img,
    );

    let dst_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Destination texture"),
        size: wgpu::Extent3d {
            width: dst_width,
            height: dst_height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let scale = ScaleUniform([
        src_width as f32 / dst_width as f32,
        src_height as f32 / dst_height as f32,
    ]);
    let scale_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Scale Buffer"),
        contents: bytemuck::cast_slice(&[scale]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let src_view = src_texture.create_view(&Default::default());
    let dst_view = dst_texture.create_view(&Default::default());

    // Bind group layout
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
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
        label: Some("Bind Group Layout"),
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&src_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&dst_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: scale_buf.as_entire_binding(),
            },
        ],
        label: Some("Bind Group"),
    });

    // Pipeline
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Resize Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Encode compute pass
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut compute_pass = encoder.begin_compute_pass(&Default::default());
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups((dst_width + 7) / 8, (dst_height + 7) / 8, 1);
    }

    // Read back
    let buffer_size = (dst_width * dst_height * 4) as wgpu::BufferAddress;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
        label: Some("Output Buffer"),
    });

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfoBase {
            texture: &dst_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfoBase {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(NonZeroU32::new(dst_width * 4).unwrap().into()),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: dst_width,
            height: dst_height,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(Some(encoder.finish()));

    // Read back and save
    let buffer_slice = output_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::Wait)?;
    let data = buffer_slice.get_mapped_range().to_vec();

    let img = RgbaImage::from_raw(dst_width, dst_height, data).unwrap();
    img.save("resized.png").unwrap();

    Ok(())
}
