// catmull_rom.wgsl

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    mean: vec3<f32>,
    avg: vec3<f32>,
}

// Catmull-Rom interpolation weight calculation
fn catmull_rom_weight(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let t3 = t2 * t;

    // Catmull-Rom coefficients
    let w0 = -0.5 * t3 + t2 - 0.5 * t;
    let w1 = 1.5 * t3 - 2.5 * t2 + 1.0;
    let w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
    let w3 = 0.5 * t3 - 0.5 * t2;

    return vec4<f32>(w0, w1, w2, w3);
}

// Sample the input texture with safety bounds
fn safe_sample(x: i32, y: i32) -> vec4<f32> {
    let x_safe = clamp(x, 0, i32(params.input_width) - 1);
    let y_safe = clamp(y, 0, i32(params.input_height) - 1);
    return textureLoad(input_texture, vec2<i32>(x_safe, y_safe), 0);
}

// Perform Catmull-Rom filtering in 1D (horizontal or vertical)
fn catmull_rom_1d(p0: vec4<f32>, p1: vec4<f32>, p2: vec4<f32>, p3: vec4<f32>, t: f32) -> vec4<f32> {
    let weights = catmull_rom_weight(t);
    return p0 * weights.x + p1 * weights.y + p2 * weights.z + p3 * weights.w;
}

fn normalize_old(color: vec4<f32>, mean: vec3<f32>, avg: vec3<f32>) -> vec4<f32> {
    let r = (color.x - mean.x) / avg.x;
    let g = (color.y - mean.y) / avg.y;
    let b = (color.z - mean.z) / avg.z;
    let a = color.w;

    return vec4<f32>(r,g,b,a);
}

fn normalize(color: vec4<f32>, mean: vec3<f32>, avg: vec3<f32>) -> vec4<f32> {
    let rgb = (color.xyz - mean) / avg;
    return vec4<f32>(rgb, color.w);
}

// Bicubic Catmull-Rom interpolation
fn bicubic_catmull_rom(pos: vec2<f32>) -> vec4<f32> {
    // Calculate the integer coordinates and fractional offsets
    let x = floor(pos.x);
    let y = floor(pos.y);
    let fx = pos.x - x;
    let fy = pos.y - y;

    // Sample the 16 surrounding pixels
    let x_int = i32(x);
    let y_int = i32(y);

    // Row 0
    let p00 = safe_sample(x_int - 1, y_int - 1);
    let p10 = safe_sample(x_int,     y_int - 1);
    let p20 = safe_sample(x_int + 1, y_int - 1);
    let p30 = safe_sample(x_int + 2, y_int - 1);

    // Row 1
    let p01 = safe_sample(x_int - 1, y_int);
    let p11 = safe_sample(x_int,     y_int);
    let p21 = safe_sample(x_int + 1, y_int);
    let p31 = safe_sample(x_int + 2, y_int);

    // Row 2
    let p02 = safe_sample(x_int - 1, y_int + 1);
    let p12 = safe_sample(x_int,     y_int + 1);
    let p22 = safe_sample(x_int + 1, y_int + 1);
    let p32 = safe_sample(x_int + 2, y_int + 1);

    // Row 3
    let p03 = safe_sample(x_int - 1, y_int + 2);
    let p13 = safe_sample(x_int,     y_int + 2);
    let p23 = safe_sample(x_int + 1, y_int + 2);
    let p33 = safe_sample(x_int + 2, y_int + 2);

    // Interpolate horizontally first
    let row0 = catmull_rom_1d(p00, p10, p20, p30, fx);
    let row1 = catmull_rom_1d(p01, p11, p21, p31, fx);
    let row2 = catmull_rom_1d(p02, p12, p22, p32, fx);
    let row3 = catmull_rom_1d(p03, p13, p23, p33, fx);

    // Then interpolate vertically
    return catmull_rom_1d(row0, row1, row2, row3, fy);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Check if within output bounds
    if (global_id.x >= params.output_width || global_id.y >= params.output_height) {
        return;
    }

    // Calculate the sample position in the input texture
    let scale_x = f32(params.input_width) / f32(params.output_width);
    let scale_y = f32(params.input_height) / f32(params.output_height);

    // Add 0.5 to sample from pixel centers
    let input_pos = vec2<f32>(
        (f32(global_id.x) + 0.5) * scale_x,
        (f32(global_id.y) + 0.5) * scale_y
    );

    // Get the interpolated color using Catmull-Rom
    let color = bicubic_catmull_rom(input_pos);
    let normalized = normalize(color, params.mean, params.avg);
    // Write the result to the output texture
    textureStore(output_texture, vec2<i32>(global_id.xy), normalized);
    //textureStore(output_texture, vec2<i32>(global_id.xy), color);
}
