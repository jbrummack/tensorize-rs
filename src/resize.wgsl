@group(0) @binding(0)
var src_tex: texture_2d<f32>;

@group(0) @binding(1)
var dst_tex: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(2)
var<uniform> scale: vec2<f32>;

fn catmull_rom_weight(x: f32) -> vec4<f32> {
    let x2 = x * x;
    let x3 = x2 * x;

    return vec4<f32>(
        (-0.5 * x3) + x2 - (0.5 * x),
        (1.5 * x3) - (2.5 * x2) + 1.0,
        (-1.5 * x3) + (2.0 * x2) + (0.5 * x),
        (0.5 * x3) - (0.5 * x2)
    );
}

fn catmull_rom_sample(pos: vec2<f32>) -> vec4<f32> {
    let base = floor(pos - 0.5);
    let f = pos - base - 0.5;
    let wx = catmull_rom_weight(f.x);
    let wy = catmull_rom_weight(f.y);

    var color = vec4<f32>(0.0);

    for (var j = 0; j < 4; j = j + 1) {
        for (var i = 0; i < 4; i = i + 1) {
            let coord = vec2<i32>(base) + vec2<i32>(i, j);
            let sample = textureLoad(src_tex, coord, 0);
            color += sample * wx[i] * wy[j];
        }
    }

    return color;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_size = textureDimensions(dst_tex);
    if (gid.x >= dst_size.x || gid.y >= dst_size.y) {
        return;
    }

    let src_pos = vec2<f32>(gid.xy) / scale;
    let out_color = catmull_rom_sample(src_pos);
    textureStore(dst_tex, vec2<i32>(gid.xy), out_color);
}
