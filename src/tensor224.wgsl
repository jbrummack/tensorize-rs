fn catmull_rom_weight(x: f32) -> vec4<f32> {
    let abs_x = abs(x);
    let abs_x2 = abs_x * abs_x;
    let abs_x3 = abs_x2 * abs_x;

    return vec4<f32>(
        (-0.5 * abs_x3) + (abs_x2) - (0.5 * abs_x),       // weight for -1
        (1.5 * abs_x3) - (2.5 * abs_x2) + 1.0,            // weight for 0
        (-1.5 * abs_x3) + (2.0 * abs_x2) + (0.5 * abs_x), // weight for 1
        (0.5 * abs_x3) - (0.5 * abs_x2)                   // weight for 2
    );
}
