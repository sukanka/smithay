#version 450

layout(push_constant) uniform PushConstants {
    vec4 mat_pos;
    vec4 pos_off_rect;
    vec4 rect_size_misc; // rect size w, h; alpha; tint
    vec4 mat_uv;
    vec4 uv_off;
    vec4 color;
} data;

layout(set = 0, binding = 0) uniform sampler2D tex;

layout(location = 0) in vec2 v_coords;
layout(location = 0) out vec4 out_color;

void main() {
    // Opaque formats are handled by an image view swizzle forcing alpha to one,
    // so a plain multiply matches both GLES texture shader variants.
    vec4 color = texture(tex, v_coords) * data.rect_size_misc.z;

    if (data.rect_size_misc.w == 1.0) {
        color = vec4(0.0, 0.2, 0.0, 0.2) + color * 0.8;
    }

    out_color = color;
}
