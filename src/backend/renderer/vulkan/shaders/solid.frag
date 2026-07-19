#version 450

layout(push_constant) uniform PushConstants {
    vec4 mat_pos;
    vec4 pos_off_rect;
    vec4 rect_size_misc;
    vec4 mat_uv;
    vec4 uv_off;
    vec4 color;
} data;

layout(location = 0) in vec2 v_coords;
layout(location = 0) out vec4 out_color;

void main() {
    out_color = data.color;
}
