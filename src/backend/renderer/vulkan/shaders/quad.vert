#version 450

// Shared push-constant block, see `PushConstants` in the renderer.
//
// Position and UV transforms are 2D affine transforms split into a 2x2 linear
// part and a translation to fit the guaranteed 128-byte push constant limit.
layout(push_constant) uniform PushConstants {
    vec4 mat_pos;        // position matrix linear part: m00, m01, m10, m11
    vec4 pos_off_rect;   // position translation x, y; rect offset x, y
    vec4 rect_size_misc; // rect size w, h; alpha; tint
    vec4 mat_uv;         // uv matrix linear part: m00, m01, m10, m11
    vec4 uv_off;         // uv translation x, y; unused
    vec4 color;          // solid color (premultiplied)
} data;

layout(location = 0) out vec2 v_coords;

void main() {
    // Triangle strip: (0,0), (1,0), (0,1), (1,1)
    vec2 corner = vec2(gl_VertexIndex & 1, gl_VertexIndex >> 1);
    vec2 pos = corner * data.rect_size_misc.xy + data.pos_off_rect.zw;

    vec2 ndc = vec2(
        data.mat_pos.x * pos.x + data.mat_pos.y * pos.y + data.pos_off_rect.x,
        data.mat_pos.z * pos.x + data.mat_pos.w * pos.y + data.pos_off_rect.y
    );
    v_coords = vec2(
        data.mat_uv.x * pos.x + data.mat_uv.y * pos.y + data.uv_off.x,
        data.mat_uv.z * pos.x + data.mat_uv.w * pos.y + data.uv_off.y
    );
    gl_Position = vec4(ndc, 0.0, 1.0);
}
