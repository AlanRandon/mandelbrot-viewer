const std = @import("std");

pub const Args = extern struct {
    width: u32,
    height: u32,
};

pub const unit_str = "\x1B[48;2;255;000;000;38;2;000;255;000m▄";

fn digitToChar(digit: u8) u8 {
    return digit + '0';
}

fn componentToBytes(component: u8) [3]u8 {
    return .{
        digitToChar(component / 100),
        digitToChar(component / 10 % 10),
        digitToChar(component % 10),
    };
}

fn colorToBytes(color: [3]u8) [3 * 3 + 2]u8 {
    const result = componentToBytes(color[0]) ++ ";" ++
        componentToBytes(color[1]) ++ ";" ++
        componentToBytes(color[2]);
    return result.*;
}

pub fn pixelsToAnsiKernel(
    ansi_image: [*]addrspace(.global) u8,
    image: [*]addrspace(.global) const u8,
    args_ptr: *addrspace(.global) const Args,
) callconv(.spirv_kernel) void {
    const args = args_ptr.*;
    const x = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
    const y = @workGroupId(1) * @workGroupSize(1) + @workItemId(1);

    const top_y = y << 1;

    const top_color = (image + (top_y * args.width + x) * 3)[0..3];
    const bottom_color = (image + ((top_y + 1) * args.width + x) * 3)[0..3];

    const buf: *const [unit_str.len]u8 =
        "\x1B[48;2;" ++
        colorToBytes(top_color.*) ++
        ";38;2;" ++
        colorToBytes(bottom_color.*) ++
        "m▄";

    @memcpy((ansi_image + (y * args.width + x) * unit_str.len)[0..unit_str.len], buf);
}
