const builtin = @import("builtin");
const std = @import("std");

const Float = f64;
const max_iterations: usize = 1_000;

const Complex = struct {
    real: Float,
    imaginary: Float,

    fn square(z: *const Complex) Complex {
        return .{
            .real = z.real * z.real - z.imaginary * z.imaginary,
            .imaginary = z.real * z.imaginary * 2,
        };
    }

    fn add(lhs: *const Complex, rhs: *const Complex) Complex {
        return .{
            .real = rhs.real + lhs.real,
            .imaginary = rhs.imaginary + lhs.imaginary,
        };
    }

    fn magnitude(z: *const Complex) Float {
        // return std.math.hypot(z.real, z.imaginary);
        return std.math.sqrt(z.real * z.real + z.imaginary * z.imaginary);
    }
};

fn hsvToRgb(h: Float, s: Float, v: Float) @Vector(3, Float) {
    if (s == 0) {
        return .{ v, v, v };
    } else {
        var var_h = h * 6;
        if (var_h == 6) var_h = 0;
        const var_i = std.math.floor(var_h);
        const v1 = v * (1 - s);
        const v2 = v * (1 - s * (var_h - var_i));
        const v3 = v * (1 - s * (1 - var_h + var_i));

        switch (@as(usize, @intFromFloat(var_i))) {
            0 => return .{ v, v3, v1 },
            1 => return .{ v2, v, v1 },
            2 => return .{ v1, v, v3 },
            3 => return .{ v1, v2, v },
            4 => return .{ v3, v1, v },
            5 => return .{ v, v1, v2 },
            else => unreachable,
        }
    }
}

pub const Args = extern struct {
    upper_left_corner: [2]Float,
    delta_x: Float,
    delta_y: Float,
    width: u32,
};

pub fn mandelbrotKernel(
    image: [*]addrspace(.global) u8,
    args_ptr: *addrspace(.global) const Args,
) callconv(.spirv_kernel) void {
    const args = args_ptr.*;
    const x = @workGroupId(0) * @workGroupSize(0) + @workItemId(0);
    const y = @workGroupId(1) * @workGroupSize(1) + @workItemId(1);
    const index = (y * args.width + x) * 3;

    const bound = 2;

    var z = Complex{ .real = 0, .imaginary = 0 };
    const c = Complex{
        .real = @mulAdd(Float, @floatFromInt(x), args.delta_x, args.upper_left_corner[0]),
        .imaginary = @mulAdd(Float, @floatFromInt(y), args.delta_y, args.upper_left_corner[1]),
    };

    var iterations: usize = 1000;
    for (0..max_iterations) |i| {
        iterations = i;
        if (z.magnitude() >= bound) {
            break;
        } else {
            z = z.square().add(&c);
        }
    }

    if (iterations >= max_iterations - 1) {
        image[index] = 0;
        image[index + 1] = 0;
        image[index + 2] = 0;
    } else {
        const h = @mod(@mulAdd(
            Float,
            @as(Float, @floatFromInt(iterations)) / @as(Float, @floatFromInt(max_iterations)),
            3,
            0.52,
        ), 1);

        const rgb = hsvToRgb(h, 0.7, 0.7);
        const bytes: @Vector(3, u8) = @intFromFloat(@as(@Vector(3, Float), @splat(255)) * rgb);
        image[index] = bytes[0];
        image[index + 1] = bytes[1];
        image[index + 2] = bytes[2];
    }
}
