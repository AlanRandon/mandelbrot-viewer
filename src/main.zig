const std = @import("std");
const c = @import("c.zig");
const checkClError = @import("error.zig").checkClError;
const RawTerm = @import("RawTerm");
const pixelsToAnsi = @import("kernels/pixelsToAnsi.zig");
const mandelbrot = @import("kernels/mandelbrot.zig");

const State = struct {
    zoom: f32,
    offset_x: f32,
    offset_y: f32,
};

pub fn render(
    raw_term: *RawTerm,
    stdout: std.fs.File,
    context: c.cl_context,
    commands: c.cl_command_queue,
    kernels: anytype,
    out: *[]u8,
    allocator: std.mem.Allocator,
    state: *const State,
) !void {
    const size = try raw_term.size();
    const width: u32 = size.width;
    const height: u32 = size.height - 1;

    var err: c.cl_int = undefined;
    const image_cl_buf = c.clCreateBuffer(
        context,
        c.CL_MEM_READ_WRITE,
        width * 2 * height * 3,
        null,
        &err,
    ) orelse {
        try checkClError(err);
        unreachable;
    };

    {
        const width_f: f32 = @floatFromInt(width);
        const height_f: f32 = @floatFromInt(height);
        const min_axis = @min(width_f, height_f * 2);
        const center: @Vector(2, f32) = .{ state.offset_x, state.offset_y };

        const delta = 1.0 / state.zoom / min_axis;

        var args = mandelbrot.Args{
            .upper_left_corner = center + @Vector(2, f32){ -delta, delta } * @Vector(2, f32){ width_f / 2, height_f },
            .delta_x = delta,
            .delta_y = -delta,
            .width = width,
        };

        const args_cl_buf = c.clCreateBuffer(
            context,
            c.CL_MEM_READ_ONLY | c.CL_MEM_USE_HOST_PTR,
            @sizeOf(mandelbrot.Args),
            @ptrCast(&args),
            &err,
        ) orelse {
            try checkClError(err);
            unreachable;
        };

        try checkClError(c.clSetKernelArg(kernels.mandelbrot, 0, @sizeOf(c.cl_mem), @ptrCast(&image_cl_buf)));
        try checkClError(c.clSetKernelArg(kernels.mandelbrot, 1, @sizeOf(c.cl_mem), @ptrCast(&args_cl_buf)));

        var cl_dimensions: [2]usize = .{ width, height * 2 };
        try checkClError(c.clEnqueueNDRangeKernel(commands, kernels.mandelbrot, 2, null, @constCast(&cl_dimensions), null, 0, null, null));
        try checkClError(c.clFinish(commands));
    }

    {
        var args = pixelsToAnsi.Args{
            .width = width,
            .height = height,
        };

        const args_cl_buf = c.clCreateBuffer(
            context,
            c.CL_MEM_READ_ONLY | c.CL_MEM_USE_HOST_PTR,
            @sizeOf(pixelsToAnsi.Args),
            @ptrCast(&args),
            &err,
        ) orelse {
            try checkClError(err);
            unreachable;
        };

        const out_len = @as(usize, width) * @as(usize, height) * pixelsToAnsi.unit_str.len;
        if (out_len > out.len) {
            out.* = try allocator.realloc(out.*, out_len);
        }

        const out_cl_buf = c.clCreateBuffer(
            context,
            c.CL_MEM_WRITE_ONLY | c.CL_MEM_HOST_READ_ONLY,
            out_len,
            null,
            &err,
        ) orelse {
            try checkClError(err);
            unreachable;
        };

        try checkClError(c.clSetKernelArg(kernels.pixelsToAnsi, 0, @sizeOf(c.cl_mem), @ptrCast(&out_cl_buf)));
        try checkClError(c.clSetKernelArg(kernels.pixelsToAnsi, 1, @sizeOf(c.cl_mem), @ptrCast(&image_cl_buf)));
        try checkClError(c.clSetKernelArg(kernels.pixelsToAnsi, 2, @sizeOf(c.cl_mem), @ptrCast(&args_cl_buf)));

        var cl_dimensions: [2]usize = .{ width, height };
        try checkClError(c.clEnqueueNDRangeKernel(commands, kernels.pixelsToAnsi, 2, null, @constCast(&cl_dimensions), null, 0, null, null));
        try checkClError(c.clFinish(commands));
        try checkClError(c.clEnqueueReadBuffer(commands, out_cl_buf, c.CL_TRUE, 0, out_len, out.ptr, 0, null, null));
        try checkClError(c.clFinish(commands));

        try stdout.writer().print("\x1B[H{s}\x1B[0m", .{out.*[0..out_len]});
    }

    try stdout.writer().print("\n\r\x1B[2KZoom: {} Offset: ({}, {})", .{ state.zoom, state.offset_x, state.offset_y });
}

fn deviceName(device: c.cl_device_id, allocator: std.mem.Allocator) ![]u8 {
    var name_len: usize = undefined;
    try checkClError(c.clGetDeviceInfo(device, c.CL_DEVICE_NAME, 0, null, &name_len));

    const name = try allocator.alloc(u8, name_len);
    errdefer allocator.free(name);

    try checkClError(c.clGetDeviceInfo(device, c.CL_DEVICE_NAME, name_len, name.ptr, null));

    return name;
}

fn deviceHasRequiredExtensions(device: c.cl_device_id, allocator: std.mem.Allocator) !bool {
    var extensions_len: usize = undefined;
    try checkClError(c.clGetDeviceInfo(device, c.CL_DEVICE_EXTENSIONS, 0, null, &extensions_len));

    const extensions = try allocator.alloc(u8, extensions_len);
    defer allocator.free(extensions);

    try checkClError(c.clGetDeviceInfo(device, c.CL_DEVICE_EXTENSIONS, extensions_len, extensions.ptr, null));

    return std.mem.indexOf(u8, extensions, "cl_khr_il_program") != null;
}

fn getClDeviceOnPlatform(allocator: std.mem.Allocator, platform_id: c.cl_platform_id, index: ?usize) !?c.cl_device_id {
    var num_devices: c.cl_uint = undefined;
    try checkClError(c.clGetDeviceIDs(platform_id, c.CL_DEVICE_TYPE_ALL, 0, null, &num_devices));

    const device_ids = try allocator.alloc(c.cl_device_id, num_devices);
    defer allocator.free(device_ids);

    try checkClError(c.clGetDeviceIDs(platform_id, c.CL_DEVICE_TYPE_ALL, num_devices, device_ids.ptr, null));

    if (index) |i| {
        if (i >= device_ids.len) {
            return error.ClDeviceIndexOutOfRange;
        }

        const id = device_ids[i];

        if (!try deviceHasRequiredExtensions(id, allocator)) {
            return error.ClDeviceMissingExtensions;
        }

        return id;
    } else {
        for (device_ids) |id| {
            if (try deviceHasRequiredExtensions(id, allocator)) {
                return id;
            }
        }

        return null;
    }
}

fn getClDevice(allocator: std.mem.Allocator) !c.cl_device_id {
    var num_platforms: c.cl_uint = undefined;
    try checkClError(c.clGetPlatformIDs(0, null, &num_platforms));

    const platform_ids = try allocator.alloc(c.cl_platform_id, num_platforms);
    defer allocator.free(platform_ids);

    try checkClError(c.clGetPlatformIDs(num_platforms, platform_ids.ptr, null));

    var env_map = try std.process.getEnvMap(allocator);
    defer env_map.deinit();

    const device_index = if (env_map.get("CL_DEVICE")) |index_str| try std.fmt.parseInt(usize, index_str, 10) else null;
    const platform_index = if (env_map.get("CL_PLATFORM")) |index_str| try std.fmt.parseInt(usize, index_str, 10) else null;

    if (platform_index) |i| {
        if (i >= platform_ids.len) {
            return error.ClPlatformIndexOutOfRange;
        }

        const platform_id = platform_ids[i];
        if (try getClDeviceOnPlatform(allocator, platform_id, device_index)) |id| {
            return id;
        }
    } else {
        for (platform_ids) |platform_id| {
            if (try getClDeviceOnPlatform(allocator, platform_id, device_index)) |id| {
                return id;
            }
        }
    }

    return error.MissingClDevice;
}

pub fn main() !void {
    const allocator = std.heap.c_allocator;

    const device_id = try getClDevice(allocator);

    var err: c.cl_int = undefined;
    const context = c.clCreateContext(0, 1, &device_id, null, null, &err) orelse {
        try checkClError(err);
        unreachable;
    };

    const commands = c.clCreateCommandQueue(context, device_id, 0, &err) orelse {
        try checkClError(err);
        unreachable;
    };

    const il = @embedFile("kernels.spv");
    const program = c.clCreateProgramWithIL(context, il.ptr, il.len, &err) orelse {
        try checkClError(err);
        unreachable;
    };

    try checkClError(c.clBuildProgram(program, 0, null, null, null, null));

    const kernels = .{
        .mandelbrot = c.clCreateKernel(program, "mandelbrotKernel", &err) orelse {
            try checkClError(err);
            unreachable;
        },
        .pixelsToAnsi = c.clCreateKernel(program, "pixelsToAnsiKernel", &err) orelse {
            try checkClError(err);
            unreachable;
        },
    };

    const stdin = std.io.getStdIn();
    const stdout = std.io.getStdOut();

    var raw_term = try RawTerm.enable(stdin, false);
    defer raw_term.disable() catch {};

    var listener = try raw_term.eventListener(allocator);
    defer listener.deinit();

    try stdout.writeAll("\x1B[?1049h\x1B[?25l"); // enter alternate screen, hide cursor
    defer stdout.writeAll("\x1B[?1049l\x1B[?25h") catch {}; // exit alternate screen, show cursor

    var out = try allocator.alloc(u8, 80 * 40 * pixelsToAnsi.unit_str.len);
    defer allocator.free(out);

    var state = State{ .zoom = 0.5, .offset_x = 0, .offset_y = 0 };

    try render(&raw_term, stdout, context, commands, &kernels, &out, allocator, &state);

    while (true) {
        const event = try listener.queue.wait();
        switch (event) {
            .char => |char| switch (char.value) {
                'q' => break,
                '+' => {
                    state.zoom *= 2.0;
                    try render(&raw_term, stdout, context, commands, &kernels, &out, allocator, &state);
                },
                '-' => {
                    state.zoom /= 2.0;
                    try render(&raw_term, stdout, context, commands, &kernels, &out, allocator, &state);
                },
                'r', '0' => {
                    state = State{ .zoom = 1.0, .offset_x = 0, .offset_y = 0 };
                    try render(&raw_term, stdout, context, commands, &kernels, &out, allocator, &state);
                },
                'h' => {
                    state.offset_x -= 0.5 / state.zoom;
                    try render(&raw_term, stdout, context, commands, &kernels, &out, allocator, &state);
                },
                'j' => {
                    state.offset_y -= 0.5 / state.zoom;
                    try render(&raw_term, stdout, context, commands, &kernels, &out, allocator, &state);
                },
                'k' => {
                    state.offset_y += 0.5 / state.zoom;
                    try render(&raw_term, stdout, context, commands, &kernels, &out, allocator, &state);
                },
                'l' => {
                    state.offset_x += 0.5 / state.zoom;
                    try render(&raw_term, stdout, context, commands, &kernels, &out, allocator, &state);
                },
                else => {},
            },
            .resize => {
                try render(&raw_term, stdout, context, commands, &kernels, &out, allocator, &state);
            },
            else => {},
        }
    }
}
