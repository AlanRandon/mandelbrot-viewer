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

pub fn main() !void {
    const allocator = std.heap.c_allocator;

    var device_id: c.cl_device_id = undefined;
    try checkClError(c.clGetDeviceIDs(null, c.CL_DEVICE_TYPE_ALL, 1, &device_id, null));

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
                    render(&raw_term, stdout, context, commands, &kernels, &out, allocator, &state) catch {};
                },
                '-' => {
                    state.zoom /= 2.0;
                    render(&raw_term, stdout, context, commands, &kernels, &out, allocator, &state) catch {};
                },
                'r', '0' => {
                    state = State{ .zoom = 1.0, .offset_x = 0, .offset_y = 0 };
                    render(&raw_term, stdout, context, commands, &kernels, &out, allocator, &state) catch {};
                },
                'h' => {
                    state.offset_x -= 0.5 / state.zoom;
                    render(&raw_term, stdout, context, commands, &kernels, &out, allocator, &state) catch {};
                },
                'j' => {
                    state.offset_y -= 0.5 / state.zoom;
                    render(&raw_term, stdout, context, commands, &kernels, &out, allocator, &state) catch {};
                },
                'k' => {
                    state.offset_y += 0.5 / state.zoom;
                    render(&raw_term, stdout, context, commands, &kernels, &out, allocator, &state) catch {};
                },
                'l' => {
                    state.offset_x += 0.5 / state.zoom;
                    render(&raw_term, stdout, context, commands, &kernels, &out, allocator, &state) catch {};
                },
                else => {},
            },
            .resize => {
                render(&raw_term, stdout, context, commands, &kernels, &out, allocator, &state) catch {};
            },
            else => {},
        }
    }
}
