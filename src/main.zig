const std = @import("std");
const c = @import("c.zig");
const checkClError = @import("error.zig").checkClError;
const RawTerm = @import("RawTerm");
const pixelsToAnsi = @import("kernels/pixelsToAnsi.zig");
const mandelbrot = @import("kernels/mandelbrot.zig");
const ansi = RawTerm.ansi;
const getClDevice = @import("device.zig").getClDevice;

const State = struct {
    zoom: f32,
    offset_x: f32,
    offset_y: f32,
};

const Renderer = struct {
    context: c.cl_context,
    commands: c.cl_command_queue,
    kernels: Kernels,
    buffers: Buffers,

    const bytes_per_pixel = 3;
    const pixels_per_cell = 2; // upper and lower
    const ansi_bytes_per_cell = pixelsToAnsi.unit_str.len;

    const Kernels = struct {
        mandelbrot: c.cl_kernel,
        pixels_to_ansi: c.cl_kernel,
    };

    const Buffers = struct {
        mandelbrot_args: c.cl_mem,
        pixels_to_ansi_args: c.cl_mem,
        image: c.cl_mem,
        ansi_image: c.cl_mem,
        cell_count: usize,

        fn resize(buffers: *Buffers, context: c.cl_context, width: u32, height: u32) !void {
            var err: c.cl_int = undefined;

            const cell_count = width * height;
            buffers.cell_count = cell_count;

            try checkClError(c.clReleaseMemObject(buffers.image));
            buffers.image = c.clCreateBuffer(
                context,
                c.CL_MEM_READ_WRITE,
                cell_count * pixels_per_cell * bytes_per_pixel,
                null,
                &err,
            ) orelse {
                try checkClError(err);
                unreachable;
            };

            try checkClError(c.clReleaseMemObject(buffers.ansi_image));
            buffers.ansi_image = c.clCreateBuffer(
                context,
                c.CL_MEM_WRITE_ONLY | c.CL_MEM_HOST_READ_ONLY,
                cell_count * ansi_bytes_per_cell,
                null,
                &err,
            ) orelse {
                try checkClError(err);
                unreachable;
            };
        }
    };

    fn init(
        context: c.cl_context,
        commands: c.cl_command_queue,
    ) !Renderer {
        var err: c.cl_int = undefined;
        const il = @embedFile("kernels.spv");
        const program = c.clCreateProgramWithIL(context, il.ptr, il.len, &err) orelse {
            try checkClError(err);
            unreachable;
        };

        try checkClError(c.clBuildProgram(program, 0, null, null, null, null));

        const kernels = Kernels{
            .mandelbrot = c.clCreateKernel(program, "mandelbrotKernel", &err) orelse {
                try checkClError(err);
                unreachable;
            },
            .pixels_to_ansi = c.clCreateKernel(program, "pixelsToAnsiKernel", &err) orelse {
                try checkClError(err);
                unreachable;
            },
        };

        const initial_cells = 80 * 40;

        const buffers = Buffers{
            .cell_count = initial_cells,
            .mandelbrot_args = c.clCreateBuffer(
                context,
                c.CL_MEM_READ_ONLY | c.CL_MEM_HOST_WRITE_ONLY,
                @sizeOf(mandelbrot.Args),
                null,
                &err,
            ) orelse {
                try checkClError(err);
                unreachable;
            },
            .pixels_to_ansi_args = c.clCreateBuffer(
                context,
                c.CL_MEM_READ_ONLY | c.CL_MEM_HOST_WRITE_ONLY,
                @sizeOf(pixelsToAnsi.Args),
                null,
                &err,
            ) orelse {
                try checkClError(err);
                unreachable;
            },
            .image = c.clCreateBuffer(
                context,
                c.CL_MEM_READ_WRITE,
                initial_cells * bytes_per_pixel * pixels_per_cell,
                null,
                &err,
            ) orelse {
                try checkClError(err);
                unreachable;
            },
            .ansi_image = c.clCreateBuffer(
                context,
                c.CL_MEM_WRITE_ONLY | c.CL_MEM_HOST_READ_ONLY,
                initial_cells * ansi_bytes_per_cell,
                null,
                &err,
            ) orelse {
                try checkClError(err);
                unreachable;
            },
        };

        return .{
            .context = context,
            .commands = commands,
            .kernels = kernels,
            .buffers = buffers,
        };
    }

    pub fn deinit(renderer: *Renderer) void {
        checkClError(c.clReleaseMemObject(renderer.buffers.mandelbrot_args)) catch unreachable;
        checkClError(c.clReleaseMemObject(renderer.buffers.pixels_to_ansi_args)) catch unreachable;
        checkClError(c.clReleaseMemObject(renderer.buffers.image)) catch unreachable;
        checkClError(c.clReleaseMemObject(renderer.buffers.ansi_image)) catch unreachable;
    }

    pub fn render(renderer: *Renderer, size: RawTerm.Size, state: *const State) ![]const u8 {
        const width: u32 = size.width;
        const height: u32 = size.height - 1;

        const cell_count = @as(usize, width) * @as(usize, height);
        const ansi_image_len = cell_count * ansi_bytes_per_cell;
        if (cell_count > renderer.buffers.cell_count) {
            try renderer.buffers.resize(renderer.context, width, height);
        }

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

            try checkClError(c.clEnqueueWriteBuffer(
                renderer.commands,
                renderer.buffers.mandelbrot_args,
                c.CL_TRUE,
                0,
                @sizeOf(mandelbrot.Args),
                &args,
                0,
                null,
                null,
            ));
        }

        try checkClError(c.clSetKernelArg(renderer.kernels.mandelbrot, 0, @sizeOf(c.cl_mem), @ptrCast(&renderer.buffers.image)));
        try checkClError(c.clSetKernelArg(renderer.kernels.mandelbrot, 1, @sizeOf(c.cl_mem), @ptrCast(&renderer.buffers.mandelbrot_args)));

        {
            var cl_dimensions: [2]usize = .{ width, height * 2 };
            try checkClError(c.clEnqueueNDRangeKernel(renderer.commands, renderer.kernels.mandelbrot, 2, null, @constCast(&cl_dimensions), null, 0, null, null));
            try checkClError(c.clFinish(renderer.commands));
        }

        {
            var args = pixelsToAnsi.Args{
                .width = width,
                .height = height,
            };

            try checkClError(c.clEnqueueWriteBuffer(
                renderer.commands,
                renderer.buffers.pixels_to_ansi_args,
                c.CL_TRUE,
                0,
                @sizeOf(pixelsToAnsi.Args),
                &args,
                0,
                null,
                null,
            ));
        }

        try checkClError(c.clSetKernelArg(renderer.kernels.pixels_to_ansi, 0, @sizeOf(c.cl_mem), @ptrCast(&renderer.buffers.ansi_image)));
        try checkClError(c.clSetKernelArg(renderer.kernels.pixels_to_ansi, 1, @sizeOf(c.cl_mem), @ptrCast(&renderer.buffers.image)));
        try checkClError(c.clSetKernelArg(renderer.kernels.pixels_to_ansi, 2, @sizeOf(c.cl_mem), @ptrCast(&renderer.buffers.pixels_to_ansi_args)));

        {
            var cl_dimensions: [2]usize = .{ width, height };
            try checkClError(c.clEnqueueNDRangeKernel(renderer.commands, renderer.kernels.pixels_to_ansi, 2, null, @constCast(&cl_dimensions), null, 0, null, null));
            try checkClError(c.clFinish(renderer.commands));
        }

        var err: c.cl_int = undefined;
        const ptr: [*]const u8 = @ptrCast(c.clEnqueueMapBuffer(
            renderer.commands,
            renderer.buffers.ansi_image,
            c.CL_TRUE,
            c.CL_MAP_READ,
            0,
            ansi_image_len,
            0,
            null,
            null,
            &err,
        ) orelse {
            try checkClError(err);
            unreachable;
        });
        const buf = ptr[0..ansi_image_len];
        errdefer renderer.unmapBuffer(buf);

        return buf;
    }

    fn unmapBuffer(renderer: *Renderer, buf: []const u8) void {
        const ptr = @constCast(buf.ptr);
        checkClError(c.clEnqueueUnmapMemObject(renderer.commands, renderer.buffers.ansi_image, ptr, 0, null, null)) catch unreachable;
    }

    fn display(renderer: *Renderer, raw_term: *RawTerm, size: RawTerm.Size, state: *const State) !void {
        const image = try renderer.render(size, state);
        defer renderer.unmapBuffer(image);

        try raw_term.out.writer().print(
            ansi.cursor.goto_top_left ++ "{s}" ++ ansi.style.reset ++ "\n\r" ++ ansi.clear.line ++ "Zoom: {} Offset: ({}, {})",
            .{
                image,
                state.zoom,
                state.offset_x,
                state.offset_y,
            },
        );
    }
};

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

    var raw_term = try RawTerm.enable(std.io.getStdIn(), std.io.getStdOut(), false);
    defer raw_term.disable() catch {};

    var listener = try raw_term.eventListener(allocator);
    defer listener.deinit();

    try raw_term.out.writeAll(ansi.alternate_screen.enable ++ ansi.cursor.hide);
    defer raw_term.out.writeAll(ansi.alternate_screen.disable ++ ansi.cursor.show) catch {};

    var state = State{ .zoom = 0.5, .offset_x = 0, .offset_y = 0 };

    var size = try raw_term.size();

    var renderer = try Renderer.init(context, commands);
    defer renderer.deinit();

    try renderer.display(&raw_term, size, &state);

    while (true) {
        const event = try listener.queue.wait();
        switch (event) {
            .char => |char| switch (char.value) {
                'q' => break,
                '+' => {
                    state.zoom *= 2.0;
                    try renderer.display(&raw_term, size, &state);
                },
                '-' => {
                    state.zoom /= 2.0;
                    try renderer.display(&raw_term, size, &state);
                },
                '0' => {
                    state.zoom = 1.0;
                    try renderer.display(&raw_term, size, &state);
                },
                'r' => {
                    state = State{ .zoom = 1.0, .offset_x = 0, .offset_y = 0 };
                    try renderer.display(&raw_term, size, &state);
                },
                'h' => {
                    state.offset_x -= 0.5 / state.zoom;
                    try renderer.display(&raw_term, size, &state);
                },
                'j' => {
                    state.offset_y -= 0.5 / state.zoom;
                    try renderer.display(&raw_term, size, &state);
                },
                'k' => {
                    state.offset_y += 0.5 / state.zoom;
                    try renderer.display(&raw_term, size, &state);
                },
                'l' => {
                    state.offset_x += 0.5 / state.zoom;
                    try renderer.display(&raw_term, size, &state);
                },
                else => {},
            },
            .resize => {
                size = try raw_term.size();
                try renderer.display(&raw_term, size, &state);
            },
            else => {},
        }
    }
}
