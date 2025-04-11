const builtin = @import("builtin");

comptime {
    exportKernels(@import("kernels/mandelbrot.zig"));
    exportKernels(@import("kernels/pixelsToAnsi.zig"));
}

fn exportKernels(K: anytype) void {
    if (builtin.os.tag == .opencl) {
        for (@typeInfo(K).@"struct".decls) |decl| {
            const kernel = @field(K, decl.name);
            switch (@typeInfo(@TypeOf(kernel))) {
                .@"fn" => |signature| {
                    if (signature.calling_convention != .spirv_kernel) {
                        @compileError("kernel `" ++ decl.name ++ "` must use callconv(.spirv_kernel)");
                    }

                    @export(&kernel, .{ .name = decl.name });
                },
                else => {},
            }
        }
    }
}
