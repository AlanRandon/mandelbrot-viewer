const builtin = @import("builtin");

pub usingnamespace @cImport({
    if (builtin.os.tag.isDarwin()) {
        @cInclude("OpenCL/opencl.h");
    } else {
        @cInclude("CL/cl.h");
    }
});
