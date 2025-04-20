const std = @import("std");
const c = @import("c.zig");
const checkClError = @import("error.zig").checkClError;

pub fn deviceName(device: c.cl_device_id, allocator: std.mem.Allocator) ![]u8 {
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

pub fn getClDevice(allocator: std.mem.Allocator) !c.cl_device_id {
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
