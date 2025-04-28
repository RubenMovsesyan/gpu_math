// X = sum(A)

// Matirx A ------------------------------------
// Buffer of the A matrix
@group(0) @binding(0)
var<storage, read> matrix_a: array<f32>;

// Uniform of the dimensions of the A matrix
@group(0) @binding(1)
var<uniform> a_dimensions: vec2<u32>;

// Uniform of the transpose of the A matrix
@group(0) @binding(2)
var<uniform> a_transpose: u32;

@group(1) @binding(0)
var<storage, read_write> reduce_buffer: atomic<u32>;

const MAX_ARRAY_SIZE: u32 = 256;

// Shared memory for the workgroups
var<workgroup> shared_reduce: array<f32, MAX_ARRAY_SIZE>;

// Functions for working with atomic value
fn u32_to_f32(val: u32) -> f32 {
    return bitcast<f32>(val);
}

fn f32_to_u32(val: f32) -> u32 {
    return bitcast<u32>(val);
}

fn atomic_add_f32(val: f32) {
    var old_val: u32;
    var new_val: u32;
    var curr_val: f32;

    let atomic_ptr = &reduce_buffer;

    loop {
        // Get the current value in the atomic location
        old_val = atomicLoad(atomic_ptr);
        curr_val = u32_to_f32(old_val);

        // Calculate the new val
        new_val = f32_to_u32(curr_val + val);

        // try to store the value back
        let result = atomicCompareExchangeWeak(atomic_ptr, old_val, new_val);
        if (result.exchanged) {
            break;
        }
    }
}

@compute @workgroup_size(256)
fn sum_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let elem = global_id.x;
    let local_elem = local_id.x;
    let rows = a_dimensions.x;
    let cols = a_dimensions.y;
    let size = rows * cols;

    // Load the group into shared memory
    if (elem < size) {
        shared_reduce[local_elem] = matrix_a[elem];
    } else {
        shared_reduce[local_elem] = 0.0;
    }

    workgroupBarrier();

    for (var half_size = MAX_ARRAY_SIZE / 2; half_size > 0; half_size /= 2) {
        if (local_elem < half_size) {
            shared_reduce[local_elem] += shared_reduce[local_elem + half_size];
        }

        workgroupBarrier();
    }

    if (elem < size && local_elem == 0) {
        atomic_add_f32(shared_reduce[local_elem]);
    }
}
