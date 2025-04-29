// X = AB

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

// Matirx B ------------------------------------
// Buffer of the B matrix
@group(1) @binding(0)
var<storage, read> matrix_b: array<f32>;

// Uniform of the dimensions of the B matrix
@group(1) @binding(1)
var<uniform> b_dimensions: vec2<u32>;

// Uniform of the transpose of the A matrix
@group(1) @binding(2)
var<uniform> b_transpose: u32;

// Matirx X ------------------------------------
// Buffer of the output matrix
@group(2) @binding(0)
var <storage, read_write> matrix_x: array<atomic<u32>>;
// var<storage, read_write> matrix_x: array<f32>;

// Uniform of the output dimensions
@group(2) @binding(1)
var<uniform> output_dimensions: vec2<u32>;

// Uniform of the transpose of the A matrix
@group(2) @binding(2)
var<uniform> x_transpose: u32;

const MIN_DIMENSION: u32 = 256;
const WORKGROUP_SIZE: u32 = 16;
const TILE_SIZE: u32 = 512;

var<workgroup> mat_a_window: array<array<f32, WORKGROUP_SIZE + 1>, WORKGROUP_SIZE>;
var<workgroup> mat_b_window: array<array<f32, WORKGROUP_SIZE + 1>, WORKGROUP_SIZE>;

fn u32_to_f32(val: u32) -> f32 {
    return bitcast<f32>(val);
}

fn f32_to_u32(val: f32) -> u32 {
    return bitcast<u32>(val);
}

fn atomic_add_f32(atomic_ptr_index: u32, val: f32) {
    var old_val: u32;
    var new_val: u32;
    var curr_val: f32;

    let atomic_ptr = &matrix_x[atomic_ptr_index];

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

fn load_into_shared_mem(a_location: vec2<u32>, b_location: vec2<u32>, workgroup_id: vec2<u32>) {
    let wg_row = workgroup_id.x;
    let wg_col = workgroup_id.y;

    let a_row = a_location.x;
    let a_col = a_location.y;

    let b_row = b_location.x;
    let b_col = b_location.y;

    let a_rows = a_dimensions.x;
    let a_cols = a_dimensions.y;
    let b_rows = b_dimensions.x;
    let b_cols = b_dimensions.y;

    // let a_index = a_row * a_cols + a_col;
    var a_index: u32 = 0;
    var b_index: u32 = 0;

    if (a_transpose == 1) {
        a_index = a_row + a_rows * a_col;
    } else {
        a_index = a_row * a_cols + a_col;
    }

    if (b_transpose == 1) {
        b_index = b_row + b_rows * b_col;
    } else {
        b_index = b_row * b_cols + b_col;
    }
    // let b_index = b_row * b_cols + b_col;

    if (a_row < a_rows && a_col < a_cols) {
        mat_a_window[wg_row][wg_col] = matrix_a[a_index];
    } else {
        mat_a_window[wg_row][wg_col] = 0.0;
    }

    if (b_row < b_rows && b_col < b_cols) {
        mat_b_window[wg_row][wg_col + 1] = matrix_b[b_index];
    } else {
        mat_b_window[wg_row][wg_col + 1] = 0.0;
    }
}

fn sum_workgroup_matrices(output_location: vec2<u32>, limit: u32, workgroup_id: vec2<u32>) {
    let wg_row = workgroup_id.x;
    let wg_col = workgroup_id.y;

    let row = output_location.x;
    let col = output_location.y;

    let output_rows = output_dimensions.x;
    let output_cols = output_dimensions.y;

    var sum = 0.0;
    for (var k: u32 = 0; k < limit; k++) {
        sum += mat_a_window[wg_row][k] * mat_b_window[k][wg_col + 1];
    }

    var output_index: u32 = 0;

    if (x_transpose == 1) {
        output_index = row + output_rows * col;
    } else {
        output_index = row * output_cols + col;
    }

    atomic_add_f32(output_index, sum);
}


@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE)
fn dot_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {

    let a_rows = a_dimensions.x;
    let a_cols = a_dimensions.y;

    let b_rows = b_dimensions.x;
    let b_cols = b_dimensions.y;

    let output_rows = output_dimensions.x;
    let output_cols = output_dimensions.y;

    let workgroup_row = local_id.x;
    let workgroup_col = local_id.y;

    let total_row_workgroups = u32(ceil(f32(a_rows) / f32(WORKGROUP_SIZE)));
    let total_col_workgroups = u32((ceil(f32(b_cols) / f32(WORKGROUP_SIZE))) * (ceil(f32(a_cols) / f32(WORKGROUP_SIZE))));

    let total_a_row_workgroups = total_row_workgroups;
    let total_a_col_workgroups = u32(ceil(f32(a_cols) / f32(WORKGROUP_SIZE)));

    let total_b_row_workgroups = total_a_col_workgroups;
    let total_b_col_workgroups = u32(ceil(f32(b_cols) / f32(WORKGROUP_SIZE)));

    let a_row_block_id = workgroup_id.x;
    let a_col_block_id = workgroup_id.y / total_b_col_workgroups;

    let b_row_block_id = workgroup_id.y / total_b_col_workgroups;
    let b_col_block_id = workgroup_id.y % total_b_col_workgroups;

    let output_row_block_id = a_row_block_id;
    let output_col_block_id = b_col_block_id;

    let a_row = (a_row_block_id * WORKGROUP_SIZE) + workgroup_row;
    let a_col = (a_col_block_id * WORKGROUP_SIZE) + workgroup_col;

    let b_row = (b_row_block_id * WORKGROUP_SIZE) + workgroup_row;
    let b_col = (b_col_block_id * WORKGROUP_SIZE) + workgroup_col;

    let output_row = (output_row_block_id * WORKGROUP_SIZE) + workgroup_row;
    let output_col = (output_col_block_id * WORKGROUP_SIZE) + workgroup_col;

    var a_row_limit = WORKGROUP_SIZE;
    var a_col_limit = WORKGROUP_SIZE;
    var b_row_limit = WORKGROUP_SIZE;
    var b_col_limit = WORKGROUP_SIZE;

    if (a_row_block_id >= a_rows / WORKGROUP_SIZE) {
        a_row_limit = a_rows % WORKGROUP_SIZE;
    }

    if (a_col_block_id >= total_a_col_workgroups) {
        a_col_limit = a_cols % WORKGROUP_SIZE;
    }

    if (b_row_block_id >= b_rows / WORKGROUP_SIZE) {
        b_row_limit = b_rows % WORKGROUP_SIZE;
    }

    if (b_col_block_id >= b_cols / WORKGROUP_SIZE) {
        b_col_limit = b_cols % WORKGROUP_SIZE;
    }


    load_into_shared_mem(vec2<u32>(a_row, a_col), vec2<u32>(b_row, b_col), local_id.xy);

    workgroupBarrier();

    if (output_row < output_rows && output_col < output_cols) {
        sum_workgroup_matrices(vec2<u32>(output_row, output_col), a_col_limit, local_id.xy);
    }
}
