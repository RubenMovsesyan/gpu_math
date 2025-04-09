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

var<workgroup> mat_a_window: array<array<f32, WORKGROUP_SIZE>, WORKGROUP_SIZE>;
var<workgroup> mat_b_window: array<array<f32, WORKGROUP_SIZE>, WORKGROUP_SIZE>;

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
    // let b_cols = b_dimensions.x;
    let b_rows = b_dimensions.x;

    let a_index = a_row * a_rows + a_col;
    let b_index = b_row * b_rows + b_col;

    mat_a_window[wg_row][wg_col] = matrix_a[a_index];
    mat_b_window[wg_row][wg_col] = matrix_b[b_index];
}

fn sum_workgroup_matrices(output_location: vec2<u32>, workgroup_id: vec2<u32>) {
    let wg_row = workgroup_id.x;
    let wg_col = workgroup_id.y;

    let row = output_location.x;
    let col = output_location.y;

    let output_rows = output_dimensions.x;

    var sum = 0.0;
    for (var k: u32 = 0; k < WORKGROUP_SIZE; k++) {
        sum += mat_a_window[wg_row][k] * mat_b_window[wg_col][k];
    }

    let output_index = row * output_rows + col;
    atomic_add_f32(output_index, sum);
    // atomic_add_f32(output_index, 1.0);
    // atomic_add_f32(output_index, f32(output_index));
    // atomic_add_f32(output_index, f32(val));
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

    let a_row_block_id = workgroup_id.x % (u32(ceil(f32(a_rows) / f32(WORKGROUP_SIZE))));
    let a_col_block_id = workgroup_id.y % (u32(ceil(f32(a_cols) / f32(WORKGROUP_SIZE))));

    let b_row_block_id = workgroup_id.x / (a_rows / WORKGROUP_SIZE);
    let b_col_block_id = workgroup_id.y / (a_cols / WORKGROUP_SIZE);

    let output_row_block_id = a_row_block_id;
    let output_col_block_id = b_col_block_id;

    let a_row = (a_row_block_id * WORKGROUP_SIZE) + workgroup_row;
    let a_col = (a_col_block_id * WORKGROUP_SIZE) + workgroup_col;

    let b_row = (b_row_block_id * WORKGROUP_SIZE) + workgroup_col;
    let b_col = (b_col_block_id * WORKGROUP_SIZE) + workgroup_row;

    let output_row = (output_row_block_id * WORKGROUP_SIZE) + workgroup_row;
    let output_col = (output_col_block_id * WORKGROUP_SIZE) + workgroup_col;

    load_into_shared_mem(vec2<u32>(a_row, a_col), vec2<u32>(b_row, b_col), local_id.xy);
    workgroupBarrier();

    if (output_row < output_rows && output_col < output_cols) {
        sum_workgroup_matrices(vec2<u32>(output_row, output_col), local_id.xy);
    }
}

    // let row = global_id.x;
    // let col = global_id.y;


    // let workgroup_row = local_id.x;
    // let workgroup_col = local_id.y;
    // // let workgroup_index = workgroup_row * WORKGROUP_SIZE + workgroup_col;

    // let a_rows = a_dimensions.x;
    // let a_cols = a_dimensions.y;

    // let b_rows = b_dimensions.x;
    // let b_cols = b_dimensions.y;

    // let output_rows = output_dimensions.x;
    // let output_cols = output_dimensions.y;

    // let dot_size = a_cols;




    // if (row < output_rows && col < output_cols) {
    //     var sum: f32 = 0.0;
    //     var x_index: u32 = 0;

    //     if (x_transpose == 1) {
    //         x_index = row + output_rows * col;
    //     } else {
    //         x_index = row * output_cols + col;
    //     }

    //     for (var k: u32 = 0; k < dot_size; k++) {
    //         var a_index: u32 = 0;
    //         var b_index: u32 = 0;

    //         if (a_transpose == 1) {
    //             a_index = k * a_rows + row;
    //         } else {
    //             a_index = row * a_cols + k;
    //         }

    //         if (b_transpose == 1) {
    //             b_index = col * b_rows + k;
    //         } else {
    //             b_index = k * b_cols + col;
    //         }

    //         sum += matrix_a[a_index] * matrix_b[b_index];
    //     }

    //     matrix_x[x_index] = sum;
    // }


    // var sum: f32 = 0.0;
    // for (var tile: u32 = 0; tile < dot_size; tile += TILE_SIZE) {
    //     let max_iter = min(tile + TILE_SIZE, dot_size);
    //     // Load from matrix b into shared memory
    //     if (col < b_cols) {
    //         for (var k: u32 = tile; k < max_iter; k += WORKGROUP_SIZE) {
    //             let b_index = (k + workgroup_row) * b_cols + col;

    //             if (k + workgroup_row < b_rows) {
    //                 workgroup_cols[(workgroup_col + workgroup_row) % TILE_SIZE][workgroup_row + (k % TILE_SIZE)] = matrix_b[b_index];
    //             }
    //         }
    //     }

    //     workgroupBarrier();

    //     // Weighted sum of the values with matrix a
    //     if (row < output_rows && col < output_cols) {
    //         for (var k: u32 = tile; k < max_iter; k++) {
    //             let a_index = row * a_cols + k;
    //             sum += workgroup_cols[(workgroup_col + workgroup_row) % TILE_SIZE][k] * matrix_a[a_index];
    //         }


    //         // matrix_x[output_index] = sum;
    //     }
    // }

    // if (row < output_rows && col < output_cols) {
    //     let output_index = row * output_cols + col;
    //     matrix_x[output_index] = sum;
    // }

    // var sum: f32 = 0.0;
    // for (var tile: u32 = 0; tile < dot_size; tile += TILE_SIZE) {
    //     let max_iter = min(tile + TILE_SIZE, dot_size);
    //     // Load matrix a into shared memory
    //     if (row < a_rows) {
    //         for (var k: u32 = tile; k < max_iter; k += WORKGROUP_SIZE) {
    //             let a_index = row * a_cols + k + workgroup_col;

    //             if (k + workgroup_col < a_cols) {
    //                 workgroup_rows[workgroup_row][workgroup_col + (k % TILE_SIZE)] = matrix_a[a_index];
    //             }
    //         }
    //     }

    //     workgroupBarrier();


    //     // Load matrix b into shared memory
    //     if (col < b_cols) {
    //         for (var k: u32 = tile; k < max_iter; k += WORKGROUP_SIZE) {
    //             let b_index = (k + workgroup_row) * b_cols + col;

    //             if (k + workgroup_row < b_rows) {
    //                 workgroup_cols[workgroup_col][workgroup_row + (k % TILE_SIZE)] = matrix_b[b_index];
    //             }
    //         }
    //     }

    //     // Syncronize workgroups
    //     workgroupBarrier();

    //     var len: u32 = TILE_SIZE;
    //     if (len + tile > dot_size) {
    //         len = dot_size - tile;
    //     }

    //     for (var k: u32 = 0; k < len; k++) {
    //         sum += workgroup_rows[workgroup_row][k] * workgroup_cols[workgroup_col][k];
    //     }
    // }

    // workgroupBarrier();

    // if (row < output_rows && col < output_cols) {
    //     let output_index = row * output_cols + col;

    //     matrix_x[output_index] = sum;
    // }
// }
