use criterion::{Criterion, criterion_group, criterion_main};
use gpu_math::{GpuMath, matrix::Matrix};

fn bench_matrix_exp(c: &mut Criterion) {
    let gpu_math = GpuMath::new();

    let mat = Matrix::new(
        &gpu_math,
        (3, 3),
        Some(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    )
    .expect("Failed");

    let dest = Matrix::new(&gpu_math, (3, 3), None).expect("Failed");

    c.bench_function("exp", |b| {
        b.iter(|| {
            Matrix::exp(&mat, &dest).expect("Failed");
        });
    });
}

fn bench_matrix_exp_in_place(c: &mut Criterion) {
    let gpu_math = GpuMath::new();

    let mat = Matrix::new(
        &gpu_math,
        (3, 3),
        Some(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    )
    .expect("Failed");

    c.bench_function("exp_in_place", |b| {
        b.iter(|| {
            Matrix::exp_in_place(&mat).expect("Failed");
        });
    });
}

fn bench_matrix_sum(c: &mut Criterion) {
    let gpu_math = GpuMath::new();

    const ROWS: u32 = 16;
    const COLS: u32 = 32;

    let vec = (0..(ROWS * COLS))
        .into_iter()
        .map(|v| v as f32)
        .collect::<Vec<f32>>();

    let mat = Matrix::new(&gpu_math, (ROWS, COLS), Some(vec.clone())).expect("Failed");

    c.bench_function("sum", |b| {
        b.iter(|| {
            let _sum = Matrix::sum(&mat).expect("Failed");
        });
    });
}

fn bench_matrix_sum_big(c: &mut Criterion) {
    let gpu_math = GpuMath::new();

    let mat = Matrix::new(
        &gpu_math,
        (1000, 1000),
        Some(
            (0..(1000 * 1000))
                .into_iter()
                .map(|v| v as f32)
                .collect::<Vec<f32>>(),
        ),
    )
    .expect("Failed");

    c.bench_function("sum_big", |b| {
        b.iter(|| {
            let _sum = Matrix::sum(&mat).expect("Failed");
        });
    });
}

criterion_group!(
    special_benches,
    bench_matrix_exp,
    bench_matrix_exp_in_place,
    bench_matrix_sum,
    bench_matrix_sum_big
);

criterion_main!(special_benches);
