use criterion::{Criterion, criterion_group, criterion_main};
use gpu_math::{GpuMath, matrix::Matrix};

pub fn bench_matrix_mult_scalar(c: &mut Criterion) {
    let gpu_math = GpuMath::new();

    let mat1 = Matrix::new(
        &gpu_math,
        (3, 3),
        Some(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    )
    .expect("Failed");

    let dest = Matrix::new(&gpu_math, (3, 3), None).expect("Failed");

    c.bench_function("mult_scalar", |b| {
        b.iter(|| {
            Matrix::mult_scalar(&mat1, 10.0, &dest).expect("Failed");
        })
    });
}

pub fn bench_matrix_mult_scalar_big(c: &mut Criterion) {
    let gpu_math = GpuMath::new();

    let mat1 = Matrix::new(
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

    let dest = Matrix::new(&gpu_math, (1000, 1000), None).expect("Failed");

    c.bench_function("mult_scalar_big", |b| {
        b.iter(|| {
            Matrix::mult_scalar(&mat1, 10.0, &dest).expect("Failed");
        })
    });
}

fn bench_matrix_mult(c: &mut Criterion) {
    let gpu_math = GpuMath::new();

    let mat1 = Matrix::new(
        &gpu_math,
        (3, 3),
        Some(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    )
    .expect("Failed");

    let mat2 = Matrix::new(
        &gpu_math,
        (3, 3),
        Some(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    )
    .expect("Failed");

    let dest = Matrix::new(&gpu_math, (3, 3), None).expect("Failed");

    c.bench_function("matrix_mult", |b| {
        b.iter(|| {
            Matrix::mult(&mat1, &mat2, &dest).expect("Failed");
        })
    });
}

fn bench_matrix_mult_big(c: &mut Criterion) {
    let gpu_math = GpuMath::new();

    let mat1 = Matrix::new(
        &gpu_math,
        (1000, 1000),
        Some({
            let mut output = Vec::new();

            for _ in 0..1000 {
                output.push(
                    (0..1000)
                        .into_iter()
                        .map(|i| i as f32)
                        .collect::<Vec<f32>>(),
                );
            }

            output.into_iter().flatten().collect::<Vec<f32>>()
        }),
    )
    .expect("Failed");

    let mat2 = Matrix::new(
        &gpu_math,
        (1000, 1000),
        Some({
            let mut output = Vec::new();

            for _ in 0..1000 {
                output.push((0..1000).into_iter().map(|_| 10.0).collect::<Vec<f32>>());
            }

            output.into_iter().flatten().collect::<Vec<f32>>()
        }),
    )
    .expect("Failed");

    let dest = Matrix::new(&gpu_math, (1000, 1000), None).expect("Failed");

    c.bench_function("matrix_mult_big", |b| {
        b.iter(|| {
            Matrix::mult(&mat1, &mat2, &dest).expect("Failed");
        });
    });
}

fn bench_matrix_mult_in_place(c: &mut Criterion) {
    let gpu_math = GpuMath::new();

    let mat1 = Matrix::new(
        &gpu_math,
        (3, 3),
        Some((0..9).into_iter().map(|v| v as f32).collect::<Vec<f32>>()),
    )
    .expect("Failed");

    let mat2 = Matrix::new(
        &gpu_math,
        (3, 3),
        Some((0..9).into_iter().map(|v| v as f32).collect::<Vec<f32>>()),
    )
    .expect("Failed");

    c.bench_function("matrix_mult_in_place", |b| {
        b.iter(|| {
            Matrix::mult_in_place(&mat1, &mat2).expect("Failed");
        });
    });
}

fn bench_matrix_mult_in_place_big(c: &mut Criterion) {
    let gpu_math = GpuMath::new();

    let mat1 = Matrix::new(
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

    let mat2 = Matrix::new(
        &gpu_math,
        (1000, 1000),
        Some({
            let mut output = Vec::new();

            for _ in 0..1000 {
                output.push((0..1000).into_iter().map(|_| 10.0).collect::<Vec<f32>>());
            }

            output.into_iter().flatten().collect::<Vec<f32>>()
        }),
    )
    .expect("Failed");

    c.bench_function("matrix_mult_in_place_big", |b| {
        b.iter(|| {
            Matrix::mult_in_place(&mat1, &mat2).expect("Failed");
        });
    });
}

criterion_group!(
    multing_benches,
    bench_matrix_mult_scalar,
    bench_matrix_mult_scalar_big,
    bench_matrix_mult,
    bench_matrix_mult_big,
    bench_matrix_mult_in_place,
    bench_matrix_mult_in_place_big
);

criterion_main!(multing_benches);
