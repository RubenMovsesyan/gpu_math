use criterion::{Criterion, criterion_group, criterion_main};
use gpu_math::{GpuMath, matrix::Matrix};

pub fn bench_matrix_add(c: &mut Criterion) {
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

    c.bench_function("add", |b| {
        b.iter(|| {
            Matrix::add(&mat1, &mat2, &dest).expect("Failed");
        })
    });
}

pub fn bench_matrix_add_in_place(c: &mut Criterion) {
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

    c.bench_function("add_in_place", |b| {
        b.iter(|| {
            Matrix::add_in_place(&mat1, &mat2).expect("Failed");
        })
    });
}

pub fn bench_matrix_add_in_place_big(c: &mut Criterion) {
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
        Some(
            (0..(1000 * 1000))
                .into_iter()
                .map(|v| v as f32)
                .collect::<Vec<f32>>(),
        ),
    )
    .expect("Failed");

    c.bench_function("add_in_place_big", |b| {
        b.iter(|| {
            Matrix::add_in_place(&mat1, &mat2).expect("Failed");
        });
    });
}

pub fn bench_matrix_add_big(c: &mut Criterion) {
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

    let dest = Matrix::new(&gpu_math, (1000, 1000), None).expect("Failed");

    c.bench_function("add_big", |b| {
        b.iter(|| {
            Matrix::add(&mat1, &mat2, &dest).expect("Failed");
        });
    });
}

pub fn bench_matrix_add_scalar(c: &mut Criterion) {
    let gpu_math = GpuMath::new();

    let mat1 = Matrix::new(
        &gpu_math,
        (3, 3),
        Some(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    )
    .expect("Failed");

    let dest = Matrix::new(&gpu_math, (3, 3), None).expect("Failed");

    c.bench_function("add_scalar", |b| {
        b.iter(|| {
            Matrix::add_scalar(&mat1, 1.0, &dest).expect("Failed");
        })
    });
}

pub fn bench_matrix_add_scalar_big(c: &mut Criterion) {
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

    c.bench_function("add_scalar_big", |b| {
        b.iter(|| {
            Matrix::add_scalar(&mat1, 1.0, &dest).expect("Failed");
        })
    });
}

pub fn bench_matrix_vectored_add(c: &mut Criterion) {
    let gpu_math = GpuMath::new();

    const ROWS: u32 = 16;
    const COLS: u32 = 16;

    let mat = Matrix::new(
        &gpu_math,
        (ROWS, COLS),
        Some({
            let mut out = Vec::with_capacity((ROWS * COLS) as usize);

            for _ in 0..ROWS {
                for i in 0..COLS {
                    out.push(i as f32);
                }
            }

            out
        }),
    )
    .expect("Failed");

    let vec = Matrix::new(
        &gpu_math,
        (1, COLS),
        Some(
            (0..COLS)
                .into_iter()
                .map(|v| v as f32)
                .collect::<Vec<f32>>(),
        ),
    )
    .expect("Failed");

    let dest = Matrix::new(&gpu_math, (ROWS, COLS), None).expect("Failed");

    c.bench_function("vectored_add", |b| {
        b.iter(|| {
            Matrix::vectored_add(&mat, &vec, &dest).expect("Failed");
        });
    });
}

pub fn bench_matrix_vectored_add_big(c: &mut Criterion) {
    let gpu_math = GpuMath::new();

    let mat = Matrix::new(
        &gpu_math,
        (1000, 900),
        Some({
            let mut out = Vec::with_capacity(1000 * 900);

            for _ in 0..1000 {
                for i in 0..900 {
                    out.push(i as f32);
                }
            }

            out
        }),
    )
    .expect("Failed");

    let vec = Matrix::new(
        &gpu_math,
        (1, 900),
        Some((0..900).into_iter().map(|v| v as f32).collect::<Vec<f32>>()),
    )
    .expect("Failed");

    let dest = Matrix::new(&gpu_math, (1000, 900), None).expect("Failed");

    c.bench_function("vectored_add_big", |b| {
        b.iter(|| {
            Matrix::vectored_add(&mat, &vec, &dest).expect("Failed");
        });
    });
}

criterion_group!(
    adding_benches,
    bench_matrix_add,
    bench_matrix_add_in_place,
    bench_matrix_add_big,
    bench_matrix_add_in_place_big,
    bench_matrix_add_scalar,
    bench_matrix_add_scalar_big,
    bench_matrix_vectored_add,
    bench_matrix_vectored_add_big
);

criterion_main!(adding_benches);
