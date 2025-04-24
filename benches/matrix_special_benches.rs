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

criterion_group!(special_benches, bench_matrix_exp, bench_matrix_exp_in_place);

criterion_main!(special_benches);
