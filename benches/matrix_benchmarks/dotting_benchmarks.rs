use criterion::Criterion;

use gpu_math::matrix::Matrix;

pub fn bench_matrix_dot(c: &mut Criterion) {
    gpu_math::init();

    let mat1 = Matrix::new(
        (3, 3),
        Some(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    )
    .expect("Failed");

    let mat2 = Matrix::new(
        (3, 3),
        Some(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    )
    .expect("Failed");

    let dest = Matrix::new((3, 3), None).expect("Failed");

    c.bench_function("dot", |b| {
        b.iter(|| {
            Matrix::dot(&mat1, &mat2, &dest).expect("Failed");
        })
    });
}

pub fn bench_matrix_dot_big(c: &mut Criterion) {
    gpu_math::init();

    let mat1 = Matrix::new(
        (1000, 1000),
        Some({
            let mut vec = Vec::with_capacity(1000 * 1000);
            for _ in 0..1000 {
                for j in 0..1000 {
                    vec.push(j as f32);
                }
            }
            vec
        }),
    )
    .expect("Failed");

    let mat2 = Matrix::new(
        (1000, 1000),
        Some({
            let mut vec = Vec::with_capacity(1000 * 1000);
            for _ in 0..1000 {
                for j in 0..1000 {
                    vec.push(j as f32);
                }
            }
            vec
        }),
    )
    .expect("Failed");

    let dest = Matrix::new((1000, 1000), None).expect("Failed");

    c.bench_function("dot_big", |b| {
        b.iter(|| {
            Matrix::dot(&mat1, &mat2, &dest).expect("Failed");
        })
    });
}
