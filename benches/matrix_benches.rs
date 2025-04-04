use criterion::{criterion_group, criterion_main};

mod matrix_benchmarks;

criterion_group!(
    matrix_benches,
    matrix_benchmarks::dotting_benchmarks::bench_matrix_dot,
    matrix_benchmarks::adding_benchmarks::bench_matrix_add
);
criterion_main!(matrix_benches);
