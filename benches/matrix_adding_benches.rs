use criterion::{criterion_group, criterion_main};

mod matrix_benchmarks;

criterion_group!(
    adding_benches,
    matrix_benchmarks::adding_benchmarks::bench_matrix_add,
    matrix_benchmarks::adding_benchmarks::bench_matrix_add_big,
    matrix_benchmarks::adding_benchmarks::bench_matrix_add_scalar,
    matrix_benchmarks::adding_benchmarks::bench_matrix_add_scalar_big
);

criterion_main!(adding_benches);
