use criterion::{criterion_group, criterion_main};

mod matrix_benchmarks;

criterion_group!(
    dotting_benches,
    matrix_benchmarks::dotting_benchmarks::bench_matrix_dot,
    matrix_benchmarks::dotting_benchmarks::bench_matrix_dot_big
);

criterion_main!(dotting_benches);
