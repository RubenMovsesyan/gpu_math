#[cfg(test)]
mod tests {
    use gpu_math::{GpuMath, matrix::Matrix};

    #[test]
    fn test_matrix_adding() {
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
        let expected = Matrix::new(
            &gpu_math,
            (3, 3),
            Some(vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]),
        )
        .expect("Failed");

        Matrix::add(&mat1, &mat2, &dest).expect("Failed");

        assert_eq!(dest, expected);
    }
}
