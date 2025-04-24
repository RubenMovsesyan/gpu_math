#[cfg(test)]
mod tests {
    use gpu_math::{GpuMath, matrix::Matrix};

    #[test]
    fn test_matrix_exp() {
        let gpu_math = GpuMath::new();

        let mat = Matrix::new(
            &gpu_math,
            (3, 3),
            Some(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        )
        .expect("Failed");

        let dest = Matrix::new(&gpu_math, (3, 3), None).expect("Failed");

        let expected = Matrix::new(
            &gpu_math,
            (3, 3),
            Some(vec![
                1.0, 2.718282, 7.389056, 20.085535, 54.59815, 148.41316, 403.4287, 1096.6334,
                2980.9578,
            ]),
        )
        .expect("Failed");

        Matrix::exp(&mat, &dest).expect("Failed");

        assert_eq!(dest, expected);
    }

    #[test]
    fn test_matrix_exp_in_place() {
        let gpu_math = GpuMath::new();

        let mat = Matrix::new(
            &gpu_math,
            (3, 3),
            Some(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        )
        .expect("Failed");

        let expected = Matrix::new(
            &gpu_math,
            (3, 3),
            Some(vec![
                1.0, 2.718282, 7.389056, 20.085535, 54.59815, 148.41316, 403.4287, 1096.6334,
                2980.9578,
            ]),
        )
        .expect("Failed");

        Matrix::exp_in_place(&mat).expect("Failed");

        assert_eq!(mat, expected);
    }
}
