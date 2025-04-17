#[cfg(test)]
mod tests {
    use gpu_math::{GpuMath, matrix::Matrix};

    #[test]
    fn test_matrix_mult_scalar() {
        let gpu_math = GpuMath::new();

        let mat1 = Matrix::new(
            &gpu_math,
            (3, 3),
            Some(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        )
        .expect("Failed");

        let dest = Matrix::new(&gpu_math, (3, 3), None).expect("Failed");
        let expected = Matrix::new(
            &gpu_math,
            (3, 3),
            Some(vec![0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]),
        )
        .expect("Failed");

        Matrix::mult_scalar(&mat1, 10.0, &dest).expect("Failed");

        assert_eq!(dest, expected);
    }

    #[test]
    fn test_matrix_mult_scalar_big() {
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
        let expected = Matrix::new(
            &gpu_math,
            (1000, 1000),
            Some(
                (0..(1000 * 1000))
                    .into_iter()
                    .map(|v| (v * 10) as f32)
                    .collect::<Vec<f32>>(),
            ),
        )
        .expect("Failed");

        Matrix::mult_scalar(&mat1, 10.0, &dest).expect("Failed");

        assert_eq!(dest, expected);
    }
}
