#[cfg(test)]
mod subbing_tests {
    use gpu_math::{GpuMath, matrix::Matrix};

    #[test]
    fn test_matrix_subing() {
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
            Some(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        .expect("Failed");

        Matrix::sub(&mat1, &mat2, &dest).expect("Failed");

        assert_eq!(dest, expected);
    }

    #[test]
    fn test_matrix_subing_big() {
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

        let expected = Matrix::new(
            &gpu_math,
            (1000, 1000),
            Some({
                let mut output = Vec::new();

                for _ in 0..1000 {
                    output.push((0..1000).into_iter().map(|_| 0.0).collect::<Vec<f32>>());
                }

                output.into_iter().flatten().collect::<Vec<f32>>()
            }),
        )
        .expect("Failed");

        let dest = Matrix::new(&gpu_math, (1000, 1000), None).expect("Failed");

        Matrix::sub(&mat1, &mat2, &dest).expect("Failed");

        assert_eq!(dest, expected);
    }

    #[test]
    fn test_matrix_subing_in_place() {
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

        let expected = Matrix::new(
            &gpu_math,
            (3, 3),
            Some((0..9).into_iter().map(|_| 0.0).collect::<Vec<f32>>()),
        )
        .expect("Failed");

        Matrix::sub_in_place(&mat1, &mat2).expect("Failed");

        assert_eq!(mat1, expected);
    }

    #[test]
    fn test_matrix_subing_in_place_big() {
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

        let expected = Matrix::new(
            &gpu_math,
            (1000, 1000),
            Some(
                (0..(1000 * 1000))
                    .into_iter()
                    .map(|_| 0.0)
                    .collect::<Vec<f32>>(),
            ),
        )
        .expect("Failed");

        Matrix::sub_in_place(&mat1, &mat2).expect("Failed");

        assert_eq!(mat1, expected);
    }

    #[test]
    fn test_matrix_subing_scalar() {
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
            Some(vec![-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        )
        .expect("Failed");

        Matrix::sub_scalar(&mat1, 1.0, &dest).expect("Failed");

        assert_eq!(dest, expected);
    }

    #[test]
    fn test_matrix_subing_scalar_big() {
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
                    .map(|v| (v - 1) as f32)
                    .collect::<Vec<f32>>(),
            ),
        )
        .expect("Failed");

        Matrix::sub_scalar(&mat1, 1.0, &dest).expect("Failed");

        assert_eq!(dest, expected);
    }

    #[test]
    fn test_matrix_vectored_sub() {
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

        let expected = Matrix::new(
            &gpu_math,
            (ROWS, COLS),
            Some({
                let mut out = Vec::with_capacity((ROWS * COLS) as usize);

                for _ in 0..ROWS {
                    for _ in 0..COLS {
                        out.push(0.0);
                    }
                }

                out
            }),
        )
        .expect("Failed");

        Matrix::vectored_sub(&mat, &vec, &dest).expect("Failed");

        assert_eq!(dest, expected);
    }

    #[test]
    fn test_matrix_vectored_sub_big() {
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

        let expected = Matrix::new(
            &gpu_math,
            (1000, 900),
            Some({
                let mut out = Vec::with_capacity(1000 * 900);

                for _ in 0..1000 {
                    for _ in 0..900 {
                        out.push(0.0);
                    }
                }

                out
            }),
        )
        .expect("Failed");

        Matrix::vectored_sub(&mat, &vec, &dest).expect("Failed");

        assert_eq!(dest, expected);
    }

    #[test]
    fn test_matrix_vectored_sub_in_place() {
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

        let expected = Matrix::new(
            &gpu_math,
            (ROWS, COLS),
            Some({
                let mut out = Vec::with_capacity((ROWS * COLS) as usize);

                for _ in 0..ROWS {
                    for _ in 0..COLS {
                        out.push(0.0);
                    }
                }

                out
            }),
        )
        .expect("Failed");

        Matrix::vectored_sub_in_place(&mat, &vec).expect("Failed");

        assert_eq!(mat, expected);
    }

    #[test]
    fn test_matrix_vectored_sub_in_place_big() {
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

        let expected = Matrix::new(
            &gpu_math,
            (1000, 900),
            Some({
                let mut out = Vec::with_capacity(1000 * 900);

                for _ in 0..1000 {
                    for _ in 0..900 {
                        out.push(0.0);
                    }
                }

                out
            }),
        )
        .expect("Failed");

        Matrix::vectored_sub_in_place(&mat, &vec).expect("Failed");

        assert_eq!(mat, expected);
    }
}
