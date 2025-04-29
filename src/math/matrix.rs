use std::{error::Error, fmt::Display, rc::Rc};

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, Buffer, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, Device, Maintain, Queue,
    util::{BufferInitDescriptor, DeviceExt},
};

use crate::{
    GpuMath,
    gpu_utils::{
        WORK_GROUP_SIZE, WORK_GROUP_SIZE_2D, compute_workgroup_size, compute_workgroup_size_2d,
        get_buffer, read_buffer,
    },
    math_errors::MatrixExpError,
    matrix_dot_pipline, matrix_matrix_2d_in_place_pipeline, matrix_matrix_2d_pipeline,
    matrix_scalar_in_place_pipline, matrix_scalar_pipline, matrix_sum_pipeline,
};

use super::{
    math_errors::{MatrixAddError, MatrixDotError},
    matrix_pipelines::MatrixPipelines,
};

const DATA_SIZE: u64 = std::mem::size_of::<f32>() as u64;

#[allow(dead_code)]
#[derive(Debug)]
pub struct Matrix {
    // WGPU
    device: Rc<Device>,
    queue: Rc<Queue>,
    pipeline_info: Rc<MatrixPipelines>,

    pub rows: u32,
    pub cols: u32,
    // GPU vals
    dimensions: Buffer,
    data: Buffer,
    transpose: Buffer,
    scalar: Buffer,
    sum: Buffer,

    // Bind Groups
    readable_bind_group: BindGroup,
    writable_bind_group: BindGroup,
    scalar_bind_group: BindGroup,
    sum_bind_group: BindGroup,
}

impl Matrix {
    /// Creates a new matrix with a specified `shape` and fills it with `data` if provided
    /// The Matrix will not be transposed by default
    pub fn new(
        gpu_math: &GpuMath,
        shape: (u32, u32),
        data: Option<Vec<f32>>,
    ) -> Result<Self, Box<dyn Error>> {
        let device = gpu_math.device.clone();
        let queue = gpu_math.queue.clone();
        let pipeline_info = gpu_math.matrix_pipelines.clone();

        let buffer = match data {
            Some(data) => device.create_buffer_init(&BufferInitDescriptor {
                label: Some("Matrix Buffer"),
                contents: bytemuck::cast_slice(&data),
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            }),
            None => device.create_buffer(&BufferDescriptor {
                label: Some("Matrix Buffer"),
                mapped_at_creation: false,
                size: shape.0 as u64 * shape.1 as u64 * DATA_SIZE,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            }),
        };

        let dimensions = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Matrix Dimensions Buffer"),
            contents: bytemuck::cast_slice::<u32, u8>(&vec![shape.0 as u32, shape.1 as u32]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let transpose = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Matrix Transpose Buffer"),
            contents: bytemuck::cast_slice(&[false as u32]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        let scalar = device.create_buffer(&BufferDescriptor {
            label: Some("Matrix Scalar Buffer"),
            mapped_at_creation: false,
            size: DATA_SIZE,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let sum = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Matrix Sum Buffer"),
            contents: bytemuck::cast_slice(&[0f32]),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });

        let readable_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Matrix Readable Bind Group"),
            layout: &pipeline_info.readable_bind_group_layout,
            entries: &[
                // Data Buffer
                BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                // Dimensions
                BindGroupEntry {
                    binding: 1,
                    resource: dimensions.as_entire_binding(),
                },
                // Transpose
                BindGroupEntry {
                    binding: 2,
                    resource: transpose.as_entire_binding(),
                },
            ],
        });

        let scalar_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Matrix Scalar Bind Group"),
            layout: &pipeline_info.scalar_bind_group_layout,
            entries: &[
                // Scalar Uniform
                BindGroupEntry {
                    binding: 0,
                    resource: scalar.as_entire_binding(),
                },
            ],
        });

        let sum_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Matrix Sum Bind Group"),
            layout: &pipeline_info.sum_bind_group_layout,
            entries: &[
                // Sub Buffer
                BindGroupEntry {
                    binding: 0,
                    resource: sum.as_entire_binding(),
                },
            ],
        });

        let writable_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Matrix Writable Bind Group"),
            layout: &pipeline_info.writable_bind_group_layout,
            entries: &[
                // Data Buffer
                BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                // Dimensions
                BindGroupEntry {
                    binding: 1,
                    resource: dimensions.as_entire_binding(),
                },
                // Transpose
                BindGroupEntry {
                    binding: 2,
                    resource: transpose.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            device,
            queue,
            pipeline_info,
            rows: shape.0,
            cols: shape.1,
            dimensions,
            data: buffer,
            scalar,
            transpose,
            sum,
            readable_bind_group,
            writable_bind_group,
            scalar_bind_group,
            sum_bind_group,
        })
    }

    pub fn is_vector(&self) -> bool {
        (self.rows == 1 || self.cols == 1) && !(self.rows == self.cols)
    }

    /// Transposes the matrix
    pub fn transpose(&mut self) -> Result<(), Box<dyn Error>> {
        let temp = self.rows;
        self.rows = self.cols;
        self.cols = temp;

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Transpose Command Encoder"),
            });

        let transpose = read_buffer(&self.transpose, DATA_SIZE, &self.device, &mut encoder);

        self.queue.write_buffer(
            &self.dimensions,
            0,
            bytemuck::cast_slice::<u32, u8>(&vec![self.rows, self.cols]),
        );

        self.queue.submit(Some(encoder.finish()));

        let transpose_val: u32 = get_buffer(&transpose, &self.device)[0].to_bits();

        match transpose_val {
            0 => self
                .queue
                .write_buffer(&self.transpose, 0, bytemuck::cast_slice(&[1])),
            _ => self
                .queue
                .write_buffer(&self.transpose, 0, bytemuck::cast_slice(&[0])),
        }

        self.queue.submit(None);

        Ok(())
    }

    /// Dots the `source1` and `source2` matrix together and stores the output in `destination`
    pub fn dot(
        source1: &Matrix,
        source2: &Matrix,
        destination: &Matrix,
    ) -> Result<(), Box<dyn Error>> {
        // Make sure that the gpu is initialized
        // Check to see if the matrix rows and columns line up
        if source1.cols != source2.rows {
            return Err(Box::new(MatrixDotError(
                "Source 1 cols do not match Source 2 rows in Matrix::dot".to_string(),
            )));
        }

        matrix_dot_pipline!(
            &destination.device,
            &destination.queue,
            "Matrix Dot",
            &destination.pipeline_info.dot_pipeline,
            source1,
            source2,
            destination
        );

        Ok(())
    }

    /// Adds the matrices in `source1` and `source2` and stores it in `destination`
    pub fn add(
        source1: &Matrix,
        source2: &Matrix,
        destination: &Matrix,
    ) -> Result<(), Box<dyn Error>> {
        // Make sure that the gpu is initialzied
        // Check to see if the matrix rows and coloumns are the same
        if source1.cols != source2.cols || source1.cols != destination.cols {
            return Err(Box::new(MatrixAddError(
                "Source 1 cols do not match Source 2 cols or destinaiton cols".to_string(),
            )));
        }
        if source1.rows != source2.rows || source1.rows != destination.rows {
            return Err(Box::new(MatrixAddError(
                "Source 1 rows do not match Source 2 rows or destinaiton rows".to_string(),
            )));
        }

        // Run the add pipeline
        matrix_matrix_2d_pipeline!(
            &destination.device,
            &destination.queue,
            "Matrix Add",
            &destination.pipeline_info.add_pipeline,
            source1,
            source2,
            destination
        );

        Ok(())
    }

    /// Adds the contents of `vector` to `matrix` and stores it in `destination`
    pub fn vectored_add(
        matrix: &Matrix,
        vector: &Matrix,
        destination: &Matrix,
    ) -> Result<(), Box<dyn Error>> {
        // Check if the `vector` matrix is actually a vector
        if !vector.is_vector() {
            return Err(Box::new(MatrixAddError(
                "Vector matrix is not a vector".to_string(),
            )));
        }

        // Check if the vector dimensions match the matrix
        if !(vector.rows == matrix.rows
            || vector.cols == matrix.cols
            || vector.rows == matrix.cols
            || vector.cols == matrix.rows)
        {
            return Err(Box::new(MatrixAddError(
                "Vector dimensions do not match Matrix".to_string(),
            )));
        }

        matrix_matrix_2d_pipeline!(
            &destination.device,
            &destination.queue,
            "Matrix Vectored Add",
            &destination.pipeline_info.vectored_add_pipeline,
            matrix,
            vector,
            destination
        );

        Ok(())
    }

    /// Adds the contents of `matrix2` to `matrix1`
    pub fn add_in_place(matrix1: &Matrix, matrix2: &Matrix) -> Result<(), Box<dyn Error>> {
        // Make sure that the rows and columns of both matrices match
        if matrix1.cols != matrix2.cols {
            return Err(Box::new(MatrixAddError(
                "Matrix 1 cols do not match Matrix 2 cols".to_string(),
            )));
        }

        if matrix1.rows != matrix2.rows {
            return Err(Box::new(MatrixAddError(
                "Matrix 1 rows do not match Matrix 2 rows".to_string(),
            )));
        }

        // Run the add in place pipeline
        matrix_matrix_2d_in_place_pipeline!(
            &matrix1.device,
            &matrix1.queue,
            "Matrix Add in Place",
            &matrix1.pipeline_info.add_in_place_pipeline,
            matrix1,
            matrix2
        );

        Ok(())
    }

    /// Adds the contents of `vector` to `matrix`
    pub fn vectored_add_in_place(matrix: &Matrix, vector: &Matrix) -> Result<(), Box<dyn Error>> {
        // Check if the `vector` matrix is actually a vector
        if !vector.is_vector() {
            return Err(Box::new(MatrixAddError(
                "Vector matrix is not a vector".to_string(),
            )));
        }

        // Check if the vector dimensions match the matrix
        if !(vector.rows == matrix.rows
            || vector.cols == matrix.cols
            || vector.rows == matrix.cols
            || vector.cols == matrix.rows)
        {
            return Err(Box::new(MatrixAddError(
                "Vector dimensions do not match Matrix".to_string(),
            )));
        }

        matrix_matrix_2d_in_place_pipeline!(
            &matrix.device,
            &matrix.queue,
            "Matrix Vectored Add In Place",
            &matrix.pipeline_info.vectored_add_in_place_pipeline,
            matrix,
            vector
        );

        Ok(())
    }

    /// Adds the scalar value to `source` and stores it in `destination`
    pub fn add_scalar(
        source: &Matrix,
        scalar: f32,
        destination: &Matrix,
    ) -> Result<(), Box<dyn Error>> {
        // Check to make sure the source and destination matrices are the same size
        if source.cols != destination.cols {
            return Err(Box::new(MatrixAddError(
                "Source cols do not match Destination cols".to_string(),
            )));
        }

        if source.rows != destination.rows {
            return Err(Box::new(MatrixAddError(
                "Source rows do not match Destination rows".to_string(),
            )));
        }

        // write the scalar to the scalar buffer
        source
            .queue
            .write_buffer(&source.scalar, 0, bytemuck::cast_slice(&[scalar]));

        // Run the add scalar pipeline
        matrix_scalar_pipline!(
            &destination.device,
            &destination.queue,
            "Matrix Add Scalar",
            &source.pipeline_info.add_scalar_pipeline,
            source,
            destination
        );

        Ok(())
    }

    /// Subtracts `source2` from `source1` and stores it in `destination`
    pub fn sub(
        source1: &Matrix,
        source2: &Matrix,
        destination: &Matrix,
    ) -> Result<(), Box<dyn Error>> {
        // Make sure that the gpu is initialzied
        // Check to see if the matrix rows and coloumns are the same
        if source1.cols != source2.cols || source1.cols != destination.cols {
            return Err(Box::new(MatrixAddError(
                "Source 1 cols do not match Source 2 cols or destinaiton cols".to_string(),
            )));
        }
        if source1.rows != source2.rows || source1.rows != destination.rows {
            return Err(Box::new(MatrixAddError(
                "Source 1 rows do not match Source 2 rows or destinaiton rows".to_string(),
            )));
        }

        // Run the add pipeline
        matrix_matrix_2d_pipeline!(
            &destination.device,
            &destination.queue,
            "Matrix Sub",
            &destination.pipeline_info.sub_pipeline,
            source1,
            source2,
            destination
        );

        Ok(())
    }

    /// Subtracts the contents of `vector` from `matrix` and stores it in `destination`
    pub fn vectored_sub(
        matrix: &Matrix,
        vector: &Matrix,
        destination: &Matrix,
    ) -> Result<(), Box<dyn Error>> {
        // Check if the `vector` matrix is actually a vector
        if !vector.is_vector() {
            return Err(Box::new(MatrixAddError(
                "Vector matrix is not a vector".to_string(),
            )));
        }

        // Check if the vector dimensions match the matrix
        if !(vector.rows == matrix.rows
            || vector.cols == matrix.cols
            || vector.rows == matrix.cols
            || vector.cols == matrix.rows)
        {
            return Err(Box::new(MatrixAddError(
                "Vector dimensions do not match Matrix".to_string(),
            )));
        }

        matrix_matrix_2d_pipeline!(
            &destination.device,
            &destination.queue,
            "Matrix Vectored Add",
            &destination.pipeline_info.vectored_sub_pipeline,
            matrix,
            vector,
            destination
        );

        Ok(())
    }

    /// Subtracts the contents of `matrix2` from `matrix1`
    pub fn sub_in_place(matrix1: &Matrix, matrix2: &Matrix) -> Result<(), Box<dyn Error>> {
        // Make sure that the rows and columns of both matrices match
        if matrix1.cols != matrix2.cols {
            return Err(Box::new(MatrixAddError(
                "Matrix 1 cols do not match Matrix 2 cols".to_string(),
            )));
        }

        if matrix1.rows != matrix2.rows {
            return Err(Box::new(MatrixAddError(
                "Matrix 1 rows do not match Matrix 2 rows".to_string(),
            )));
        }

        // Run the add in place pipeline
        matrix_matrix_2d_in_place_pipeline!(
            &matrix1.device,
            &matrix1.queue,
            "Matrix Sub in Place",
            &matrix1.pipeline_info.sub_in_place_pipeline,
            matrix1,
            matrix2
        );

        Ok(())
    }

    /// Adds the contents of `vector` to `matrix`
    pub fn vectored_sub_in_place(matrix: &Matrix, vector: &Matrix) -> Result<(), Box<dyn Error>> {
        // Check if the `vector` matrix is actually a vector
        if !vector.is_vector() {
            return Err(Box::new(MatrixAddError(
                "Vector matrix is not a vector".to_string(),
            )));
        }

        // Check if the vector dimensions match the matrix
        if !(vector.rows == matrix.rows
            || vector.cols == matrix.cols
            || vector.rows == matrix.cols
            || vector.cols == matrix.rows)
        {
            return Err(Box::new(MatrixAddError(
                "Vector dimensions do not match Matrix".to_string(),
            )));
        }

        matrix_matrix_2d_in_place_pipeline!(
            &matrix.device,
            &matrix.queue,
            "Matrix Vectored Add In Place",
            &matrix.pipeline_info.vectored_sub_in_place_pipeline,
            matrix,
            vector
        );

        Ok(())
    }

    /// Subtracts the scalar value from `source` and stores it in `destination`
    pub fn sub_scalar(
        source: &Matrix,
        scalar: f32,
        destination: &Matrix,
    ) -> Result<(), Box<dyn Error>> {
        // Check to make sure the source and destination matrices are the same size
        if source.cols != destination.cols {
            return Err(Box::new(MatrixAddError(
                "Source cols do not match Destination cols".to_string(),
            )));
        }

        if source.rows != destination.rows {
            return Err(Box::new(MatrixAddError(
                "Source rows do not match Destination rows".to_string(),
            )));
        }

        // write the scalar to the scalar buffer
        source
            .queue
            .write_buffer(&source.scalar, 0, bytemuck::cast_slice(&[scalar]));

        // Run the add scalar pipeline
        matrix_scalar_pipline!(
            &destination.device,
            &destination.queue,
            "Matrix Sub Scalar",
            &source.pipeline_info.sub_scalar_pipeline,
            source,
            destination
        );

        Ok(())
    }

    /// Multiplies the scalar value to `source` and stores it in `destination`
    pub fn mult_scalar(
        source: &Matrix,
        scalar: f32,
        destination: &Matrix,
    ) -> Result<(), Box<dyn Error>> {
        // Check to make sure the source and destination matrices are the same size
        if source.cols != destination.cols {
            return Err(Box::new(MatrixAddError(
                "Source cols do not match Destination cols".to_string(),
            )));
        }

        if source.rows != destination.rows {
            return Err(Box::new(MatrixAddError(
                "Source rows do not match Destination rows".to_string(),
            )));
        }

        // write the scalar to the scalar buffer
        source
            .queue
            .write_buffer(&source.scalar, 0, bytemuck::cast_slice(&[scalar]));

        // Run the add scalar pipeline
        matrix_scalar_pipline!(
            &destination.device,
            &destination.queue,
            "Matrix Mult Scalar",
            &source.pipeline_info.mult_scalar_pipeline,
            source,
            destination
        );

        Ok(())
    }

    /// Multiplies the `matrix` by the `scalar` in place
    pub fn mult_scalar_in_place(matrix: &Matrix, scalar: f32) -> Result<(), Box<dyn Error>> {
        matrix
            .queue
            .write_buffer(&matrix.scalar, 0, bytemuck::cast_slice(&[scalar]));

        matrix_scalar_in_place_pipline!(
            &matrix.device,
            &matrix.queue,
            "Matrix Mult Scalar In Place",
            &matrix.pipeline_info.mult_scalar_in_place_pipeline,
            matrix
        );

        Ok(())
    }

    /// Multiplies every element of `source1` by `source2` and stores it in `destination`
    pub fn mult(
        source1: &Matrix,
        source2: &Matrix,
        destination: &Matrix,
    ) -> Result<(), Box<dyn Error>> {
        // Make sure that the gpu is initialzied
        // Check to see if the matrix rows and coloumns are the same
        if source1.cols != source2.cols || source1.cols != destination.cols {
            return Err(Box::new(MatrixAddError(
                "Source 1 cols do not match Source 2 cols or destinaiton cols".to_string(),
            )));
        }
        if source1.rows != source2.rows || source1.rows != destination.rows {
            return Err(Box::new(MatrixAddError(
                "Source 1 rows do not match Source 2 rows or destinaiton rows".to_string(),
            )));
        }

        // Run the add pipeline
        matrix_matrix_2d_pipeline!(
            &destination.device,
            &destination.queue,
            "Matrix Mult",
            &destination.pipeline_info.mult_pipeline,
            source1,
            source2,
            destination
        );

        Ok(())
    }

    /// Multiplies the contents of `matrix2` to `matrix1`
    pub fn mult_in_place(matrix1: &Matrix, matrix2: &Matrix) -> Result<(), Box<dyn Error>> {
        // Make sure that the rows and columns of both matrices match
        if matrix1.cols != matrix2.cols {
            return Err(Box::new(MatrixAddError(
                "Matrix 1 cols do not match Matrix 2 cols".to_string(),
            )));
        }

        if matrix1.rows != matrix2.rows {
            return Err(Box::new(MatrixAddError(
                "Matrix 1 rows do not match Matrix 2 rows".to_string(),
            )));
        }

        // Run the add in place pipeline
        matrix_matrix_2d_in_place_pipeline!(
            &matrix1.device,
            &matrix1.queue,
            "Matrix Mult in Place",
            &matrix1.pipeline_info.mult_in_place_pipeline,
            matrix1,
            matrix2
        );

        Ok(())
    }

    /// Exponents the `matrix` and stores it in `destination`
    pub fn exp(matrix: &Matrix, destination: &Matrix) -> Result<(), Box<dyn Error>> {
        // Make sure that the rows and columns of both matrices match
        if matrix.cols != destination.cols {
            return Err(Box::new(MatrixExpError(
                "Matrix cols do not match Destination cols".to_string(),
            )));
        }

        if matrix.rows != destination.rows {
            return Err(Box::new(MatrixExpError(
                "Matrix rows do not match Destination rows".to_string(),
            )));
        }

        matrix_scalar_pipline!(
            &destination.device,
            &destination.queue,
            "Matrix Exp",
            &matrix.pipeline_info.exp_pipeline,
            matrix,
            destination
        );

        Ok(())
    }

    pub fn exp_in_place(matrix: &Matrix) -> Result<(), Box<dyn Error>> {
        matrix_scalar_in_place_pipline!(
            &matrix.device,
            &matrix.queue,
            "Matrix Exp In Place",
            &matrix.pipeline_info.exp_in_place_pipeline,
            matrix
        );

        Ok(())
    }

    pub fn sum(matrix: &Matrix) -> Result<f32, Box<dyn Error>> {
        #[allow(unused_assignments)]
        let mut sum = 0.0;

        matrix
            .queue
            .write_buffer(&matrix.sum, 0, bytemuck::cast_slice(&[0f32]));

        matrix_sum_pipeline!(
            &matrix.device,
            &matrix.queue,
            "Matrix Sum Pipeline",
            &matrix.pipeline_info.sum_pipeline,
            matrix,
            sum
        );

        Ok(sum)
    }
}

impl Drop for Matrix {
    fn drop(&mut self) {
        self.device.poll(Maintain::Wait);
        self.data.destroy();
        self.transpose.destroy();
        self.scalar.destroy();
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }

        let device = &self.device;
        let queue = &self.queue;

        let (mat1, mat2) = {
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Matrix Reading Command Encoder"),
            });

            let buffer_1 = read_buffer(
                &self.data,
                self.rows as u64 * self.cols as u64 * DATA_SIZE,
                device,
                &mut encoder,
            );

            let buffer_2 = read_buffer(
                &other.data,
                other.rows as u64 * other.cols as u64 * DATA_SIZE,
                device,
                &mut encoder,
            );

            queue.submit(Some(encoder.finish()));

            (get_buffer(&buffer_1, device), get_buffer(&buffer_2, device))
        };

        // println!("mat1: {:#?}", mat1);
        // println!("mat2: {:#?}", mat2);
        // TODO: Check for transpose
        for ((v1_ind, val1), (v2_ind, val2)) in mat1.iter().enumerate().zip(mat2.iter().enumerate())
        {
            if val1 != val2 {
                println!(
                    "{} != {}, v1_ind: {} v2_ind: {}",
                    val1, val2, v1_ind, v2_ind
                );
                return false;
            }
        }

        true
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let device = &self.device;
        let queue = &self.queue;

        let mat = {
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Matrix Reading Command Encoder"),
            });

            let buffer = read_buffer(
                &self.data,
                self.rows as u64 * self.cols as u64 * DATA_SIZE,
                device,
                &mut encoder,
            );

            queue.submit(Some(encoder.finish()));

            get_buffer(&buffer, device)
        };

        writeln!(f)?;
        for i in 0..self.rows {
            write!(f, "| ")?;
            for j in 0..self.cols {
                let index = (i * self.cols + j) as usize;

                write!(f, "{}, ", mat[index])?;
            }
            writeln!(f, "|")?;
        }

        Ok(())
    }
}
