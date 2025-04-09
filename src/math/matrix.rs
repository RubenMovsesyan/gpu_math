use std::{borrow::Cow, collections::HashMap, error::Error, fmt::Display, rc::Rc};

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, PipelineCompilationOptions, PipelineLayout, PipelineLayoutDescriptor, Queue,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
    util::{BufferInitDescriptor, DeviceExt},
};

use crate::{
    GpuMath,
    errors::GpuMathNotInitializedError,
    gpu_utils::{
        WORK_GROUP_SIZE, WORK_GROUP_SIZE_2D, compute_workgroup_size, compute_workgroup_size_2d,
        get_buffer, read_buffer,
    },
    matrix_dot_pipline, matrix_matrix_1d_pipeline, matrix_matrix_2d_pipeline,
};

use super::math_errors::{MatrixAddError, MatrixDotError};

const DATA_SIZE: u64 = std::mem::size_of::<f32>() as u64;
const MIN_DIMENSION: f64 = WORK_GROUP_SIZE as f64;

#[allow(dead_code)]
#[derive(Debug)]
pub struct MatrixPipelines {
    // Bind Group Layouts
    readable_bind_group_layout: BindGroupLayout,
    writable_bind_group_layout: BindGroupLayout,

    // Pipeline Layouts
    matrix_matrix_pipeline_layout: PipelineLayout,

    // Maximum number of rows or columns in a matrix for computation efficiency
    max_dimension: f64,

    // Pipelines
    dot_pipeline: ComputePipeline,
    add_pipeline: ComputePipeline,
}

impl MatrixPipelines {
    fn compile_pipelines(
        device: &Device,
        max_dimension: f64,
        matrix_matrix_pipeline_layout: &PipelineLayout,
    ) -> (ComputePipeline, ComputePipeline) {
        let pipeline_compilaiton_hm = {
            let /* mut */ options = HashMap::new();
            // options.insert("MAX_DIMENSION".to_string(), max_dimension);
            options
        };

        let dot_pipeline = {
            // let shader = device.create_shader_module(include_wgsl!("shaders/dotting.wgsl"));
            let shader = device.create_shader_module(ShaderModuleDescriptor {
                label: Some("Matrix Dot Shader"),
                source: ShaderSource::Wgsl(Cow::Owned(
                    include_str!("shaders/dotting.wgsl")
                        .replace("MAX_DIMENSION", &max_dimension.to_string()),
                )),
            });

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Dot Pipeline"),
                module: &shader,
                layout: Some(matrix_matrix_pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions {
                    constants: &pipeline_compilaiton_hm,
                    ..Default::default()
                },
                entry_point: Some("dot_main"),
            })
        };

        let add_pipeline = {
            // let shader = device.create_shader_module(include_wgsl!("shaders/adding.wgsl"));
            let shader = device.create_shader_module(ShaderModuleDescriptor {
                label: Some("Matrix Add Shader"),
                source: ShaderSource::Wgsl(Cow::Owned(
                    include_str!("shaders/adding.wgsl")
                        .replace("MAX_DIMENSION", &max_dimension.to_string()),
                )),
            });

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Add Pipeline"),
                module: &shader,
                layout: Some(matrix_matrix_pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions {
                    constants: &pipeline_compilaiton_hm,
                    ..Default::default()
                },
                entry_point: Some("add_main"),
            })
        };

        (dot_pipeline, add_pipeline)
    }

    pub fn init(device: &Device) -> Result<Self, GpuMathNotInitializedError> {
        // Create the readable bind group layout for the pipelines
        let readable_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Readable Bind Group Layout"),
                entries: &[
                    // Matrix Buffer
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Dimensions
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Transpose
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Create the writable bind group layout for the pipelines
        let writable_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Readable Bind Group Layout"),
                entries: &[
                    // Matrix Buffer
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Dimensions
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Transpose
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStages::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        // This is the pipeline layout for a Matrix Matrix operation
        let matrix_matrix_pipeline_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Matrix Matrix Pipeline Layout"),
                bind_group_layouts: &[
                    &readable_bind_group_layout,
                    &readable_bind_group_layout,
                    &writable_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let (dot_pipeline, add_pipeline) =
            Self::compile_pipelines(device, MIN_DIMENSION, &matrix_matrix_pipeline_layout);

        Ok(Self {
            readable_bind_group_layout,
            writable_bind_group_layout,
            matrix_matrix_pipeline_layout,
            max_dimension: MIN_DIMENSION,
            dot_pipeline,
            add_pipeline,
        })
    }

    pub unsafe fn recompile(&mut self, device: &Device, new_dimension: f64) {
        // test_init("MatrixPipelines::recompile")?;
        let nearest_power_of_2 = 2f64.powi(f64::log2(new_dimension).ceil() as i32);

        if nearest_power_of_2 > self.max_dimension {
            let (dot_pipeline, add_pipeline) = Self::compile_pipelines(
                device,
                nearest_power_of_2,
                &self.matrix_matrix_pipeline_layout,
            );

            self.dot_pipeline = dot_pipeline;
            self.add_pipeline = add_pipeline;

            self.max_dimension = nearest_power_of_2;
        }
    }
}

#[derive(Debug)]
pub struct Matrix {
    // WGPU
    device: Rc<Device>,
    queue: Rc<Queue>,
    pipeline_info: Rc<MatrixPipelines>,

    rows: u32,
    cols: u32,
    // GPU vals
    dimensions: Buffer,
    data: Buffer,
    transpose: Buffer,

    // Bind Groups
    readable_bind_group: BindGroup,
    writable_bind_group: BindGroup,
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
            contents: bytemuck::cast_slice(&vec![shape.0 as u32, shape.1 as u32]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let transpose = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Matrix Transpose Buffer"),
            contents: bytemuck::cast_slice(&[false as u32]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
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
            transpose,
            readable_bind_group,
            writable_bind_group,
        })
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
