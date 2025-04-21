use wgpu::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    BufferBindingType, ComputePipeline, ComputePipelineDescriptor, Device,
    PipelineCompilationOptions, PipelineLayout, PipelineLayoutDescriptor, ShaderStages,
    include_wgsl,
};

use crate::errors::GpuMathNotInitializedError;

#[allow(dead_code)]
#[derive(Debug)]
pub struct MatrixPipelines {
    // Bind Group Layouts
    pub readable_bind_group_layout: BindGroupLayout,
    pub scalar_bind_group_layout: BindGroupLayout,
    pub writable_bind_group_layout: BindGroupLayout,

    // Pipeline Layouts
    matrix_matrix_pipeline_layout: PipelineLayout,
    matrix_scalar_pipeline_layout: PipelineLayout,
    matrix_matrix_in_place_pipeline_layout: PipelineLayout,

    // Pipelines
    pub dot_pipeline: ComputePipeline,
    pub add_pipeline: ComputePipeline,
    pub add_in_place_pipeline: ComputePipeline,
    pub add_scalar_pipeline: ComputePipeline,
    pub sub_pipeline: ComputePipeline,
    pub sub_in_place_pipeline: ComputePipeline,
    pub sub_scalar_pipeline: ComputePipeline,
    pub mult_scalar_pipeline: ComputePipeline,

    // Vectored Pipelines
    pub vectored_add_pipeline: ComputePipeline,
    pub vectored_add_in_place_pipeline: ComputePipeline,
}

impl MatrixPipelines {
    fn compile_pipelines(
        device: &Device,
        matrix_matrix_pipeline_layout: &PipelineLayout,
        matrix_scalar_pipeline_layout: &PipelineLayout,
        matrix_matrix_in_place_pipeline_layout: &PipelineLayout,
    ) -> (
        ComputePipeline,
        ComputePipeline,
        ComputePipeline,
        ComputePipeline,
        ComputePipeline,
        ComputePipeline,
        ComputePipeline,
        ComputePipeline,
        ComputePipeline,
        ComputePipeline,
    ) {
        let dot_pipeline = {
            let shader = device.create_shader_module(include_wgsl!("shaders/dotting.wgsl"));

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Dot Pipeline"),
                module: &shader,
                layout: Some(matrix_matrix_pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("dot_main"),
            })
        };

        let add_pipeline = {
            let shader = device.create_shader_module(include_wgsl!("shaders/adding.wgsl"));

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Add Pipeline"),
                module: &shader,
                layout: Some(matrix_matrix_pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("add_main"),
            })
        };

        let add_in_place_pipeline = {
            let shader = device.create_shader_module(include_wgsl!("shaders/adding_in_place.wgsl"));

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Add In Place Pipeline"),
                module: &shader,
                layout: Some(matrix_matrix_in_place_pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("add_in_place_main"),
            })
        };

        let add_scalar_pipeline = {
            let shader = device.create_shader_module(include_wgsl!("shaders/adding_scalar.wgsl"));

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Add Scalar Pipeline"),
                module: &shader,
                layout: Some(&matrix_scalar_pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("add_scalar_main"),
            })
        };

        let sub_pipeline = {
            let shader = device.create_shader_module(include_wgsl!("shaders/subing.wgsl"));

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Sub Pipeline"),
                module: &shader,
                layout: Some(matrix_matrix_pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("sub_main"),
            })
        };

        let sub_in_place_pipeline = {
            let shader = device.create_shader_module(include_wgsl!("shaders/subing_in_place.wgsl"));

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Sub In Place Pipeline"),
                module: &shader,
                layout: Some(matrix_matrix_in_place_pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("sub_in_place_main"),
            })
        };

        let sub_scalar_pipeline = {
            let shader = device.create_shader_module(include_wgsl!("shaders/subing_scalar.wgsl"));

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Sub Scalar Pipeline"),
                module: &shader,
                layout: Some(&matrix_scalar_pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("sub_scalar_main"),
            })
        };

        let mult_scalar_pipeline = {
            let shader = device.create_shader_module(include_wgsl!("shaders/mult_scalar.wgsl"));

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Mult Scalar Pipeline"),
                module: &shader,
                layout: Some(&matrix_scalar_pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("mult_scalar_main"),
            })
        };

        // Vectored pipelines
        let vectored_add_pipeline = {
            let shader = device.create_shader_module(include_wgsl!("shaders/vectored_add.wgsl"));

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Vectored Add Pipeline"),
                module: &shader,
                layout: Some(&matrix_matrix_pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("vectored_add_main"),
            })
        };

        let vectored_add_in_place_pipeline = {
            let shader =
                device.create_shader_module(include_wgsl!("shaders/vectored_add_in_place.wgsl"));

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Matrix Vectored Add In Place Pipeline"),
                module: &shader,
                layout: Some(&matrix_matrix_in_place_pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("vectored_add_in_place_main"),
            })
        };

        (
            dot_pipeline,
            add_pipeline,
            add_in_place_pipeline,
            add_scalar_pipeline,
            sub_pipeline,
            sub_in_place_pipeline,
            sub_scalar_pipeline,
            mult_scalar_pipeline,
            // Vectored Pipelines
            vectored_add_pipeline,
            vectored_add_in_place_pipeline,
        )
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

        // Create the readable bind group layout with the scalar buffer too
        let scalar_bind_group_layout =
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("Scalar Readable Bind Group Layout"),
                entries: &[
                    // Matrix Buffer
                    BindGroupLayoutEntry {
                        binding: 0,
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

        // This is the pipeline layout for a matrix scalar operation
        let matrix_scalar_pipeline_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Matrix Pipeline Layout"),
                bind_group_layouts: &[
                    &readable_bind_group_layout,
                    &scalar_bind_group_layout,
                    &writable_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        // This is the pipeline layout for matrix matrix operation in place
        let matrix_matrix_in_place_pipeline_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Matrix Matrix In Place Pipeline Layout"),
                bind_group_layouts: &[&writable_bind_group_layout, &readable_bind_group_layout],
                push_constant_ranges: &[],
            });

        let (
            dot_pipeline,
            add_pipeline,
            add_in_place_pipeline,
            add_scalar_pipeline,
            sub_pipeline,
            sub_in_place_pipeline,
            sub_scalar_pipeline,
            mult_scalar_pipeline,
            vectored_add_pipeline,
            vectored_add_in_place_pipeline,
        ) = Self::compile_pipelines(
            device,
            &matrix_matrix_pipeline_layout,
            &matrix_scalar_pipeline_layout,
            &matrix_matrix_in_place_pipeline_layout,
        );

        Ok(Self {
            readable_bind_group_layout,
            scalar_bind_group_layout,
            writable_bind_group_layout,
            matrix_matrix_pipeline_layout,
            matrix_scalar_pipeline_layout,
            matrix_matrix_in_place_pipeline_layout,
            dot_pipeline,
            add_pipeline,
            add_in_place_pipeline,
            add_scalar_pipeline,
            sub_pipeline,
            sub_in_place_pipeline,
            sub_scalar_pipeline,
            mult_scalar_pipeline,
            // Vector pipelines
            vectored_add_pipeline,
            vectored_add_in_place_pipeline,
        })
    }
}
