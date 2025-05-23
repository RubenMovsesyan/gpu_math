use std::borrow::Cow;

use anyhow::Result;
use wgpu::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    BufferBindingType, ComputePipeline, ComputePipelineDescriptor, Device,
    PipelineCompilationOptions, PipelineLayout, PipelineLayoutDescriptor, ShaderModuleDescriptor,
    ShaderSource, ShaderStages, include_wgsl,
};

use crate::create_matrix_pipelines;

#[allow(dead_code)]
#[derive(Debug)]
pub struct MatrixPipelines {
    // Bind Group Layouts
    pub readable_bind_group_layout: BindGroupLayout,
    pub scalar_bind_group_layout: BindGroupLayout,
    pub sum_bind_group_layout: BindGroupLayout,
    pub writable_bind_group_layout: BindGroupLayout,

    // Pipeline Layouts
    matrix_matrix_pipeline_layout: PipelineLayout,
    matrix_scalar_pipeline_layout: PipelineLayout,
    matrix_scalar_in_place_pipeline_layout: PipelineLayout,
    matrix_sum_pipeline_layout: PipelineLayout,
    matrix_matrix_in_place_pipeline_layout: PipelineLayout,
    // Custom Layouts
    matrix_in_place_pipeline_layout: PipelineLayout,
    matrix_pipeline_layout: PipelineLayout,

    // Pipelines
    pub dot_pipeline: ComputePipeline,
    pub add_pipeline: ComputePipeline,
    pub add_in_place_pipeline: ComputePipeline,
    pub add_scalar_pipeline: ComputePipeline,
    pub add_scalar_in_place_pipeline: ComputePipeline,
    pub sub_pipeline: ComputePipeline,
    pub sub_in_place_pipeline: ComputePipeline,
    pub sub_scalar_pipeline: ComputePipeline,
    pub sub_scalar_in_place_pipeline: ComputePipeline,
    pub mult_scalar_pipeline: ComputePipeline,
    pub mult_scalar_in_place_pipeline: ComputePipeline,
    pub mult_pipeline: ComputePipeline,
    pub mult_in_place_pipeline: ComputePipeline,
    pub exp_pipeline: ComputePipeline,
    pub exp_in_place_pipeline: ComputePipeline,
    pub sum_pipeline: ComputePipeline,

    // Vectored Pipelines
    pub vectored_add_pipeline: ComputePipeline,
    pub vectored_add_in_place_pipeline: ComputePipeline,
    pub vectored_sub_pipeline: ComputePipeline,
    pub vectored_sub_in_place_pipeline: ComputePipeline,

    // Custom pipelines
    pub custom_pipelines: Vec<ComputePipeline>,
}

impl MatrixPipelines {
    pub fn init(device: &Device) -> Result<Self> {
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
                    // Scalar Buffer
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

        // Create the writable bind group layout with the sum buffer
        let sum_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Sum Writable Bind Group Layout"),
            entries: &[
                // Sum Buffer
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
                label: Some("Matrix Scalar Pipeline Layout"),
                bind_group_layouts: &[
                    &readable_bind_group_layout,
                    &scalar_bind_group_layout,
                    &writable_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        // This is the pipeline layout for a matrix sum operation
        let matrix_sum_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Maitrx Sum Pipeline Layout"),
            bind_group_layouts: &[&readable_bind_group_layout, &sum_bind_group_layout],
            push_constant_ranges: &[],
        });

        // This is the pipeline layout for matrix matrix operation in place
        let matrix_matrix_in_place_pipeline_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Matrix Matrix In Place Pipeline Layout"),
                bind_group_layouts: &[&writable_bind_group_layout, &readable_bind_group_layout],
                push_constant_ranges: &[],
            });

        // This is the pipeline layout for a matrix scalar operation in place
        let matrix_scalar_in_place_pipeline_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Matrix Scalar In Place Pipeline Layout"),
                bind_group_layouts: &[&writable_bind_group_layout, &scalar_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Pipeline layouts for custom pipelines
        let matrix_in_place_pipeline_layout =
            device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Maitrx in Place Pipeline Layout"),
                bind_group_layouts: &[&writable_bind_group_layout],
                push_constant_ranges: &[],
            });

        let matrix_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Matrix Pipeline Layout"),
            bind_group_layouts: &[&readable_bind_group_layout, &writable_bind_group_layout],
            push_constant_ranges: &[],
        });

        let (
            dot_pipeline,
            add_pipeline,
            add_in_place_pipeline,
            add_scalar_pipeline,
            add_scalar_in_place_pipeline,
            sub_pipeline,
            sub_in_place_pipeline,
            sub_scalar_pipeline,
            mult_scalar_pipeline,
            mult_scalar_in_place_pipeline,
            mult_pipeline,
            mult_in_place_pipeline,
        ) = create_matrix_pipelines!(
            device,
            (
                "shaders/dotting.wgsl",
                "Matrix Dot Pipeline",
                matrix_matrix_pipeline_layout,
                "dot_main"
            ),
            (
                "shaders/adding.wgsl",
                "Matrix Add Pipeline",
                matrix_matrix_pipeline_layout,
                "add_main"
            ),
            (
                "shaders/adding_in_place.wgsl",
                "Matrix Add In Place Pipeline",
                matrix_matrix_in_place_pipeline_layout,
                "add_in_place_main"
            ),
            (
                "shaders/adding_scalar.wgsl",
                "Matrix Add Scalar Pipeline",
                matrix_scalar_pipeline_layout,
                "add_scalar_main"
            ),
            (
                "shaders/adding_scalar_in_place.wgsl",
                "Matrix Add Scalar In Place Pipeline",
                matrix_scalar_in_place_pipeline_layout,
                "add_scalar_in_place_main"
            ),
            (
                "shaders/subing.wgsl",
                "Matrix Sub Pipeline",
                matrix_matrix_pipeline_layout,
                "sub_main"
            ),
            (
                "shaders/subing_in_place.wgsl",
                "Matrix Sub In Place Pipeline",
                matrix_matrix_in_place_pipeline_layout,
                "sub_in_place_main"
            ),
            (
                "shaders/subing_scalar.wgsl",
                "Matrix Sub Scalar Pipeline",
                matrix_scalar_pipeline_layout,
                "sub_scalar_main"
            ),
            (
                "shaders/mult_scalar.wgsl",
                "Matrix Mult Scalar Pipeline",
                matrix_scalar_pipeline_layout,
                "mult_scalar_main"
            ),
            (
                "shaders/mult_scalar_in_place.wgsl",
                "Matrix Mult Scalar In Place Pipeline",
                matrix_scalar_in_place_pipeline_layout,
                "mult_scalar_in_place_main"
            ),
            (
                "shaders/mult.wgsl",
                "Matrix Mult Pipeline",
                matrix_matrix_pipeline_layout,
                "mult_main"
            ),
            (
                "shaders/mult_in_place.wgsl",
                "Matrix Mult In Place Pipeline",
                matrix_matrix_in_place_pipeline_layout,
                "mult_in_place_main"
            )
        );

        // Vector pipelines
        let (
            vectored_add_pipeline,
            vectored_add_in_place_pipeline,
            vectored_sub_pipeline,
            vectored_sub_in_place_pipeline,
        ) = create_matrix_pipelines!(
            device,
            (
                "shaders/vectored_add.wgsl",
                "Matrix Vectored Add Pipeline",
                matrix_matrix_pipeline_layout,
                "vectored_add_main"
            ),
            (
                "shaders/vectored_add_in_place.wgsl",
                "Matrix Vectored Add In Place Pipeline",
                matrix_matrix_in_place_pipeline_layout,
                "vectored_add_in_place_main"
            ),
            (
                "shaders/vectored_sub.wgsl",
                "Matrix Vectored Sub Pipeline",
                matrix_matrix_pipeline_layout,
                "vectored_sub_main"
            ),
            (
                "shaders/vectored_sub_in_place.wgsl",
                "Matrix Vectored Sub In Place Pipeline",
                matrix_matrix_in_place_pipeline_layout,
                "vectored_sub_in_place_main"
            )
        );

        // Extra pipelines
        let (exp_pipeline, exp_in_place_pipeline, sum_pipeline, sub_scalar_in_place_pipeline) = create_matrix_pipelines!(
            device,
            (
                "shaders/exp.wgsl",
                "Matrix Exp Pipeline",
                matrix_scalar_pipeline_layout,
                "exp_main"
            ),
            (
                "shaders/exp_in_place.wgsl",
                "Matrix Exp In Place Pipeline",
                matrix_scalar_in_place_pipeline_layout,
                "exp_in_place_main"
            ),
            (
                "shaders/sum.wgsl",
                "Matrix Sum Pipeline",
                matrix_sum_pipeline_layout,
                "sum_main"
            ),
            (
                "shaders/subing_scalar_in_place.wgsl",
                "Matrix Sub Scalar In Place Pipeline",
                matrix_scalar_in_place_pipeline_layout,
                "sub_scalar_in_place_main"
            )
        );

        Ok(Self {
            readable_bind_group_layout,
            scalar_bind_group_layout,
            sum_bind_group_layout,
            writable_bind_group_layout,
            matrix_matrix_pipeline_layout,
            matrix_scalar_pipeline_layout,
            matrix_scalar_in_place_pipeline_layout,
            matrix_sum_pipeline_layout,
            matrix_matrix_in_place_pipeline_layout,
            matrix_in_place_pipeline_layout,
            matrix_pipeline_layout,
            dot_pipeline,
            add_pipeline,
            add_in_place_pipeline,
            add_scalar_pipeline,
            add_scalar_in_place_pipeline,
            sub_pipeline,
            sub_in_place_pipeline,
            sub_scalar_pipeline,
            sub_scalar_in_place_pipeline,
            mult_scalar_pipeline,
            mult_scalar_in_place_pipeline,
            mult_pipeline,
            mult_in_place_pipeline,
            exp_pipeline,
            exp_in_place_pipeline,
            sum_pipeline,
            // Vector pipelines
            vectored_add_pipeline,
            vectored_add_in_place_pipeline,
            vectored_sub_pipeline,
            vectored_sub_in_place_pipeline,
            // Custom pipelines
            custom_pipelines: Vec::new(),
        })
    }

    /// Creates a custom in place pipeline from the shader and returns the index of that pipeline
    pub fn create_custom_matrix_in_place_pipeline(
        &mut self,
        device: &Device,
        shader: &str,
    ) -> usize {
        let index = self.custom_pipelines.len();

        self.custom_pipelines.push({
            let pipeline_shader = device.create_shader_module(ShaderModuleDescriptor {
                label: Some("Custom Pipeline Shader"),
                source: ShaderSource::Wgsl(Cow::Borrowed(shader)),
            });

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Custom Compute Pipeline"),
                layout: Some(&self.matrix_in_place_pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("op_main"),
                module: &pipeline_shader,
            })
        });

        index
    }

    /// Creates a custom matrix scalar in place pipeline from the shader and returns the index of that pipeline
    pub fn create_custom_matrix_scalar_in_place_pipeline(
        &mut self,
        device: &Device,
        shader: &str,
    ) -> usize {
        let index = self.custom_pipelines.len();

        self.custom_pipelines.push({
            let pipeline_shader = device.create_shader_module(ShaderModuleDescriptor {
                label: Some("Custome Pipeline Shader"),
                source: ShaderSource::Wgsl(Cow::Borrowed(shader)),
            });

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Custome Compute Pipeline"),
                layout: Some(&self.matrix_scalar_in_place_pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("op_main"),
                module: &pipeline_shader,
            })
        });

        index
    }

    /// Creates a custom single op pipeline from the shader and returns the index of that pipeline
    pub fn create_custom_matrix_pipeline(&mut self, device: &Device, shader: &str) -> usize {
        let index = self.custom_pipelines.len();

        self.custom_pipelines.push({
            let pipeline_shader = device.create_shader_module(ShaderModuleDescriptor {
                label: Some("Custom Pipeline Shader"),
                source: ShaderSource::Wgsl(Cow::Borrowed(shader)),
            });

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Custom Compute Pipeline"),
                layout: Some(&self.matrix_pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("op_main"),
                module: &pipeline_shader,
            })
        });

        index
    }

    /// Creates a custom multi op pipeline from the shader and returns the index of that pipeline
    pub fn create_custom_matrix_matrix_pipeline(&mut self, device: &Device, shader: &str) -> usize {
        let index = self.custom_pipelines.len();

        self.custom_pipelines.push({
            let pipeline_shader = device.create_shader_module(ShaderModuleDescriptor {
                label: Some("Custom Pipeline Shader"),
                source: ShaderSource::Wgsl(Cow::Borrowed(shader)),
            });

            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Custom Compute Pipeline"),
                layout: Some(&self.matrix_matrix_pipeline_layout),
                cache: None,
                compilation_options: PipelineCompilationOptions::default(),
                entry_point: Some("op_main"),
                module: &pipeline_shader,
            })
        });

        index
    }
}
