#![recursion_limit = "512"]
mod gpu_utils;

use std::rc::Rc;

// Logging
#[allow(unused_imports)]
use log::*;

// error modules
mod errors;

// Publically exposed modules
pub mod math;

use math::matrix_pipelines::MatrixPipelines;
// Include these when the library is imported
pub use math::*;

// Library inner imports
use pollster::FutureExt;
use wgpu::{
    Adapter, Backends, Device, DeviceDescriptor, Features, Instance, InstanceDescriptor, Limits,
    PowerPreference, Queue, RequestAdapterOptions,
};

// static mut GPU_MATHS: Option<GpuMath> = None;

#[allow(dead_code)]
pub struct GpuMath {
    instance: Instance,
    adapter: Adapter,
    device: Rc<Device>,
    queue: Rc<Queue>,

    // Math Pipelines
    matrix_pipelines: Rc<MatrixPipelines>,
}

impl GpuMath {
    pub fn new() -> Self {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .block_on()
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("Device and Queue"),
                    required_features: Features::default(),
                    required_limits: Limits {
                        max_compute_workgroup_storage_size: 32768,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                None,
            )
            .block_on()
            .unwrap();

        let (device, queue) = (Rc::new(device), Rc::new(queue));

        // Create the pipelines for matrix math
        let matrix_pipelines =
            unsafe { Rc::new(MatrixPipelines::init(&device).unwrap_unchecked()) };

        Self {
            instance,
            adapter,
            device,
            queue,
            matrix_pipelines,
        }
    }
}

// This initializes all the required gpu devices for the gpu math library
// This must be called before anything in this library is used
// pub fn init() {
//     // Initialize all the gpu required values
//     let instance = Instance::new(&InstanceDescriptor {
//         backends: Backends::all(),
//         ..Default::default()
//     });

//     let adapter = instance
//         .request_adapter(&RequestAdapterOptions {
//             power_preference: PowerPreference::HighPerformance,
//             force_fallback_adapter: false,
//             compatible_surface: None,
//         })
//         .block_on()
//         .unwrap();

//     let (device, queue) = adapter
//         .request_device(
//             &DeviceDescriptor {
//                 label: Some("Device and Queue"),
//                 required_features: Features::default(),
//                 required_limits: Limits {
//                     max_compute_workgroup_storage_size: 32768,
//                     ..Default::default()
//                 },
//                 ..Default::default()
//             },
//             None,
//         )
//         .block_on()
//         .unwrap();

//     // Create the pipelines for matrix math
//     let matrix_pipelines = unsafe { MatrixPipelines::init(&device).unwrap_unchecked() };

//     unsafe {
//         GPU_MATHS = Some(GpuMath {
//             instance,
//             adapter,
//             device,
//             queue,
//             matrix_pipelines,
//         });
//     }
// }

// pub fn cleanup() {
//     unsafe {
//         #[allow(static_mut_refs)]
//         if let Some(_math) = GPU_MATHS.take() {
//             info!("GPU_MATHS has been cleaned up");
//         }
//     }
// }

// fn test_init(location: &str) -> Result<(), GpuMathNotInitializedError> {
//     unsafe {
//         #[allow(static_mut_refs)]
//         if GPU_MATHS.is_none() {
//             Err(GpuMathNotInitializedError(location.to_string()))
//         } else {
//             Ok(())
//         }
//     }
// }

// // IMPORTANT: make sure to call test_init before this function
// unsafe fn get_device<'a>() -> &'a Device {
//     #[allow(static_mut_refs)]
//     unsafe {
//         &GPU_MATHS.as_ref().unwrap_unchecked().device
//     }
// }

// // IMPORTANT: make sure to call test_init before this function
// unsafe fn get_queue<'a>() -> &'a Queue {
//     #[allow(static_mut_refs)]
//     unsafe {
//         &GPU_MATHS.as_ref().unwrap_unchecked().queue
//     }
// }

// // IMPORTANT: make sure to call test_init before this function
// unsafe fn get_pipelines<'a>() -> &'a MatrixPipelines {
//     #[allow(static_mut_refs)]
//     unsafe {
//         &GPU_MATHS.as_ref().unwrap_unchecked().matrix_pipelines
//     }
// }

// // IMPORTANT: make sure to call test_init before this function
// unsafe fn recompile_pipelines(new_dimension: f64) {
//     #[allow(static_mut_refs)]
//     unsafe {
//         GPU_MATHS
//             .as_mut()
//             .unwrap_unchecked()
//             .matrix_pipelines
//             .recompile(new_dimension);
//     }
// }
