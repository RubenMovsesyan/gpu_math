#[macro_export]
macro_rules! matrix_matrix_2d_pipeline {
    ($device:expr, $queue:expr, $label:expr, $pipeline:expr, $source1:expr, $source2:expr, $destination:expr) => {
        let mut encoder = $device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some(&format!("{} Command Encoder", $label)),
        });

        {
            // Get the workgroup size
            let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                ($destination.rows, $destination.cols),
                (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
            );

            // Begin the compute pass
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(&format!("{} Compute Pass", $label)),
                timestamp_writes: None,
            });

            // Set the pipelines
            compute_pass.set_pipeline($pipeline);

            // Set the bind groups
            compute_pass.set_bind_group(0, &$source1.readable_bind_group, &[]);
            compute_pass.set_bind_group(1, &$source1.readable_bind_group, &[]);
            compute_pass.set_bind_group(2, &$destination.writable_bind_group, &[]);

            compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
        }

        $queue.submit(Some(encoder.finish()));
    };
}

#[macro_export]
macro_rules! matrix_matrix_1d_pipeline {
    ($device:expr, $queue:expr, $label:expr, $pipeline:expr, $source1:expr, $source2:expr, $destination:expr) => {
        let mut encoder = $device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some(&format!("{} Command Encoder", $label)),
        });

        {
            // Get the workgroup size
            let dispatch_size =
                compute_workgroup_size($destination.rows * $destination.cols, WORK_GROUP_SIZE);

            // Begin the compute pass
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(&format!("{} Compute Pass", $label)),
                timestamp_writes: None,
            });

            // Set the pipelines
            compute_pass.set_pipeline($pipeline);

            // Set the bind groups
            compute_pass.set_bind_group(0, &$source1.readable_bind_group, &[]);
            compute_pass.set_bind_group(1, &$source1.readable_bind_group, &[]);
            compute_pass.set_bind_group(2, &$destination.writable_bind_group, &[]);

            compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
        }

        $queue.submit(Some(encoder.finish()));
    };
}

#[macro_export]
macro_rules! matrix_scalar_pipline {
    ($device:expr, $queue:expr, $label:expr, $pipeline:expr, $source1:expr, $destination:expr) => {
        let mut encoder = $device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Matrix Scalar Command Encoder"),
        });

        {
            // Get the workgroup size
            let (dispatch_width, dispatch_height) = compute_workgroup_size_2d(
                ($source1.rows, $source1.cols),
                (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
            );

            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Matrix Scalar Command Encoder"),
                timestamp_writes: None,
            });

            // Set the pipeline
            compute_pass.set_pipeline($pipeline);

            // Set the bind groups
            compute_pass.set_bind_group(0, &$source1.readable_bind_group, &[]);
            compute_pass.set_bind_group(1, &$source1.scalar_bind_group, &[]);
            compute_pass.set_bind_group(2, &$destination.writable_bind_group, &[]);

            compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
        }

        $queue.submit(Some(encoder.finish()));
    };
}

#[macro_export]
macro_rules! matrix_dot_pipline {
    ($device:expr, $queue:expr, $label:expr, $pipeline:expr, $source1:expr, $source2:expr, $destination:expr) => {
        let mut encoder = $device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Matrix Dot Command Encdoer"),
        });

        {
            // Get the workgroup size
            let (dispatch_width, dispatch_height) = {
                let (source1_dispatch_width, source1_dispatch_height) = compute_workgroup_size_2d(
                    ($source1.rows, $source1.cols),
                    (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                );

                let (source2_dispatch_width, _source2_dispatch_height) = compute_workgroup_size_2d(
                    ($source2.rows, $source2.cols),
                    (WORK_GROUP_SIZE_2D, WORK_GROUP_SIZE_2D),
                );

                (
                    source1_dispatch_width,
                    source1_dispatch_height * source2_dispatch_width,
                )
            };

            // Begin the compute pass
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some(&format!("{} Compute Pass", $label)),
                timestamp_writes: None,
            });

            // Set the pipeline
            compute_pass.set_pipeline($pipeline);

            // Set the bind groups
            compute_pass.set_bind_group(0, &$source1.readable_bind_group, &[]);
            compute_pass.set_bind_group(1, &$source2.readable_bind_group, &[]);
            compute_pass.set_bind_group(2, &$destination.writable_bind_group, &[]);

            compute_pass.dispatch_workgroups(dispatch_width, dispatch_height, 1);
        }

        $queue.submit(Some(encoder.finish()));
    };
}
