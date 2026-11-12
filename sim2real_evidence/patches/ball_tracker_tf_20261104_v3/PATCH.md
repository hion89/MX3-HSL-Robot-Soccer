# Ball tracker TF fix, revision 3 - 2026-11-04

The robot compiled revision 2 successfully, then the isolated ROS test failed.
The revision-2 constructor `tf_listener_(tf_buf_, this, false)` disabled the real
TF executor thread. Humble Buffer::canTransform checks for a dedicated thread
even when its ROS-time overload receives a zero timeout. This was an error in
the revision-2 patch, distinct from the original missing-frame symptom.

## Reviewed correction

Use `tf_listener_(tf_buf_, this, true)`. Humble TransformListener then creates a
dedicated callback group and executor thread and sets the buffer's dedicated-thread
flag itself. Merely setting the flag without creating a thread is not the fix.

Primary implementation references:

- https://github.com/ros2/geometry2/blob/humble/tf2_ros/src/buffer.cpp
- https://github.com/ros2/geometry2/blob/humble/tf2_ros/include/tf2_ros/transform_listener.hpp

Retain revision-2 timestamp checks and diagnostics. Frames, kinematics, motor code,
gains and model files are unchanged. Latest TF is diagnostic only; detection-time
TF is required for processing. The original buffer-only listener constructor was
also a valid API; the original persistent missing-frame cause is not yet proven.

## Apply and validate

Keep hardware, camera, leg_odometry and RSP running. The applier refuses to update
a running ball_tracker. It accepts only the reviewed original, revision-2 or
revision-3 source hash, backs up the current source and atomically replaces it.
Repeated application is idempotent; failed readback restores the previous source.

`bash apply_and_build.sh` builds only hsl_world_model, runs the installed C++ node
in isolated domain 77, and starts the real tracker in domain 42 only after the
test succeeds. The isolated test uses no motor topics or actuator services.

The strengthened test requires:

1. Static TF published before C++ startup must be received by that C++ buffer.
2. Future detections remain invalid while latest odom and base TF are demonstrably
   available in the same buffer. No unavailable-TF result is counted as a pass.
3. Correctly timestamped detections produce valid, finite base coordinates (1.1,0).
4. The installed executable emits the revision-3 startup marker and no dedicated
   thread requirement error.

Seven stdlib helper tests cover inspection, backup/idempotence, revision-2 upgrade,
unreviewed-source rejection, active-process rejection, failed-readback rollback
and Linux process detection. Python 3.10 syntax, Bash syntax and archive integrity
are checked on Windows. Actual revision-3 C++ compilation and ROS integration
remain pending on the Jetson; no Windows result substitutes for those checks.
