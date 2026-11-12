# Ball tracker TF reception and diagnostics — 2026-11-04

Observed: external tf2_echo receives odom/base_link to camera_color_optical_frame, but the ball
tracker continually reports that its camera source frame does not exist. Querying BallEstimate
also failed in a separate terminal with a message-import error. These are distinct observations.
The deployed TF frame definitions exist. The runtime cause of the private buffer failure has
not yet been isolated from node environment, DDS delivery, and listener execution.

## Reviewed change

- Attach TransformListener subscriptions to the ball-tracker node and its executor using
  `tf_listener_(tf_buf_, this, false)`, a Humble-supported constructor. The original buffer-only
  constructor is also a valid API; its presence alone is not proof of the root cause.
- All lookups remain non-blocking. Detection transforms still use the detection timestamp.
  Latest TF is checked for diagnosis only, never substituted for a missing timestamped transform.
- Log configured frames, first exact-time TF availability, and first initialized ball position.
- Capture TF errors and report publisher counts, cached frames, timestamps, and latest-versus-exact
  availability every three seconds while waiting, instead of unactionable warnings on every frame.
- Do not create or rename TF frames or change kinematics, model/gain parameters, or motor code.

The same executor can process TF callbacks because cycle/processPending do not wait for TF.
Self-localizer is not modified by this patch.

Humble API reference:
https://github.com/ros2/geometry2/blob/humble/tf2_ros/include/tf2_ros/transform_listener.hpp
https://github.com/ros2/geometry2/blob/humble/tf2_ros/include/tf2_ros/buffer.hpp

## Apply and validate

Stop only the ball_tracker process first. Keep hardware, camera, leg_odometry, and RSP running.
Run `bash apply_and_build.sh` from the extracted bundle. apply.py accepts only the reviewed old
hash or the already-patched hash; it makes a source backup, performs atomic replacement and
readback, and restores the old source if readback fails.

The script builds only hsl_world_model, sources the installation, runs an isolated ROS domain 77
test, then starts the real ball tracker in domain 42. The isolated test does not publish motor
commands or call actuator services. Its static TF is published before the C++ node starts;
future-stamped detections must remain invalid, and correctly timestamped detections must produce
a valid base position (1.1, 0). The startup marker checks that the patched executable was tested.

The script aborts on failure. An invalid message import will show the Python traceback, rather
than hiding it behind the ros2 topic generic error. Actual target compilation and runtime must
be assessed on the Jetson; they have not been executed on the Windows preparation host.
