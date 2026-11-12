"""Read-only model identity and CPU ONNXRuntime loading check; no ROS nodes."""
import hashlib
from pathlib import Path

from ament_index_python.packages import get_package_share_directory


def main():
    selected = Path(get_package_share_directory('hsl_bringup')) / 'models/walk_v0_6_13500.onnx'
    default = Path(get_package_share_directory('humanoid_policy_runner')) / 'models/walk_policy.onnx'
    expected = '72e2b8d88bac39feb97bc7789e0a67e4675650713c6e4528988a32f96831fbb1'
    for label, path in [('SELECTED', selected), ('LAUNCH_DEFAULT', default)]:
        print(label + ':', path, flush=True)
        print('exists:', path.is_file(), flush=True)
        if path.is_file():
            print('SHA256:', hashlib.sha256(path.read_bytes()).hexdigest(), flush=True)
    if not selected.is_file():
        raise SystemExit('FAIL: selected model is missing')
    if hashlib.sha256(selected.read_bytes()).hexdigest() != expected:
        raise SystemExit('FAIL: selected model differs from the reviewed source snapshot')
    import numpy as np
    import onnxruntime as ort
    print('NumPy:', np.__version__, np.__file__, flush=True)
    print('ONNXRuntime:', ort.__version__, flush=True)
    print('available providers:', ort.get_available_providers(), flush=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(selected), sess_options=options,
                                  providers=['CPUExecutionProvider'])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    print('session providers:', session.get_providers(), flush=True)
    print('inputs:', [(v.name, v.shape, v.type) for v in inputs], flush=True)
    print('outputs:', [(v.name, v.shape, v.type) for v in outputs], flush=True)
    if len(inputs) != 1 or len(outputs) != 1:
        raise SystemExit('FAIL: expected one policy input and one output')
    if inputs[0].shape[-1] != 240 or outputs[0].shape[-1] != 12:
        raise SystemExit('FAIL: expected input 240 and output 12')
    print('WALK_MODEL_LOAD_CHECK_PASSED (CPU load/shape only; no motor commands)', flush=True)


if __name__ == '__main__':
    main()
