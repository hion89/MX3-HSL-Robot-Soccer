"""Verify critical downloadable sources and policy hashes without ROS or motor commands."""
import hashlib
from pathlib import Path
import sys

sys.dont_write_bytecode = True
root = Path(__file__).resolve().parents[1]
canonical = root / 'MX3_Match_Sim'
sys.path.insert(0, str(canonical / 'src/hsl_src/hsl_behavior'))
from hsl_behavior.geometry import Field

models = canonical / 'src/hsl_src/hsl_bringup/models'
expected = {
    'walk_v0_6_13500.onnx': '72e2b8d88bac39feb97bc7789e0a67e4675650713c6e4528988a32f96831fbb1',
    'walkkick_v1_5b_22300.onnx': 'f71c4db1ecc0db2dd3f40de4a8008c87d88106b1ed89881c30d2722f48c3779f',
}
for name, digest in expected.items():
    actual = hashlib.sha256((models / name).read_bytes()).hexdigest()
    if actual != digest:
        raise SystemExit('POLICY_HASH_MISMATCH: ' + name)
    print('PASS policy hash:', name)
f = Field.for_profile('practice')
for name, value in {'length': 3.9, 'width': 2.6, 'center_circle_radius': .35,
                    'penalty_area_depth': .87, 'penalty_area_width': 2.17,
                    'goal_area_depth': .43, 'goal_area_width': 1.3,
                    'goal_width': 1.13, 'border_x': .55, 'border_y': .20}.items():
    if abs(getattr(f, name) - value) > 1e-9:
        raise SystemExit('FIELD_MISMATCH: ' + name)
print('PASS practice field geometry')
pose = root / 'sim2real_evidence/deploy/init_practice_pose_20261112.sh'
if 'x: -1.95, y: 0.8675, z: 0.0' not in pose.read_text(encoding='utf-8'):
    raise SystemExit('INITIAL_POSE_MISMATCH')
if not (canonical / 'src/hsl_src/hsl_world_model/config').is_dir():
    raise SystemExit('MISSING_CMAKE_CONFIG_DIRECTORY')
print('PASS initial pose and required config directory')
print('CHECKOUT_VERIFIED (offline; no ROS runtime or robot deployment claimed)')
