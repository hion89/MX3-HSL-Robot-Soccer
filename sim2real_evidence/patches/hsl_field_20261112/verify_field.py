"""Verify Python strategy geometry and legacy landmark geometry against the photo."""
import importlib
import json
import math
from pathlib import Path
import sys

bundle = Path(__file__).resolve().parent
root = Path('/home/robo04/sim2real_ws')
if '--source' in sys.argv:
    sys.path.insert(0, str(root / 'src/hsl_src/hsl_behavior'))
    sys.path.insert(0, str(root / 'src/mx3_src/humanoid_localization'))
from hsl_behavior.geometry import Field
from humanoid_localization.vision.field_geometry import practice_field_geometry, s_field_geometry
from humanoid_localization.vision.field_landmarks import generate_field_landmarks

expected = json.loads((bundle / 'practice_photo.json').read_text())
field = Field.for_profile('practice')
legacy = practice_field_geometry()
mapping = {'length':'length_m', 'width':'width_m', 'center_circle_radius':'center_circle_radius_m',
           'penalty_mark_dist':'penalty_mark_distance_from_goal_line_m',
           'penalty_area_depth':'penalty_area_depth_m', 'penalty_area_width':'penalty_area_width_m',
           'goal_area_depth':'goal_area_depth_m', 'goal_area_width':'goal_area_width_m', 'goal_width':'goal_width_m'}
for name, value in expected.items():
    if name == 'source':
        continue
    assert math.isclose(getattr(field, name), value, abs_tol=1e-9), name
for name, other in mapping.items():
    assert math.isclose(getattr(field, name), getattr(legacy, other), abs_tol=1e-9), other
points = generate_field_landmarks(legacy)
has = lambda kind,x,y: any(abs(p.x-x)<1e-9 and abs(p.y-y)<1e-9 for p in points[kind])
assert has(1,-1.95,.565) and has(6,-1.95,1.085) and has(6,-1.95,.65)
assert has(7,-1.08,1.085) and has(7,-1.52,.65) and has(5,-1.30,0) and has(5,0,.35)
assert sum(map(len,points.values())) == 31
s = Field.for_profile('s_field')
assert (s.length,s.width,s.goal_width,s.penalty_area_width,s.center_circle_radius) == (9.,6.,1.8,4.,.75)
print('PASS: Python practice geometry and 31 legacy landmarks match photo; S-Field unchanged')
print('MODULE=' + str(Path(importlib.import_module('hsl_behavior.geometry').__file__).resolve()))
print('PRACTICE=' + json.dumps({k:getattr(field,k) for k in expected if k != 'source'}))
