"""Offline Python/source consistency checks; C++ checker must run on robot."""
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parent
BUNDLE=ROOT/'patches/hsl_field_20261112'
CANONICAL = ROOT.parent / 'MX3_Match_Sim'
if not (CANONICAL / 'src').is_dir():
    CANONICAL = Path('C:/Users/jes/Desktop/MX3_Match_Sim/MX3_Match_Sim')
sys.path.insert(0,str(CANONICAL/'src/hsl_src/hsl_behavior'))
sys.path.insert(0,str(CANONICAL/'src/mx3_src/humanoid_localization'))
from hsl_behavior.geometry import Field
from hsl_behavior.strategy import goalkeeper_pose
from humanoid_localization.vision.field_geometry import practice_field_geometry
from humanoid_localization.vision.field_landmarks import generate_field_landmarks


class FieldTests(unittest.TestCase):
    def test_python_and_cpp_constants_match_photo(self):
        expected=json.loads((BUNDLE/'practice_photo.json').read_text())
        f=Field.for_profile('practice')
        for k,v in expected.items():
            if k!='source':self.assertAlmostEqual(getattr(f,k),v)
        header=(CANONICAL/'src/hsl_src/hsl_world_model/include/hsl_world_model/field_model.hpp').read_text(encoding='utf-8-sig')
        block=header.split('if(profile == "practice")')[1].split('return f;')[0]
        values={k:float(v) for k,v in re.findall(r'f\.(\w+)\s*=\s*([0-9.]+)\s*;',block)}
        for k in ['length','width','center_circle_radius','penalty_mark_dist','penalty_area_depth',
                  'penalty_area_width','goal_area_depth','goal_area_width','goal_width','border_x','border_y']:
            self.assertAlmostEqual(values[k],expected[k],msg=k)
        self.assertIn('hl() + border_x',header)
        self.assertIn('hw() + border_y',header)

    def test_actual_legacy_landmark_map_and_start_position(self):
        legacy=practice_field_geometry();f=Field.for_profile('practice')
        pairs={'length':'length_m','width':'width_m','center_circle_radius':'center_circle_radius_m',
               'penalty_mark_dist':'penalty_mark_distance_from_goal_line_m',
               'penalty_area_depth':'penalty_area_depth_m','penalty_area_width':'penalty_area_width_m',
               'goal_area_depth':'goal_area_depth_m','goal_area_width':'goal_area_width_m','goal_width':'goal_width_m'}
        for a,b in pairs.items():self.assertAlmostEqual(getattr(f,a),getattr(legacy,b))
        landmarks=generate_field_landmarks(legacy)
        has=lambda kind,x,y:any(abs(p.x-x)<1e-9 and abs(p.y-y)<1e-9 for p in landmarks[kind])
        for kind,x,y in [(1,-1.95,.565),(6,-1.95,1.085),(6,-1.95,.65),
                          (7,-1.08,1.085),(7,-1.52,.65),(5,-1.30,0),(5,0,.35)]:
            self.assertTrue(has(kind,x,y),(kind,x,y))
        self.assertEqual(sum(map(len,landmarks.values())),31)
        self.assertTrue(.65<.7583333<1.085)
        self.assertAlmostEqual((.65+1.085)/2,.8675)

    def test_s_field_and_derived_strategy_effects(self):
        s=Field.for_profile('s_field')
        self.assertEqual((s.length,s.width,s.center_circle_radius,s.penalty_area_width,s.goal_width),
                         (9.,6.,.75,4.,1.8))
        self.assertEqual((s.carpet_length,s.carpet_width),(11.,8.))
        f=Field.for_profile('practice')
        self.assertAlmostEqual(goalkeeper_pose((0,100),f).y,.415)
        self.assertAlmostEqual(goalkeeper_pose((0,-100),f).y,-.415)
        self.assertAlmostEqual(min(.4,f.border_y*.5),.1)
        self.assertAlmostEqual(min(.4,s.border_y*.5),.4)

    def test_apply_is_guarded_backed_up_and_repeatable(self):
        manifest=json.loads((BUNDLE/'manifest.json').read_text())
        for name,digest in manifest['bundle_files'].items():
            self.assertEqual(hashlib.sha256((BUNDLE/name).read_bytes()).hexdigest(),digest)
        source=(BUNDLE/'apply_field.py').read_text()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);proc=root/'proc';proc.mkdir()
            for entry in manifest['sources']:
                before=BUNDLE/'before'/entry['payload']
                data=before.read_bytes() if before.exists() else (BUNDLE/entry['payload']).read_bytes()
                p=root/entry['path'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
            code=source.replace("Path('/home/robo04/sim2real_ws')",'Path('+repr(str(root))+')')\
                .replace("Path('/proc')",'Path('+repr(str(proc))+')')
            def run():
                try:exec(compile(code,'installer-test','exec'),{'__file__':str(BUNDLE/'apply_field.py')})
                except SystemExit as e:return e.code
            run()
            for entry in manifest['sources']:
                self.assertEqual((root/entry['path']).read_bytes(),(BUNDLE/entry['payload']).read_bytes())
            self.assertTrue((root/'runtime_logs/COLCON_IGNORE').exists())
            self.assertTrue((root/'src/hsl_src/hsl_world_model/config').is_dir())
            self.assertTrue(list((root/'runtime_logs').glob('practice_photo_*/field_model.hpp')))
            self.assertEqual(run(),0)
            p=root/manifest['sources'][0]['path'];p.write_bytes(b'unknown=1\n')
            self.assertIn('Source differs',run());self.assertEqual(p.read_bytes(),b'unknown=1\n')
            active=proc/'123';active.mkdir();(active/'cmdline').write_bytes(b'/x/self_localizer\0')
            self.assertIn('STOPPED',run())


if __name__=='__main__':
    unittest.main()
