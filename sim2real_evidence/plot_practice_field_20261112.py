"""Render before/after geometry from the actual Python field definitions."""
import importlib.util
import math
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Rectangle

root=Path(__file__).resolve().parent
bundle=root/'patches/hsl_field_20261112'
for module_name,file in [('old_field',bundle/'before/geometry.py'),('new_field',bundle/'geometry.py')]:
    spec=importlib.util.spec_from_file_location(module_name,file)
    module=importlib.util.module_from_spec(spec);sys.modules[module_name]=module;spec.loader.exec_module(module)
    globals()[module_name]=module
font=Path('C:/Windows/Fonts/malgun.ttf')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family']='Malgun Gothic'
plt.rcParams['axes.unicode_minus']=False
fig,axes=plt.subplots(1,2,figsize=(14,7.5))
for ax,mod,title in zip(axes,[old_field,new_field],['수정 전: HSL 위치 추정·전략 지도','수정 후: 제공한 사진의 명시 치수']):
    f=mod.Field.for_profile('practice')
    cl,cw=(f.carpet_length,f.carpet_width) if hasattr(f,'carpet_length') else (4.9,3.6)
    ax.set_facecolor('#202020')
    ax.add_patch(Rectangle((-cl/2,-cw/2),cl,cw,facecolor='#1f3926',edgecolor='#95ac95',ls='--'))
    ax.add_patch(Rectangle((-f.hl,-f.hw),f.length,f.width,fill=False,edgecolor='white',lw=1.5))
    ax.plot([0,0],[-f.hw,f.hw],color='white',lw=1.5)
    ax.add_patch(Circle((0,0),f.center_circle_radius,fill=False,edgecolor='white',lw=1.5))
    for sign in [-1,1]:
        for depth,width in [(f.penalty_area_depth,f.penalty_area_width),(f.goal_area_depth,f.goal_area_width)]:
            x=sign*(f.hl-depth)
            ax.plot([sign*f.hl,x,x,sign*f.hl],[-width/2,-width/2,width/2,width/2],color='white')
        ax.scatter([sign*(f.hl-f.penalty_mark_dist)],[0],c='white',s=18)
        ax.scatter([sign*f.hl]*2,[-f.goal_width/2,f.goal_width/2],c='#ffd43b',s=45,zorder=4)
    ax.scatter([0],[0],c='white',s=18)
    ax.set_title(title,fontsize=13)
    ax.text(0,2.05,'라인 %.2f×%.2fm / 잔디·유효 경계 %.2f×%.2fm'%(f.length,f.width,cl,cw),ha='center',fontsize=10,color='white')
    ax.text(0,-2.05,'페널티 구역 %.2f×%.2fm | 골 구역 %.2f×%.2fm\n골대 폭 %.2fm | 센터서클 r=%.3fm' %
            (f.penalty_area_depth,f.penalty_area_width,f.goal_area_depth,f.goal_area_width,f.goal_width,f.center_circle_radius),
            ha='center',va='top',fontsize=10,color='white')
    ax.set_aspect('equal');ax.set_xlim(-2.8,2.8);ax.set_ylim(-2.35,2.35)
    ax.set_xlabel('지도 x [m], +x = 상대 골대');ax.set_ylabel('지도 y [m], +y = 왼쪽')
    ax.tick_params(colors='#222222')
fig.suptitle('사진 기준 연습장 지도 대조 — 실제 코드에서 생성한 치수',fontsize=16)
fig.tight_layout(rect=[0,0,1,.94])
out=root/'practice_field_20261112';out.mkdir(exist_ok=True)
fig.savefig(out/'practice_field_comparison.png',dpi=170)
print(out/'practice_field_comparison.png')
