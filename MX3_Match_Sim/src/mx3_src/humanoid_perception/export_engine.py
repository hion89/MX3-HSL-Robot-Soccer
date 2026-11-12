#!/usr/bin/env python3
"""
export_engine.py
================

YOLOv8 .pt 모델을 Jetson 용 TensorRT engine 으로 변환 (FP16).

⚠ 주의:
  - 빌드한 머신/GPU/JetPack 조합에서만 동작. 다른 Jetson 으로 옮기면 재빌드 필요.
  - 첫 빌드는 5~20분 소요 (TensorRT 최적화). 두 번째부터는 캐시 활용.
  - VRAM 4GB 정도 잡으므로 다른 GPU 작업 중이면 OOM 가능.

실행:
  python3 export_engine.py /path/to/best.pt
  python3 export_engine.py /path/to/best.pt --imgsz 640 --batch 1

결과:
  best.pt 옆에 best.engine 생성됨.
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pt_path', type=str,
                    help='Input .pt model path')
    ap.add_argument('--imgsz', type=int, default=640,
                    help='Inference input size (default 640)')
    ap.add_argument('--batch', type=int, default=1,
                    help='Batch size for engine (default 1)')
    ap.add_argument('--workspace', type=int, default=4,
                    help='TensorRT workspace GB (default 4)')
    ap.add_argument('--fp32', action='store_true',
                    help='Use FP32 instead of FP16 (slower but higher precision)')
    args = ap.parse_args()

    pt_path = Path(args.pt_path).expanduser().resolve()
    if not pt_path.exists():
        raise FileNotFoundError(pt_path)

    print(f"Loading: {pt_path}")
    model = YOLO(str(pt_path))

    print(f"Exporting to TensorRT (imgsz={args.imgsz}, batch={args.batch}, "
          f"half={'False' if args.fp32 else 'True'})...")
    print("이 작업은 5~20분 걸릴 수 있습니다. 진행 중...")

    out = model.export(
        format='engine',
        half=not args.fp32,
        device=0,
        imgsz=args.imgsz,
        workspace=args.workspace,
        batch=args.batch,
        simplify=True,
        verbose=True,
    )

    print(f"\n✓ 완료: {out}")
    print("다음 단계: launch 에서 engine_path 로 이 파일 경로 사용")


if __name__ == '__main__':
    main()