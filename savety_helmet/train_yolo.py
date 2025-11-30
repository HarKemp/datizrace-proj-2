# Treniņa skripts YOLOv8 modeļiem priekš Safety Helmet datu kopas

# Izmanto priekšapmācītus svarus (yolov8s.pt vai yolov8m.pt) - šeit pēc default yolov8s, bet to mainu iekš
# komandrindas, palaižot python savety_helmet/train_yolo.py --model yolov8s.pt vai 
# python savety_helmet/train_yolo.py --model yolov8m.pt 
# Trenē uz safety-helmet.yaml datiem

import argparse
import os
from ultralytics import YOLO

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def pick_device(d: str | None) -> str:
    # Izvēlas ierīci: ja nenorāda komandrindā, mēģina cuda (klasterī), tad mps (Apple GPU) un tad cpu
    if d:
        return d
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="savety_helmet/safety-helmet.yaml", help="YOLO datu YAML")
    ap.add_argument("--model", default="yolov8s.pt", help="Priekšapmācītais YOLO modelis")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="cuda/mps/cpu vai tukšs auto-izvēlei")
    ap.add_argument("--name", default=None, help="YOLO runa mapes nosaukums (runs/detect/<name>)")
    ap.add_argument("--project", default="runs/detect", help="YOLO projekta mape")
    args = ap.parse_args()

    device = pick_device(args.device)
    print("Izmanto ierīci:", device)

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
