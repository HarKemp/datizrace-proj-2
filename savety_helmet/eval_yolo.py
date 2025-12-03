# Evaluation skripts YOLOv8 un YOLO11 modeļiem priekš Safety Helmet datu kopas
# Palaiž val uz norādītā svaru faila, piem, iekš komandrindas 
# python savety_helmet/eval_yolo.py --weights runs/detect/<run_name>/weights/best.pt
# Un saglabā prognozes (save_txt/save_conf/save)

import argparse
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Modeļa svaru fails (piem, runs/detect/.../best.pt)")
    ap.add_argument("--data", default="savety_helmet/safety-helmet.yaml", help="YOLO datu YAML")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=None, help="cuda/mps/cpu vai tukšs auto-izvēlei")
    args = ap.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save_txt=True,
        save_conf=True,
        save=True,
    )
    print(metrics)


if __name__ == "__main__":
    main()
