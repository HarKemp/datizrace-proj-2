# Vizualizācija Faster R-CNN (Detectron2) prognozēm uz val attēliem
# Piemērs palaižot komandrinda:
# python savety_helmet/visualize_faster_rcnn.py --weights runs/faster_rcnn/model_final.pth --num 5

import argparse
import os
import random
import cv2
import detectron2
import torch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode


def register_datasets(root: str) -> None:
    train_json = os.path.join(root, "annotations_train.json")
    val_json = os.path.join(root, "annotations_val.json")
    train_img = os.path.join(root, "images", "train")
    val_img = os.path.join(root, "images", "val")
    register_coco_instances("helmet_train", {}, train_json, train_img)
    register_coco_instances("helmet_val", {}, val_json, val_img)
    MetadataCatalog.get("helmet_train").thing_classes = ["helmet", "head", "person"]
    MetadataCatalog.get("helmet_val").thing_classes = ["helmet", "head", "person"]


def pick_device(d: str | None) -> str:
    # Izvēlas ierīci: ja nenorāda komandrindā, mēģina cuda (klasterī) un tad cpu
    if d:
        return d
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_cfg(args: argparse.Namespace) -> detectron2.config.CfgNode:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TEST = ("helmet_val",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.INPUT.MIN_SIZE_TEST = args.imgsz
    cfg.INPUT.MAX_SIZE_TEST = args.imgsz
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.conf
    cfg.MODEL.DEVICE = args.device
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Trenētie svari (.pth)")
    ap.add_argument("--data-root", default="datasets/helmet", help="COCO anotāciju un attēlu mape")
    ap.add_argument("--output", default="runs/faster_rcnn_vis", help="Izvades mape vizualizācijām")
    ap.add_argument("--imgsz", type=int, default=640, help="Ievades izmērs (kvadrāts)")
    ap.add_argument("--conf", type=float, default=0.25, help="Konfidences slieksnis vizualizācijām")
    ap.add_argument("--num", type=int, default=5, help="Cik val attēlus saglabāt")
    ap.add_argument("--device", default=None, help="cuda/cpu vai tukšs auto-izvēlei")
    args = ap.parse_args()

    args.device = pick_device(args.device)
    print("Izmanto ierīci:", args.device)

    os.makedirs(args.output, exist_ok=True)
    register_datasets(args.data_root)
    cfg = build_cfg(args)
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get("helmet_val")

    val_dir = os.path.join(args.data_root, "images", "val")
    imgs = sorted([p for p in os.listdir(val_dir) if p.lower().endswith(".png")])
    if args.num < len(imgs):
        imgs = random.sample(imgs, args.num)

    for name in imgs:
        img_path = os.path.join(val_dir, name)
        image = cv2.imread(img_path)
        outputs = predictor(image)
        v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0, instance_mode=ColorMode.IMAGE)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_path = os.path.join(args.output, name)
        cv2.imwrite(out_path, v.get_image()[:, :, ::-1])
        print("Saglabāts:", out_path)


if __name__ == "__main__":
    main()
