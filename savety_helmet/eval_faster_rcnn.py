# Faster R-CNN (Detectron2) validation uz helmet_val
# Piemērs palaižot komandrindai: python savety_helmet/eval_faster_rcnn.py --weights runs/faster_rcnn/model_final.pth


import argparse
import os
import detectron2
import torch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


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
    cfg.DATASETS.TRAIN = ("helmet_train",)
    cfg.DATASETS.TEST = ("helmet_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.INPUT.MIN_SIZE_TEST = args.imgsz
    cfg.INPUT.MAX_SIZE_TEST = args.imgsz
    cfg.MODEL.WEIGHTS = args.weights
    cfg.SOLVER.IMS_PER_BATCH = args.batch
    cfg.OUTPUT_DIR = args.output
    cfg.MODEL.DEVICE = args.device
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Trenētie svari (.pth)")
    ap.add_argument("--data-root", default="datasets/helmet", help="COCO anotāciju un attēlu mape")
    ap.add_argument("--output", default="runs/faster_rcnn_eval", help="Izvades mape")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="cuda/cpu vai tukšs auto-izvēlei")
    args = ap.parse_args()

    args.device = pick_device(args.device)
    print("Izmanto ierīci:", args.device)

    register_datasets(args.data_root)
    cfg = build_cfg(args)

    evaluator = COCOEvaluator("helmet_val", cfg, False, output_dir=args.output)
    val_loader = build_detection_test_loader(cfg, "helmet_val")
    trainer = DefaultTrainer(cfg)
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    print(results)


if __name__ == "__main__":
    main()
