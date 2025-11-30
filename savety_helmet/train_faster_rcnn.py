# Faster R-CNN (Detectron2, R50-FPN) treniņa skripts priekš Safety Helmet datu kopas (COCO anotācijas)

import argparse
import os
import detectron2
import torch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances


def register_datasets(root: str) -> None:
    # Reģistrē COCO anotācijas train/val un klašu iestatīšana
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
    # Uzstāda Detectron2 config no COCO base, coco anotācijām train un test, batch/LR/iter/imgsize, klases=3
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("helmet_train",)
    cfg.DATASETS.TEST = ("helmet_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = args.batch
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.STEPS = []  # nav learning rate samazināšana treniņa laikā, jo priekš pirmā run max iter tikai 2000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.INPUT.MIN_SIZE_TRAIN = (args.imgsz,)
    cfg.INPUT.MAX_SIZE_TRAIN = args.imgsz
    cfg.INPUT.MIN_SIZE_TEST = args.imgsz
    cfg.INPUT.MAX_SIZE_TEST = args.imgsz
    cfg.OUTPUT_DIR = args.output
    cfg.MODEL.DEVICE = args.device
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg


def main() -> None:
    # cli parametri ar datu sakni, output mapi, batch/LR/iter/imgsz, resume
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="datasets/helmet", help="COCO anotāciju un attēlu mape")
    ap.add_argument("--output", default="runs/faster_rcnn", help="Izvades mape")
    ap.add_argument("--batch", type=int, default=4, help="IMS_PER_BATCH")
    ap.add_argument("--lr", type=float, default=0.0025, help="Bāzes learning rate")
    ap.add_argument("--max-iter", type=int, default=2000, help="Iterāciju skaits")
    ap.add_argument("--imgsz", type=int, default=640, help="Ievades izmērs (kvadrāts)")
    ap.add_argument("--resume", action="store_true", help="Turpina no OUTPUT_DIR")
    ap.add_argument("--device", default=None, help="cuda/cpu vai tukšs auto-izvēlei")
    args = ap.parse_args()

    args.device = pick_device(args.device)
    print("Izmanto ierīci:", args.device)

    register_datasets(args.data_root)
    cfg = build_cfg(args)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
