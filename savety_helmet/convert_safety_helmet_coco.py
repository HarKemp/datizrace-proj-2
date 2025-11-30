
# Šeit tiek ģenerētas COCO anotācijas priekš Safety Helmet datu kopas (supervisely json uz COCO json)
# Attēli tiek kopēti uz datasets/helmet/images/{train,val}, ja nav jau no convert_safety_helmet_yolo.py skripta
# Izvade anotācijām datasets/helmet/annotations_train.json un datasets/helmet/annotations_val.json

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

CLASSES = ["helmet", "head", "person"]


def load_bbox(obj: Dict) -> Tuple[float, float, float, float]:
    # Nolasa (x1,y1,x2,y2) koordinātas no supervisely anotācijas
    (x1, y1), (x2, y2) = obj["points"]["exterior"]
    return float(x1), float(y1), float(x2), float(y2)


def convert_split(split: str, src_root: Path, out_root: Path) -> None:
    # source ceļi (img + ann) un izvades mape attēliem
    src_img = src_root / split / "img"
    src_ann = src_root / split / "ann"
    out_img = out_root / "images" / split
    out_img.mkdir(parents=True, exist_ok=True)

    # COCO struktūra ar attēliem, anotācijām un kategorijām
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": c} for i, c in enumerate(CLASSES)],
    }
    ann_id = 1

    # Katram png nolasa atbilstošo json un veido COCO ierakstus
    for idx, img_path in enumerate(sorted(src_img.glob("*.png"))):
        ann_path = src_ann / f"{img_path.name}.json"
        if not ann_path.exists():
            continue
        data = json.loads(ann_path.read_text())
        width = float(data["size"]["width"])
        height = float(data["size"]["height"])

        # COCO images ieraksts
        image_id = idx + 1
        coco["images"].append(
            {"id": image_id, "file_name": img_path.name, "width": width, "height": height}
        )

        # COCO annotations ieraksti visiem objektiem klasēs (helmet, head, person)
        for obj in data.get("objects", []):
            cls = obj.get("classTitle")
            if cls not in CLASSES:
                continue
            cid = CLASSES.index(cls)
            x1, y1, x2, y2 = load_bbox(obj)
            bw, bh = x2 - x1, y2 - y1
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cid + 1,
                    "bbox": [x1, y1, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        # mokopē attēlu uz izvades mapi, ja vēl nav (convert_safety_helmet_yolo.py palaiži pirmo)
        dst = out_img / img_path.name
        if not dst.exists():
            shutil.copy2(img_path, dst)

    # Saglabā COCO anotācijas priekš split
    (out_root / f"annotations_{split}.json").write_text(json.dumps(coco))
    print(f"[{split}] images: {len(coco['images'])}, annotations: {len(coco['annotations'])}")


def main() -> None:
    # cli parametri: source, izvade, izvēlētais split
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="Data/1.3. Savety helmet", help="Avota datu kopa")
    ap.add_argument("--out", default="datasets/helmet", help="Izvades mape (images + COCO anotācijas)")
    ap.add_argument("--split", choices=["train", "val", "both"], default="both")
    args = ap.parse_args()

    src_root = Path(args.src)
    out_root = Path(args.out)
    splits = ["train", "val"] if args.split == "both" else [args.split]
    for split in splits:
        convert_split(split, src_root, out_root)
    print("COCO anotācijas saglabātas mapē:", out_root)


if __name__ == "__main__":
    main()
