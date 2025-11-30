# Šeit tiek konvertēta Safety Helmet datu kopa (json) uz YOLO txt formātu
# Nolasa Data/1.3. Savety helmet/{train,val}/img + ann
# Izvada datasets/helmet/images/{train,val} un labels/{train,val}

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Tuple


# Klases fiksētā secībā - tās būs YOLO indeksi 0,1,2
CLASSES = ["helmet", "head", "person"]


def box_to_yolo(
    x1: float, y1: float, x2: float, y2: float, w: float, h: float
) -> Tuple[float, float, float, float]:
    # Normalizē box uz YOLO formātu (cx, cy, bw, bh)/ attēla izmēriem
    bw, bh = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    return cx, cy, bw / w, bh / h


def convert_split(split: str, src_root: Path, out_root: Path) -> None:
    # Ceļi avota un izvades mapēm
    src_img = src_root / split / "img"
    src_ann = src_root / split / "ann"
    out_img = out_root / "images" / split
    out_lbl = out_root / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    kept, skipped = 0, 0
    # Katram PNG paņem atbilstošo JSON, uzraksta YOLO txt un nokopē attēlu
    for img_path in sorted(src_img.glob("*.png")):
        ann_path = src_ann / f"{img_path.name}.json"
        if not ann_path.exists():
            skipped += 1
            continue

        data = json.loads(ann_path.read_text())
        width = float(data["size"]["width"])
        height = float(data["size"]["height"])

        lines: List[str] = []
        for obj in data.get("objects", []):
            cls = obj.get("classTitle")
            if cls not in CLASSES:
                continue
            cid = CLASSES.index(cls)
            (x1, y1), (x2, y2) = obj["points"]["exterior"]
            cx, cy, bw, bh = box_to_yolo(float(x1), float(y1), float(x2), float(y2), width, height)
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # Uzraksta YOLO anotāciju un kopē attēlu
        (out_lbl / f"{img_path.stem}.txt").write_text("\n".join(lines))
        shutil.copy2(img_path, out_img / img_path.name)
        kept += 1

    print(f"[{split}] converted: {kept}, skipped: {skipped}")
    print(f"images -> {out_img}")
    print(f"labels -> {out_lbl}")


def main() -> None:
    # CLI parametri - avots, izvade, izvēlētais splits
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="Data/1.3. Savety helmet", help="Avota datu kopa")
    ap.add_argument("--out", default="datasets/helmet", help="Izvades mape YOLO datiem")
    ap.add_argument(
        "--split", choices=["train", "val", "both"], default="both", help="Spliti konvertēšanai"
    )
    args = ap.parse_args()

    src_root = Path(args.src)
    out_root = Path(args.out)

    splits = ["train", "val"] if args.split == "both" else [args.split]
    for split in splits:
        convert_split(split, src_root, out_root)


if __name__ == "__main__":
    main()
