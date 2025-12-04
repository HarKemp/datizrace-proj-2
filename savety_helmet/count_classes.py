# Šis skripts saskaita klases (helmet/head/person) oriģinālajās supervisely anotācijās

import json
import pathlib
import collections


def count_split(split: str) -> collections.Counter:
    ann_dir = pathlib.Path("Data/1.3. Savety helmet") / split / "ann"
    counts = collections.Counter()
    for fp in ann_dir.glob("*.json"):
        data = json.loads(fp.read_text())
        for o in data.get("objects", []):
            counts[o.get("classTitle")] += 1
    return counts


def main() -> None:
    for split in ["train", "val"]:
        counts = count_split(split)
        print(split, counts)


if __name__ == "__main__":
    main()
