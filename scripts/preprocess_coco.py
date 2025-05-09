"""
COCO JSON を SAM2 用に整理し、train/val を 9:1 に分割
usage: python preprocess_coco.py \
         --src ./data/annotations/manga_balloon.json \
         --dst ./data/annotations
"""

import json, random, argparse, pathlib
from sklearn.model_selection import train_test_split

def main(args):
    with open(args.src) as f:
        coco = json.load(f)

    imgs = coco["images"]
    anns = coco["annotations"]

    # train/val を画像単位で分割
    train_imgs, val_imgs = train_test_split(
        imgs, test_size=0.1, random_state=42)

    def subset(img_subset):
        img_ids = {im["id"] for im in img_subset}
        subset_anns = [a for a in anns if a["image_id"] in img_ids]
        return {"images": img_subset,
                "annotations": subset_anns,
                "categories": coco["categories"]}

    pathlib.Path(args.dst).mkdir(parents=True, exist_ok=True)
    with open(f"{args.dst}/train.json", "w") as f:
        json.dump(subset(train_imgs), f)
    with open(f"{args.dst}/val.json", "w") as f:
        json.dump(subset(val_imgs), f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    main(ap.parse_args())
