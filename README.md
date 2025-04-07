# SAM2 漫画セグメンテーション ファインチューニング

このリポジトリは、Meta AIのSegment Anything Model 2 (SAM2)を漫画画像のセグメンテーション用にファインチューニングするためのコードを提供します。漫画のコマ、吹き出し、キャラクターなどを自動的にセグメンテーションすることができます。

## ディレクトリ構成

```
sam2_comic_fine-tune/
├── data/
│   ├── images/          # 元の漫画画像
│   │   ├── train/       # 学習用画像
│   │   └── val/         # 検証用画像
│   ├── masks/           # セグメンテーションマスク
│   │   ├── train/       # 学習用マスク
│   │   └── val/         # 検証用マスク
│   └── annotations.json # アノテーションデータ（COCO形式）
├── configs/             # 設定ファイル
│   └── config.yaml      # 学習設定
├── models/              # ファインチューニングしたモデルの保存先
├── scripts/             # 学習・評価用スクリプト
│   ├── train.py         # 学習スクリプト
│   ├── evaluate.py      # 評価スクリプト
│   └── utils.py         # ユーティリティ関数
├── requirements.txt     # 必要なパッケージリスト
└── README.md            # プロジェクト説明（このファイル）
```

## 環境構築

必要なパッケージをインストールします。

```bash
pip install -r requirements.txt
```

## データセットの準備

### データセットの構造

学習には以下のような構成のデータセットが必要です：

1. **画像**: `data/images/train/` と `data/images/val/` に元の漫画画像を配置します
2. **マスク**: `data/masks/train/` と `data/masks/val/` に対応するセグメンテーションマスク画像を配置します
3. **アノテーション**: `data/annotations.json` にCOCO形式のアノテーションを作成します

### アノテーションファイルの形式

アノテーションファイルはCOCO形式でJSON形式です：

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "manga_page_001.jpg",
      "width": 1024,
      "height": 1536,
      "split": "train"
    },
    ...
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "segmentation_file": "manga_page_001_mask_001.png",
      "iscrowd": 0,
      "area": 10000,
      "bbox": [100, 100, 200, 300]
    },
    ...
  ],
  "categories": [
    {"id": 1, "name": "panel"},
    {"id": 2, "name": "bubble"},
    {"id": 3, "name": "character"},
    {"id": 4, "name": "text"}
  ]
}
```

アノテーションの各項目は以下を示します：
- `id`: アノテーションの一意なID
- `image_id`: このアノテーションが関連する画像のID
- `category_id`: オブジェクトのカテゴリID
- `segmentation_file`: マスク画像のファイル名
- `iscrowd`: 群集アノテーションかどうか（通常は0）
- `area`: マスクの面積（ピクセル数）
- `bbox`: [x, y, width, height] 形式のバウンディングボックス

### 既存データセットからの変換

既存のCOCOデータセットがある場合、以下のような手順でマスク画像を生成できます：

```python
import json
import numpy as np
import cv2
import pycocotools.mask as mask_util

# COCOアノテーションファイルを読み込み
with open('coco_annotations.json', 'r') as f:
    annotations = json.load(f)

# 各アノテーションに対してマスク画像を生成
for ann in annotations['annotations']:
    if 'segmentation' in ann:
        if isinstance(ann['segmentation'], list):
            # ポリゴン形式のセグメンテーション
            img = np.zeros((ann['height'], ann['width']), dtype=np.uint8)
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape(-1, 2)
                cv2.fillPoly(img, [poly.astype(np.int32)], 255)
        else:
            # RLE形式のセグメンテーション
            img = mask_util.decode(ann['segmentation'])
        
        # マスク画像を保存
        mask_filename = f"{ann['image_id']:012d}_{ann['id']:012d}.png"
        cv2.imwrite(f"data/masks/{mask_filename}", img)
        
        # アノテーションを更新
        ann['segmentation_file'] = mask_filename
```

## モデルの学習

### 設定ファイルの調整

`configs/config.yaml` ファイルを編集して、ハイパーパラメータなどを調整します。

### 学習の実行

```bash
python scripts/train.py --config configs/config.yaml
```

学習中は、以下のようなプロセスが実行されます：

1. データの読み込みと前処理
2. SAM2モデルへの入力形式への変換
3. モデルのファインチューニング
4. 検証データでの評価
5. 最良モデルの保存

WandBを使用して学習の進捗を可視化することができます（設定ファイルで `use_wandb: true` に設定）。

## 評価と推論

ファインチューニングしたモデルを使って新しい漫画画像をセグメンテーションするには：

### 単一画像の評価

```bash
python scripts/evaluate.py --model_path models/sam2_manga_final --image_path your_manga_image.jpg --output_dir outputs
```

### ディレクトリ内の全画像を一括評価

```bash
python scripts/evaluate.py --model_path models/sam2_manga_final --image_dir your_manga_dir --output_dir outputs
```

### インタラクティブモード

ユーザーがクリックして対象オブジェクト（コマや吹き出し、キャラクターなど）を指定してセグメンテーションを行うインタラクティブモードも使用できます：

```bash
python scripts/evaluate.py --model_path models/sam2_manga_final --image_path your_manga_image.jpg --output_dir outputs --interactive
```

## 注意事項

- GPUでの学習を推奨します。CPU環境でも動作しますが、非常に時間がかかります。
- モデルのサイズによって必要なGPUメモリが変わります：
  - SAM2-ViT-B: 8GB以上のGPUメモリを推奨
  - SAM2-ViT-L: 16GB以上のGPUメモリを推奨
  - SAM2-ViT-H: 24GB以上のGPUメモリを推奨
- バッチサイズは使用可能なGPUメモリに応じて調整してください。

## 参考リンク

- [Segment Anything Model 2 (SAM2)](https://github.com/facebookresearch/segment-anything)
- [SAM2 Hugging Face](https://huggingface.co/docs/transformers/model_doc/sam2)