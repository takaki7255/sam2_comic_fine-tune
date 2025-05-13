import os; os.environ["TRANSFORMERS_NO_TF"] = "1"
import json, torch, albumentations as A
from torch.utils.data import DataLoader
from torchvision import transforms
import torch, torch.nn.functional as F
from torchmetrics.classification import BinaryJaccardIndex
from transformers import Trainer
from PIL import Image
from datasets import load_dataset
from transformers import SamProcessor, SamModel, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import cv2


class SamSegTrainer(Trainer):
    """SAM2 用 – outputs.pred_masks を使って loss を計算"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # IoU を学習中にログしたい場合
        self.iou_metric = BinaryJaccardIndex().to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch: int | None = None):
        # Trainer が batch を dict で渡す → 必要項目だけ取り出す
        pixel_values = inputs.pop("pixel_values").to(device)  # ★
        labels       = inputs.pop("labels").to(device)        # ★
        prompts      = inputs.pop("prompts")
        prompts["points"] = prompts["points"].to(device)      # ★

        outputs = model(pixel_values=pixel_values, **prompts)
        # ──★ ① SAM2 は 3 枚返す → 最初の 1 枚だけ使う
        logits = outputs.pred_masks[:, 0, 0, ...].unsqueeze(1)

        labels = F.interpolate(labels.float(), size=logits.shape[-2:], mode="nearest")

        # ---- 損失関数：BCE + Dice の複合例 ----------------------
        probs = logits.sigmoid()
        bce   = F.binary_cross_entropy_with_logits(logits, labels.float())
        # Dice = 1 - (2 * |X∩Y|) / (|X|+|Y|)
        eps   = 1e-6
        inter = (probs * labels).sum(dim=[1,2,3])
        union = (probs + labels).sum(dim=[1,2,3])
        dice  = 1 - (2*inter + eps) / (union + eps)
        loss  = bce + dice.mean()
        # --------------------------------------------------------

        # ロギング用 IoU
        if self.state.is_world_process_zero:
            self.iou_metric.update((probs > 0.5).long(),
                       (labels > 0.5).long())

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):        # 可変長で受け取る
        if getattr(self.iou_metric, "tp", None) is not None and self.iou_metric.tp.numel() > 0:
            logs["train_iou"] = self.iou_metric.compute().item()
            self.iou_metric.reset()
        super().log(logs, *args, **kwargs)

# ---- 1. COCO を読み込んで “画像ごとのアノテ一覧” を作る ----
with open("data/annotations/train.json") as f:
    coco_train = json.load(f)
with open("data/annotations/val.json") as f:
    coco_val = json.load(f)

def build_ann_map(coco):
    ann_map = {}
    for ann in coco["annotations"]:
        ann_map.setdefault(ann["image_id"], []).append(ann)
    return ann_map

train_ann_map = build_ann_map(coco_train)
val_ann_map   = build_ann_map(coco_val)

# images の配列だけを Dataset 化（file_name が取れる）
train_ds = load_dataset(
    "json",
    data_files="data/annotations/train.json",
    field="images"          # ← 追加！
)["train"]
val_ds = load_dataset(
    "json",
    data_files="data/annotations/val.json",
    field="images"
)["train"]

# 1. ベースモデルをロード
# model = SamModel.from_pretrained("facebook/sam-vit-huge")
# processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
# huge → base へ
model = SamModel.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)        # => cuda:0 / cpu
model.to(device)

# 2. ──★ ここでエンコーダ／プロンプトエンコーダを凍結 ──
for p in model.vision_encoder.parameters():
    p.requires_grad = False
for p in model.prompt_encoder.parameters():
    p.requires_grad = False

# 3. LoRA をデコーダの q_proj / v_proj にだけ適用
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION  # 汎用タスク
)
model = get_peft_model(model, lora_cfg)    # ← この後は model.base_model が SAM2


for name, _ in model.named_modules():
    if name.endswith(("q_proj", "v_proj")):   # 任意のフィルタ
        print(name)


# エンコーダ・プロンプトエンコーダは凍結
for n, p in list(model.vision_encoder.named_parameters()) + \
            list(model.prompt_encoder.named_parameters()):
    p.requires_grad = False
# ---- 2) データセット ------------------------
def albumentations_transform():
    return A.Compose([
        A.HorizontalFlip(0.5),
        A.ShiftScaleRotate(0.05,0.05,15, p=0.5),
        A.Resize(1024,1024)               # SAM2 の想定解像度
    ])

# ---- 2. encode_example で ann_map から注釈を引く ----
def encode_example(example, ann_map):
    img_path = os.path.join("data/images", example["file_name"])
    image = Image.open(img_path).convert("RGB")
    w, h = image.size                     # PIL は (width, height)
    mask = np.zeros((h, w), dtype=np.uint8)

    # 該当画像のアノテーション
    for ann in ann_map[example["id"]]:
        for poly in ann["segmentation"]:
            pts = np.array(poly).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)

    aug = albumentations_transform()(image=np.array(image), mask=mask)
    pixel_values = processor(images=aug["image"], return_tensors="pt").pixel_values[0]
    label = torch.tensor(aug["mask"][None])

    ys, xs = torch.where(label[0] > 0)
    rand = torch.randint(0, len(xs), (1,))
    prompt = {"points": torch.stack([xs[rand], ys[rand]], dim=-1)}

    return {"pixel_values": pixel_values,
            "labels": label,
            "prompts": prompt}
    
def sam_collator(features):
    # pixel_values, labels → そのままテンソルスタック
    batch = {
        "pixel_values": torch.stack([f["pixel_values"] for f in features]),
        "labels":       torch.stack([f["labels"]       for f in features]),
        # prompts は keys を保ったままネストをまとめる
        "prompts": {
            "points": torch.stack([f["prompts"]["points"] for f in features])
        }
    }
    return batch

train_ds = train_ds.map(encode_example, fn_kwargs={"ann_map": train_ann_map})
val_ds   = val_ds.map(encode_example, fn_kwargs={"ann_map": val_ann_map})

train_ds.set_format(type="torch")
val_ds.set_format(type="torch")

# ---- 3) Trainer ----------------------------
args = TrainingArguments(
    output_dir="checkpoints",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=20,
    learning_rate=3e-5,
    warmup_ratio=0.1,
    save_strategy="steps",
    save_steps=500,
    # ↓ ここを変更
    eval_strategy="steps",
    eval_steps=100,
    fp16=torch.cuda.is_available(),
    logging_strategy="steps",
    logging_steps=20,
    label_names=["labels"],
    remove_unused_columns=False,
)


trainer = SamSegTrainer(                    # ← ここだけ変更
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=sam_collator
)

trainer.train()
model.save_pretrained("checkpoints/sam2-balloon-ft")
