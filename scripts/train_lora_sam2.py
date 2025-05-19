import os; os.environ["TRANSFORMERS_NO_TF"] = "1"
import json, torch, albumentations as A
from torch.utils.data import DataLoader
from torchvision import transforms
import torch, torch.nn.functional as F
from torchmetrics.classification import BinaryJaccardIndex
from transformers import Trainer
from PIL import Image
from datasets import load_dataset
from transformers import SamProcessor, SamModel, TrainingArguments, Trainer, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import cv2

torch.backends.cudnn.benchmark = True


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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)        # => cuda:0 / cpu
# model = SamModel.from_pretrained("facebook/sam-vit-huge")
# processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
# huge → base へ
# model = SamModel.from_pretrained("facebook/sam-vit-base")
# processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
base = Path("sam2_hiera_base_plus_local")
model = SamModel.from_pretrained(base, local_files_only=True)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
ckpt = torch.load("./sam2_hiera_base_plus_local/pytorch_model.bin", map_location="cpu")
if "model" in ckpt:
    model.load_state_dict(ckpt["model"], strict=False)

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
for n, p in model.named_parameters():
    if n.startswith("vision_encoder") or n.startswith("prompt_encoder"):
        p.requires_grad = False   # エンコーダは凍結
    else:
        p.requires_grad = True    # ★ decoder は解放
# ---- 2) データセット ------------------------
def albumentations_transform():
    return A.Compose([
        A.HorizontalFlip(0.5),
        A.ShiftScaleRotate(0.05,0.05,15, p=0.5),
        #A.Resize(1024,1024)               # SAM2 の想定解像度．重すぎるなら512,512
        A.Resize(512,512),               # SAM2 の想定解像度．重すぎるなら512,512
    ])

# ---- 2. encode_example で ann_map から注釈を引く ----
def encode_example(example, ann_map):
    img_path = os.path.join("data/images", example["file_name"])
    image = Image.open(img_path).convert("RGB")
    h, w = image.size[::-1]
    mask = np.zeros((h, w), dtype=np.uint8)

    # 該当画像のアノテーション
    for ann in ann_map[example["id"]]:
        for poly in ann["segmentation"]:
            pts = np.array(poly).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)

    aug = albumentations_transform()(image=np.array(image), mask=mask)
    pixel_values = processor(images=aug["image"], return_tensors="pt").pixel_values[0]
    label = torch.tensor(aug["mask"][None])

    ys, xs = torch.where(label[0] > 0)
    cx, cy = xs.float().mean().long(), ys.float().mean().long()
    prompt = {"points": torch.tensor([[cx, cy]])}
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

class SamSegTrainer(Trainer):
    """SAM2 用 – outputs.pred_masks を使って loss を計算"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # IoU を学習中にログしたい場合
        self.iou_metric = BinaryJaccardIndex().to(self.args.device)
        self._iou_has_data = False      # ★ 追加

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
        eps = 1e-6
        probs       = logits.sigmoid()            # (B,1,H,W)
        labels_bin  = (labels > 0.5).float()      # 0/1

        bce  = F.binary_cross_entropy_with_logits(logits, labels_bin)
        eps  = 1e-6
        dice_vec   = 1 - (2*(probs*labels_bin).sum((1,2,3))+eps) / \
                  ((probs+labels_bin).sum((1,2,3))+eps)      # shape = (B,)
        dice       = dice_vec.mean()                                 # ★ ここで平均

        loss = 0.3 * bce + 1.2 * dice      # ← dice は scalar Tensor
        # --------------------------------------------------------
        
        # ---------- デバッグ ---------- #
        if self.state.global_step % 50 == 0:       # 50stepごとで十分
            thr = 0.5                              # IoU計算に使っている閾値
            pred_pix  = (probs > thr).sum().item()
            label_pix = (labels_bin > 0).sum().item()
            print(f"[dbg] step {self.state.global_step:4d}  "
                  f"pred_pix={pred_pix:6d}  label_pix={label_pix:6d}  "
                  f"bce={bce.item():.3f}  dice={dice.item():.3f}  "
                  f"iou_has_data={self._iou_has_data}")
        # -------------------------------- #


        # ロギング用 IoU
        if model.training:
            thr = 0.5                                  # ★ 閾値を下げる
            self.iou_metric.update(
                (probs > thr).long().to(self.args.device),
                (labels > 0.5).long().to(self.args.device)
            )
            self._iou_has_data = True
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if self._iou_has_data:   # ★
            try:
                logs["train_iou"] = self.iou_metric.compute().item()
                self.iou_metric.reset()          # 成功したときだけリセット
                self._iou_has_data = False  # 次の蓄積を待つ
            except (RuntimeError, ValueError):   # まだ update されていない場合
                pass
        super().log(logs, *args, **kwargs)

# EarlyStopping設定
# trainer.add_callback(EarlyStoppingCallback(
#     early_stopping_patience = 5,         # eval/loss が 5 回改善しなければ停止
#     early_stopping_threshold = 0.001))   # 改善幅 0.001 以下を無視

train_ds = train_ds.map(encode_example, fn_kwargs={"ann_map": train_ann_map})
val_ds   = val_ds.map(encode_example, fn_kwargs={"ann_map": val_ann_map})

train_ds.set_format(type="torch")
val_ds.set_format(type="torch")

# ---- 3) Trainer ----------------------------
args = TrainingArguments(
    output_dir="checkpoints",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=50,
    learning_rate=3e-5,
    warmup_ratio=0.0,
    save_strategy="epoch",
    save_steps=500,
    # ↓ ここを変更
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model = "eval_loss",
    greater_is_better=False,
    eval_steps=100,
    fp16=torch.cuda.is_available(),
    logging_strategy="epoch",
    logging_steps=20,
    label_names=["labels"],
    remove_unused_columns=False,
    max_grad_norm = 1.0,
    report_to="all",
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
