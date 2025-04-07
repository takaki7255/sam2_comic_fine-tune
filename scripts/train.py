import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import wandb
from accelerate import Accelerator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import load_config, set_seed, get_dataloader, prepare_model_and_processor, visualize_masks


def dice_loss(inputs, targets, smooth=1.0):
    """Dice Lossを計算する関数"""
    inputs = F.sigmoid(inputs)
    
    # フラット化
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    
    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    return 1 - dice


def focal_loss(inputs, targets, alpha=0.8, gamma=2):
    """Focal Lossを計算する関数"""
    inputs = F.sigmoid(inputs)
    
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    
    # focal lossの計算
    bce = F.binary_cross_entropy(inputs, targets, reduction='none')
    p_t = targets * inputs + (1 - targets) * (1 - inputs)
    loss = bce * ((1 - p_t) ** gamma)
    
    # alphaを適用
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    
    return loss.mean()


def combined_loss(pred_masks, true_masks, alpha=0.5):
    """DiceとFocalを組み合わせた損失関数"""
    dice = dice_loss(pred_masks, true_masks)
    focal = focal_loss(pred_masks, true_masks)
    return alpha * dice + (1 - alpha) * focal


def train_one_epoch(model, dataloader, optimizer, accelerator, epoch):
    """1エポックのトレーニングを行う関数"""
    model.train()
    total_loss = 0
    
    for step, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch}")):
        # バッチデータを取得
        pixel_values = batch.pop("pixel_values")
        masks = batch.pop("masks", None)
        
        if masks is None or masks.shape[1] == 0:
            # マスクがない場合はスキップ
            continue
        
        # モデルの入力を準備
        inputs = {"pixel_values": pixel_values}
        
        if "points" in batch and batch["points"].shape[1] > 0:
            inputs["input_points"] = batch["points"]
            if "point_labels" in batch:
                inputs["input_labels"] = batch["point_labels"]
        
        # 前方伝播
        outputs = model(**inputs)
        pred_masks = outputs.pred_masks
        
        # マスクがバッチ全体で存在する場合のみ損失を計算
        loss = combined_loss(pred_masks, masks)
        
        # 勾配の計算と更新
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        
        # ログ記録
        if accelerator.is_main_process and step % 10 == 0:
            accelerator.log({"train_loss": loss.item(), "step": step + epoch * len(dataloader)})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, accelerator, epoch, save_samples=False):
    """検証データでの評価を行う関数"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc=f"Validation Epoch {epoch}")):
            # バッチデータを取得
            pixel_values = batch.pop("pixel_values")
            masks = batch.pop("masks", None)
            original_images = batch.pop("original_image", None)
            
            if masks is None or masks.shape[1] == 0:
                continue
            
            # モデルの入力を準備
            inputs = {"pixel_values": pixel_values}
            
            if "points" in batch and batch["points"].shape[1] > 0:
                inputs["input_points"] = batch["points"]
                if "point_labels" in batch:
                    inputs["input_labels"] = batch["point_labels"]
            
            # 前方伝播
            outputs = model(**inputs)
            pred_masks = outputs.pred_masks
            
            # 損失計算
            loss = combined_loss(pred_masks, masks)
            total_loss += loss.item()
            
            # サンプル画像の保存
            if accelerator.is_main_process and save_samples and step < 2:  # 最初の2バッチだけ保存
                for i in range(min(2, len(pixel_values))):  # 各バッチから最大2枚
                    if original_images is not None:
                        img = original_images[i].cpu().numpy()
                        pred_mask = F.sigmoid(pred_masks[i][0]).cpu().numpy() > 0.5
                        true_mask = masks[i][0].cpu().numpy() > 0.5
                        
                        # 予測と正解を比較する画像を保存
                        save_dir = os.path.join("models", "samples")
                        os.makedirs(save_dir, exist_ok=True)
                        
                        # 元画像に予測マスクを重ねる
                        visualize_masks(
                            img, 
                            [pred_mask], 
                            save_path=os.path.join(save_dir, f"epoch{epoch}_batch{step}_img{i}_pred.png")
                        )
                        
                        # 元画像に正解マスクを重ねる
                        visualize_masks(
                            img, 
                            [true_mask], 
                            save_path=os.path.join(save_dir, f"epoch{epoch}_batch{step}_img{i}_true.png")
                        )
    
    return total_loss / len(dataloader)


def main(config_path):
    """メイン関数"""
    # 設定ファイルを読み込み
    config = load_config(config_path)
    
    # シードを設定
    set_seed(config["training"]["seed"])
    
    # Acceleratorの初期化
    accelerator = Accelerator(
        mixed_precision=config["training"]["mixed_precision"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        log_with="wandb" if config["logging"]["use_wandb"] else None
    )
    
    if accelerator.is_main_process and config["logging"]["use_wandb"]:
        accelerator.init_trackers(
            project_name=config["logging"]["project_name"],
            config=config,
            init_kwargs={"wandb": {"name": config["logging"]["run_name"]}}
        )
    
    # モデルとプロセッサを準備
    model, processor = prepare_model_and_processor(config)
    
    # データローダーを準備
    train_dataloader = get_dataloader(config, processor, split='train', shuffle=True)
    val_dataloader = get_dataloader(config, processor, split='val', shuffle=False)
    
    # オプティマイザとスケジューラを準備
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"],
        eta_min=1e-6
    )
    
    # Acceleratorを使用して準備
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )
    
    # トレーニングループ
    best_val_loss = float('inf')
    
    for epoch in range(config["training"]["epochs"]):
        # トレーニング
        train_loss = train_one_epoch(model, train_dataloader, optimizer, accelerator, epoch)
        
        # 検証
        val_loss = validate(model, val_dataloader, accelerator, epoch, save_samples=(epoch % config["training"]["eval_interval"] == 0))
        
        # スケジューラのステップ
        scheduler.step()
        
        # ログ記録
        if accelerator.is_main_process:
            accelerator.log({
                "train_loss_epoch": train_loss,
                "val_loss_epoch": val_loss,
                "epoch": epoch
            })
            
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # モデル保存
        if accelerator.is_main_process and val_loss < best_val_loss and epoch % config["training"]["save_interval"] == 0:
            best_val_loss = val_loss
            unwrapped_model = accelerator.unwrap_model(model)
            
            # モデル保存ディレクトリ
            os.makedirs("models", exist_ok=True)
            unwrapped_model.save_pretrained(
                os.path.join("models", f"sam2_manga_epoch{epoch}_val{val_loss:.4f}")
            )
            processor.save_pretrained(
                os.path.join("models", f"sam2_manga_epoch{epoch}_val{val_loss:.4f}")
            )
    
    # 最終モデルの保存
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(os.path.join("models", "sam2_manga_final"))
        processor.save_pretrained(os.path.join("models", "sam2_manga_final"))
    
    if accelerator.is_main_process and config["logging"]["use_wandb"]:
        accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2の漫画データセットでのファインチューニング")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="設定ファイルのパス")
    args = parser.parse_args()
    
    main(args.config)