import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything_2 import Sam2Model
from transformers import AutoProcessor
import cv2
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import load_config, visualize_masks

def evaluate_image(image_path, model, processor, output_dir=None, interactive=False):
    """単一の画像を評価する関数"""
    # 画像を読み込み
    image = Image.open(image_path).convert("RGB")
    orig_size = image.size
    
    # 画像をモデル入力形式に変換
    inputs = processor(images=image, return_tensors="pt")
    
    if torch.cuda.is_available():
        model = model.cuda()
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # インタラクティブモードの場合
    if interactive:
        # クリックイベントを処理する関数
        points = []
        point_labels = []
        
        def onclick(event):
            if event.xdata is not None and event.ydata is not None:
                x, y = int(event.xdata), int(event.ydata)
                # 左クリック: 前景点
                if event.button == 1:
                    points.append([x, y])
                    point_labels.append(1)
                    plt.plot(x, y, 'ro', markersize=10)
                # 右クリック: 背景点
                elif event.button == 3:
                    points.append([x, y])
                    point_labels.append(0)
                    plt.plot(x, y, 'bo', markersize=10)
                plt.draw()
        
        # プロットの設定
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(np.array(image))
        ax.set_title("左クリック: 前景点 / 右クリック: 背景点 / Enter: 実行")
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        
        if points:
            # 点情報をモデル入力形式に変換
            inputs["input_points"] = torch.tensor([points]).to(inputs["pixel_values"].device)
            inputs["input_labels"] = torch.tensor([point_labels]).to(inputs["pixel_values"].device)
    
    # モデルの推論
    with torch.no_grad():
        outputs = model(**inputs)
    
    # マスクを取得
    masks = outputs.pred_masks
    masks = torch.sigmoid(masks) > 0.5  # 閾値を適用
    
    # 結果の可視化・保存
    image_np = np.array(image)
    masks_np = [mask[0].cpu().numpy() for mask in masks]
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 元の画像に各マスクを重ねて保存
        for i, mask in enumerate(masks_np):
            output_path = os.path.join(output_dir, f"{base_name}_mask{i}.png")
            visualize_masks(image_np, [mask], save_path=output_path)
        
        # すべてのマスクを重ねた画像も保存
        output_path = os.path.join(output_dir, f"{base_name}_all_masks.png")
        visualize_masks(image_np, masks_np, save_path=output_path)
    else:
        # 表示のみ
        visualize_masks(image_np, masks_np)
    
    return masks_np


def batch_process(image_dir, model, processor, output_dir):
    """ディレクトリ内の複数画像を処理する関数"""
    os.makedirs(output_dir, exist_ok=True)
    
    # サポートする画像拡張子
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # 画像ファイルをリストアップ
    image_files = []
    for file in os.listdir(image_dir):
        ext = os.path.splitext(file)[1].lower()
        if ext in supported_extensions:
            image_files.append(os.path.join(image_dir, file))
    
    print(f"{len(image_files)}枚の画像を処理します。")
    
    for img_path in tqdm(image_files, desc="画像処理中"):
        try:
            evaluate_image(img_path, model, processor, output_dir)
        except Exception as e:
            print(f"画像{img_path}の処理に失敗しました。エラー: {e}")


def get_manga_objects(masks, min_size=100):
    """マスクから漫画要素のタイプを推定する関数"""
    objects = []
    
    for i, mask in enumerate(masks):
        # マスクの面積を計算
        area = np.sum(mask)
        if area < min_size:
            continue  # 小さすぎるマスクを無視
        
        # マスクの外接矩形を計算
        y, x = np.where(mask)
        x1, y1, x2, y2 = np.min(x), np.min(y), np.max(x), np.max(y)
        width, height = x2 - x1, y2 - y1
        aspect_ratio = width / height if height > 0 else 0
        
        # マスクの形状から要素タイプを推定
        obj_type = "unknown"
        if aspect_ratio > 3:  # 横長
            obj_type = "text_line"
        elif 0.8 < aspect_ratio < 1.2 and area < width * height * 0.7:
            obj_type = "bubble"
        elif area > width * height * 0.7:
            obj_type = "panel"
        else:
            obj_type = "character"
            
        objects.append({
            "id": i,
            "type": obj_type,
            "bbox": [int(x1), int(y1), int(width), int(height)],
            "area": int(area)
        })
    
    return objects


def main(args):
    """メイン関数"""
    # モデルとプロセッサの読み込み
    model = Sam2Model.from_pretrained(args.model_path)
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    print(f"モデルを読み込みました: {args.model_path}")
    
    if args.image_path:
        # 単一画像処理モード
        evaluate_image(args.image_path, model, processor, args.output_dir, interactive=args.interactive)
    elif args.image_dir:
        # バッチ処理モード
        batch_process(args.image_dir, model, processor, args.output_dir)
    else:
        print("エラー: image_pathまたはimage_dirのいずれかを指定してください")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2を使用した漫画のセグメンテーション評価")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="ファインチューニング済みモデルのパス")
    parser.add_argument("--image_path", type=str, 
                        help="評価する単一画像のパス")
    parser.add_argument("--image_dir", type=str, 
                        help="評価する画像が含まれるディレクトリ")
    parser.add_argument("--output_dir", type=str, default="outputs", 
                        help="出力画像を保存するディレクトリ")
    parser.add_argument("--interactive", action="store_true", 
                        help="インタラクティブモードを有効にする（単一画像のみ）")
    
    args = parser.parse_args()
    main(args)