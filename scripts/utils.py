import os
import json
import yaml
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from torchvision import transforms
from segment_anything_2 import Sam2Model
from transformers import AutoProcessor
import matplotlib.pyplot as plt


def set_seed(seed):
    """再現性のためにシードを固定する関数"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """設定ファイルを読み込む関数"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def visualize_masks(image, masks, points=None, save_path=None):
    """セグメンテーションマスクを可視化する関数"""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    if masks is not None:
        for mask in masks:
            mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4))
            # ランダムな色を生成
            color = np.concatenate([np.random.random(3), [0.5]])  # RGBA形式で透明度を0.5に設定
            mask_rgba[mask > 0] = color
            plt.imshow(mask_rgba)
    
    if points is not None:
        for point in points:
            plt.plot(point[0], point[1], 'r.', markersize=10)
    
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


class MangaSegmentationDataset(Dataset):
    """漫画セグメンテーション用のカスタムデータセット"""
    def __init__(self, image_dir, mask_dir, annotations_file, processor, image_size=1024, 
                 num_points_per_mask=10, max_masks_per_image=20, transform=None, split='train'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.image_size = image_size
        self.num_points_per_mask = num_points_per_mask
        self.max_masks_per_image = max_masks_per_image
        self.transform = transform
        self.split = split
        
        # アノテーションファイルから情報を読み込む
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
            
        # このsplitに対応するアノテーションだけを抽出
        self.image_data = [item for item in self.annotations['images'] 
                          if item.get('split', 'train') == split]
        
        # 各画像のマスク情報を取得
        self.mask_info = {}
        for item in self.annotations['annotations']:
            image_id = item['image_id']
            if image_id not in self.mask_info:
                self.mask_info[image_id] = []
            self.mask_info[image_id].append(item)
    
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, idx):
        # 画像情報を取得
        img_info = self.image_data[idx]
        image_id = img_info['id']
        file_name = img_info['file_name']
        
        # 画像を読み込む
        image_path = os.path.join(self.image_dir, file_name)
        image = Image.open(image_path).convert('RGB')
        
        # 画像をリサイズ
        if self.transform:
            image = self.transform(image)
        
        # この画像に対応するマスク情報を取得
        mask_list = self.mask_info.get(image_id, [])
        if len(mask_list) > self.max_masks_per_image:
            # マスクの最大数を制限
            mask_list = random.sample(mask_list, self.max_masks_per_image)
        
        masks = []
        points = []
        point_labels = []
        
        for mask_info in mask_list:
            # マスクを読み込む
            mask_path = os.path.join(self.mask_dir, mask_info['segmentation_file'])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 0).astype(np.uint8)  # バイナリマスクに変換
            masks.append(mask)
            
            # マスクからランダムな点を抽出
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) > 0:
                # マスク内の点を取得
                point_indices = np.random.choice(len(y_indices), 
                                               min(self.num_points_per_mask, len(y_indices)), 
                                               replace=False)
                mask_points = np.array([[x_indices[i], y_indices[i]] for i in point_indices])
                points.extend(mask_points.tolist())
                point_labels.extend([1] * len(mask_points))  # 1はマスク内の点
        
        # SAM2のプロセッサを使用して入力を準備
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # マスクとプロンプト情報の追加
        if masks:
            masks_tensor = torch.stack([torch.from_numpy(mask) for mask in masks])
            points_tensor = torch.tensor(points) if points else torch.zeros((0, 2))
            point_labels_tensor = torch.tensor(point_labels) if point_labels else torch.zeros(0)
            
            inputs["masks"] = masks_tensor
            inputs["points"] = points_tensor
            inputs["point_labels"] = point_labels_tensor
        else:
            # マスクがない場合、ダミーデータを作成
            inputs["masks"] = torch.zeros((0, self.image_size, self.image_size))
            inputs["points"] = torch.zeros((0, 2))
            inputs["point_labels"] = torch.zeros(0)
        
        inputs["original_image"] = np.array(image)  # 可視化用に元の画像を保存
        
        return inputs


def get_dataloader(config, processor, split='train', shuffle=True):
    """データローダーを作成する関数"""
    if split == 'train':
        image_dir = config['data']['train_image_dir']
        mask_dir = config['data']['train_mask_dir']
    else:
        image_dir = config['data']['val_image_dir']
        mask_dir = config['data']['val_mask_dir']
    
    transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
    ])
    
    dataset = MangaSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        annotations_file=config['data']['annotations_file'],
        processor=processor,
        image_size=config['data']['image_size'],
        num_points_per_mask=config['data']['num_points_per_mask'],
        max_masks_per_image=config['data']['max_masks_per_image'],
        transform=transform,
        split=split
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=shuffle,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader


def prepare_model_and_processor(config):
    """モデルとプロセッサを準備する関数"""
    processor = AutoProcessor.from_pretrained(config['model']['checkpoint'])
    model = Sam2Model.from_pretrained(config['model']['checkpoint'])
    
    # 画像エンコーダーをフリーズするかどうか
    if config['model']['freeze_image_encoder']:
        for param in model.vision_encoder.parameters():
            param.requires_grad = False
    
    return model, processor