import torch, cv2
from transformers import SamModel, SamProcessor
from peft import PeftModel, PeftConfig
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# ① LoRA をロード
base_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
lora_model = PeftModel.from_pretrained(base_model,
                                       "checkpoints/sam2-balloon-ft").to(device)
processor  = SamProcessor.from_pretrained("facebook/sam-vit-base")

# ② 検証用画像で推論
img_path = "data/images/005.jpg"
image    = Image.open(img_path).convert("RGB")
img_np = np.array(image)
inputs   = processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    logits = lora_model(**inputs).pred_masks[:,0,0,...].unsqueeze(1)
mask = (logits.sigmoid() > 0.5).cpu().numpy().astype("uint8")[0,0]

h, w = img_np.shape[:2]
mask_up  = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
mask_rgb = cv2.cvtColor(mask_up * 255, cv2.COLOR_GRAY2BGR)

# ③ 可視化
overlay = cv2.addWeighted(img_np, 0.7, mask_rgb, 0.3, 0)
cv2.imwrite("overlay.png", cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.imshow(overlay); plt.axis("off"); plt.show()
