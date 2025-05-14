import torch, cv2
from transformers import SamModel, SamProcessor
from peft import PeftModel, PeftConfig
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# ① LoRA をロード
base_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
lora_model = PeftModel.from_pretrained(base_model,
                                       "checkpoints/sam2-balloon-ft").to(device)
processor  = SamProcessor.from_pretrained("facebook/sam-vit-base")

# ② 検証用画像で推論
img_path = "data/images/003.jpg"
image    = Image.open(img_path).convert("RGB")
inputs   = processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    logits = lora_model(**inputs).pred_masks[:,0,0,...].unsqueeze(1)
mask = (logits.sigmoid() > 0.5).cpu().numpy().astype("uint8")[0,0]

# ③ 可視化
overlay = cv2.addWeighted(np.array(image), 0.7,
                          cv2.cvtColor(mask*255, cv2.COLOR_GRAY2RGB), 0.3, 0)
plt.imshow(overlay); plt.axis("off"); plt.show()
