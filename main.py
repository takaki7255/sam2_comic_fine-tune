from PIL import Image
import torch
from transformers import SamProcessor, SamModel

processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
model = SamModel.from_pretrained("checkpoints/sam2-balloon-ft").eval()

img = Image.open("test.jpg").convert("RGB")
inputs = processor(images=img, return_tensors="pt").to(model.device)
with torch.no_grad():
    preds = model(**inputs).pred_masks
mask = (preds[0,0] > 0).cpu().numpy().astype("uint8")*255
Image.fromarray(mask).save("balloon_mask.png")
