from PIL import Image, ImageEnhance
import requests
import torch
import torch.nn as nn
import numpy as np
from safetensors import torch
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

def get_masks(segmentation):
    obj_ids = torch.unique(segmentation)
    obj_ids = obj_ids[1:]
    masks = segmentation == 7
    return masks, obj_ids

processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

url = "http://localhost:9090/IMG_1572.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits.cpu()

upsampled_logits = nn.functional.interpolate(
    logits,
    size=image.size[::-1],
    mode="bilinear",
    align_corners=False,
)

pred_seg = upsampled_logits.argmax(dim=1)[0]

segments = torch.unique(pred_seg)
for i in segments:
    mask = pred_seg == i
    name = model.config.id2label[i.item()]
    if name == "Dress":
        img_dress = Image.fromarray((mask * 255).numpy().astype(np.uint8))
    elif name == "Background":
        img_background = Image.fromarray((mask * 255).numpy().astype(np.uint8))
    elif name == "Left-leg":
        img_leftleg = Image.fromarray((mask * 255).numpy().astype(np.uint8))
    elif name == "Right-leg":
        img_rightleg = Image.fromarray((mask * 255).numpy().astype(np.uint8))

img_background = img_background.convert("L")
enhancer_background = ImageEnhance.Contrast(img_background)
img_background = enhancer_background.enhance(10.0)

img_dress = img_dress.convert("L")
enhancer_dress = ImageEnhance.Contrast(img_dress)
img_dress = enhancer_dress.enhance(10.0)

enhancer_leftleg = ImageEnhance.Contrast(img_leftleg)
img_leftleg = enhancer_leftleg.enhance(10.0)

enhancer_rightleg = ImageEnhance.Contrast(img_rightleg)
img_rightleg = enhancer_rightleg.enhance(10.0)

result_legs = Image.blend(img_rightleg, img_leftleg, 0.5)
result = Image.blend(img_dress, result_legs, 0.5)

result_gray = result.convert('L')
enhancer = ImageEnhance.Contrast(result_gray)
result_enhancer = enhancer.enhance(50.0)

result_enhancer.save("d:\SDM\sendReq2DB\mask\dress1025.png")
