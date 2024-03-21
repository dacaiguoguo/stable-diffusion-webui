from safetensors import torch
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
from PIL import ImageEnhance
import requests
import torch.nn as nn
import torch
import numpy as np
def get_masks(segmentation):
    obj_ids = torch.unique(segmentation)
    obj_ids = obj_ids[1:]
    # masks = segmentation == obj_ids[:, None, None]
    masks = segmentation == 7
    return masks, obj_ids
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
url = "http://127.0.0.1:9090/IMG_1572.jpg" #用python -m http.server 9090 &定义了当前主机服务器
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")
# plt.imshow(image)
# plt.axis('off')
# pylab.show()
outputs = model(**inputs)
logits = outputs.logits.cpu()
upsampled_logits = nn.functional.interpolate(
    logits,
    size=image.size[::-1],
    mode="bilinear",
    align_corners=False,
    )
pred_seg = upsampled_logits.argmax(dim=1)[0]
# plt.imshow(pred_seg)
# pylab.show()
# Background
# Hair
# Upper-clothes
# Pants
# Left-shoe
# Face
# Left-arm
# Right-arm


segments = torch.unique(pred_seg) # Get a list of all the predicted items
print(segments)
for i in segments:
    mask = pred_seg == i # Filter out anything that isn't the current item
    name = model.config.id2label[i.item()] # get the item name
    print(name)
    if name == "Dress" :
        img_dress = Image.fromarray((mask * 255).numpy().astype(np.uint8))
    # plt.imshow(img_dress)
    # plt.title(name)
    # plt.show()
    elif name == "Background" :
        img_background = Image.fromarray((mask * 255).numpy().astype(np.uint8))
    # plt.imshow(img_face)
    # plt.title(name)
    # plt.show()
    elif name == "Left-leg":
        img_leftleg = Image.fromarray((mask * 255).numpy().astype(np.uint8))
    elif name == "Right-leg":
        img_rightleg = Image.fromarray((mask * 255).numpy().astype(np.uint8))


        
#将图片二值化
img_background = img_background.convert("L")
enhancer_background = ImageEnhance.Contrast(img_background)
img_background = enhancer_background.enhance(10.0)
img_dress = img_dress.convert("L")
enhancer_dress = ImageEnhance.Contrast(img_dress)
img_dress = enhancer_dress.enhance(10.0)
# img_leftleg = img_leftleg.convert("L")
enhancer_leftleg = ImageEnhance.Contrast(img_leftleg)
img_leftleg = enhancer_leftleg.enhance(10.0)
# img_rightleg = img_rightleg.convert("L")
enhancer_rightleg = ImageEnhance.Contrast(img_rightleg)
img_rightleg = enhancer_rightleg.enhance(10.0)
#如果没有混需求
# result = img_dress
# 下一步有两个问题要解决，第一，如何读取本地文件；第二，如何将不同的蒙版合并
# result_dress = Image.blend(img_dress,img_background,0.5)
result_legs = Image.blend(img_rightleg,img_leftleg,0.5)
result = Image.blend(img_dress,result_legs,0.5)
# plt.imshow(result)
# plt.axis('off')
# plt.show()
#转换为灰度图
result_gray = result.convert('L')
#加大对比度
enhancer = ImageEnhance.Contrast(result_gray)
result_enhancer = enhancer.enhance(50.0) #增加两倍的对比度
#显示图片
# result_enhancer.show()
#保存文件到mask文件夹，需要对应文件名方便加载蒙版
result_enhancer.save("/Users/yanguosun/Developer/stable-diffusion-webui/outputs/dress1025.png")
