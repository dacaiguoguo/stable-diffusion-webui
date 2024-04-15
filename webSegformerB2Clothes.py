from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch.nn as nn

color_mapping = {
    0: (0, 0, 0),       # 背景: 黑色
    1: (255, 0, 0),     # 类别1: 红色
    2: (0, 255, 0),     # 类别2: 绿色
    3: (0, 0, 255),     # 类别3: 蓝色
    4: (255, 255, 0),   # 类别4: 黄色
    5: (255, 0, 255),   # 类别5: 洋红
    6: (0, 255, 255),   # 类别6: 青色
    7: (192, 192, 192), # 类别7: 银色
    8: (128, 0, 0),     # 类别8: 暗红
    9: (128, 128, 0),   # 类别9: 橄榄
    10: (0, 128, 0),    # 类别10: 暗绿
    11: (128, 0, 128),  # 类别11: 紫色
    12: (0, 128, 128),  # 类别12: 深青
    13: (0, 0, 128),    # 类别13: 深蓝
    14: (255, 165, 0),  # 类别14: 橙色
    15: (255, 105, 180),# 类别15: 粉红
    16: (75, 0, 130),    # 类别16: 靛青
    17: (255, 215, 0)    # 类别17: 金色
}



processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

# url = "http://127.0.0.1:9090/Business.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("/Users/yanguosun/Developer/aistudyGio/iphone-card-50-specialist-help-202309.jpeg")

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
# plt.imshow(pred_seg)

import numpy as np

# 需要排除的类别标签
exclude_labels = [2, 11]

# 创建一个空的彩色图像，大小和 pred_seg 一样，但是有三个颜色通道
color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)

# 将需要排除的标签对应的像素设置为背景色（这里假设背景色是黑色）
for label in exclude_labels:
    color_mapping.pop(label, None)  # 移除不需要的标签颜色映射

# 应用颜色映射
for label, color in color_mapping.items():
    # 找到所有该类别的像素点，然后将它们设置为对应的颜色
    color_seg[pred_seg == label] = color

# 将 numpy 数组转换为 PIL 图像
color_seg_img = Image.fromarray(color_seg)

# 显示彩色分割图像（可选）
color_seg_img.show()

# 保存彩色分割图像到文件
color_seg_img.save('color_segmentation_result.png')

