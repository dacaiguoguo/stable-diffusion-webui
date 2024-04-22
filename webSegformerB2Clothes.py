from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import os

# 色彩映射定义
color_mapping = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
    5: (255, 0, 255),
    6: (0, 255, 255),
    7: (192, 192, 192),
    8: (128, 0, 0),
    9: (128, 128, 0),
    10: (0, 128, 0),
    12: (0, 128, 128),
    13: (0, 0, 128),
    14: (255, 165, 0),
    15: (255, 105, 180),
    16: (75, 0, 130),
    17: (255, 215, 0)
}

# 初始化模型和处理器
processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

# 图像文件夹路径
directory = "/Users/yanguosun/Developer/aiserver/configurations/collect_images"

# 遍历目录中的所有文件
for filename in os.listdir(directory):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):  # 检查文件是否为图像
        image_path = os.path.join(directory, filename)
        image = Image.open(image_path).convert('RGB')  # 确保图像为 RGB 格式

        # 处理图像并获取模型输出
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits.cpu()

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],  # 图像大小
            mode="bilinear",
            align_corners=False
        )

        # 获取最可能的类别
        pred_seg = upsampled_logits.argmax(dim=1)[0]

        # 创建彩色分割图像
        color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
        for label, color in color_mapping.items():
            color_seg[pred_seg == label] = color

        # 转换为 PIL 图像并保存覆盖原图
        color_seg_img = Image.fromarray(color_seg)
        color_seg_img.save(image_path)

print("处理完成，所有图像已更新。")