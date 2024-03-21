from PIL import Image
import io
import os
import json
import base64
import random
import requests

# 假设的其他模块函数
from imageCrop import cropImage2Video
from pythonJpgToPng import image2png
from translateFromCh2EnYoudao import translateYouDao

def encode_pil_to_base64(image):
    with io.BytesIO() as output_bytes:
        image.save(output_bytes, format="PNG")
        bytes_data = output_bytes.getvalue()
    return base64.b64encode(bytes_data).decode("utf-8")

def submit_post(url: str, data: dict):
    return requests.post(url, data=json.dumps(data))

def save_encoded_image(b64_image: str, output_path: str):
    with open(output_path, "wb") as image_file:
        image_file.write(base64.b64decode(b64_image))

def joinLoraList(loraName01, weight01, loraName02, weight02, loraName03, weight03):
    loraPart01 = "<lora:%s:%2.1f>" % (loraName01, weight01) if loraName01 else ""
    loraPart02 = "<lora:%s:%2.1f>" % (loraName02, weight02) if loraName02 else ""
    loraPart03 = "<lora:%s:%2.1f>" % (loraName03, weight03) if loraName03 else ""
    loraList = loraPart01 + loraPart02 + loraPart03
    print(loraList)
    return loraList

def maskImage2Image(text_prompt, loraList, denoisingStrength, localPythonExePath, gifImageNum, seed):
    localFilePath = "D:\SDM\sendReq2DB\image\%d.jpg" % gifImageNum
    localCroppedFilePath = localPythonExePath + "\image\i2i\croppedimage\croppedpromptimage.jpg"
    
    cropImage2Video(localFilePath, localCroppedFilePath)
    promptImageFilePath = localPythonExePath + "\image\i2i\croppedimage"
    image2png(promptImageFilePath, "png")
    
    promptimage_temp = Image.open(localPythonExePath + "\prompt_image\croppedpromptimage.png")
    promptimage_temp_code = encode_pil_to_base64(promptimage_temp)
    
    cropImage2Video(localPythonExePath + "\mask\dress1025.png", localPythonExePath + "\mask\dress1025.png")
    mask_temp = Image.open(localPythonExePath + "\mask\dress1025.png")
    mask_temp_code = encode_pil_to_base64(mask_temp)
    
    os.remove("%s\prompt_image\croppedpromptimage.png" % localPythonExePath)
    
    image2Image_url = 'http://127.0.0.1:7861/sdapi/v1/img2img'
    data = {
        "init_images": [promptimage_temp_code],
        "mask": mask_temp_code,
        "mask_blur_x": 4,
        "mask_blur_y": 4,
        "mask_blur": 0,
        "prompt": text_prompt + loraList,
        'brach_size': 1,
        "steps": 20,
        "denoising_strength": denoisingStrength,
        "cfg_scale": 8,
        "width": 512,
        "height": 910,
        "seed": seed,
        "restore_faces": "true",
        "negative_prompt": "nsfw,blurry,bad anatomy,low quality,worst quality,normal quality",
        "alwayson_scripts": {
            "ControlNet": {
                "args": [{
                    "enabled": "true",
                    "pixel_perfect": "true",
                    "module": "none",
                    "model": "control_v11p_sd15_openpose",
                    "weight": 1.5
                }, {
                    "enabled": "true",
                    "pixel_perfect": "true",
                    "module": "none",
                    "model": "control_v11f1e_sd15_tile",
                    "weight": 0.6
                }]
            }
        }
    }
    
    response = submit_post(image2Image_url, data)
    save_encoded_image(response.json()['images'][0], 'FramesNew\%d.png' % (gifImageNum))
    print("图片已经生成,并保存在image目录中")


if __name__ == '__main__':

    # text_prompt = "美女，JK，(short-sleeved JK_shirt),JK_style,黑色裙子"
    text_prompt = "美女,mahalaiuniform,站立姿势, ((短发))" #mahalaiuniform-000001泰国风格
    text_prompt = translateYouDao(text_prompt)
    loraList = joinLoraList("mahalaiuniform-000001", 1.0, '', 1.0, '', 1.0)
    denoisingStrength = 0.8
    seed = random.randint(1, 10000000)
    localPythonExePath = "D:\SDM\sendReq2DB"
    gifImageNum = 20230925151939 #原始图片文件名编号

    maskImage2Image(text_prompt, loraList, denoisingStrength, localPythonExePath, gifImageNum,seed) 
    
    # 作者：北大BIM老龙 https://www.bilibili.com/read/cv27441533/ 出处：bilibili