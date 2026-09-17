# Object Detection (concise, correct Content-Type). Requires: pillow, requests, config.py (HF_API_KEY).
import os, io, time, random, requests, mimetypes
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from config import HF_API_KEY

Model="facebook/detr-resnet-101"
API=f"https://router.huggingface.co/hf-inference/models/{MODEL}"

ALLOWED, MAX_MB = {".jpg",".jpeg",".png",".bmp",".gif",".webp",".tiff"}, 8
Emoji={"person":"🧍","car":"🚗","truck":"🚚","bus":"🚌",
       "bicycle":"🚲","motorcycle":"🏍️","dog":"🐶","cat":"🐱",
       "bird":"🐦","horse":"🐴","sheep":"🐑","cow":"🐮",
       "bear":"🐻","giraffe":"🦒","zebra":"🦓","banana":"🍌",
       "apple":"🍎","orange":"🍊","pizza":"🍕","broccoli":"🥦",
       }
def font(sz=18):
    return ImageFont.truetype("arial.ttf", sz)
def ask_image():
    return Image.open(io.BytesIO(requests.get("https://picsum.photos/640/480").content))    
def infer(path, img_bytes, tries=8):
    for _ in range(tries):
        try:
            resp = requests.post(API, headers={"Authorization": f"Bearer {HF_API_KEY}"}, files={"image": ("image.jpg", img_bytes, "image/jpeg")})
            if resp.status_code == 200: return resp.json()
        except: time.sleep(random.uniform(0.5, 1.5))
def draw(img, dets, thr=0.5):
    path=""
    draw = ImageDraw.Draw(img)
    for det in dets:
        if det["score"] < thr: continue

        x1, y1, x2, y2 = det["box"]
        draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=3)
        draw.text((x1, y1-20), f"{det['class']} {det['score']:.2f}", font=font(16), fill="red")
        draw.text((x1, y1-40), f"{Emoji[det['class']]} {det['class']}", font=font(16), fill="red")
    img.save(path)

if __name__ == "__main__":
    img = ask_image()
