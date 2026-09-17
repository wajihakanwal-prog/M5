import os,io,time,requests,random,mimetypes
from xmlrpc import client
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter,ImageFont,ImageDraw
from config import HF_API_KEY
model="facebook/detr-resnet-101"
API_URL = f"https://api-inference.huggingface.co/models/{model}"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}
ALLOWED_MIME_TYPES , max_db= ["image/jpeg", "image/png", "image/bmp", "image/gif", "image/webp"]
EMOJI_SIZE = {"person": 0.05, "car": 0.1, "dog": 0.08, "cat": 0.08, "bicycle": 0.1, "motorcycle": 0.1,
               "airplane": 0.15, "bus": 0.15, "train": 0.15, "truck": 0.15, "boat": 0.15, "horse": 0.08,
               "sheep": 0.08, "cow": 0.08, "elephant": 0.15, "bear": 0.15, "zebra": 0.15, "giraffe": 0.15, "backpack": 0.05, "umbrella": 0.05,
               "handbag": 0.05, "tie": 0.05, "suitcase": 0.05, "frisbee": 0.05, "skis": 0.05, "snowboard": 0.05, "sports ball": 0.05,
               "kite": 0.05, "baseball bat": 0.05, "baseball glove": 0.05, "skateboard": 0.05, "surfboard": 0.05,
               "tennis racket": 0.05, "bottle": 0.05}
def font_size(sz=18):
 for f in ("dejaVuSans.ttf", "arial.ttf"):
  try:
   return ImageFont.truetype(f, sz)
  except Exception:
   continue
 return ImageFont.load_default()

def ask_image():
 while True:
  p=input("Enter image path or URL (or 'q' to quit): ").strip()
  if p.lower() in ["q", "quit", "exit"]:
    return None
  try:
   im = Image.open(p)
   return im
  except Exception as e:
   print(f"Error opening image: {e}")

def infer(path , img_bytes=None,tries=9):
      mime_type = mimetypes.guess_type(path)[0]
      for _ in range(tries):
       try:
        if img_bytes is None:
         with open(path, "rb") as f:
          img_bytes = f.read()
        response = client.infer(path, mime_type, img_bytes)
        return response
       except Exception as e:
        print(f"Error: {e}")
        time.sleep(1)
      return None
    
