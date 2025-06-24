import json
import tarfile

import gradio as gr
import numpy as np

from os import path, remove
from PIL import Image as PImage
from urllib import request

def download_extract(url):
  target_path = url.split("/")[-1]

  with request.urlopen(request.Request(url), timeout=15.0) as response:
    if response.status == 200:
      with open(target_path, "wb") as f:
        f.write(response.read())
  
  tar = tarfile.open(target_path, "r:gz")
  tar.extractall()
  tar.close()
  remove(target_path)

download_extract("https://digitais.acervos.at.eu.org/imgs/herbario/arts/full.tgz")

DATA_FILE = "./metadata/json/20250619_processed.json"
IMG_DIR = "./full"

with open(DATA_FILE, "r") as ifp:
  all_data = json.load(ifp)

def get_min_height_and_size(idObjIdxs_data):
  height_min = 1e6
  sizes = {}

  for idObjIdxs in idObjIdxs_data:
    if len(idObjIdxs["objIdxs"]) < 1:
      continue
    id = idObjIdxs["id"]
    img = PImage.open(path.join(IMG_DIR, f"{id}.jpg"))
    iw,ih = img.size
    sizes[id] = (iw,ih)

    for idx in idObjIdxs["objIdxs"]:
      (x0,y0,x1,y1) = all_data[id]["objects"][idx]["box"]
      crop_h = ih * (y1 - y0)
      height_min = min(height_min, crop_h)

  return height_min, sizes

def get_mosaic_size(idObjIdxs_data, height_min, sizes):
  width_sum = 0

  for idObjIdxs in idObjIdxs_data:
    if len(idObjIdxs["objIdxs"]) < 1:
      continue
    id = idObjIdxs["id"]
    iw,ih = sizes[id]
    for idx in idObjIdxs["objIdxs"]:
      (x0,y0,x1,y1) = all_data[id]["objects"][idx]["box"]
      crop_w = iw * (x1 - x0)
      crop_h = ih * (y1 - y0)
      width_sum += (height_min / crop_h) * crop_w

  mos_w = int((width_sum * height_min) ** 0.5)
  mos_h = int(1.2 * mos_w)
  return mos_w, mos_h

def get_mosaic(idObjIdxs_data):
  height_min, sizes = get_min_height_and_size(idObjIdxs_data)
  mos_w, mos_h = get_mosaic_size(idObjIdxs_data, height_min, sizes)

  mos_img = PImage.fromarray(np.zeros((mos_h, mos_w))).convert("RGB")
  cur_x, cur_y = 0, 0

  for idObjIdxs in idObjIdxs_data:
    if len(idObjIdxs["objIdxs"]) < 1:
      continue
    id = idObjIdxs["id"]
    img = PImage.open(path.join(IMG_DIR, f"{id}.jpg"))
    iw,ih = img.size
    for idx in idObjIdxs["objIdxs"]:
      (x0,y0,x1,y1) = all_data[id]["objects"][idx]["box"]
      crop_w = iw * (x1 - x0)
      crop_h = ih * (y1 - y0)

      scale_factor = height_min / crop_h
      crop_w, crop_h = int(scale_factor * crop_w), int(scale_factor * crop_h)

      crop_img = img.crop((int(x0 * iw), int(y0 * ih), int(x1 * iw), int(y1 * ih))).resize((crop_w, crop_h))

      if cur_y >= mos_h:
        print("break")
        break

      mos_img.paste(crop_img, (cur_x, cur_y))
      cur_x += crop_w

      if cur_x > mos_w:
        overflow_x = cur_x - mos_w
        crop_img = crop_img.crop((crop_w - overflow_x, 0, crop_w, crop_h))
        cur_x = 0
        cur_y += crop_h
        mos_img.paste(crop_img, (cur_x, cur_y))
        cur_x += overflow_x

  if cur_x < mos_w and cur_y < mos_h:
    empty_w = mos_w - cur_x
    row = mos_img.crop((0, 0, empty_w, height_min))
    mos_img.paste(row, (cur_x, cur_y))

  mos_img = mos_img.crop((0, 0, mos_w, cur_y + crop_h))
  mos_img.thumbnail((1024,1024))
  return mos_img

with gr.Blocks() as demo:
  gr.Interface(
    fn=get_mosaic,
    inputs="json",
    outputs="image",
    flagging_mode="never",
  )

if __name__ == "__main__":
   demo.launch()
