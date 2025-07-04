import json
import tarfile

import gradio as gr
import numpy as np

from os import makedirs, path, remove
from PIL import Image as PImage
from urllib import request

OBJS_URLS = "https://raw.githubusercontent.com/acervos-digitais/herbario-data/main/json/20250705_processed.json"
IMG_URL = "https://digitais.acervos.at.eu.org/imgs/herbario/arts"
IMG_DIR = "./imgs/full"
MAX_PIXELS = 2**25
PImage.MAX_IMAGE_PIXELS = 2 * MAX_PIXELS


### define functions
def download_file(url, local_path="."):
  file_name = url.split("/")[-1]
  file_path = path.join(local_path, file_name)

  with request.urlopen(request.Request(url), timeout=30.0) as response:
    if response.status == 200:
      with open(file_path, "wb") as f:
        f.write(response.read())
  return file_path

def download_extract(url, target_path):
  tar_path = download_file(url)

  tar = tarfile.open(tar_path, "r:gz")
  tar.extractall(target_path, filter="data")
  tar.close()
  remove(tar_path)


def download_image(id):
  image_url = f"{IMG_URL}/full/{id}.jpg"
  filename = f"{IMG_DIR}/{id}.jpg"
  request.urlretrieve(image_url, filename)


def get_min_height_and_size(idObjIdxs_data, min_min_height=64):
  heights = []
  sizes = {}

  for idObjIdxs in idObjIdxs_data:
    id = idObjIdxs["id"]

    img = PImage.open(path.join(IMG_DIR, f"{id}.jpg"))
    iw,ih = img.size
    sizes[id] = (iw,ih)

    for idx in idObjIdxs["objIdxs"]:
      (x0,y0,x1,y1) = all_data[id]["objects"][idx]["box"]
      crop_h = ih * (y1 - y0)
      heights.append(max(crop_h, min_min_height))

  heights_sorted = list(set(sorted(heights)))

  return heights_sorted[0], sizes


def get_mosaic_size(idObjIdxs_data, height_min, sizes):
  width_sum = 0

  for idObjIdxs in idObjIdxs_data:
    id = idObjIdxs["id"]

    iw,ih = sizes[id]

    for idx in idObjIdxs["objIdxs"]:
      (x0,y0,x1,y1) = all_data[id]["objects"][idx]["box"]
      crop_w = iw * (x1 - x0)
      crop_h = ih * (y1 - y0)
      width_sum += (height_min / crop_h) * crop_w

  total_area = width_sum * height_min
  scale = 1.0
  mos_w = total_area ** 0.5
  mos_h = 1.2 * mos_w

  if total_area > MAX_PIXELS:
    scale = (MAX_PIXELS / total_area) ** 0.5
    mos_w *= scale
    mos_h *= scale

  return int(mos_w), int(mos_h), scale


def get_mosaic(idObjIdxs_all):
  idObjIdxs_data = [x for x in idObjIdxs_all if len(x["objIdxs"]) > 0]

  for id in [x["id"] for x in idObjIdxs_data]:
    if not path.isfile(path.join(IMG_DIR, f"{id}.jpg")):
      download_image(id)

  height_min, sizes = get_min_height_and_size(idObjIdxs_data)
  mos_w, mos_h, limit_scale = get_mosaic_size(idObjIdxs_data, height_min, sizes)

  mos_img = PImage.fromarray(np.zeros((mos_h, mos_w))).convert("RGB")
  cur_x, cur_y = 0, 0

  for idObjIdxs in idObjIdxs_data:
    id = idObjIdxs["id"]

    img = PImage.open(path.join(IMG_DIR, f"{id}.jpg"))
    iw,ih = img.size

    for idx in idObjIdxs["objIdxs"]:
      (x0,y0,x1,y1) = all_data[id]["objects"][idx]["box"]
      crop_w = iw * (x1 - x0)
      crop_h = ih * (y1 - y0)

      scale_factor = limit_scale * (height_min / crop_h)
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
  mos_img.thumbnail((2048, 2048))
  return mos_img


### prep files and dirs
makedirs(IMG_DIR, exist_ok=True)
objs_file_path = download_file(OBJS_URLS)
download_extract(f"{IMG_URL}/full.tgz", "./imgs")

with open(objs_file_path, "r") as ifp:
  all_data = json.load(ifp)


### start Gradio
with gr.Blocks() as demo:
  gr.Interface(
    fn=get_mosaic,
    inputs="json",
    outputs="image",
    flagging_mode="never",
  )

if __name__ == "__main__":
   demo.launch()
