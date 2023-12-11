import os, sys
sam_dir = os.path.join(os.getcwd(), 'segment-anything')
sys.path.insert(0, sam_dir)
os.chdir(sam_dir)

import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from segment_anything import SamPredictor, sam_model_registry

input_image = './logs/llff_pe/flower_lg_pe/k0.png'
input_point = np.array([[400, 400], [600, 400]]) # [x, y]
input_label = np.array([1, 1]) # 1 for fg, 0 for bg

image = np.array(Image.open(input_image))
sam = sam_model_registry["default"](checkpoint="./models/sam_vit_h_4b8939.pth").cuda()
predictor = SamPredictor(sam)
predictor.set_image(image)
masks, _, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
)
mask_fg = masks[0]
mask_bg = ~mask_fg
rgb_fg = image * mask_fg[..., None]
rgb_bg = image * mask_bg[..., None]
rgba_fg = np.concatenate([image, mask_fg[..., None] * 255], axis=-1).astype(np.uint8)
rgba_bg = np.concatenate([image, mask_bg[..., None] * 255], axis=-1).astype(np.uint8)
Image.fromarray(rgb_fg).save(input_image.replace('.png', '_extract_fg.png'))
Image.fromarray(rgb_bg).save(input_image.replace('.png', '_extract_bg.png'))
