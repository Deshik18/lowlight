import os
import sys
import argparse
import torch
import torch.nn as nn
import torchvision
import torch.optim

from . import model_small  # Relative import if in the same package
import numpy as np
from PIL import Image
import glob
import time

# --- Parse hyper-parameters  --- #
# parser = argparse.ArgumentParser(description='PyTorch implementation of CLIP-LIT (liang. 2023)')
# parser.add_argument('-i', '--input', help='directory of input folder', default='./input/')
# parser.add_argument('-o', '--output', help='directory of output folder', default='./inference_result/')
# parser.add_argument('-c', '--ckpt', help='test ckpt path', default='./pretrained_models/enhancement_model.pth')


def lowlight(image_path):
    args_ckpt = "algorithms/CLIP/enhancement_model.pth"
    args_output = "results/"
    # U_net = model_small.UNet_emb_oneBranch_symmetry_noreflect(3,1)
    U_net = model_small.UNet_emb_oneBranch_symmetry(3,1)

    state_dict = torch.load(args_ckpt, map_location=torch.device('cpu'))

    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    U_net.load_state_dict(new_state_dict)

    data_lowlight = Image.open(image_path)#.convert("RGB")
    if data_lowlight.mode == 'RGBA':
        data_lowlight = data_lowlight.convert("RGB")

    data_lowlight = (np.asarray(data_lowlight)/255.0)

    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2,0,1)
    data_lowlight = data_lowlight.unsqueeze(0)

    light_map = U_net(data_lowlight)
    enhanced_image = torch.clamp((data_lowlight / light_map), 0, 1)

    image_path = args_output + os.path.basename(image_path)
    image_path = image_path.replace('.jpg','.png')
    image_path = image_path.replace('.JPG','.png')

    result_path = image_path
    if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
        os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))

    torchvision.utils.save_image(enhanced_image, result_path)
    return result_path

if __name__ == '__main__':
    with torch.no_grad():
        filePath = args.input
        file_list = os.listdir(filePath)
        #print(file_list)

        for file_name in file_list:
            image = filePath + file_name
            #print(image)
            lowlight(image)
