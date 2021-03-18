# demo for evaluator
import argparse
import os
from utils import transforms, render
import numpy as np
from PIL import Image
from scipy import ndimage as ndimg
import matplotlib.pyplot as plt

import model.model as module_arch
import torch
from tqdm import tqdm

from parse_config import ConfigParser

def inference(config):
    """
    inference the image
    :param img_path: image file name or image dir
    """
    # step1: load the image filename
    # determine whether img_path is a file or dir
    img_path = config.config['img']
    if os.path.isfile(img_path):
        img_list = [img_path]
    else:
        img_list = os.listdir(img_path)
        img_list = [os.path.join(img_path, f) for f in img_list]

    # filter the img_list elements that not image format
    image_format = ('.png', '.jpg', '.tif')
    img_list = [f for f in img_list if f.endswith(image_format)]

    # step2: load the model and model weight
    model = config.init_obj('arch', module_arch)
    state_dict = torch.load(config.resume)
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare models for inferencing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # step3: start loop inference
    for img_i in tqdm(img_list):
        inference_single(img_i, model, device)


def inference_single(img_path, model, device):
    # 3.1 read the image
    image = Image.open(img_path)
    image = np.array(image.convert('RGB'))

    # 3.2 pre-process the image
    img = transforms.reshape_and_normalize_data(image, channels=[0, 0], normalize=True)
    img, slc = transforms.pad_image_ND(img)
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img)

    # 3.3 model forward
    output, style = model(img.to(device))
    output = output[0].detach().cpu().numpy()
    style = style[0].detach().cpu().numpy()

    # 3.4 post-process the model output
    # remove padding
    output = output[slc]
    # transpose so the channel is last axis
    output = np.transpose(output, (1, 2, 0))

    # flow to mask
    lab = flow2msk(output)
    render.show(image, output, lab)

    # 3.5 save the results
    pass

def flow2msk(flowp, level=0.5, grad=0.5, area=None, volume=None):
    flowp = np.asarray(flowp)
    shp, dim = flowp.shape[:-1], flowp.ndim - 1
    l = np.linalg.norm(flowp[:,:,:2], axis=-1)
    flow = flowp[:,:,:2]/l.reshape(shp+(1,))
    flow[(flowp[:,:,2]<level)|(l<grad)] = 0
    ss = ((slice(None),) * (dim) + ([0,-1],)) * 2
    for i in range(dim):flow[ss[dim-i:-i-2]+(i,)]=0
    sn = np.sign(flow); sn *= 0.5; flow += sn;
    dn = flow.astype(np.int32).reshape(-1, dim)
    strides = np.cumprod(np.array((1,)+shp[::-1]))
    dn = (strides[-2::-1] * dn).sum(axis=-1)
    rst = np.arange(flow.size//dim); rst += dn
    for i in range(10): rst = rst[rst]
    hist = np.bincount(rst, None, len(rst))
    hist = hist.astype(np.uint32).reshape(shp)
    lab, n = ndimg.label(hist, np.ones((3,)*dim))
    volumes = ndimg.sum(hist, lab, np.arange(n+1))
    areas = np.bincount(lab.ravel())
    mean, std = estimate_volumes(volumes, 2)
    if not volume: volume = max(mean-std*3, 50)
    if not area: area = volumes // 3
    msk = (areas<area) & (volumes>volume)
    lut = np.zeros(n+1, np.uint32)
    lut[msk] = np.arange(1, msk.sum()+1)
    return lut[lab].ravel()[rst].reshape(shp)
    return hist, lut[lab], mask


def estimate_volumes(arr, sigma=3):
    msk = arr > 50
    idx = np.arange(len(arr), dtype=np.uint32)
    idx, arr = idx[msk], arr[msk]
    for k in np.linspace(5, sigma, 5):
       std = arr.std()
       dif = np.abs(arr - arr.mean())
       msk = dif < std * k
       idx, arr = idx[msk], arr[msk]
    return arr.mean(), arr.std()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-i', '--img', default='data/test1.png', type=str,
                      help='image path or image dir')
    args.add_argument('-r', '--resume', default='saved/models/CellposeNet/cytotorch', type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    config = ConfigParser.from_args(args)
    inference(config)