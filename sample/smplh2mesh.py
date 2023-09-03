import parser

import matplotlib
import torch
from tqdm import tqdm
from body_models.smplh import SMPLH
import os
import pickle
import numpy as np
from data_loaders.p2m.tools import inverse
from torch.utils.data import DataLoader
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', help='data path')
    args = parser.parse_args()
    return args


args = parse_args()

smplh = SMPLH(
    path='/mnt/disk_1/jinpeng/motion-diffusion-model/body_models/',
    input_pose_rep='rot6d',
    batch_size=1,
    gender='neutral').to('cuda').eval()
source_path = args.source_path
source = np.load(source_path, allow_pickle=True).item()
target = {}
# pre
for file in list(source['sample'].keys()):
    target[file] = []
    loader = DataLoader(source['sample'][file], batch_size=32, shuffle=False, num_workers=8)
    print(len(source['sample'][file]))
    for i, data in enumerate(tqdm(loader)):
        data = data.squeeze(1).to('cuda')
        trans_gt = data[:, 0:3].permute(0, 3, 2, 1)
        trans_gt = inverse(trans_gt)
        pose_gt = data[:, 3:].permute(0, 3, 2, 1).reshape(data.shape[0], 64, -1, 6)
        vertices_gt = smplh(data.shape[0], pose_gt,
                            trans_gt).cpu().numpy().astype(np.float16)
        target[file].append(np.split(vertices_gt, vertices_gt.shape[0]))
    target[file] = np.concatenate(target[file], axis=0)
save_path = source_path.replace('results', 'smplh')
save_path = save_path.replace('npy', 'pkl')
save_path = open(save_path, 'wb')
pickle.dump(target, save_path, protocol=4)

# ## gt
# target = []
# loader = DataLoader(source['gt'], batch_size=32, shuffle=False, num_workers=8)
# print(len(source['gt']))
# for i, data in enumerate(tqdm(loader)):
#     data = data.squeeze(1).to('cuda')
#     trans_gt = data[:, 0:3].permute(0, 3, 2, 1)
#     trans_gt = inverse(trans_gt)
#     pose_gt = data[:, 3:].permute(0, 3, 2, 1).reshape(data.shape[0], 64, -1, 6)
#     vertices_gt = smplh(data.shape[0], pose_gt,
#                         trans_gt).cpu().numpy().astype(np.float16)
#     target.append(np.split(vertices_gt, vertices_gt.shape[0]))
# target = np.concatenate(target, axis=0)
# save_path = source_path.replace('results.npy', 'smplh_gt.pkl')
# save_path = open(save_path, 'wb')
# pickle.dump(target, save_path, protocol=4)