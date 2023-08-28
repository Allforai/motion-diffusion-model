import matplotlib
import torch
from tqdm import tqdm
from body_models.smplh import SMPLH
import os
import numpy as np
from data_loaders.p2m.tools import inverse
from torch.utils.data import DataLoader
smplh = SMPLH(
    path='/mnt/disk_1/jinpeng/motion-diffusion-model/body_models/',
    input_pose_rep='rot6d',
    batch_size=1,
    gender='neutral').to('cuda').eval()

source = np.load('/mnt/disk_1/jinpeng/motion-diffusion-model/save/0813_cross/samples_0813_cross_000300000_seed10/t2m_results.npy',allow_pickle=True).item()
target = {}

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
np.save('/mnt/disk_1/jinpeng/motion-diffusion-model/GPT_response_8022_smplh/total_batch.npy', target)
