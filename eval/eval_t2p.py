from utils.fixseed import fixseed
from torch.utils.data import DataLoader
import os
import torch
from utils.parser_util import generate_args
from utils import dist_util
import numpy as np
from collections import OrderedDict
from datetime import datetime
from scipy import linalg
from model.temos_encoder import ActorAgnosticEncoder
from model.pose_encoder import PoseEncoder
from data_loaders.p2m.eval_dataset import Pose2Motion

torch.set_default_dtype(torch.float32)
from sklearn.metrics import mean_squared_error
from eval.eval_p2m import calculate_activation_statistics, calculate_frechet_distance
import numpy as np


def main():
    dataset_gt = Pose2Motion('/mnt/disk_1/jinpeng/motion-diffusion-model/GPT_response_8022_smplh/motion_data.npy',
                             0)
    dataloader_gt = DataLoader(dataset_gt, batch_size=32, shuffle=False, num_workers=8)
    dataset_pre = Pose2Motion(
        '/mnt/disk_1/jinpeng/motion-diffusion-model/GPT_response_8022_smplh/motion_data_gptpose.npy',
        0)
    dataloader_pre = DataLoader(dataset_pre, batch_size=32, shuffle=False, num_workers=8)
    fid, mse = evaluate_fid_and_mse(dataloader_gt, dataloader_pre)
    return None


def evaluate_fid_and_mse(dataloader_gt, dataloader_pre):
    gt_pose_embeddings = []
    pre_pose_embeddings = []
    mse_loss = []
    pose_encoder, _ = build_models()
    loss_fn1 = torch.nn.MSELoss(reduction='none')
    with torch.no_grad():
        for (_, cond_gt, _, _), (_, cond_pre, _, _) in zip(dataloader_gt, dataloader_pre):
            cond_gt = cond_gt.to('cuda').squeeze(1).squeeze(-2).permute(0, 2, 1).contiguous()
            '''GT Pose Encoding'''
            pose_embedding = pose_encoder(cond_gt)
            gt_pose_embeddings.append(pose_embedding.cpu().numpy())
            cond_pre = cond_pre.to('cuda').squeeze(1).squeeze(-2).permute(0, 2, 1).contiguous()
            '''Pre Pose Encoding'''
            pose_embedding = pose_encoder(cond_pre)
            pre_pose_embeddings.append(pose_embedding.cpu().numpy())
            " MSE Loss "
            loss1 = loss_fn1(cond_gt, cond_pre)
            mse_loss.append(loss1)
    gt_pose_embeddings = np.concatenate(gt_pose_embeddings, axis=0)
    pre_pose_embeddings = np.concatenate(pre_pose_embeddings, axis=0)
    mse_loss = torch.concatenate(mse_loss, axis=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(gt_pose_embeddings)
    mu, cov = calculate_activation_statistics(pre_pose_embeddings)
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    print(f'--->  FID: {fid:.8f}')
    print(f"=========Evaluating MSE: ============")
    mse = mse_loss.mean()
    print(f'--->  MSE: {mse:.4f}')
    return fid, mse


def build_models():
    pose_enc = PoseEncoder(num_neurons=512, num_neurons_mini=32, latentD=256, role="retrieval")
    motion_enc = ActorAgnosticEncoder(nfeats=135, vae=False, latent_dim=256, ff_size=1024, num_layers=4, num_heads=4,
                                      dropout=0.1, activation="gelu")
    checkpoint = torch.load('/mnt/disk_1/jinpeng/motion-diffusion-model/save/pmm/0816_1821/finest.tar',
                            map_location='cuda')
    pose_enc.load_state_dict(checkpoint['pose_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    pose_enc.eval()
    motion_enc.eval()
    return pose_enc.to('cuda'), motion_enc.to('cuda')


if __name__ == '__main__':
    main()
