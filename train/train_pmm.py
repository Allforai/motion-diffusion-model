from utils.parser_util import train_args
from utils.fixseed import fixseed
import os
import json
from utils import dist_util
from os.path import join as pjoin
from argparse import ArgumentParser
import torch
from data_loaders.p2m.dataset import HumanML3D
from torch.utils.data import DataLoader
from model import temos_encoder, pose_encoder
from data_loaders.p2m.trainers import PoseMotionMatchTrainer
from utils.model_util import create_model_and_diffusion
from torch.nn.utils import clip_grad_norm_


def build_models():
    pose_enc = pose_encoder.PoseEncoder(num_neurons=512, num_neurons_mini=32, latentD=256, role="retrieval")
    motion_enc = temos_encoder.ActorAgnosticEncoder(nfeats=135, vae=False, latent_dim=256, ff_size=1024, num_layers=2, num_heads=4, dropout=0.1, activation="gelu")

    # if not opt.is_continue:
    #    checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.decomp_name, 'model', 'latest.tar'),
    #                            map_location=opt.device)
    return pose_enc, motion_enc


def parser():
    parser = ArgumentParser()
    parser.add_argument("--negative_margin", default=3.0, type=float)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--max_epoch", default=1000, type=int)
    parser.add_argument("--log_every", default=5, type=int)
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--model_dir", default='/mnt/disk_1/jinpeng/motion-diffusion-model/save/pmm/0817_1350_batch512_2layer_encoder', type=str)
    parser.add_argument("--eval_dir", default='/mnt/disk_1/jinpeng/motion-diffusion-model/save/pmm/0817_1350_batch512_2layer_encoder', type=str)
    parser.add_argument("--save_every_e", default=5, type=int)
    parser.add_argument("--eval_every_e", default=5, type=int)
    # parser.add_argument("--save_latest", default=100, type=int)

    return parser


if __name__ == '__main__':
    argparser = parser()
    args = argparser.parse_args()
    batch_size = 512
    print("creating data loader...")
    training_data = HumanML3D(datapath='dataset/p2m_humanml_opt.txt')
    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=8)

    testing_data = HumanML3D(datapath='dataset/p2m_humanml_opt.txt', split='test')
    test_loader = DataLoader(testing_data, batch_size=batch_size, shuffle=True, num_workers=8)

    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    pose_enc, motion_enc = build_models()

    trainer = PoseMotionMatchTrainer(args=args, pose_encoder=pose_enc, motion_encoder=motion_enc)
    trainer.train(train_loader, test_loader)

