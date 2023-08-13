# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from random import random

from tqdm import tqdm
from data_loaders.p2m.dataset import HumanML3D
from utils.fixseed import fixseed
from torch.utils.data import DataLoader
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from data_loaders.tensors import collate
from body_models.smplh import SMPLH
from data_loaders.p2m.tools import inverse
torch.set_default_dtype(torch.float32)


def main():
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 64
    fps = 12.5 if args.dataset == 'kit' else 20
    n_frames = min(max_frames, int(args.motion_length*fps))
    is_using_data = True
    # is_using_data = not any([args.input_text, args.text_prompt, args.action_file, args.action_name])
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')

    # this block must be called BEFORE the dataset is loaded
    if args.text_prompt != '':
        texts = [args.text_prompt]
        args.num_samples = 1
    elif args.input_text != '':
        assert os.path.exists(args.input_text)
        with open(args.input_text, 'r') as fr:
            texts = fr.readlines()
        texts = [s.replace('\n', '') for s in texts]
        args.num_samples = len(texts)
    elif args.action_name:
        action_text = [args.action_name]
        args.num_samples = 1
    elif args.action_file != '':
        assert os.path.exists(args.action_file)
        with open(args.action_file, 'r') as fr:
            action_text = fr.readlines()
        action_text = [s.replace('\n', '') for s in action_text]
        args.num_samples = len(action_text)

    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples


    print("Loading Dataset")
    data = HumanML3D(datapath='dataset/p2m_humanml_opt.txt', split='test')
    train_loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=8)
    # data = load_dataset(args, max_frames, n_frames)
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, train_loader)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking
    iterator = iter(data)
    smplh = SMPLH(
        path='/mnt/disk_1/jinpeng/motion-diffusion-model/body_models/',
        input_pose_rep='rot6d',
        batch_size=1,
        gender='neutral').to(dist_util.dev()).eval()
    all_motions = []
    all_motions_gt = []
    for file in tqdm(range(len(data))):
        source, model_kwargs = next(iterator)

        for rep_i in range(args.num_repetitions):
            print(f'### Sampling [repetitions #{rep_i}]')

            # add CFG scale to batch
            if args.guidance_param != 1:
                model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
            model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in
                                 model_kwargs['y'].items()}
            source = source.to(dist_util.dev())
            if len(model_kwargs['y']['pose_feature'].shape) == 2:
                model_kwargs['y']['pose_feature'] = model_kwargs['y']['pose_feature'].unsqueeze(0)
            sample_fn = diffusion.p_sample_loop

            final = sample_fn(
                model,
                # (args.batch_size, model.njoints, model.nfeats, n_frames),  # BUG FIX - this one caused a mismatch between training and inference
                (args.batch_size, model.njoints, model.nfeats, max_frames),  # BUG FIX
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            sample = final['pred_xstart']
            trans = sample[:, 0:3].permute(0, 3, 2, 1)
            trans = inverse(trans)
            pose = sample[:, 3:].permute(0, 3, 2, 1).reshape(1, 64, -1, 6)
            vertices = smplh(1, pose,
                             trans).cpu().numpy()

            source = source.unsqueeze(0)
            trans_gt = source[:, 0:3].permute(0, 3, 2, 1)
            trans_gt = inverse(trans_gt)
            pose_gt = source[:, 3:].permute(0, 3, 2, 1).reshape(1, 64, -1, 6)
            vertices_gt = smplh(1, pose_gt,
                                trans_gt).cpu().numpy()
            all_motions.append(vertices)
            all_motions_gt.append(vertices_gt)
    all_motions = np.concatenate(all_motions, axis=0)
    all_motions_gt = np.concatenate(all_motions_gt, axis=0)

    # if os.path.exists(out_path):
    #     shutil.rmtree(out_path)
    # os.makedirs(out_path)

    npy_path = os.path.join(out_path, 'test_results.npy')
    npy_path_gt = os.path.join(out_path, 'test_gt_results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path, all_motions)
    np.save(npy_path_gt, all_motions_gt)


if __name__ == "__main__":
    main()
