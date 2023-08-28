import argparse
import torch
from utils.parser_util import generate_args
from utils.fixseed import fixseed
import os
from utils import dist_util
from data_loaders.p2m.dataset import HumanML3D
from text2pose.generative.evaluate_generative import load_model
from T2M.text_to_pose.tools import compute_text2poses_similarity, search_optimal_path
from text2pose.vocab import Vocabulary  # needed
from data_loaders.p2m.tools import axis_angle_to
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from torch.utils.data import DataLoader
from model.cfg_sampler import ClassifierFreeSampleModel
from tqdm import tqdm
from body_models.smplh import SMPLH
import numpy as np
CUDA_LAUNCH_BLOCKING=1

def main():
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print("Creating model and diffusion...")
    data = HumanML3D(datapath='dataset/p2m_humanml_opt.txt', split='test')
    test_loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=8)
    model, diffusion = create_model_and_diffusion(args, test_loader)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    dist_util.setup_dist(args.device)
    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)  # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    # loading model
    pose_model, _ = load_model(args.pose_model, dist_util.dev())
    model.eval()  # disable random masking
    pose_model.eval()
    namelist = []
    cond_all = {}
    false_name = []
    true_name = []
    # Text Data
    for file in os.listdir('/mnt/disk_1/jinpeng/motion-diffusion-model/GPT_response_0822'):
        namelist.append(file)
    for name in tqdm(namelist):
        try:
            Fs = np.load('/mnt/disk_1/jinpeng/motion-diffusion-model/GPT_response_0822/' + name).item().split('\n')
            pose_dists = []
            text_dists = []
            poses = []
            # sampling pose  # batch sampling
            with torch.no_grad():
                for f in Fs:
                    f = f.split(':')[1]
                    pose = pose_model.sample_str_nposes(f, n=args.n_generate)['pose_body'][0].view(args.n_generate, -1)
                    text_dist = pose_model.text_encoder(pose_model.tokenizer(f).to(dist_util.dev()).view(1, -1),
                                                        torch.tensor([len(pose_model.tokenizer(f).to(dist_util.dev()))]))
                    pose_dist = pose_model.pose_encoder(pose)
                    poses.append(pose)
                    pose_dists.append(pose_dist)
                    text_dists.append(text_dist)

            # [(n_generate x 1), (n_generate x 1), ...]
            text2poses_similarity = compute_text2poses_similarity(text_dists, pose_dists)

            path = search_optimal_path(pose_dists, text2poses_similarity, dist_util.dev(), args.n_generate, args.op)

            cond = []
            for i, p in enumerate(path):
                pose_i = poses[i][p, :].reshape(-1, 3)[1:22][None]
                cond.append(pose_i)
            cond = torch.concatenate(cond, axis=0)
            assert cond.shape[0] == 8
            cond = axis_angle_to("rot6d", cond)
            cond_all[name] = cond
        except:
            false_name.append(name)
            pass
    np.save('/mnt/disk_1/jinpeng/motion-diffusion-model/GPT_response_0822_false_name.npy', false_name)
    # print('==============Stage 1 Finished=============')
    # print('==============Stage 2 Begin=============')
    cond_all = np.load(
    '/mnt/disk_1/jinpeng/motion-diffusion-model/save/0813_cross/samples_0813_cross_000300000_seed10/t2m_cond_all.npy',
    allow_pickle=True).item()
    all_motions = {}
    for i in range(args.num_repetitions):
        repeat_time = 'repeat_' + str(i)
        all_motions[repeat_time] = []
    # train_loader = DataLoader(list(cond_all.values()), batch_size=args.batch_size, shuffle=False, num_workers=1)
    iterator = iter(list(cond_all.values()))
    np.save(os.path.join(out_path, 't2m_cond_all.npy'), cond_all)
    pose_gt = []
    for i, cond_i in enumerate(tqdm(iterator)):
        cond_i = cond_i.unsqueeze(0)
        model_kwargs = {'y': {'pose_feature': cond_i, 'lengths': 64 * torch.ones(1).type(torch.IntTensor),
                              'mask': torch.ones(64, dtype=bool)}}
        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(cond_i.shape[0], device=dist_util.dev()) * args.guidance_param
        model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in
                             model_kwargs['y'].items()}
        pose_gt.append(np.split(model_kwargs['y']['pose_feature'].cpu().numpy(), cond_i.shape[0]))
        for rep_i in range(args.num_repetitions):
            print(f'### Sampling [repetitions #{rep_i}]')
            sample_fn = diffusion.p_sample_loop
            final = sample_fn(
                model,
                (cond_i.shape[0], model.njoints, model.nfeats, 64),  # BUG FIX
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            all_motions['repeat_' + str(rep_i)].append(np.split(final.cpu().numpy(), final.shape[0]))
    for i in range(args.num_repetitions):
        all_motions['repeat_' + str(i)] = np.concatenate(all_motions['repeat_' + str(i)], axis=0)
    pose_gt = np.concatenate(pose_gt, axis=0)

    npy_path = os.path.join(out_path, 't2m_results.npy')
    npy_pose_path = os.path.join(out_path, 't2m_pose_results.npy')
    print(f"saving results file to [{npy_path}]")
    results = {'sample': all_motions}
    np.save(npy_path, results)
    np.save(npy_pose_path, pose_gt)


if __name__ == "__main__":
    main()
