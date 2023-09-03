# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
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

torch.set_default_dtype(torch.float32)


def main():
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 64
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        if args.text_prompt != '':
            out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
        elif args.input_text != '':
            out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples

    print("Loading Dataset")
    data = HumanML3D(pose_length=args.pose_length, datapath='dataset/p2m_humanml_opt.txt', split='test')
    train_loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=8)

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, train_loader)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)  # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking
    all_motions = {}
    for i in range(args.num_repetitions):
        repeat_time = 'repeat_' + str(i)
        all_motions[repeat_time] = []
    all_motions_gt = []
    pose_gt = []
    namelist = []
    for i, (source, model_kwargs) in enumerate(tqdm(train_loader)):
        # add CFG scale to batch
        if args.guidance_param != 1:
            model_kwargs['y']['scale'] = torch.ones(source.shape[0], device=dist_util.dev()) * args.guidance_param
        model_kwargs['y'] = {key: val.to(dist_util.dev()) if torch.is_tensor(val) else val for key, val in
                             model_kwargs['y'].items()}
        source = source.to(dist_util.dev())
        all_motions_gt.append(np.split(source.cpu().numpy(), source.shape[0]))
        pose_gt.append(np.split(model_kwargs['y']['pose_feature'].cpu().numpy(), source.shape[0]))
        namelist.append(model_kwargs['y']['key_id'])
        for rep_i in range(args.num_repetitions):
            print(f'### Sampling [repetitions #{rep_i}]')
            sample_fn = diffusion.p_sample_loop
            final = sample_fn(
                model,
                (source.shape[0], model.njoints, model.nfeats, max_frames),  # BUG FIX
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            all_motions['repeat_'+str(rep_i)].append(np.split(final.cpu().numpy(), final.shape[0]))
    for i in range(args.num_repetitions):
        all_motions['repeat_' + str(i)] = np.concatenate(all_motions['repeat_' + str(i)], axis=0)
    all_motions_gt = np.concatenate(all_motions_gt, axis=0)
    pose_gt = np.concatenate(pose_gt, axis=0)
    namelist = np.concatenate(namelist, axis=0)

    npy_path = os.path.join(out_path, 'results_cfg' + str(args.guidance_param) + '.npy')
    print(f"saving results file to [{npy_path}]")
    results = {'gt': all_motions_gt, 'pose': pose_gt, 'name': namelist, 'sample': all_motions}
    np.save(npy_path, results)


if __name__ == "__main__":
    main()
