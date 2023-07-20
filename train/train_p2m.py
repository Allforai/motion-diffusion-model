from utils.parser_util import train_args
from utils.fixseed import fixseed
import os
import json
from utils import dist_util
from data_loaders.p2m.dataset import HumanML3D
from torch.utils.data import DataLoader
from utils.model_util import create_model_and_diffusion
from train.training_loop import TrainLoop
import pydevd_pycharm
# pydevd_pycharm.settrace('10.8.32.196', port=17778, stdoutToServer=True, stderrToServer=True)
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation

def main():
    args = train_args()
    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    training_data = HumanML3D(datapath='dataset/p2m_humanml_opt.txt')
    train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, num_workers=8)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, train_loader)
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, train_loader).run_loop()
    train_platform.close()


if __name__=="__main__":
    main()