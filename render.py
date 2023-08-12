import argparse
import logging
from data_loaders.p2m.dataset import HumanML3D
from torch.utils.data import DataLoader
logger = logging.getLogger(__name__)
import os
import sys
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-name', default='t2m', help='dataset name')
    parser.add_argument('--batch-size', default=32, help='batch_size')
    parser.add_argument('--min-motion-len', default=24, type=int, help='the minimum of motion length')
    parser.add_argument('--mode', default="video", help="mode of rendering")
    parser.add_argument('--npy',
                        default="/mnt/disk_1/jinpeng/motion-diffusion-model/save/p2m_humanml_trans_enc_512_126_temos/samples_p2m_humanml_trans_enc_512_126_temos_000050000_seed10_the_person_walked_forward_and_is_picking_up_his_toolbox/results.npy",
                        type=str, help="mode of rendering")
    args = parser.parse_args()
    return args


def render_cli(path, output, mode, downsample):
    init = True
    import numpy as np
    from visualize.render.blender import render
    data = np.load(path)[0]
    if not os.path.exists(output):
        # os.system(r"touch {}".format(output))
        os.makedirs(output, mode=0o777)

    frames_folder = render(data, frames_folder=output,
                           denoising=True,
                           oldrender=True,
                           canonicalize=True,
                           exact_frame=0.5,
                           num=8, mode=mode,
                           faces_path='/mnt/disk_1/jinpeng/motion-diffusion-model/body_models/smplh.faces',
                           downsample=True,
                           always_on_floor=False,
                           init=init,
                           gt=False)
    print(frames_folder)
    init = False


if __name__ == '__main__':
    a = np.load('/mnt/disk_1/jinpeng/motion-diffusion-model/dataset/debug/data_dict.npy',
                allow_pickle=True).item()
    mode = "video"
    output = './compare/crab_walk'
    render_cli(
        path='/mnt/disk_1/jinpeng/motion-diffusion-model/wenxun/crab_walk_smplh.npy',
        output=output, downsample=False, mode=mode)
