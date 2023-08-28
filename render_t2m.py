import argparse
import logging

# import torch

# from data_loaders.p2m.tools import inverse
logger = logging.getLogger(__name__)

import numpy as np
import os
# from body_models.smplh import SMPLH

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


def render_cli(data, output, mode, downsample):
    init = True
    # import numpy as np
    print("Begining Rendering")
    from visualize.render.blender import render
    # data = np.load(path)[0]
    # if not os.path.exists(output):
    #     # os.system(r"touch {}".format(output))
    #     os.makedirs(output, mode=0o777)

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
    init = False


if __name__ == '__main__':
    #   data shape: ( 64, 6890, 3)

    mode = "sequence"
    output = "/mnt/disk_1/jinpeng/motion-diffusion-model/0822_GPT_render"
    baby = np.load('/mnt/disk_1/jinpeng/motion-diffusion-model/GPT_response_8022_smplh/total_batch.npy',
                   allow_pickle=True).item()
    namelist = np.load('/mnt/disk_1/jinpeng/motion-diffusion-model/GPT_response_8022_smplh/name.npy')
    for i in range(3):
        baby_i = baby['repeat_' + str(i)]
        for j, file in enumerate(baby_i):
            render_cli(
                data=file.squeeze(0),
                output=os.path.join(output, namelist[j].split('.')[0] + '_repeat_' + str(i)), downsample=False, mode=mode)