import argparse
import logging
import pickle
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

    mode = "video"
    output = "/mnt/disk_1/jinpeng/motion-diffusion-model/save/0813_cross/samples_0813_cross_000300000_seed10/smplh_cfg2.5"
    with open('/mnt/disk_1/jinpeng/motion-diffusion-model/save/0813_cross/samples_0813_cross_000300000_seed10/smplh.pkl', 'rb') as f:
        baby = pickle.load(f)
    namelist = np.load('/mnt/disk_1/jinpeng/motion-diffusion-model/save/0813_cross/samples_0813_cross_000300000_seed10/results.npy', allow_pickle=True).item()['name']
    for i in range(1):
        baby_i = baby['repeat_' + str(i)]
        for j, file in enumerate(baby_i):
            if j < 50:
                render_cli(
                    data=file.squeeze(0),
                    output=os.path.join(output, namelist[j].split('.')[0] + '_repeat_' + str(i)), downsample=False, mode=mode)
                os.system(r"cp {} {}".format(os.path.join('dataset/HumanML3D', 'texts', namelist[j] + '.txt'),
                                             os.path.join(output, namelist[j].split('.')[0] + '_repeat_' + str(i) + '/')))
