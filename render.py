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

    # Testing set
    # pr = np.load('/mnt/disk_1/jinpeng/motion-diffusion-model/save/0813_cross/samples_0813_cross_000300000_seed10'
    #              '/test_results.npy',
    #              allow_pickle=True)
    # gt = np.load('/mnt/disk_1/jinpeng/motion-diffusion-model/save/0813_cross/samples_0813_cross_000300000_seed10'
    #              '/test_gt_results.npy', allow_pickle=True)
    # namelist = np.load('/mnt/disk_1/jinpeng/motion-diffusion-model/dataset/namelist.npy', allow_pickle=True)
    #
    # # Training Code test set
    # # pr = np.load('/mnt/disk_1/jinpeng/motion-diffusion-model/save/p2m_temos_0812_test_loss'
    # #              '/samples_p2m_temos_0812_test_loss_000100000_seed10/train_codetest.npy',
    # #              allow_pickle=True)
    # # print("Rendering")
    # # gt = np.load('/mnt/disk_1/jinpeng/motion-diffusion-model/save/p2m_temos_0812_test_loss'
    # #              '/samples_p2m_temos_0812_test_loss_000100000_seed10/train_codetest_gt.npy', allow_pickle=True)
    # # namelist = np.load('/mnt/disk_1/jinpeng/motion-diffusion-model/dataset/namelist_train_codetest_0813.npy', allow_pickle=True)
    # mode = 'sequence'
    # output_gt = "/mnt/disk_1/jinpeng/motion-diffusion-model/0814_cross_test_gt"
    # for i, file in enumerate(range(len(namelist))):
    #     render_cli(
    #         data=gt[file],
    #         output=os.path.join(output_gt, namelist[file]), downsample=False, mode=mode)
    # data shape: ( 64, 6890, 3)
    mode = 'video'
    path = '/mnt/disk_1/jinpeng/motion-diffusion-model/0831_unseen'
    data_path = []
    for file in os.listdir(path):
        for name_1 in os.listdir(os.path.join(path, file)):
            for name_2 in os.listdir(os.path.join(path, file, name_1)):
                if 'prompt' not in name_2 and 'npy' in name_2 and 'ballet' in name_2:
                    data_path.append(os.path.join(path, file, name_1, name_2))
    for npy_path in data_path:
        file = np.load(npy_path)[0]
        render_cli(
            data=file,
            output=npy_path.replace(npy_path.split('/')[-1], npy_path.split('/')[-1].split('.')[0]), downsample=False, mode=mode)
