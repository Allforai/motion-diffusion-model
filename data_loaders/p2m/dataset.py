from torch.utils import data
from os.path import join as pjoin
from data_loaders.humanml.utils.get_opt import get_opt
import codecs as cs
from tqdm import tqdm
import numpy as np
import random
from .tools import axis_angle_to, matrix_to
import torch
from .geometry import axis_angle_to_matrix, matrix_to_axis_angle
from einops import rearrange

#
# class Pose2MotionDataset(data.dataset):
#     def __init__(self, opt, split_file):
#         self.opt = opt
#         self.motion_length = opt.max_motion_length
#
#         data_dict = {}
#         id_list = []
#         with cs.open(split_file, 'r') as f:
#             for line in f.readlines():
#                 id_list.append(line.strip())
#
#         new_name_list = []
#         length_list = []
#         for name in tqdm(id_list):
#             try:
#                 motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
#                 if len(motion) < self.motion_length:
#                     continue
#                 motion = smpl_data_to_matrix_and_trans(motion)
#
#                 data_dict[name] = {'motion': motion,
#                                    'length': len(motion['features'])}
#                 new_name_list.append(name)
#                 length_list.append(len(motion))
#             except:
#                 pass
#
#         name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
#
#         self.length_arr = np.array(length_list)
#         self.data_dict = data_dict
#         self.name_list = name_list
#
#     def __len__(self):
#         return len(self.data_dict)
#
#     def __getitem__(self, item):
#         data = self.data_dict[self.name_list[item]]
#         motion, m_length = data['motion'], data['length']
#
#         idx = random.randint(0, len(motion) - self.motion_length)
#         features = motion['features'][idx:idx + self.min_motion_len]
#         pose_feature = motion['pose_feature'][idx:idx + self.min_motion_len]
#         trans_feature = motion['trans_feature'][idx:idx + self.min_motion_len]
#
#         return {'features': features, 'pose_feature': pose_feature, 'trans_feature': trans_feature, 'length': m_length}


class HumanML3D(data.Dataset):
    def __init__(self, datapath='./dataset/humanml_opt.txt', split="train"):

        self.dataset_name = 'p2m'
        self.dataname = 'p2m'

        abs_base_path = f'.'
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = None
        opt = get_opt(dataset_opt_path, device)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = './dataset'
        self.opt = opt
        print('Loading dataset %s ...' % opt.dataset_name)
        self.motion_length = opt.max_motion_length

        self.split_file = pjoin(opt.data_root, f'{split}.txt')
        # data_dict = {}
        # id_list = []
        # with cs.open(self.split_file, 'r') as f:
        #     for line in f.readlines():
        #         id_list.append(line.strip())
        #
        # new_name_list = []
        # length_list = []
        # for name in tqdm(id_list):
        #     try:
        #         motion = np.load(pjoin(opt.motion_dir, name + '.npy'), allow_pickle=True).item()
        #         if len(motion['trans']) < self.motion_length:
        #             continue
        #         motion = smpl_data_to_matrix_and_trans(motion)
        #
        #         data_dict[name] = {'motion': motion,
        #                            'length': len(motion['features'])}
        #         new_name_list.append(name)
        #         length_list.append(len(motion['features']))
        #     except:
        #         pass
        #
        # name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        self.min_motion_len = 64  # data length
        self.data_dict = np.load('/mnt/disk_1/jinpeng/motion-diffusion-model/dataset/debug/data_dict.npy', allow_pickle=True).item()
        self.name_list = np.load('/mnt/disk_1/jinpeng/motion-diffusion-model/dataset/debug/name_list.npy', allow_pickle=True)
        self.length_list = np.load('/mnt/disk_1/jinpeng/motion-diffusion-model/dataset/debug/length_list.npy', allow_pickle=True)
        # self.length_arr = np.array(length_list)
        # self.data_dict = data_dict
        # self.name_list = name_list
        # np.save('/mnt/disk_1/jinpeng/motion-diffusion-model/dataset/debug/data_dict.npy', data_dict)
        # np.save('/mnt/disk_1/jinpeng/motion-diffusion-model/dataset/debug/name_list.npy', name_list)
        # np.save('/mnt/disk_1/jinpeng/motion-diffusion-model/dataset/debug/length_list.npy', np.array(length_list))

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        motion, m_length = data['motion'], data['length']

        idx = random.randint(0, len(motion['features']) - self.motion_length)
        features = motion['features'][idx:idx + self.motion_length]
        features = features.T.unsqueeze(1)
        pose_feature = motion['pose_feature'][np.arange(idx, idx + self.motion_length, 8)]
        padding_feature = torch.zeros(8, 3)
        pose_feature = torch.cat((padding_feature, pose_feature), dim=-1)
        # trans_feature = motion['trans_feature'][idx:idx + self.motion_len]
        # return {'features': features, 'pose_feature': pose_feature, 'trans_feature': trans_feature, 'length': m_length}
        return features, {'y': {'pose_feature': pose_feature, 'mask': torch.ones(8, dtype=bool)}}

    def __len__(self):
        return len(self.data_dict)


def smpl_data_to_matrix_and_trans(data):
    trans = torch.from_numpy(data['trans'])
    root_orient = torch.from_numpy(data['root_orient'])
    pose_body = torch.from_numpy(data["pose_body"])
    nframes = len(trans)

    axis_angle_poses = torch.cat((root_orient.reshape(nframes, -1, 3), pose_body.reshape(nframes, -1, 3)), dim = 1)

    matrix_poses = axis_angle_to("matrix", axis_angle_poses)


    # extract the root gravity axis
    # for smpl it is the last coordinate
    root_y = trans[..., 2]
    trajectory = trans[..., [0, 1]]

    # Comoute the difference of trajectory (for X and Y axis)
    vel_trajectory = torch.diff(trajectory, dim=-2)
    # 0 for the first one => keep the dimentionality
    vel_trajectory = torch.cat((0 * vel_trajectory[..., [0], :], vel_trajectory), dim=-2)

    # first normalize the data
    global_orient = matrix_poses[..., 0, :, :]
    # remove the rotation
    rot2d = matrix_to_axis_angle(global_orient[..., 0, :, :])
    # Remove the fist rotation along the vertical axis
    # construct this by extract only the vertical component of the rotation
    rot2d[..., :2] = 0

    # add a bit more rotation
    rot2d[..., 2] += torch.pi / 2

    rot2d = axis_angle_to_matrix(rot2d)

    # turn with the same amount all the rotations
    global_orient = torch.einsum("...kj,...kl->...jl", rot2d, global_orient)

    matrix_poses = torch.cat((global_orient[..., None, :, :],
                              matrix_poses[..., 1:, :, :]), dim=-3)

    # Turn the trajectory as well
    vel_trajectory = torch.einsum("...kj,...lk->...lj", rot2d[..., :2, :2], vel_trajectory)

    poses = matrix_to('rot6d', matrix_poses)
    trans_feature = torch.cat((root_y[..., None],
                               vel_trajectory),
                              dim=-1)
    pose_feature = rearrange(poses, "... joints rot -> ... (joints rot)")
    padding_feature = torch.zeros(64, 3)
    pose_feature = torch.cat((pose_feature, padding_feature), dim=-1)
    features = torch.cat((root_y[..., None],
                          vel_trajectory,
                          rearrange(poses, "... joints rot -> ... (joints rot)")),
                         dim=-1)
    return {'features': features, 'pose_feature': pose_feature, 'trans_feature': trans_feature}