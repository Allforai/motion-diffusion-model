import contextlib
import torch.nn as nn
import torch
from data_loaders.p2m.tools import to_matrix


def slice_or_none(data, cslice):
    if data is None:
        return data
    else:
        return data[cslice]


class SMPLH(nn.Module):
    def __init__(self, path: str,
                 input_pose_rep: str = "matrix",
                 batch_size: int = 512,
                 gender="neutral",
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.input_pose_rep = input_pose_rep
        self.training = False
        from smplx.body_models import SMPLHLayer

        # Remove annoying print
        with contextlib.redirect_stdout(None):
            self.smplh = SMPLHLayer(path, ext="npz", gender=gender).eval()

        self.faces = self.smplh.faces
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, batch_size, poses, trans):
        poses = poses
        trans = trans

        # Convert any rotations to matrix
        matrix_poses = to_matrix('rot6d', poses)

        from functools import reduce
        import operator
        save_shape_bs_len = matrix_poses.shape[:-3]
        nposes = reduce(operator.mul, save_shape_bs_len, 1)

        # Reshaping
        matrix_poses = matrix_poses.reshape((nposes, *matrix_poses.shape[-3:]))
        global_orient = matrix_poses[:, 0]

        trans_all = trans.reshape((nposes, *trans.shape[-1:]))

        body_pose = matrix_poses[:, 1:22]

        # still axis angle
        left_hand_pose = self.smplh.left_hand_mean.reshape(15, 3)
        left_hand_pose = to_matrix("axisangle", left_hand_pose)
        left_hand_pose = left_hand_pose[None].repeat((nposes, 1, 1, 1))

        right_hand_pose = self.smplh.right_hand_mean.reshape(15, 3)
        right_hand_pose = to_matrix("axisangle", right_hand_pose)
        right_hand_pose = right_hand_pose[None].repeat((nposes, 1, 1, 1))

        n = len(body_pose)
        outputs = []
        for chunk in range(int((n - 1) / batch_size) + 1):
            chunk_slice = slice(chunk * batch_size, (chunk + 1) * batch_size)
            smpl_output = self.smplh(global_orient=slice_or_none(global_orient, chunk_slice),
                                     body_pose=slice_or_none(body_pose, chunk_slice),
                                     left_hand_pose=slice_or_none(left_hand_pose, chunk_slice),
                                     right_hand_pose=slice_or_none(right_hand_pose, chunk_slice),
                                     transl=slice_or_none(trans_all, chunk_slice))
            output_chunk = smpl_output.vertices
            outputs.append(output_chunk)
        outputs = torch.cat(outputs)
        outputs = outputs.reshape((*save_shape_bs_len, *outputs.shape[1:]))

        return outputs
