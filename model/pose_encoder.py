import torch.nn as nn
import torch
from human_body_prior.models.vposer_model import NormalDistDecoder, VPoser
import numpy as np

class L2Norm(nn.Module):
    def forward(self, x):
        return x / x.norm(dim=-1, keepdim=True)


class Object(object):
    pass


class PoseEncoder(nn.Module):

    def __init__(self, num_frame=8, num_neurons=512, num_neurons_mini=32, latentD=512, role=None):
        super(PoseEncoder, self).__init__()

        self.input_dim = num_frame * 126

        # use VPoser pose encoder architecture...
        vposer_params = Object()
        vposer_params.model_params = Object()
        vposer_params.model_params.num_neurons = num_neurons
        vposer_params.model_params.latentD = latentD
        vposer = VPoser(vposer_params)
        encoder_layers = list(vposer.encoder_net.children())
        # change first layers to have the right data input size
        encoder_layers[1] = nn.BatchNorm1d(self.input_dim)
        encoder_layers[2] = nn.Linear(self.input_dim, num_neurons)
        # remove last layer; the last layer.s depend on the task/role
        encoder_layers = encoder_layers[:-1]

        # output layers
        if role == "retrieval":
            encoder_layers += [
                nn.Linear(num_neurons, num_neurons_mini),
                # keep the bottleneck while adapting to the joint embedding size
                nn.ReLU(),
                nn.Linear(num_neurons_mini, latentD),
                L2Norm()]
        elif role == "generative":
            encoder_layers += [NormalDistDecoder(num_neurons, latentD)]
        else:
            raise NotImplementedError

        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, pose):
        pose_embedding = self.encoder(pose)
        return pose_embedding


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
#         super().__init__()
#         self.batch_first = batch_first
#
#         self.dropout = nn.Dropout(p=dropout)
#
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         # not used in the final model
#         if self.batch_first:
#             x = x + self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
#         else:
#             x = x + self.pe[:x.shape[0], :]
#         return self.dropout(x)
