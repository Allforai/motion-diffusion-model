from torch.utils import data
import numpy as np


class Pose2Motion(data.Dataset):
    def __init__(self, datapath, replication_times):
        motion_data = np.load(datapath, allow_pickle=True).item()
        self.gt = motion_data['gt']
        self.pose = motion_data['pose']
        self.name = motion_data['name']
        self.sample = motion_data['sample']['repeat_'+str(replication_times)]

    def __getitem__(self, item):
        return self.gt[item], self.pose[item], self.name[item], self.sample[item]

    def __len__(self):
        return len(self.gt)
