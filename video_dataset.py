import torch
import numpy as np
import cv2
from collections import OrderedDict
import os
import glob
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import scipy.io as scio

def sort_file_names(frame_name):
    frame_name = int(frame_name.split(os.sep)[-1].split('.')[0])
    return frame_name

def img_tensor2numpy(img):
    # mutual transformation between ndarray-like imgs and Tensor-like images
    # both intensity and rgb images are represented by 3-dim data
    if isinstance(img, np.ndarray):
        return torch.from_numpy(np.transpose(img, [2, 0, 1]))
    else:
        return np.transpose(img, [1, 2, 0]).numpy()
    
class VideoDatasetWithFlows(Dataset):
    def __init__(self, dataset_name, root, train=True, sequence_length=0, mode="last",
                 all_bboxes=None, patch_size=224, normalize=False, bboxes_extractions=False):
        super(VideoDatasetWithFlows, self).__init__()
        self.dataset_name = dataset_name
        self.root = os.path.join(root, dataset_name)
        self.train = train
        self.videos = OrderedDict()
        self.frame_addresses = []
        self.frame_video_idx = []
        self.sequence_length = sequence_length
        self.mode = mode
        self.all_bboxes = all_bboxes
        self.patch_size = patch_size
        self.normalize = normalize
        self.bboxes_extractions = bboxes_extractions
        self.__initialize()
        self.num_of_frames = len(self.frame_addresses)

    def __len__(self):
        return self.num_of_frames

    def __initialize(self):
        if self.train:
            self.root_frames = os.path.join(self.root, 'training', 'frames/*')
        else:
            self.root_frames = os.path.join(self.root, 'testing', 'frames/*')

        videos = glob.glob(self.root_frames)
        for i, video_path in enumerate(sorted(videos)):
            self.videos[i] = {}
            self.videos[i]['path'] = video_path
            self.videos[i]['frames'] = glob.glob(os.path.join(video_path, '*.jpg'))
            self.videos[i]['frames'].sort(key=sort_file_names)
            self.frame_addresses += self.videos[i]['frames']
            self.videos[i]['length'] = len(self.videos[i]['frames'])
            self.frame_video_idx += [i] * self.videos[i]['length']
        self.num_of_frames = len(self.frame_addresses)
        if not self.train:
            self.get_gt()

    def get_gt(self):
        if self.dataset_name == 'ped2':
            mat_name = self.dataset_name + '.mat'
            mat_path = os.path.join(self.root, mat_name)
            abnormal_mat = scio.loadmat(mat_path, squeeze_me=True)['gt']
            self.all_gt = []
            for i, (_, video) in enumerate(self.videos.items()):
                length = video['length']
                sub_video_gt = np.zeros((length,), dtype=np.int8)
                one_abnormal = abnormal_mat[i]
                if one_abnormal.ndim == 1:
                    one_abnormal = one_abnormal.reshape((one_abnormal.shape[0], -1))
                for j in range(one_abnormal.shape[1]):
                    start = one_abnormal[0, j] - 1
                    end = one_abnormal[1, j]
                    sub_video_gt[start: end] = 1
                sub_video_gt = torch.from_numpy(sub_video_gt)
                self.all_gt.append(sub_video_gt)
            self.all_gt = torch.cat(self.all_gt, 0)

        elif self.dataset_name == 'avenue':
            ids = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17',
                   '18', '19', '20', '21']
            mat_name = self.dataset_name + '.mat'
            mat_path = os.path.join(self.root, mat_name)
            abnormal_mat = scio.loadmat(mat_path)
            self.all_gt = []
            for id in ids:
                self.all_gt.append(abnormal_mat[id][0])
            self.all_gt = np.concatenate(self.all_gt, axis=0)
            self.all_gt = torch.from_numpy(self.all_gt)

        elif self.dataset_name == 'shanghaitech':
            frame_mask = os.path.join(self.root, 'testing/test_frame_mask/*')
            frame_mask = glob.glob(frame_mask)
            frame_mask.sort()
            self.all_gt = []
            for i, (_, video) in enumerate(self.videos.items()):
                gt = torch.from_numpy(np.load(frame_mask[i]))
                self.all_gt.append(gt)
            self.all_gt = torch.cat(self.all_gt, 0)

    def __getitem__(self, index):
        img_path = self.frame_addresses[index]
        flow_path = self.frame_addresses_flows[index]

        img = np.transpose(cv2.imread(img_path), [2, 0, 1])
        flows = np.transpose(np.load(flow_path), [2, 0, 1])

        if self.normalize:
            img = img / 255.0

        return img, flows