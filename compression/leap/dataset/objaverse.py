import os
import pickle
import json
import tqdm
import cv2
import random
import torch
import numpy as np
import random
import math
import json
import time
from torch.utils.data import Dataset
from torchvision.transforms import functional as func_transforms
from torchvision import transforms
import torchvision
import glob
from utils.geo_utils import quat2mat, quat2mat_transform, get_relative_pose, canonicalize_poses, transform_relative_pose

from PIL import Image, ImageFile, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Objaverse(Dataset):
    def __init__(self, config, split='train', root='/blob/data/3dmodels/views_release'):
        '''
        root: root of the dataset (use rendered subset).
        split: split of the dataset, in 'train', 'test'.
        Use all categories for training and testing.
        For each category, randomly split 10% instances for test.
        For training, randomly select N views for training.
        For testing, use images sampled by equal intervals as inputs, and use another group of sampled images for evaluation.
        '''
        self.config = config
        self.split = split
        self.root = root
        assert split in ['train', 'val', 'test']

        self.image_height = config.dataset.img_size
        self.image_width = config.dataset.img_size

        self.num_frames_per_seq = config.dataset.num_frame if self.split == 'train' else 10


        '''                                                      Up
                 | Kubric | OpenCV | Pytorch3d                    |
        X-axis   | Right  | Right  |   Left                       |________ Right
        Y-axis   | Up     | Down   |   Up                        /
        Z-axis   | Out    | In     |   In                    Out/
        Kubric:    right-handed frame, default is the same as openGL
        OpenCV:    right-handed frame
        pytorch3d: right-handed frame
        See: https://stackoverflow.com/questions/44375149/opencv-to-opengl-coordinate-system-transform
        '''
        self.kubric_to_cv2 = torch.tensor([[1.0, 0.0, 0.0, 0.0],   # inverse y-axis and z-axis, no translation
                                           [0.0, -1.0, 0.0, 0.0],
                                           [0.0, 0.0, -1.0, 0.0],
                                           [0.0, 0.0, 0.0, 1.0]])
        self.cv2_to_torch3d = torch.tensor([[-1.0, 0.0, 0.0, 0.0], # inverse x-axis and y-axis, no translation
                                           [0.0, -1.0, 0.0, 0.0],
                                           [0.0, 0.0, 1.0, 0.0],
                                           [0.0, 0.0, 0.0, 1.0]])
        self.torch3d_to_cv2 = torch.inverse(self.cv2_to_torch3d)
        self.cv2_to_kubric = torch.inverse(self.kubric_to_cv2)

        # set canonical camera extrinsics and pose
        self.canonical_extrinsics_cv2 = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                                      [0.0, 1.0, 0.0, 0.0],
                                                      [0.0, 0.0, 1.0, self.config.render.camera_z],
                                                      [0.0, 0.0, 0.0, 1.0]])
        self.canonical_pose_cv2 = torch.inverse(self.canonical_extrinsics_cv2)

        # intrinsics
        FoV_degree = 49.1
        FoV_radians = math.radians(FoV_degree)
        focal_length = 0.5 / math.tan(0.5 * FoV_radians)
        self.K = torch.tensor([[self.image_height * focal_length, 0., self.image_height / 2.0],
                                [0., self.image_height * focal_length, self.image_width / 2.0],
                                [0., 0., 1.]])

        # split data for traing and testing
        self.data_split = {}
        self._load_dataset()
        self.seq_names = []
        if self.split == 'train':
            self.seq_names += self.data_split['train']
        else:
            self.seq_names += self.data_split['test']
            if self.split == 'val':
                self.seq_names = self.seq_names[::config.eval_vis_freq]

    
    def _load_dataset(self):
        data_root = './data_objaverse'
        os.makedirs(data_root, exist_ok=True)
        data_split_file_path = os.path.join(data_root, 'split_info_test_1123.json')

        if not os.path.exists(data_split_file_path):
            self._split_data(data_split_file_path)

        with open(data_split_file_path, 'r') as f:
            data_split_file = json.load(f)
        #import pdb; pdb.set_trace()
        
        
        print('Objaverse dataset instances: train {}, test {}'.format(len(data_split_file['train']),
                                                                         len(data_split_file['test'])))
        
        self.data_split.update(data_split_file)

    
    def _split_data(self, data_split_file_path):
        all_instances = os.listdir(self.root)
        all_instances_valid = []
        for it in all_instances:
            it_path = os.path.join(self.root, it)
            if len(os.listdir(it_path)) == 24:
                all_instances_valid.append(it)
        test_instances = random.sample(all_instances_valid, 2000)
        train_instances = list(set(all_instances_valid) - set(test_instances))

        all_info = {'train': train_instances, 'test': test_instances}

        with open(data_split_file_path, 'w') as f:
            json.dump(all_info, f)
            
    
    def __len__(self):
        return len(self.seq_names)
    
    def _check_empty(self, idx):
        seq_name = self.seq_names[idx]
        seq_path = os.path.join(self.root, seq_name)

        # set seen flag, all testing objects are unseen objects
        if self.split == 'train':
            seen_flag = torch.tensor([1.0])
        else:
            seen_flag = torch.tensor([-1.0])

        # get image names
        rgb_files = glob.glob(os.path.join(seq_path, '*.png'))
        len_seq = len(rgb_files)        
        return rgb_files, len_seq, seen_flag,seq_name
    
    def __getitem__(self, idx):
        rgb_files, len_seq, seen_flag, seq_name = self._check_empty(idx)
        while len_seq <= 0:
            idx += 1
            print ("check", idx)
            rgb_files, len_seq, seen_flag, seq_name = self._check_empty(idx)
            

        if self.split == 'train':
            try:
                chosen_idx = random.sample(range(len_seq), self.num_frames_per_seq)
            except:
                print ("chosen_idx = random.sample(range(len_seq), len_seq)",range(len_seq), self.num_frames_per_seq)
                chosen_idx = random.sample(range(len_seq), len_seq)
        else:
            chosen_idx = list(range(self.num_frames_per_seq))
        chosen_rgb_files = [rgb_files[it] for it in chosen_idx] 
        chosen_npy_files = [it.replace('.png', '.npy') for it in chosen_rgb_files]
        
        # load image and mask    
        imgs, masks = [], []
        for rgb_file in chosen_rgb_files:
            img, mask = self._load_frame(rgb_file)
            img = torch.tensor(img)
            mask = torch.tensor(mask)
            imgs.append(img)
            masks.append(mask)
        imgs = torch.stack(imgs)     # [t,c,h,w]
        masks = torch.stack(masks)   # [t,1,h,w]

        # get camera poses
        chosen_npy_files = [it.replace('.png', '.npy') for it in chosen_rgb_files]
        cam_extrinsics = np.array([np.load(it) for it in chosen_npy_files])
        cam_extrinsics = torch.from_numpy(cam_extrinsics).float()
        cam_extrinsics = torch.cat([cam_extrinsics, 
                                    torch.tensor([0.,0.,0.,1.]).reshape(1,1,4).repeat(self.num_frames_per_seq,1,1)],
                                    dim=1)
        cam_poses = torch.inverse(cam_extrinsics)

        # get relative camera poses to the first frame (in kubric frame)
        cam_poses_rel = get_relative_pose(cam_poses[0], cam_poses)                                   # [t,4,4]
        cam_poses_rel[0] = torch.eye(4)

        # canonicalize camera poses (in opencv frame)
        cam_poses_cv2 = torch.matmul(cam_poses, self.kubric_to_cv2.unsqueeze(0))                     # [t,4,4]
        cam_extrinsics_cv2 = torch.inverse(cam_poses_cv2)                                            # [t,4,4]
        cam_poses_rel_cv2 = get_relative_pose(cam_poses_cv2[0], cam_poses_cv2)
        cam_poses_rel_cv2[0] = torch.eye(4)
        canonical_extrinsics_cv2 = self.canonical_extrinsics_cv2                                     # [4,4]
        canonical_pose_cv2 = self.canonical_pose_cv2
        cam_poses_cv2_canonicalized = canonicalize_poses(canonical_pose_cv2, cam_poses_rel_cv2)      # [t,4,4]
        cam_extrinsics_cv2_canonicalized = torch.inverse(cam_poses_cv2_canonicalized)

        sample = {
            'images': imgs.float(),                                                 # img observation
            'fg_probabilities': masks.float(),                                      # mask observation
            'K_cv2': self.K.unsqueeze(0).repeat(self.num_frames_per_seq,1,1),       # for initializing cameras
            'cam_extrinsics_cv2_canonicalized': cam_extrinsics_cv2_canonicalized,   # for initializing cameras, canonicalized setting
            'cam_extrinsics_cv2': cam_extrinsics_cv2,                               # for initializing cameras, uncanonicalized setting
            'cam_poses_cv2': cam_poses_cv2,                                         # uncanonicalized camera poses in cv2
            'cam_poses_cv2_canonicalized': cam_poses_cv2_canonicalized,             # canonicalized camera poses in cv2
            'cam_poses_rel_cv2': cam_poses_rel_cv2,                                 # relative camera poses in cv2
            'seq_name': seq_name,
        }

        if self.config.train.use_uncanonicalized_pose:
            sample['cam_extrinsics_cv2_canonicalized'] = cam_extrinsics_cv2
            sample['cam_poses_cv2'] = cam_poses_cv2

        if self.split != 'train':
            sample['seen_flag'] = seen_flag

        return sample


    def _load_frame(self, file_path, **kwargs):
        img_pil = Image.open(file_path)
        img_np = np.asarray(img_pil)
        # try:
        #     mask = Image.fromarray((img_np[:,:,3] > 0).astype(float))
        # except:
        # mask = Image.fromarray(np.logical_or(img_np[:,:,0]>254.,
        #                                           img_np[:,:,1]>254.,
        #                                           img_np[:,:,2]>254.).astype(float))
        mask = Image.fromarray((img_np[:,:,-1].astype(float) > 0).astype(float))

        # if not self.config.dataset.mask_images:
        #     # white background
        #     new_image = Image.new("RGBA", img_pil.size, "WHITE")
        #     new_image.paste(img_pil, (0, 0), img_pil)
        #     new_image = new_image.convert('RGB')
        #     rgb = new_image
        # else:
        #     # black background
        #     rgb = Image.fromarray(img_np[:,:,:3])

        rgb = Image.fromarray(img_np[:,:,:3])

        rgb = rgb.resize((self.image_height, self.image_width),Image.Resampling.LANCZOS)# Image.ANTIALIAS)
        mask = mask.resize((self.image_height, self.image_width), Image.NEAREST)

        rgb = np.asarray(rgb).transpose((2,0,1)) / 255.0                            # [3,H,W], in range [0,1]
        mask = np.asarray(mask)[:,:,np.newaxis].transpose((2,0,1))                  # [1,H,W], in range [0,1]

        # if self.config.dataset.mask_images:
        #     rgb *= mask
        
        if self.config.train.normalize_img:
            normalization = transforms.Compose([
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250])),
            ])
            rgb = torch.from_numpy(rgb)
            rgb = normalization(rgb).numpy()

        return rgb, mask
    
    def get_canonical_extrinsics_cv2(self, device='cpu'):
        return self.canonical_extrinsics_cv2.to(device)    # [4,4]

    def get_canonical_pose_cv2(self, device='cpu'):
        return self.canonical_pose_cv2.to(device)

    def pose_rel_cv2_to_torch3d(self, pose_cv2):
        device = pose_cv2.device
        return transform_relative_pose(pose_cv2, self.cv2_to_torch3d.to(device))

    def pose_cv2_to_torch3d(self, pose_cv2):
        device = pose_cv2.device
        return pose_cv2 @ self.cv2_to_torch3d.to(device).unsqueeze(0)




            


