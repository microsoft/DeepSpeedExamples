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
from utils.geo_utils import quat2mat, quat2mat_transform, get_relative_pose, canonicalize_poses, transform_relative_pose

from PIL import Image, ImageFile, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DTU(Dataset):
    def __init__(self, config, split='train', root='/vision/vision_data/DTU/rs_dtu_4/DTU'):
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

        self.image_height = config.dataset.img_size_height
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

        # transformations from PixelNeRF
        self._coord_trans_world = torch.tensor(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )
        self._coord_trans_cam = torch.tensor(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
        )

        # split data for traing and testing
        if self.split == 'train':
            file = os.path.join(root, 'new_train.lst')
        else:
            file = os.path.join(root, 'new_val.lst')
        with open(file, 'r') as f:
            self.seq_names = [it.strip() for it in f.readlines()]

        print('DTU uses {} scenes for {}'.format(len(self.seq_names), self.split))

        if self.split == 'train':
            self.seq_names = self.seq_names * 100   # make each epoch larger
            
    
    def __len__(self):
        return len(self.seq_names)
    

    def __getitem__(self, idx):
        seq_name = self.seq_names[idx]
        seq_path = os.path.join(self.root, seq_name)
        meta = np.load(os.path.join(seq_path, 'cameras.npz'))

        # set seen flag, all testing objects are unseen objects
        if self.split == 'train':
            seen_flag = torch.tensor([1.0])
        else:
            seen_flag = torch.tensor([-1.0])

        # get images
        imgs_path = os.path.join(seq_path, 'image')
        rgb_files = ['{:06d}.png'.format(idx) for idx in range(49)]
        if self.split == 'train':
            chosen_index = random.sample(range(len(rgb_files)), self.num_frames_per_seq)
        else:
            #chosen_index = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
            chosen_index = [25, 5, 15, 35, 45, 20, 0, 10, 30, 40]
            #chosen_index = [0, 10, 20, 30, 40, 5, 15, 25, 35, 45] #[0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
        chosen_rgb_files = [rgb_files[it] for it in chosen_index]
        imgs, masks = [], []
        for rgb_file in chosen_rgb_files:
            img, mask = self._load_frame(os.path.join(imgs_path, rgb_file))
            img = torch.tensor(img)
            mask = torch.tensor(mask)
            imgs.append(img)
            masks.append(mask)
        imgs = torch.stack(imgs)    # [t,c,h,w]
        masks = torch.stack(masks)  # [t,1,h,w]

        # get camera poses and intrinsics
        # poses are defined in OpenGL coordinate (dataloader requires opencv coordinate)
        K, cam_poses = self._load_camera(meta, chosen_index) # intrinsics are defined with resolution 300x400
        K = self._process_intrinsics(K)[:,:3,:3] # to the target resolution
        cam_poses = self._process_poses(cam_poses)  # make it to OpenGL 

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
            'K_cv2': K,                                                             # for initializing cameras
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


    def _load_frame(self, file_path):
        img_pil = Image.open(file_path)
        img_pil = img_pil.resize((self.image_width, self.image_height), Image.ANTIALIAS)
        rgb = np.asarray(img_pil)[:,:,:3].transpose((2,0,1)) / 255.0    # [c,h,w]
        mask = np.logical_or(rgb[0]>0.05, rgb[1]>0.05, rgb[2]>0.05).astype(float) # [h,w]
        mask = mask[np.newaxis,:,:]
        if mask.mean() > 0.999:
            mask[0,0,0] = 0.0
        
        if self.config.train.normalize_img:
            normalization = transforms.Compose([
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250])),
            ])
            rgb = torch.from_numpy(rgb)
            rgb = normalization(rgb).numpy()
        return rgb, mask
    
    def _load_camera(self, meta, idxs):
        # the function is from sparf
        intrinsics = []
        poses_c2w = []
        for idx in idxs:
            P = meta[f"world_mat_{idx}"] # Projection matrix 
            P = P[:3]  # (3x4) projection matrix
            K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
            K /= K[2, 2]  # 3x3 intrinsics matrix

            pose_c2w_ = np.eye(4, dtype=np.float32) # camera to world
            pose_c2w_[:3, :3] = R.transpose()
            pose_c2w_[:3, 3] = (t[:3] / t[3])[:, 0]

            intrinsics_ = np.eye(4)
            intrinsics_[:3, :3] = K
            scale_mat = meta.get(f"scale_mat_{idx}")
            if scale_mat is not None:
                norm_trans = scale_mat[:3, 3:]
                pose_c2w_[:3, 3:] -= norm_trans
                # 1/300, scale the world
                norm_scale = np.diagonal(scale_mat[:3, :3])[..., None]
                # here it is 3 values, but equal to each other!
                assert norm_scale.mean() == 300.
                # I directly use this scaling factor to scale the depth
                # it is hardcoded in self.scaling_factor 
                # If this assertion doesn't hold, them self.scaling_factor should be equal to 1./norm_scale
                # Importantly, the norm_scale must be equal for all directions, otherwise that wouldn't scale
                # the depth map properly. 
            pose_c2w_[:3, 3:] *= (1 / 300.)
            poses_c2w.append(pose_c2w_)
            intrinsics.append(intrinsics_)
        poses_c2w = np.stack(poses_c2w, axis=0).astype(np.float32)
        intrinsics = np.stack(intrinsics, axis=0).astype(np.float32)
        return intrinsics, poses_c2w
    
    def _process_intrinsics(self, K):
        '''
        K in shape [N,3,3]
        '''
        target_resolution_height = self.image_height
        raw_resolution_height = 300
        down_size = raw_resolution_height / target_resolution_height
        K[:,:2] = K[:,:2] / down_size
        return K
    
    def _process_poses(self, poses):
        '''
        poses in shape [N,4,4]
        '''
        # center and radius are pre-computed
        # as the cameras are actually on a sphere
        center = torch.tensor([0.27199481,  0.08126513, -0.12928901]).float()
        radius = 2.028138

        # then make the poses from opencv to opengl, also apply the world transformation in pixelnerf
        poses = torch.tensor(poses)
        poses[:,:3,3] -= center.reshape(1,3)    # make the sphere zero-centered
        poses = self._coord_trans_world.unsqueeze(0) @ poses @ self._coord_trans_cam.unsqueeze(0)

        # make the sphere axis-align
        poses = rotate_camera_poses(poses, np.radians(22), np.radians(-7), np.radians(0))
        return poses
    
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



def rotation_matrices(theta, phi, psi):
    # Rotation around x-axis
    Rx = torch.tensor([[1, 0, 0],
                   [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta), np.cos(theta)]])
    
    # Rotation around y-axis
    Ry = torch.tensor([[np.cos(phi), 0, np.sin(phi)],
                   [0, 1, 0],
                   [-np.sin(phi), 0, np.cos(phi)]])
    
    # Rotation around z-axis
    Rz = torch.tensor([[np.cos(psi), -np.sin(psi), 0],
                   [np.sin(psi), np.cos(psi), 0],
                   [0, 0, 1]])
    
    return Rx, Ry, Rz

def rotate_camera_poses(poses, theta, phi, psi):
    Rx, Ry, Rz = rotation_matrices(theta, phi, psi)
    R_combined = (Rz @ (Ry @ Rx)).float()  # Combined rotation matrix
    
    rotated_poses = poses.clone().float() #np.copy(poses)
    for i in range(poses.shape[0]):
        rotated_poses[i, :3, :3] = R_combined @ poses[i, :3, :3]
        rotated_poses[i, :3, 3] = R_combined @ poses[i, :3, 3]
    return rotated_poses
            


