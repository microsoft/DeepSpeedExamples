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


ShapeNet_ids = {
    'table': '04379243', 'car': '02958343', 'chair': '03001627', 'airplane': '02691156', 'sofa': '04256520',
    'rifle': '04090263', 'lamp': '03636649', 'watercraft': '04530566', 'bench': '02828884', 'loudspeaker': '03691459',
    'cabinet': '02933112', 'display': '03211117', 'telephone': '04401088', 'bus': '02924116', 'bathtub': '02808440',
    'guitar': '03467517', 'faucet': '03325088', 'clock': '03046257', 'flowerport': '03991062', 'jar': '03593526',
    'bottle': '02876657', 'bookshelf': '02871439', 'laptop': '03642806', 'knife': '03624134', 'train': '04468005',
    'trash bin': '02747177', 'motorbike': '03790512', 'pistol': '03948459', 'file cabinet': '03337140',
    'bed': '02818832', 'piano': '03928116', 'stove': '04330267', 'mug': '03797390', 'bowl': '02880940',
    'washer': '04554684', 'printer': '04004475', 'helmet': '03513137', 'microwaves': '03761084',
    'skateboard': '04225987', 'tower': '04460130', 'camera': '02942699', 'basket': '02801938', 'can': '02946921',
    'pillow': '03938244', 'mailbox': '03710193', 'dishwasher': '03207941', 'rocket': '04099429', 'bag': '02773838',
    'birdhouse': '02843684', 'earphone': '03261776', 'microphone': '03759954', 'remote': '04074963',
    'keyboard': '03085013', 'bicycle': '02834778', 'cap': '02954340'
}

ShapeNet_general_train = ['airplane', 'bench', 'cabinet', 'car', 'chair', 'display', 'lamp', 'loudspeaker', 'rifle',
    'sofa', 'table', 'telephone', 'watercraft']

ShapeNet_general_test_unseen = ['bus', 'guitar', 'clock', 'bottle', 'train', 'mug', 'washer', 'skateboard', 'dishwasher', 'pistol']


class Kubric(Dataset):
    def __init__(self, config, split='train', root='/vision/vision_data/kubric_synthetic/shapenet'):
        '''
        root: root of the dataset.
        split: split of the dataset, in 'train', 'test'.
        Data for each category include 5500 randomly sampled sequences of instances (can have duplication), each seq contains 10 frames
        Data is splitted into train, 
                              test_seen (seen instances during training, but from different camera viewpoints)
                              test_unseen (unseen instances)
        '''
        self.config = config
        self.split = split

        self.image_height = config.dataset.img_size
        self.image_width = config.dataset.img_size

        self.is_general = False
        self.category = config.dataset.category
        assert self.category in ['car', 'chair', 'general', 'general_unseen_category']
        self.category_name = self.category
        if self.category == 'general' or self.category == 'general_unseen_category':
            if self.category == 'general':
                self.category = ShapeNet_general_train
            else:
                self.category = ShapeNet_general_test_unseen
            self.category_id = [ShapeNet_ids[it] for it in self.category]
            self.root = '/vision/vision_data/kubric_synthetic/shapenet_general'
            self.is_general = True
        else:
            self.category_id = ShapeNet_ids[self.category]
            self.root = '/vision/vision_data/kubric_synthetic/shapenet' # category-specific dataset

        self.num_frames_per_seq = 10 if ((config.test.compute_metric and split!='train') or config.dataset.train_all_frame) else config.dataset.num_frame


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

        self.data_split = {}
        if not self.is_general:
            self.__load_dataset_category__()
        else:
            self.__load_dataset_general__()
        self.seq_names = []
        if self.split == 'train':
            self.seq_names += self.data_split['train']
        else:
            if self.category_name != 'general_unseen_category':
                self.seq_names += self.data_split['test_seen']
                self.seq_names += self.data_split['test_unseen']
            else:
                self.seq_names += self.data_split['test_unseen']
            if self.split == 'val':
                self.seq_names = self.seq_names[::config.eval_vis_freq]

    def __load_dataset_category__(self):

        data_root = './data_kubric'  # root for saving data split
        data_split_path = os.path.join(data_root, self.category)
        os.makedirs(data_split_path, exist_ok=True)
        data_split_file_path = os.path.join(data_root, self.category, 'split_info.json')

        if not os.path.exists(data_split_file_path):
            self.__process_category_data__(data_split_file_path)
        
        print('For category {}, use pre-processed data split from {}'.format(self.category, data_split_file_path))
        with open(data_split_file_path, 'r') as f:
            data_split_file = json.load(f)

        print('Category {}: train {}, test_seen {}, test_unseen {}'.format(self.category, 
                                                                           len(data_split_file['train']),
                                                                           len(data_split_file['test_seen']),
                                                                           len(data_split_file['test_unseen'])))
        self.data_split.update(data_split_file)


    def __process_category_data__(self, data_split_file_path):
        '''
        Each categroy has 5500 sequences, split them into 5000 training, ~250 test_seen, ~250 test_unseen
        '''
        print('Splitting category {} data into train, test_seen, test_unseen'.format(self.category))

        all_instances = os.listdir(os.path.join(self.root, self.category_id))
        num_instances_all = len(all_instances)
        seq_all = []
        for instance in all_instances:
            seqs = os.listdir(os.path.join(os.path.join(self.root, self.category_id, instance)))
            seq_all += [os.path.join(self.category_id, instance, seq) for seq in seqs]
        print('Load {} sequences for category {}'.format(len(seq_all), self.category))
        
        test_unseen_rate = 250.0 / 5000.0
        num_instance_test_unseen = int(test_unseen_rate * num_instances_all)
        test_unseen_instances = random.sample(all_instances, num_instance_test_unseen)
        seq_test_unseen = []
        for test_unseen_instance in test_unseen_instances:
            seqs = os.listdir(os.path.join(os.path.join(self.root, self.category_id, test_unseen_instance)))
            seq_test_unseen += [os.path.join(self.category_id, test_unseen_instance, seq) for seq in seqs]

        num_instance_test_seen = 500 - num_instance_test_unseen
        seq_train_and_test_seen = list(set(seq_all) - set(seq_test_unseen))
        seq_test_seen = random.sample(seq_train_and_test_seen, num_instance_test_seen)
        seq_train = list(set(seq_train_and_test_seen) - set(seq_test_seen))

        data_split = {
            'train': seq_train,
            'test_unseen': seq_test_unseen,
            'test_seen': seq_test_seen,
        }

        print('Splitting done')

        with open(data_split_file_path, 'w') as f:
            json.dump(data_split, f)


    def __load_dataset_general__(self):
        '''
        Category 'general':
            Each categroy has 1000 sequences training, 50 seq for seen instance testing, 50 seq for unseen instance testing
        Category 'general_test_unseen':
            Each category has ~100 instances of unseen categroy for testing
        '''
        data_root = './data_kubric'
        data_split_path = os.path.join(data_root, self.category_name)
        os.makedirs(data_split_path, exist_ok=True)
        data_split_file_path = os.path.join(data_root, self.category_name, 'split_info.json')

        if not os.path.exists(data_split_file_path):
            if self.category_name == 'general':
                self.__process_category_data__(data_split_file_path)
            elif self.category_name == 'general_unseen_category':
                self.__process_category_unseen_data__(data_split_file_path)

        with open(data_split_file_path, 'r') as f:
            data_split_file = json.load(f)

        if self.category_name == 'general':
            print('General dataset: train {}, test_seen {}, test_unseen {}'.format(
                                                                            len(data_split_file['train']),
                                                                            len(data_split_file['test_seen']),
                                                                            len(data_split_file['test_unseen'])))
        elif self.category_name == 'general_unseen_category':
            print('General dataset: test_unseen {}'.format(len(data_split_file['test_unseen'])))
        self.data_split.update(data_split_file)

    def __process_category_data__(self, data_split_file_path):
        all_info = {
                'train': [],
                'test_seen': [],
                'test_unseen': [],
            }

        for category in self.category:
            category_info = {
                'train': [],
                'test_seen': [],
                'test_unseen': [],
            }
            # get train data
            train_instances = []
            category_path = os.path.join(self.root, 'train', ShapeNet_ids[category])
            all_instance = os.listdir(category_path)
            for instance in all_instance:
                seq_path = os.path.join(category_path, instance, '0')
                if len(os.listdir(seq_path)) != 31:
                    continue
                else:
                    category_info['train'] += [os.path.join('train', ShapeNet_ids[category], instance, '0')]
                    train_instances.append(instance)

            # get test data
            category_path = os.path.join(self.root, 'test', ShapeNet_ids[category])
            all_instance = os.listdir(category_path)
            for instance in all_instance:
                seq_path = os.path.join(category_path, instance, '0')
                if len(os.listdir(seq_path)) != 31:
                    continue
                else:
                    if instance in train_instances:
                        category_info['test_seen'] += [os.path.join('test', ShapeNet_ids[category], instance, '0')]
                    else:
                        category_info['test_unseen'] += [os.path.join('test', ShapeNet_ids[category], instance, '0')]
        
            for k in category_info.keys():
                all_info[k] += category_info[k]

        print('Splitting done')
        with open(data_split_file_path, 'w') as f:
            json.dump(all_info, f)

    def __process_category_unseen_data__(self, data_split_file_path):
        all_info = {
                'test_unseen': [],
            }

        for category in self.category:
            category_info = {
                'test_unseen': [],
            }
            
            # get test data
            category_path = os.path.join(self.root, 'test_unseen', ShapeNet_ids[category])
            all_instance = os.listdir(category_path)
            for instance in all_instance:
                seq_path = os.path.join(category_path, instance, '0')
                if len(os.listdir(seq_path)) != 31:
                    continue
                else:
                    category_info['test_unseen'] += [os.path.join('test_unseen', ShapeNet_ids[category], instance, '0')]
        
            for k in category_info.keys():
                all_info[k] += category_info[k]

        print('Splitting done')
        with open(data_split_file_path, 'w') as f:
            json.dump(all_info, f)


    def __len__(self):
        return len(self.seq_names)

    def __getitem__(self, idx):
        seq_name = self.seq_names[idx]
        seq_path = os.path.join(self.root, seq_name)
        
        # load metadata
        with open(os.path.join(seq_path, 'metadata.json'), 'r') as f:
            meta = json.load(f)
        sensor_width = meta['camera']['sensor_width']       # 36
        focal_length_normalized = meta['camera']['K'][0][0] * self.image_width

        if seq_name in self.data_split['test_unseen']:
            seen_flag = torch.tensor([-1.0])
        else:
            seen_flag = torch.tensor([1.0])

        all_files = os.listdir(seq_path)

        rgb_files = []
        for it in all_files:
            if 'rgb' in it:
                rgb_files.append(it)
        len_seq = len(rgb_files)
        rgb_files_withIndices = [(it, int(it.replace('rgba_', '').replace('.png', ''))) for it in rgb_files]  # (name in rgba_xxxxx.png, index)
        rgb_files_withIndices = sorted(rgb_files_withIndices, key=lambda x: x[1])
        rgb_files = [it[0] for it in rgb_files_withIndices]
     
        if self.split == 'train':
            chosen_index = random.sample(range(len_seq), self.num_frames_per_seq)
            if self.config.dataset.train_shuffle:
                random.shuffle(chosen_index)
        else:
            chosen_index = range(self.num_frames_per_seq)

        # load image and mask
        chosen_rgb_files = [rgb_files[it] for it in chosen_index]        
        imgs, masks, depths = [], [], []
        for rgb_file in chosen_rgb_files:
            img, mask, depth = self._load_frame(seq_path, rgb_file, sensor_width=sensor_width, focal_length_normalized=focal_length_normalized)
            img = torch.tensor(img) #torch.from_numpy(img)
            mask = torch.tensor(mask) #torch.from_numpy(mask)
            depth = torch.tensor(depth) # torch.from_numpy(depth)
            imgs.append(img)
            masks.append(mask)
            depths.append(depth)
        
        imgs = torch.stack(imgs)     # [t,c,h,w]
        masks = torch.stack(masks)   # [t,1,h,w]
        depths = torch.stack(depths) # [t,1,h,w]
        
        # get camera intrinsics
        K = torch.tensor(meta['camera']['K'])               # f and p are normalized by image size, [3,3]
        K = torch.mm(K, self.kubric_to_cv2[:3,:3].t())      # kubric -> opencv, [3,3] @ [3,3] -> [3,3]
        K[0] *= self.image_height
        K[1] *= self.image_width                            # scale intrinsics by image size

        # get camera poses (in kubric frame)
        '''
            3D world frame to 2D projection: p^2d_unscaled = K @ T^c_cTow @ p^w_homo
                superscript: the coordinate frame
                subscript  : the index (name) of objects or transformation
            T^c_cTow (camera extrinsics) = T^w_wToc.inv() = |R, t|.inv() = |R.T, -R.T @ t|, T^w_wToc is camera pose
                                                            |0, 1|         |0  ,      1  |
        '''
        positions = torch.tensor(meta['camera']['positions'])[chosen_index]      # [t,3]
        quaternions = torch.tensor(meta['camera']['quaternions'])[chosen_index]  # [t,4]
        rotations = quat2mat_transform(quaternions)                              # [t,3,3]
        cam_poses = torch.zeros(self.num_frames_per_seq, 4, 4)                   # [t,4,4]
        cam_poses[:,:3,:3] = rotations
        cam_poses[:,:3,3] = positions
        cam_poses[:,3,3] = 1.0

        # get relative camera poses to the first frame (in kubric frame)
        '''
            Reference: https://haosulab.github.io/ml-for-robotics/SP21/lectures/L2_Robot_Geometry.pdf
            Transformation between two cameras: 
                P^c2 = T^c2_c2Toc1 @ P^c1, or 
                P^c1 = T^c1_c1Toc2 @ P^c2
            If we set cam1 as canonical:
                We can see cam1 as a new 'world' frame, cam2 as a camera frame
                T^c2_c2Toc1 is the (relative) extrinsics of cam2 (used in 'world' to cam transformation)
                T^c1_c1Toc2 is the (relative)    pose    of cam2 (used in cam to 'world' transformation)
        '''
        cam_poses_rel = get_relative_pose(cam_poses[0], cam_poses)                                   # [t,4,4]
        cam_poses_rel[0] = torch.eye(4)

        # canonicalize camera poses (in opencv frame)
        cam_poses_cv2 = torch.matmul(cam_poses, self.kubric_to_cv2.unsqueeze(0))                     # [t,4,4]
        cam_extrinsics_cv2 = torch.inverse(cam_poses_cv2)                                            # [t,4,4]
        cam_poses_rel_cv2 = get_relative_pose(cam_poses_cv2[0], cam_poses_cv2)
        cam_poses_rel_cv2[0] = torch.eye(4)
        cam_poses_rel_every2_cv2 = get_relative_pose(cam_poses_cv2[:-1], cam_poses_cv2[1:])
        #cam_poses_rel_cv2 = transform_relative_pose(cam_poses_rel, self.kubric_to_cv2)
        canonical_extrinsics_cv2 = self.canonical_extrinsics_cv2                                     # [4,4]
        canonical_pose_cv2 = self.canonical_pose_cv2
        cam_poses_cv2_canonicalized = canonicalize_poses(canonical_pose_cv2, cam_poses_rel_cv2)      # [t,4,4]
        cam_extrinsics_cv2_canonicalized = torch.inverse(cam_poses_cv2_canonicalized)

        sample = {
            'images': imgs.float(),                                                 # img observation
            'fg_probabilities': masks.float(),                                      # mask observation
            'depths': depths.float(),                                               # depth observation
            'K_cv2': K.unsqueeze(0).repeat(self.num_frames_per_seq,1,1),            # for initializing cameras
            'cam_extrinsics_cv2_canonicalized': cam_extrinsics_cv2_canonicalized,   # for initializing cameras, canonicalized setting
            'cam_extrinsics_cv2': cam_extrinsics_cv2,                               # for initializing cameras, uncanonicalized setting
            'cam_poses_cv2': cam_poses_cv2,                                         # uncanonicalized camera poses in cv2
            'cam_poses_cv2_canonicalized': cam_poses_cv2_canonicalized,             # canonicalized camera poses in cv2
            'cam_poses_rel_cv2': cam_poses_rel_cv2,                                 # relative camera poses in cv2
            'cam_poses_rel_every2_cv2': cam_poses_rel_every2_cv2,                   # relative camera poses between every two frame
            'seq_name': seq_name,
        }

        if self.config.train.use_uncanonicalized_pose:
            sample['cam_extrinsics_cv2_canonicalized'] = cam_extrinsics_cv2
            sample['cam_poses_cv2'] = cam_poses_cv2

        if self.split != 'train':
            sample['seen_flag'] = seen_flag

        return sample


    def _load_frame(self, seq_path, file_name, **kwargs):
        file_path = os.path.join(seq_path, file_name)
        img_pil = Image.open(file_path)
        img_np = np.asarray(img_pil)

        mask = Image.fromarray((img_np[:,:,3] > 0).astype(float))
        if not self.config.dataset.mask_images:
            # white background
            new_image = Image.new("RGBA", img_pil.size, "WHITE")
            new_image.paste(img_pil, (0, 0), img_pil)
            new_image = new_image.convert('RGB')
            rgb = new_image
        else:
            # black background
            rgb = Image.fromarray(img_np[:,:,:3])

        rgb = rgb.resize((self.image_height, self.image_width), Image.ANTIALIAS)
        mask = mask.resize((self.image_height, self.image_width), Image.NEAREST)

        rgb = np.asarray(rgb).transpose((2,0,1)) / 255.0                            # [3,H,W], in range [0,1]
        mask = np.asarray(mask)[:,:,np.newaxis].transpose((2,0,1))                  # [1,H,W], in range [0,1]

        if self.config.dataset.mask_images:
            rgb *= mask
        
        if self.config.train.normalize_img:
            normalization = transforms.Compose([
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250])),
            ])
            rgb = torch.from_numpy(rgb)
            rgb = normalization(rgb).numpy()

        depth = cv2.imread(file_path.replace('rgba', 'depth').replace('png', 'tiff'), cv2.IMREAD_UNCHANGED)
        depth = depth_to_z(depth, kwargs['sensor_width'], kwargs['sensor_width'], kwargs['focal_length_normalized']*depth.shape[0]) 
        depth = cv2.resize(depth, [self.image_height, self.image_width]) * mask[0]
        depth = depth[:,:,np.newaxis].transpose((2,0,1))    # [1,H,W]
        depth[depth>(1.6+0.5)] = 0.0    # change background pixels

        return rgb, mask, depth


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




def get_color_params(brightness=0, contrast=0, saturation=0, hue=0):
    if brightness > 0:
        brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
    else:
        brightness_factor = None

    if contrast > 0:
        contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
    else:
        contrast_factor = None

    if saturation > 0:
        saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
    else:
        saturation_factor = None

    if hue > 0:
        hue_factor = random.uniform(-hue, hue)
    else:
        hue_factor = None
    return brightness_factor, contrast_factor, saturation_factor, hue_factor


def color_jitter_seq(imgs, brightness=0, contrast=0, saturation=0, hue=0):
    res = []

    brightness, contrast, saturation, hue = get_color_params(brightness=brightness,
                                                             contrast=contrast,
                                                             saturation=saturation,
                                                             hue=hue)
    # Create img transform function sequence
    img_transforms = []
    if brightness is not None:
        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
    if saturation is not None:
        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
    if hue is not None:
        img_transforms.append(
            lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
    if contrast is not None:
        img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
    random.shuffle(img_transforms)

    for img in imgs:
        jittered_img = img
        for func in img_transforms:
            jittered_img = func(jittered_img)
        res.append(jittered_img)
    return res


def depth_to_z(z, sensor_width, sensor_height, focal_length) -> np.ndarray:
    z = np.array(z)
    assert z.ndim == 2
    h, w = z.shape

    pixel_centers_x = (np.arange(-w / 2, w / 2, dtype=np.float32) + 0.5) / w * sensor_width
    pixel_centers_y = (np.arange(-h / 2, h / 2, dtype=np.float32) + 0.5) / h * sensor_height
    squared_distance_from_center = np.sum(
        np.square(
            np.meshgrid(
                pixel_centers_x,  # X-Axis (columns)
                pixel_centers_y,  # Y-Axis (rows)
                indexing="xy",
            )
        ),
        axis=0,
    )
    depth_scaling = np.sqrt(1 + squared_distance_from_center / focal_length**2)
    return z / depth_scaling
