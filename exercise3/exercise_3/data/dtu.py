from pathlib import Path
import os

import numpy as np
import torch
from PIL import Image


class DTUDataset(torch.utils.data.Dataset):
    @staticmethod
    def near_far_from_sphere(rays_o, rays_d):
        # Get the near & far plane
        a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    @staticmethod
    def move_to_device(data, device):
        return [d.to(device) for d in data]


class DTUTrainDataset(DTUDataset):
    def __init__(self, num_rays):
        super().__init__()
        self.num_rays = num_rays

        self.images_all = []
        self.masks_all = []
        self.poses_all = []
        self.intrinsics_all = []
        self.image_ids = []

        root_path = Path("exercise_3/data/dtu_scan65")
        image_list = os.listdir(root_path / "train/images")
        intrinsics_path = root_path / "intrinsics.npy"
        self.intrinsics_all = np.load(intrinsics_path)
        pose_path = root_path / "pose.npy"
        # Camera-to-world matrix
        self.poses_all = np.load(pose_path)

        for image_name in image_list:
            image_path = root_path / "train/images" / image_name
            mask_path = root_path / "train/masks" / image_name
            self.image_ids.append(image_name.split(".")[0])

            image = np.array(Image.open(image_path)).astype(np.float32) / 255.0
            mask = np.array(Image.open(mask_path)).astype(np.float32) / 255.0

            self.images_all.append(image)
            self.masks_all.append(mask)

    def __len__(self):
        return len(self.images_all)

    def get_image_width(self):
        return self.images_all[0].shape[1]

    def get_image_height(self):
        return self.images_all[0].shape[0]

    def __getitem__(self, index):
        """Return the random sampled rays of one image
        Returns:
            rays_o: (num_rays, 3) rays origin of the image pixels
            rays_d: (num_rays, 3) rays direction of the image pixels
            rgb: (num_rays, 3) image color of the image pixels
            mask: (num_rays, 1) mask of the image pixels (valid=1.0, invalid=0.0)
        """
        image = self.images_all[index]
        mask = self.masks_all[index]

        image_id = int(self.image_ids[index])
        intrinsic = self.intrinsics_all[image_id][:3, :3]
        intrinsic_inv = np.linalg.inv(intrinsic)
        intrinsic_inv = torch.from_numpy(intrinsic_inv)

        pose = self.poses_all[image_id]
        pose = torch.from_numpy(pose)

        # Randomly sample the pixels
        pixels_x = torch.randint(low=0, high=self.get_image_width(), size=(self.num_rays,))
        pixels_y = torch.randint(low=0, high=self.get_image_height(), size=(self.num_rays,))

        rgb = torch.from_numpy(image[pixels_y, pixels_x])
        mask = torch.from_numpy(mask[pixels_y, pixels_x][:, :1])
        mask = (mask > 0.5).float()

        # K^-1 @ [u, v, 1]
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()
        p = torch.matmul(intrinsic_inv[None, :, :], p[:, :, None]).squeeze(-1)
        # Normalize the ray direction to have unit length
        rays_d = p / torch.norm(p, dim=-1, keepdim=True)
        # Rotate the ray direction by the camera pose
        rays_d = torch.matmul(pose[None, :3, :3], rays_d[:, :, None]).squeeze(-1)

        # Rays origin is the camera position
        rays_o = pose[None, :3, 3].repeat(self.num_rays, 1)
        return rays_o, rays_d, rgb, mask


class DTUValDataset(DTUDataset):
    def __init__(self):
        super().__init__()

        self.images_all = []
        self.masks_all = []
        self.poses_all = []
        self.intrinsics_all = []
        self.image_ids = []

        root_path = Path("exercise_3/data/dtu_scan65")
        image_list = os.listdir(root_path / "val/images")
        intrinsics_path = root_path / "intrinsics.npy"
        self.intrinsics_all = np.load(intrinsics_path)
        pose_path = root_path / "pose.npy"
        # Camera-to-world matrix
        self.poses_all = np.load(pose_path)

        for image_name in image_list:
            image_path = root_path / "val/images" / image_name
            mask_path = root_path / "val/masks" / image_name
            self.image_ids.append(image_name.split(".")[0])

            image = np.array(Image.open(image_path)).astype(np.float32) / 255.0
            mask = np.array(Image.open(mask_path)).astype(np.float32) / 255.0

            self.images_all.append(image)
            self.masks_all.append(mask)

    def __len__(self):
        return len(self.images_all)

    def get_image_width(self):
        return self.images_all[0].shape[1]

    def get_image_height(self):
        return self.images_all[0].shape[0]

    def __getitem__(self, index):
        """Return the ray of one whole image
        Returns:
            rays_o: (h, w, 3) rays origin of the image pixels
            rays_d: (h, w, 3) rays direction of the image pixels
            rgb: (h, w, 3) image color of the image pixels
            mask: (h, w, 1) mask of the image pixels (valid=1.0, invalid=0.0)
        """
        image = self.images_all[index]
        mask = self.masks_all[index]

        image_id = int(self.image_ids[index])
        intrinsic = self.intrinsics_all[image_id][:3, :3]
        intrinsic_inv = np.linalg.inv(intrinsic)
        intrinsic_inv = torch.from_numpy(intrinsic_inv)

        pose = self.poses_all[image_id]
        pose = torch.from_numpy(pose)

        # Sample all the pixels
        h, w = image.shape[:2]
        pixels_y, pixels_x = torch.meshgrid([torch.arange(h), torch.arange(w)])
        pixels_x = pixels_x.flatten()
        pixels_y = pixels_y.flatten()

        rgb = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        mask = (mask > 0.5).float()[..., :1]

        # K^-1 @ [u, v, 1]
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()    # [u, v, 1]
        p = torch.matmul(intrinsic_inv[None, :, :], p[:, :, None]).squeeze(-1)
        # Normalize the ray direction to have unit length
        rays_d = p / torch.norm(p, dim=-1, keepdim=True)
        # Rotate the ray direction by the camera pose
        rays_d = torch.matmul(pose[None, :3, :3], rays_d[:, :, None]).squeeze(-1)
        rays_d = rays_d.reshape(h, w, 3)

        # Rays origin is the camera position
        rays_o = pose[None, None, :3, 3].repeat(h, w, 1)
        return rays_o, rays_d, rgb, mask
