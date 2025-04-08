import torch
import torch.utils.data as data
import pytorch3d.io
import os
import json

class PointCloudDataset(data.Dataset):
    def __init__(self, ply_files, device='cuda'):
        """
        Args:
            ply_files (list): List of paths to .ply files.
            device (str): Device where tensors should be loaded ('cuda' or 'cpu').
        """
        self.ply_files = ply_files
        self.device = device

    def __len__(self):
        return len(self.ply_files)

    def __getitem__(self, idx):
        ply_file = self.ply_files[idx]
        
        # Load the point cloud from the .ply file (one by one)
        point_cloud = self.load_point(ply_file)

        return point_cloud

    def load_point(self, ply_f):
        """Load point cloud from a .ply file."""
        verts, _ = pytorch3d.io.load_ply(ply_f)
        assert isinstance(verts, torch.Tensor) and verts.shape[1] == 3, "The point cloud should be 3D"
        verts = verts.to(self.device)
        return verts

def custom_collate_fn(batch):
    """ Collate function to combine point clouds into a batch. """
    return torch.stack(batch, dim=0)