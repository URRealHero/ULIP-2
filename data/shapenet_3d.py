import torch
import torch.utils.data as data
import pytorch3d.io
import os
import json

class PointCloudDataset(data.Dataset):
    def __init__(self, json_file, device='cuda'):
        """
        Args:
            json_file (str): Path to the JSON file that maps model IDs to .ply file paths and categories.
            device (str): Device where tensors should be loaded ('cuda' or 'cpu').
        """
        self.model2ply = {}
        self.model2category = {}  # Added to store categories
        self.model_ids = []  # List of model IDs to preserve order
        self.post_process(json_file)
        self.device = device

    def __len__(self):
        return len(self.model_ids)

    def __getitem__(self, idx):
        model_id = self.model_ids[idx]
        ply_file = self.model2ply[model_id]
        category = self.model2category[model_id]  # Retrieve category
        
        # Load the point cloud from the .ply file (one by one)
        point_cloud = self.load_point(ply_file)

        return point_cloud, model_id, category  # Return the category as well

    def post_process(self, json_file):
        """Process the JSON file to create the mapping between model ID, category, and .ply path."""
        with open(json_file, 'r') as f:
            data = json.load(f)
        for model_id, v in data.items():
            self.model2ply[model_id] = v["ply_path"]
            self.model2category[model_id] = v["category"]  # Store category
            self.model_ids.append(model_id)

    def load_point(self, ply_f):
        """Load point cloud from a .ply file."""
        verts, _ = pytorch3d.io.load_ply(ply_f)
        assert isinstance(verts, torch.Tensor) and verts.shape[1] == 3, "The point cloud should be 3D"
        verts = verts.to(self.device)
        return verts

def custom_collate_fn(batch):
    """Collate function to combine point clouds, model IDs, and categories into a batch."""
    pc_batch = torch.stack([item[0] for item in batch], dim=0)
    model_id_batch = [item[1] for item in batch]
    category_batch = [item[2] for item in batch]  # Collect categories
    return pc_batch, model_id_batch, category_batch  # Return category as well
