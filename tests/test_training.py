from torch.utils.data import Dataset
import torch
from src.my_project.data import myDataset, corrupt_mnist


def test_my_dataset():
    """Test the myDataset class."""
    dataset = myDataset("data/raw")
    assert len(dataset)== 8
