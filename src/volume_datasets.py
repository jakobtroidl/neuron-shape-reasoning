import torch
import pandas as pd

from abc import ABC, abstractmethod
from pocaduck import Query, StorageConfig
from torch.utils.data import Dataset


class DataInterface(ABC):
    def __init__(self, pth: str):
        self.pth = pth

    @abstractmethod
    def load_pc(self, root_id: int):
        pass

class PoCADuckInterface(DataInterface):
    def __init__(self, pth: str):
        super().__init__(pth)

    def load_pc(self, root_id: int) -> tuple[torch.tensor, torch.tensor]:
        config = StorageConfig(base_path=self.pth)
        query = Query(config)
        data = query.get_points(label=root_id)
        query.close()

        points = data[..., :3]
        sv_ids = data[..., 3]

        points = torch.tensor(points, dtype=torch.float32)
        sv_ids = torch.tensor(sv_ids, dtype=torch.int64)

        return points, sv_ids

class SegmentationDataset(Dataset):
    def __init__(self, path: str, di_path: str, max_neurons: int = 5):
        self.df = pd.read_csv(path)
        self.di = PoCADuckInterface(di_path)
        self.max_neurons = max_neurons

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # pick random number between 1 and 5
        n = torch.randint(1, self.max_neurons + 1, (1,)).item()

        print(f"Loading {n} neurons for index {idx}")

        all_points = []
        all_labels = []

        # pick n random neurons from the dataframe
        selected_neurons = self.df.sample(n=n, random_state=idx)
        for i, row in selected_neurons.iterrows():
            root_id = row["root_id"]
            points, sv_ids = self.di.load_pc(root_id)
            labels = torch.ones(points.size(0), dtype=torch.float32) * i
            all_points.append(points)
            all_labels.append(labels)
        
        # concatenate all points
        all_points = torch.cat(all_points, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        shuffle_idx = torch.randperm(all_points.size(0))
        all_points = all_points[shuffle_idx]
        all_labels = all_labels[shuffle_idx]
        
        return all_points, all_labels