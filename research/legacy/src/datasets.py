
import glob
import torch
import navis
import pandas as pd
import json
import os
import torch_geometric.transforms as Transforms

from torch.utils.data import Dataset
from torch_geometric.data import Data


    
def build_reproducible_dataset(path):
    dataset = MultiNeuronDatasetReproducable(path=path)
    return dataset

def build_affinity_dataset(
        neuron_path, 
        root_id_path, 
        samples_per_neuron=512, 
        scale=1.0, 
        train=True, 
        max_neurons_merged=4, 
        shuffle_output=True, 
        fam_to_id=None, 
        translate=20.0, 
        n_dust_neurons=6,
        n_dust_nodes_per_neuron=32
    ): 
    print("Building Dataset ...")
    dataset = MultiNeuronDataset(
        neuron_path=neuron_path,
        root_id_path=root_id_path,
        samples_per_neuron=samples_per_neuron,
        scale=scale,
        train=train,
        max_neurons_merged=max_neurons_merged, 
        shuffle_output=shuffle_output, 
        fam_id_mapping_path=fam_to_id,
        translate=translate, 
        n_dust_neurons=n_dust_neurons,
        n_dust_nodes_per_neuron=n_dust_nodes_per_neuron
    )

    return dataset


class MultiNeuronDatasetReproducable(Dataset):
    def __init__(self, path) -> None:
        self.path = path
        self.n_dust_neurons = 6
        self.n_dust_nodes_per_neuron = 32

    def __len__(self):
        n = len(glob.glob(f"{self.path}/pc_*.pt"))
        return n

    def __getitem__(self, idx):
        # return a set of predefined data
        pc = torch.load(f"{self.path}/pc_{idx}.pt")
        labels = torch.load(f"{self.path}/labels_{idx}.pt")
        mask = torch.load(f"{self.path}/mask_{idx}.pt")
        root_ids = torch.load(f"{self.path}/root_ids_{idx}.pt")
        families = torch.load(f"{self.path}/families_{idx}.pt")
        pairs = torch.load(f"{self.path}/pairs_{idx}.pt")
        pairs_labels = torch.load(f"{self.path}/pairs_labels_{idx}.pt")
        return pc, labels, mask, root_ids, families, pairs, pairs_labels


class MultiNeuronDataset(Dataset):

    def __init__(
        self,
        neuron_path: str,
        root_id_path: str = None, # path to the types file as .csv
        samples_per_neuron: int = 1024,
        unit: str = "um", # units of the neuron can be "um" or "nm"
        scale: float = 1.0, # global scale factor
        max_neurons_merged: int = 4, # maximum number of neurons to merge
        # gnn_radius_transform = None,
        fam_id_mapping_path = None,
        train=True,
        shuffle_output = True,
        translate = 20.0, # translation factor for augmentation 
        n_dust_neurons = 6,
        n_dust_nodes_per_neuron = 32,
    ):
        self.neuron_path = neuron_path
        self.root_id_path = root_id_path
        self.samples_per_neuron = samples_per_neuron
        self.unit = unit
        self.scale = scale
        self.train = train 
        self.max_neurons_merged = max_neurons_merged
        self.n_neurons = torch.randint(1, max_neurons_merged + 1, (1,))
        self.root_ids = pd.read_csv(root_id_path)
        self.neurons = glob.glob(f"{self.neuron_path}/*.swc")
        self.fam_to_id_path = fam_id_mapping_path
        self.fam_to_id = json.load(open(self.fam_to_id_path, "r"))
        self.shuffle_output = shuffle_output
        self.store_items = False

        self.n_dust_neurons = n_dust_neurons
        self.n_dust_nodes_per_neuron = n_dust_nodes_per_neuron
        self.translate = translate

    def compute_dust(self, neuron, dust_size=8): 
        """
        Computes the dust point cloud of a neuron. Dust by terminal fragments that are smaller than dust_size (um).
        """
        
        pruned = neuron.prune_twigs(dust_size)
        diff = neuron.nodes[~neuron.nodes["node_id"].isin(pruned.nodes["node_id"])]
        dust = torch.tensor(diff[["x", "y", "z"]].values, dtype=torch.float32)
        return dust

    
    def augment(self, points):

        t = torch.rand(3) * 2 - 1
        t = (t / torch.norm(t)) * self.translate

        points = Data(pos=points)
        aug = Transforms.Compose([
            Transforms.RandomJitter(1.0),
            Transforms.Center(),
        ])
        pc = aug(points)

        if self.translate is not None:
            pc.pos += t
        return pc.pos
    
    def random_rotation(self, points):
        points = Data(pos=points)
        aug = Transforms.Compose([
            Transforms.RandomRotate(200, axis=torch.randint(0, 3, (1,)).item())
        ])
        pc = aug(points)
        return pc.pos

    def __len__(self):
        """
        Returns the number of unique neurons in the dataset.
        """
        length = len(self.root_ids)
        return length


    def __getitem__(self, idx):
        """
        Returns a neuron from the dataset.
        """
        partners = torch.arange(0, len(self.root_ids), dtype=torch.int64)
        partners = partners[partners != idx] # exclude the current neuron
        
        n = torch.randint(0, self.max_neurons_merged, (1,)) # randomly pick number of additional neurons to merge
        selected_partner = torch.randint(0, len(partners), (n,)) # select the partners
        neuron_ids = torch.tensor([idx])
        if selected_partner.numel() > 0: # only concatenate if there are partners
            neuron_ids = torch.cat([neuron_ids, selected_partner])

        n_dust_nodes = self.n_dust_neurons * self.n_dust_nodes_per_neuron
        dust_ids = torch.randint(0, len(partners), (self.n_dust_neurons,)) # randomly pick a dust neuron

        if n_dust_nodes > 0:
            all_dust = []
            for d in dust_ids:
                root_id = self.root_ids.iloc[d.item()]["root_id"]
                skeleton = navis.read_swc(self.neuron_path + f"/{root_id}.swc")
                skeleton = skeleton.convert_units(self.unit, inplace=True)
                dust = self.compute_dust(skeleton)

                if dust.size(0) > 0:
                    selected, _ = torch.sort(torch.randint(0, dust.size(0), (self.n_dust_nodes_per_neuron,)))
                    selected_dust = dust[selected]
                else:
                    selected_dust = torch.randn(self.n_dust_nodes_per_neuron, 3) * self.scale

                selected_dust = self.augment(selected_dust)
                all_dust.append(selected_dust)        
            dust = torch.cat(all_dust, dim=0)

        # store all nodes, labels and root_ids
        all_nodes = []
        all_labels = []
        all_root_ids = []
        all_families = []

        # iterate over all element in ix
        l = 1
        for n_ix in neuron_ids.flatten():
            root_id = self.root_ids.iloc[n_ix.item()]["root_id"]
            family_name = self.root_ids.iloc[n_ix.item()]["family"]
            family_id = self.fam_to_id[family_name]

            skeleton = navis.read_swc(self.neuron_path + f"/{root_id}.swc")
            skeleton = skeleton.convert_units(self.unit, inplace=True)
            
            nodes = torch.tensor(skeleton.nodes[["x", "y", "z"]].values, dtype=torch.float32)
            nodes = self.augment(nodes)
            
            selected, _ = torch.sort(torch.randint(0, nodes.size(0), (self.samples_per_neuron,)))
            selected_nodes = nodes[selected]
            all_nodes.append(selected_nodes)

            labels = torch.ones(selected_nodes.size(0), dtype=torch.float32) * l
            all_labels.append(labels)

            root_id = torch.tensor(root_id, dtype=torch.int64).unsqueeze(-1)
            all_root_ids.append(root_id)

            family = torch.tensor(family_id, dtype=torch.int64).unsqueeze(-1)
            all_families.append(family)

            l += 1

        if n_dust_nodes > 0:
            all_nodes.append(dust)
            all_labels.append(torch.ones(dust.size(0), dtype=torch.float32) * l)

        n_points = neuron_ids.size(0) * self.samples_per_neuron + n_dust_nodes
        point_cloud_size = self.samples_per_neuron * self.max_neurons_merged + n_dust_nodes

        pc = torch.zeros(point_cloud_size, 3) - self.scale
        pc[:n_points] = torch.cat(all_nodes, dim=0)

        labels = torch.zeros(point_cloud_size)
        labels[:n_points] = torch.cat(all_labels, dim=0)
        
        root_ids = torch.zeros(self.max_neurons_merged).to(torch.int64)
        root_ids[:neuron_ids.size(0)] = torch.cat(all_root_ids, dim=0)

        families = torch.zeros(self.max_neurons_merged).to(torch.int64)
        families[:neuron_ids.size(0)] = torch.cat(all_families, dim=0)

        mask = torch.zeros(point_cloud_size)
        mask[:n_points] = 1.0
        mask = mask.bool()

        pc = self.random_rotation(pc)
        # normalize pc
        pc = pc / self.scale

        if self.train:
            # shuffle the point cloud, labels and mask
            if self.shuffle_output:
                shuffle_idx = torch.randperm(pc.size(0))
                pc = pc[shuffle_idx]
                labels = labels[shuffle_idx]
                mask = mask[shuffle_idx]

            pair_idx_1 = torch.randint(0, point_cloud_size, (point_cloud_size,))
            pair_idx_2 = torch.randint(0, point_cloud_size, (point_cloud_size,)) 
            pairs = torch.stack([pair_idx_1, pair_idx_2], dim=1)        
        else:
            pair_idx_1 = torch.arange(0, point_cloud_size)
            x, y = torch.meshgrid(pair_idx_1, pair_idx_1, indexing='ij')
            pairs = torch.stack([x, y], dim=-1)
            pairs = pairs.reshape(-1, 2)
        
        is_same_neuron = labels[pairs[:, 0]] == labels[pairs[:, 1]]
        is_non_padding = torch.logical_and(labels[pairs[:, 0]] > 0, labels[pairs[:, 1]] > 0)
        is_non_dust = torch.logical_and(labels[pairs[:, 0]] < l, labels[pairs[:, 1]] < l)
        pairs_labels = is_same_neuron & is_non_padding & is_non_dust
        pairs_labels = pairs_labels.float()

        if self.store_items:
            # get parent of root id path
            parent = os.path.dirname(self.root_id_path)
            rep_items_folder = os.path.join(parent, "reproducable_items")
            if not os.path.exists(rep_items_folder):
                os.makedirs(rep_items_folder)
            
            torch.save(pc, os.path.join(rep_items_folder, f"pc_{idx}.pt"))
            torch.save(labels, os.path.join(rep_items_folder, f"labels_{idx}.pt"))
            torch.save(mask, os.path.join(rep_items_folder, f"mask_{idx}.pt"))
            torch.save(root_ids, os.path.join(rep_items_folder, f"root_ids_{idx}.pt"))
            torch.save(families, os.path.join(rep_items_folder, f"families_{idx}.pt"))
            torch.save(pairs, os.path.join(rep_items_folder, f"pairs_{idx}.pt"))
            torch.save(pairs_labels, os.path.join(rep_items_folder, f"pairs_labels_{idx}.pt"))
            

        return pc, labels, mask, root_ids, families, pairs, pairs_labels
