from .seg_data import SegmentationData, SegmentationDataReproducable

# class AxisScaling(object):

#     def __init__(self, interval=(0.75, 1.25), jitter=True, device="cuda"):
#         assert isinstance(interval, tuple)
#         self.interval = interval
#         self.jitter = jitter
#         self.device = device

#     def __call__(self, surface, point):
#         scaling = torch.rand(1, 3).to(self.device) * 0.5 + 0.75
#         surface = surface * scaling
#         point = point * scaling

#         scale = (1 / torch.abs(surface).max().item()) * 0.999999
#         surface *= scale
#         point *= scale

#         if self.jitter:
#             surface += 0.005 * torch.randn_like(surface)
#             surface.clamp_(min=-1, max=1)

#         return surface, point
    
def build_reproducible_dataset(path, radius_transform = None):
    dataset = SegmentationDataReproducable(path=path, radius_transform=radius_transform)
    return dataset

def build_affinity_dataset(
        neuron_path, 
        root_id_path, 
        samples_per_neuron=512, 
        scale=1.0, 
        train=True, 
        gnn_radius_transform=None, 
        max_neurons_merged=4, 
        shuffle_output=True, 
        fam_to_id=None, 
        translate=20.0, 
        n_dust_neurons=6,
        n_dust_nodes_per_neuron=32
    ): 
    print("Building Affinity ...")
    dataset_train = SegmentationData(
        neuron_path=neuron_path,
        root_id_path=root_id_path,
        samples_per_neuron=samples_per_neuron,
        scale=scale,
        train=train,
        gnn_radius_transform=gnn_radius_transform,
        max_neurons_merged=max_neurons_merged, 
        shuffle_output=shuffle_output, 
        fam_id_mapping_path=fam_to_id,
        translate=translate, 
        n_dust_neurons=n_dust_neurons,
        n_dust_nodes_per_neuron=n_dust_nodes_per_neuron
    )

    return dataset_train
