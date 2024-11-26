# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist
from torch import inf
from typing import List
import navis
import numpy as np

from matplotlib import pyplot as plt
import shutil


from numpy.typing import NDArray

from typing import List, Tuple, Dict, Set

import plotly.express as px
import plotly.graph_objects as go





def project_pc_to_image(
    pc_coords: torch.Tensor,
    pc_values: torch.Tensor,
    axis: int = 2,
) -> torch.Tensor:
    """
    Project the values of a point cloud to a 2D image along a specified axis.
    """

    im_size = 256
    batch_size = pc_coords.size(0)
    indices = torch.arange(pc_coords.size(-1)) != axis
    coords = pc_coords[..., indices]

    print("min max of pc coords", coords.min(), coords.max())

    coords = (coords - coords.min()) / (coords.max() - coords.min()) * (im_size - 1)
    coords = coords.int()

    print("min max of coords", coords.min(), coords.max())

    # create the image
    image = torch.zeros((batch_size, im_size, im_size), dtype=pc_values.dtype, device=pc_values.device)
    image[..., coords[..., 0], coords[..., 1]] = pc_values


    return image


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def save_im(
    path: str,  # path to save the image
    tensor: torch.Tensor = None,  # tensor to save as image, if none the image is black
    overlays: List[torch.Tensor] = None,  # list of tensors to overlay on the image
    overlay_colors: List[List[float]] = None,  # list of colors for the overlays
):
    """
    Save tensor as image to disk. If not None, overlay is added to the image. Tensor and image must have the same dimensions.
    """
    import torchvision.transforms as transforms

    if tensor is None:
        if len(overlays) == 0:
            raise ValueError("Either tensor or overlays must be provided.")
        tensor = torch.zeros_like(overlays[0])

    tensor = tensor.detach().cpu()
    # tensor = torch.tensor(tensor > 0.0).float()
    if overlays != None and len(overlays) > 0:
        tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)
        for overlay, overlay_color in zip(overlays, overlay_colors):
            # assert tensor.shape[:1] == overlay.shape
            overlay = overlay.detach().cpu()
            idx = overlay > 0
            tensor[idx, :] = torch.tensor(overlay_color, dtype=tensor.dtype).view(1, 3)
        tensor = tensor.permute(2, 0, 1)
    tensor = transforms.ToPILImage()(tensor)
    tensor.save(path)


def project_neuron(
    neuron: navis.TreeNeuron,
    sample_density: float = 1.0,
    x_length: int = None,
    y_length: int = None,
    z_length: int = None,
) -> List[torch.Tensor]:
    """
    Project a neuron onto a 2D plane.

    Args:
    - neuron (TreeNeuron): A navis.TreeNeuron object.
    - sample_density (float): The density of the projection.

    Returns:
    - heatmap (Tensor): A 2D histogram tensor representing the projection.
    """
    bbox = torch.tensor(neuron.bbox)
    if x_length is None:
        x_length = int((bbox[0, 1] - bbox[0, 0]) * sample_density)
    if y_length is None:
        y_length = int((bbox[1, 1] - bbox[1, 0]) * sample_density)
    if z_length is None:
        z_length = int((bbox[2, 1] - bbox[2, 0]) * sample_density)

    points = neuron.nodes[["x", "y", "z"]].values
    points = torch.tensor(points, dtype=torch.float32)
    xy = project_pointcloud(
        points, "xy", bins=[int(x_length), int(y_length)], range=None
    )
    yz = project_pointcloud(
        points, "yz", bins=[int(y_length), int(z_length)], range=None
    )
    xz = project_pointcloud(
        points, "xz", bins=[int(x_length), int(z_length)], range=None
    )
    return [xy, yz, xz]


def project_pointcloud(
    points: torch.Tensor, plane: str, bins=100, range=None
) -> torch.Tensor:
    """
    Project points onto a specified plane using histogram2d.

    Args:
    - points (Tensor): A Nx3 tensor of XYZ points.
    - plane (str): A string 'xy', 'yz', or 'xz' indicating the projection plane.
    - bins (int): Number of bins in the histogram (resolution of the projection).
    - range (list of tuples): Ranges of bins in each dimension.

    Returns:
    - heatmap (Tensor): A 2D histogram tensor representing the projection.
    """
    if plane == "xy":
        indices = (0, 1)
    elif plane == "yz":
        indices = (1, 2)
    elif plane == "xz":
        indices = (0, 2)
    else:
        raise ValueError("Plane must be 'xy', 'yz', or 'xz'")

    # Select the appropriate columns based on plane
    selected_points = points[:, indices]

    # Move tensor to CPU for compatibility with torch.histogramdd
    selected_points = selected_points.detach().cpu()

    print(selected_points.shape)
    print(bins)
    print(range)

    # Create histogram
    heatmap, _ = torch.histogramdd(selected_points, bins=bins, range=range)
    return heatmap


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

def comp_euclidean_distance(coords: torch.tensor, pairs: torch.tensor) -> torch.tensor:
    """
    Compute the euclidean distance between pairs of points in a point cloud.
    """
    # compute the difference between the pairs of points

    ix = pairs[..., 0].unsqueeze(-1).expand(-1, -1, 3)
    from_coords = torch.gather(coords, 1, ix)
    ix = pairs[..., 1].unsqueeze(-1).expand(-1, -1, 3)
    to_coords = torch.gather(coords, 1, ix)
    dists = torch.pow(from_coords - to_coords, 2).sum(-1).sqrt()

    return dists

def save_density(path: str, x: torch.tensor, y: torch.tensor, root_ids: torch.tensor, suffix="", folder="density", bins=30) -> None:
    """
    Save density plot to disk.

    Args:
    - path (str): The path to save the scatterplot.
    - x (Tensor): A tensor of x coordinates.
    - y (Tensor): A tensor of y coordinates.
    - root_ids (Tensor): A tensor of root_ids.

    """
    
    x = x.cpu().numpy()
    y = y.cpu().numpy()

    # filter for values smaller than the threshold
    ix = x < 1.0
    x_below_threshold = np.expand_dims(x[ix], axis=0)
    y_below_threshold = np.expand_dims(y[ix], axis=0)

    ix = x == 1.0
    y_above_threshold = np.expand_dims(y[ix], axis=0)

    root_ids = root_ids.cpu().numpy()
    path = os.path.join(path, folder)
    # make path if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(x.shape[0]):

        # compute histogram of x[i] and y[i]
        plt.hist(y_above_threshold[i], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Distribution of Predicted Distances with GT Distances = 1.0')
        plt.xlabel('Predicted Distance')
        plt.ylabel('Frequency')
        plt.savefig(f'{path}/{root_ids[i]}{"_histogram_y"}.png')
        plt.close()

        fig, ax = plt.subplots()
        ax.hexbin(x_below_threshold[i], y_below_threshold[i], gridsize=bins, cmap='Reds')
        ax.set_xlabel('Ground Truth Distances')
        ax.set_ylabel('Predicted Distances')
        ax.set_title('GT vs. Predicted Distances (for GT Distances < 1.0)')
        plt.savefig(f'{path}/{root_ids[i]}{suffix}.png')
        plt.close()


def save_scatter(path: str, x: torch.tensor, y: torch.tensor, root_ids: torch.tensor, suffix="", folder="scatter") -> None:
    """
    Save scatterplot to disk.

    Args:
    - path (str): The path to save the scatterplot.
    - x (Tensor): A tensor of x coordinates.
    - y (Tensor): A tensor of y coordinates.
    - root_ids (Tensor): A tensor of root_ids.

    """
    
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    root_ids = root_ids.cpu().numpy()
    path = os.path.join(path, folder)
    # make path if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(x.shape[0]):
        fig, ax = plt.subplots()
        ax.scatter(x[i], y[i], s=1, alpha=0.2)
        ax.set_xlabel('Ground Truth Distances')
        ax.set_ylabel('Predicted Distances')
        ax.set_title('Ground Truth vs Predicted Distances')
        plt.savefig(f'{path}/{root_ids[i]}{suffix}.png')
        plt.close()

def remove_small_clusters(labels: torch.tensor, min_size: int) -> torch.tensor:
    """
    Remove small clusters from a clustering.

    Args:
    - labels (Tensor): A tensor of cluster labels.
    - min_size (int): The minimum size of a cluster, if lower cluster will be considered background class. 

    Returns:
    - labels (Tensor): A tensor of cluster labels with small clusters removed.
    """
    cluster_sizes = torch.bincount(labels)
    labels_adjusted = labels.clone()
    for i in range(len(labels_adjusted)):
        l = labels[i]
        if cluster_sizes[l] < min_size:
            labels_adjusted[i] = -1
    _, labels_adjusted = torch.unique(labels_adjusted, sorted=True, return_inverse=True)
    return labels_adjusted



def get_mapped_labels(gt_labels : torch.tensor, labels : torch.tensor) -> torch.tensor:
    """get new labels for the predicted labels such that the classes map to the ground truth classes if the overlap is maximal"""

    gt_labels = gt_labels.cpu().numpy().astype(np.int64)
    labels = labels.cpu().numpy().astype(np.int64)


    gt_labels = gt_labels.reshape(-1)
    labels = labels.reshape(-1)
    new_labels = np.zeros_like(labels)
    used_gt_classes :  Set[int] = set()
    used_labels : Set[int] = set()
    #for each predicted label, find the ground truth class with the highest overlap and store the number of overlapping pixels
    #potential_mappings: (predicted_label, gt_label, num_overlapping_pixels)
    potential_mappings : List[Tuple[int, int, int]] = []

    for i in range(np.max(labels)+1):
        gt_classes = np.bincount(gt_labels[labels == i])
        for j in range(len(gt_classes)):
            if gt_classes[j] > 0:
                potential_mappings.append((int(i), int(j), int(gt_classes[j])))
    #sort the potential mappings by the number of overlapping pixels
    potential_mappings.sort(key=lambda x: x[2], reverse=True)
    #for each potential mapping, assign the predicted label to the ground truth class if it has not been used yet
    for i, j, _ in potential_mappings:
        if j not in used_gt_classes and i not in used_labels:
            new_labels[labels == i] = j
            used_gt_classes.add(j)
            used_labels.add(i)

    next_class = np.max(gt_labels) + 1
    for i in range(np.max(labels)+1):
        if i in used_labels:
            continue
        new_labels[labels == i] = next_class
        next_class += 1

    new_labels = torch.tensor(new_labels, dtype=torch.int64)
    return new_labels



def qual_plot(path: str, pc: torch.tensor, labels: torch.tensor, root_ids: torch.tensor, suffix="", folder="qual_res", x_res = 1000, y_res = 1000) -> None:

    pc = pc.detach().cpu()
    labels = labels.detach().cpu()

    labels = labels.numpy()

    # Qualitative colorscale (e.g., 'Set1')
    colorscale = px.colors.qualitative.Bold
    color_map = {label: colorscale[i % len(colorscale)] for i, label in enumerate(np.unique(labels))}
    mapped_colors = [color_map[label] for label in labels]

    fig = go.Figure(data=[go.Scatter3d(
    x=pc[:, 0].numpy(),
    y=pc[:, 1].numpy(),
    z=pc[:, 2].numpy(),
    mode='markers',
    marker=dict(
            size=4,
            # color=labels.numpy(),  # Set color to the z-values
            color=mapped_colors,
            # colorscale='Viridis',  # Color scale
            opacity=1.0
        )
    )])

    # Customize the layout for background colors
    fig.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor="white",
                    gridcolor="lightgrey",
                    #showbackground=False,
                    title='',
                    dtick=0.1,
                    showticklabels=False, 
                    zeroline=True),
            yaxis=dict(backgroundcolor="white",
                    gridcolor="lightgrey",
                    #showbackground=False,
                    title='',
                    dtick=0.1,
                    showticklabels=False,
                    zeroline=True),
            zaxis=dict(backgroundcolor="white",
                    gridcolor="lightgrey",
                    #showbackground=False, 
                    title='',
                    dtick=0.1,
                    showticklabels=False,
                    zeroline=True)
        ),
    )

    output_dir = os.path.join(path, folder)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    name = str(root_ids.tolist()) + suffix + ".png"
    fig.write_image(output_dir + "/" + name, width=x_res, height=y_res, scale=1)  # Adjust width, height, and scale

def save_points(path: str, points: torch.tensor, root_ids: torch.tensor, suffix="", folder="points", rand_hash=None) -> None:
    path = os.path.join(path, folder)
    points = points.detach().cpu().numpy()
    # make path if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(points.shape[0]):
        point_ = points[i]
        root_id = str(root_ids[i].tolist())
        if rand_hash:
            root_id = root_id + "_" + rand_hash[i]
        np.save(f"{path}/{root_id}{suffix}", point_)


def save_images(path: str, images: torch.tensor, root_ids: torch.tensor, suffix="", folder="images") -> None:
    """
    Save images to disk.

    Args:
    - path (str): The path to save the images.
    - images (Tensor): A tensor of images.
    - root_ids (Tensor): A tensor of root_ids.
    """
    # iterate over all batches
    path = os.path.join(path, folder)
    # make path if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(images.shape[0]):
        image = images[i]
        root_id = str(root_ids[i].tolist())
        save_im(f"{path}/{root_id}{suffix}.png", tensor=image)

def save_embeddings(path, batch, embeddings, labels, root_ids = None):
    output_dir = os.path.join(path, "embeddings")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    emb_path = os.path.join(output_dir, 'embeddings_batch_{}.pth'.format(batch))
    labels_path = os.path.join(output_dir, 'labels_batch_{}.pth'.format(batch))
    torch.save(embeddings, emb_path)
    torch.save(labels, labels_path)

    if root_ids is not None:
        neurons_path = os.path.join(output_dir, 'root_ids_batch_{}.pth'.format(batch))
        torch.save(root_ids, neurons_path)



def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    output_dir = os.path.join(args.output_dir, "ckpt")
    # make dir if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_path = os.path.join(output_dir, 'checkpoint-{}.pth'.format(epoch_name))
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args,
        }

        save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x
