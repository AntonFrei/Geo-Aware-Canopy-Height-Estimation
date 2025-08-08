import os
import platform
import shutil
import sys
import time
from collections import OrderedDict
from typing import Optional, Any

import numpy as np
import segmentation_models_pytorch as smp
import torch
import wandb
from torch.cuda.amp import autocast
from torch.utils.data import WeightedRandomSampler
from torchmetrics import MeanMetric
from torchvision.transforms import transforms
from tqdm.auto import tqdm

import visualization
from config import PreprocessedSatelliteDataset, FixValDataset
from config import means as meanDict
from config import percentiles as percentileDict
from config import stds as stdDict
from metrics import MetricsClass
from utilities import JointRandomRotationTransform
from utilities import SequentialSchedulers

import encoder as coord_encoders 
from encoder import (
    transform_coordinates,
    extract_epsg,
    ENCODER_MAP
)

from coord_injection_model import CoordInjectionModelWrapper



class Runner:
    """Base class for all runners, defines the general functions"""

    def __init__(self, config: Any, tmp_dir: str, debug: bool):
        """
        Initialize useful variables using config.
        :param config: wandb run config
        :type config: wandb.config.Config
        :param debug: Whether we are in debug mode or not
        :type debug: bool
        """
        self.config = config
        self.debug = debug
        """ Original GPU Selection Code:
        n_gpus = torch.cuda.device_count()
        if n_gpus > 0:
            config.update(dict(device='cuda:0'))
        else:
            config.update(dict(device='cpu'))

        self.dataParallel = (torch.cuda.device_count() > 1)
        if not self.dataParallel:
            self.device = torch.device(config.device)
            if 'gpu' in config.device:
                torch.cuda.set_device(self.device)
        else:
            # Use all visible GPUs
            self.device = torch.device("cuda:0")
            torch.cuda.device(self.device)
        torch.backends.cudnn.benchmark = True
        """
        #GPU Selection, currently not necessary
        #torch.cuda.set_device(1)
        self.device = torch.device("cuda:0")
        self.dataParallel = False  # Make sure only one GPU is used
        #print("Selected GPU after:", torch.cuda.current_device())


        # Set a couple useful variables
        self.seed = int(self.config.seed)
        self.loss_name = self.config.loss_name or 'shift_l1'
        sys.stdout.write(f"Using loss: {self.loss_name}.\n")
        self.use_amp = self.config.fp16
        self.tmp_dir = tmp_dir
        print(f"Using temporary directory {self.tmp_dir}.")
        self.label_rescaling_factor = 1.
        if self.config.use_label_rescaling:
            self.label_rescaling_factor = 60.
            sys.stdout.write(f"Using label rescaling of {self.label_rescaling_factor} - this is hardcoded.\n")
        
        if self.config.use_coord_encoding:
            wandb.run.summary["coord_encoder"] = self.config.coord_encoder 
       
        # Variables to be set
        self.loader = {loader_type: None for loader_type in ['train', 'val']}
        self.loss_criteria = {loss_name: self.get_loss(loss_name=loss_name) for loss_name in ['shift_l1', 'shift_l2', 'shift_huber', 'l1', 'l2', 'huber']}
        for threshold in [5, 10, 15, 20, 25, 30]:
            self.loss_criteria[f"l1_{threshold}"] = self.get_loss(loss_name=f"l1", threshold=threshold)

        # Coord encoder
        self.use_coord_encoding = self.config.use_coord_encoding
        self.coord_encoder = getattr(self.config, 'coord_encoder') # , 'raw'
        self.coord_injection_mode = getattr(self.config, 'coord_injection_mode')
        self.encoder_fn = ENCODER_MAP.get(getattr(self, "coord_encoder"))

        self.optimizer = None
        self.scheduler = None
        self.model = None
        self.artifact = None
        self.model_paths = {model_type: None for model_type in ['initial', 'trained']}
        self.model_metrics = {  # Organized way of saving metrics needed for retraining etc.
        }

        self.metrics = {mode: {'loss': MeanMetric().to(device=self.device),
                               'shift_l1': MeanMetric().to(device=self.device),
                               'shift_l2': MeanMetric().to(device=self.device),
                               'shift_huber': MeanMetric().to(device=self.device),
                               'l1': MeanMetric().to(device=self.device),
                               'l2': MeanMetric().to(device=self.device),
                               'huber': MeanMetric().to(device=self.device),
                               }
                        for mode in ['train', 'val']}

        for mode in ['train', 'val']:
            for threshold in [5, 10, 15, 20, 25, 30]:
                self.metrics[mode][f"l1_{threshold}"] = MeanMetric().to(device=self.device)

        self.metrics['train']['ips_throughput'] = MeanMetric().to(device=self.device)

    @staticmethod
    def set_seed(seed: int):
        """
        Sets the seed for the current run.
        :param seed: seed to be used
        """
        # Set a unique random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Remark: If you are working with a multi-GPU model, this function is insufficient to get determinism. To seed all GPUs, use manual_seed_all().
        torch.cuda.manual_seed(seed)  # This works if CUDA not available

    def reset_averaged_metrics(self):
        """Resets all metrics"""
        for mode in self.metrics.keys():
            for metric in self.metrics[mode].values():
                metric.reset()

    def get_metrics(self) -> dict:
        """
        Returns the metrics for the current epoch.
        :return: dict containing the metrics
        :rtype: dict
        """
        with torch.no_grad():
            loggingDict = dict(
                # Model metrics
                n_params=MetricsClass.get_parameter_count(model=self.model),

                # Optimizer metrics
                learning_rate=float(self.optimizer.param_groups[0]['lr']),
            )
            # Add metrics
            for split in ['train', 'val']:
                for metric_name, metric in self.metrics[split].items():
                    try:
                        # Catch case where MeanMetric mode not set yet
                        loggingDict[f"{split}/{metric_name}"] = metric.compute()
                    except Exception as e:
                        continue

        return loggingDict

    @staticmethod
    def get_dataset_root(dataset_name: str) -> str:

        """Returns the hardcoded root path for the dataset."""
        rootPath = f"/home/ubuntu/work/satellite_data/{dataset_name}/"
        if not os.path.isdir(rootPath):
            raise FileNotFoundError(f"Dataset path does not exist: {rootPath}")
    
        return rootPath

    

    def get_dataloaders(self):
        rootPath = self.get_dataset_root(dataset_name=self.config.dataset)
        print(f"Loading {self.config.dataset} dataset from {rootPath}.")

        data_path = rootPath
        train_dataframe = os.path.join(rootPath, 'train.csv')
        val_dataframe = os.path.join(rootPath, 'val.csv')
        #fix_val_dataframe = os.path.join(rootPath, 'fix_val.csv')

        transformDict = {split: None for split in ['train', 'val']}
        base_transform = transforms.ToTensor()
        transforms_list = [base_transform]   # Convert to tensor (this changes the order of the channels)
        if self.config.use_standardization:
            assert not self.config.use_input_clipping, "Cannot use both standardization and input clipping simultaneously."
            assert self.config.dataset in meanDict.keys(), f"Mean of Dataset {self.config.dataset} not implemented."
            assert self.config.dataset in stdDict.keys(), f"Std of Dataset {self.config.dataset} not implemented."
            mean, std = meanDict[self.config.dataset], stdDict[self.config.dataset]
            normalize_transform = transforms.Normalize(mean=mean, std=std)
            transforms_list.append(normalize_transform)
        elif self.config.use_input_clipping not in [False, None, 'None']:
            assert not self.config.use_standardization, "Mutually exclusive options: use_standardization and use_input_clipping."
            use_input_clipping = int(self.config.use_input_clipping)
            assert use_input_clipping in [1, 2, 5], "use_input_clipping must be in [False, None, 1, 2, 5]."
            sys.stdout.write(f"Using input clipping of in the {use_input_clipping}-{100-use_input_clipping}-range.\n")
            input_clipping_lower_bound = percentileDict[self.config.dataset][use_input_clipping]
            input_clipping_upper_bound = percentileDict[self.config.dataset][100 - use_input_clipping]

            # Convert the bounds to tensors
            input_clipping_lower_bound = torch.tensor(input_clipping_lower_bound, dtype=torch.float).view(-1, 1, 1) # View to make it a 3D tensor
            input_clipping_upper_bound = torch.tensor(input_clipping_upper_bound, dtype=torch.float).view(-1, 1, 1) # View to make it a 3D tensor

            # Define the clipping transform over the channels, i.e. each channel is clipped individually using the bounds from percentileDict
            clipping_transform = transforms.Lambda(lambda x: torch.clamp(x, min=input_clipping_lower_bound, max=input_clipping_upper_bound))
            transforms_list.append(clipping_transform)


        transformDict['train'] = transforms.Compose(transforms_list)
        transformDict['val'] = transforms.Compose(transforms_list)

        # Create the label transform to rescale the labels
        label_transforms = transforms.Compose([base_transform, lambda x: x * (1./self.label_rescaling_factor)])

        joint_transforms = None # Train transforms that are both applied to the image and the label
        if self.config.use_augmentation:
            sys.stdout.write(f"Using JointRandomRotationTransform.\n")
            joint_transforms = JointRandomRotationTransform()

        use_weighted_sampler = self.config.use_weighted_sampler or False

        remove_corrupt = not self.debug
        trainData = PreprocessedSatelliteDataset(data_path=data_path, dataframe=train_dataframe, image_transforms=transformDict['train'], label_transforms=label_transforms, joint_transforms=joint_transforms,
                                                 use_weighted_sampler=use_weighted_sampler, use_weighting_quantile=self.config.use_weighting_quantile, use_memmap=self.config.use_memmap, remove_corrupt=remove_corrupt, 
                                                 use_coord_encoding=self.use_coord_encoding, coord_encoder=self.coord_encoder, coord_injection_mode=self.coord_injection_mode)
        valData = PreprocessedSatelliteDataset(data_path=data_path, dataframe=val_dataframe,
                                               image_transforms=transformDict['val'], label_transforms=label_transforms, use_memmap=self.config.use_memmap, remove_corrupt=remove_corrupt,
                                                use_coord_encoding=self.use_coord_encoding, coord_encoder=self.coord_encoder, coord_injection_mode=self.coord_injection_mode)
        

        #fixvalData = FixValDataset(data_path=data_path, dataframe=fix_val_dataframe,
        #                                       image_transforms=transformDict['val'])

        # Custom collate function for feature map mode
        def collate_fn(batch):
            if self.use_coord_encoding and self.coord_injection_mode == "feature_maps":
                # Check if batch actually contains coordinates
                if len(batch[0]) == 3:  # (image, label, coordinates)
                    images, labels, coords = zip(*batch)
                    images = torch.stack(images)
                    labels = torch.stack(labels)
                    coords = torch.stack(coords)
                    return images, labels, coords
                else:  # Fallback to standard collation
                    return torch.utils.data.dataloader.default_collate(batch)
            else:
                # Standard collation for (image, label) tuples
                return torch.utils.data.dataloader.default_collate(batch)

        sys.stdout.write(f"Length of train and val splits: {len(trainData)}, {len(valData)}.\n")
        cut_off = 3000
        if len(valData) >= cut_off:
            sys.stdout.write(f"Validation dataset is large, reducing to a maximum of {cut_off} samples.\n")
            # Reduce the size of the validation dataset using self.seed as the random seed
            # Perform a random split using a generator with self.seed as the seed
            valData, _ = torch.utils.data.random_split(valData, [cut_off, len(valData) - cut_off], generator=torch.Generator().manual_seed(self.seed))
        sys.stdout.write(f"New length of train and val splits: {len(trainData)}, {len(valData)}.\n")


        num_workers_default = self.config.num_workers_per_gpu if self.config.num_workers_per_gpu is not None else 8
        num_workers = num_workers_default * torch.cuda.device_count() * int(not self.debug)
        sys.stdout.write(f"Using {num_workers} workers.\n")
        train_sampler = None
        shuffle = True
        if use_weighted_sampler:
            train_sampler = WeightedRandomSampler(trainData.weights, len(trainData))
            shuffle = False

        trainLoader = torch.utils.data.DataLoader(trainData, batch_size=self.config.batch_size, sampler=train_sampler,
                                                  pin_memory=torch.cuda.is_available(), num_workers=num_workers,
                                                  shuffle=shuffle, collate_fn=collate_fn)
        valLoader = torch.utils.data.DataLoader(valData, batch_size=self.config.batch_size, shuffle=False,
                                                pin_memory=torch.cuda.is_available(), num_workers=num_workers, collate_fn=collate_fn)
        #fixvalLoader = torch.utils.data.DataLoader(fixvalData, batch_size=2, shuffle=False, # This only works with batch_size=2
        #                                        pin_memory=torch.cuda.is_available(), num_workers=num_workers)

        return trainLoader, valLoader#, fixvalLoader

    def get_model(self, reinit: bool, model_path: Optional[str] = None) -> torch.nn.Module:
        """Returns the model with optional coordinate injection wrapper."""
        
        # Determine input channels based on injection mode
        if self.coord_injection_mode == "input" and self.use_coord_encoding:
            extra_channels = self.encoder_fn.num_output_channels
        else:
            extra_channels = 0
        
        print(f"Loading model - reinit: {reinit} | path: {model_path if model_path else 'None specified'}.")
        print(f"Coordinate injection mode: {self.coord_injection_mode}")
        
        if reinit:
            # Define the base model
            arch = self.config.arch or 'unet'
            backbone = self.config.backbone or 'resnet50'
            
            network_config = {
                "encoder_name": backbone,
                "encoder_weights": None if not self.config.use_pretrained_model else 'imagenet',
                "in_channels": 14 + extra_channels,  # Only add channels for input mode
                "classes": 1,
            }
            
            if arch == 'unet':
                base_model = smp.Unet(**network_config)
            elif arch == 'unetpp':
                base_model = smp.UnetPlusPlus(**network_config)
            elif arch == 'manet':
                base_model = smp.MAnet(**network_config)
            elif arch == 'linknet':
                base_model = smp.Linknet(**network_config)
            elif arch == 'fpn':
                base_model = smp.FPN(**network_config)
            elif arch == 'pspnet':
                base_model = smp.PSPNet(**network_config)
            elif arch == 'pan':
                base_model = smp.PAN(**network_config)
            elif arch == 'deeplabv3':
                base_model = smp.DeepLabV3(**network_config)
            elif arch == 'deeplabv3p':
                base_model = smp.DeepLabV3Plus(**network_config)
            else:
                raise NotImplementedError(f"Architecture {arch} not implemented")
            
            # Apply wrapper BEFORE loading weights
            if self.use_coord_encoding and self.coord_injection_mode == "feature_maps":
                model = CoordInjectionModelWrapper(
                    base_model=base_model,
                    coord_encoder_name=self.coord_encoder,
                    device=self.device
                )
            else:
                model = base_model
            
            # Load weights into the wrapped model (not base_model)
            if model_path is not None:
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                new_state_dict = OrderedDict()
                require_DP_format = isinstance(model, torch.nn.DataParallel)
                
                for k, v in state_dict.items():
                    is_in_DP_format = k.startswith("module.")
                    if require_DP_format and is_in_DP_format:
                        name = k
                    elif require_DP_format and not is_in_DP_format:
                        name = "module." + k
                    elif not require_DP_format and is_in_DP_format:
                        name = k[7:]
                    elif not require_DP_format and not is_in_DP_format:
                        name = k
                    new_state_dict[name] = v
                
                # Load into the correct model (wrapped or base)
                if isinstance(model, CoordInjectionModelWrapper):
                    model.base_model.load_state_dict(new_state_dict)
                else:
                    model.load_state_dict(new_state_dict)
        else:
            model = self.model

        if self.dataParallel and reinit and not isinstance(model, torch.nn.DataParallel):
            # Apply DataParallel to the wrapper or base model
            model = torch.nn.DataParallel(model)
        
        model = model.to(device=self.device)
        return model

    def get_loss(self, loss_name: str, threshold: float = None):
        assert loss_name in ['shift_l1', 'shift_l2', 'shift_huber', 'l1', 'l2', 'huber'], f"Loss {loss_name} not implemented."
        if threshold is not None:
            assert loss_name == 'l1', f"Threshold only implemented for l1 loss, not {loss_name}."
        # Dim 1 is the channel dimension, 0 is batch.
        # Sums up to get average height, could be mean without zeros
        remove_sub_track = lambda out, target: (out, torch.sum(target, dim=1))

        if loss_name == 'shift_l1':
            from losses.shift_l1_loss import ShiftL1Loss
            loss = ShiftL1Loss(ignore_value=0)
        elif loss_name == 'shift_l2':
            from losses.shift_l2_loss import ShiftL2Loss
            loss = ShiftL2Loss(ignore_value=0)
        elif loss_name == 'shift_huber':
            from losses.shift_huber_loss import ShiftHuberLoss
            loss = ShiftHuberLoss(ignore_value=0)
        elif loss_name == 'l1':
            from losses.l1_loss import L1Loss
            # Rescale the threshold to account for the label rescaling
            if threshold is not None:
                threshold = threshold / self.label_rescaling_factor
            loss = L1Loss(ignore_value=0, pre_calculation_function=remove_sub_track, lower_threshold=threshold)
        elif loss_name == 'l2':
            from losses.l2_loss import L2Loss
            loss = L2Loss(ignore_value=0, pre_calculation_function=remove_sub_track)
        elif loss_name == 'huber':
            from losses.huber_loss import HuberLoss
            loss = HuberLoss(ignore_value=0, pre_calculation_function=remove_sub_track, delta=3.0)
        loss = loss.to(device=self.device)
        return loss
    
    class EarlyStopping:
        def __init__(self, patience=10, min_delta=0.0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = float("inf")
            self.should_stop = False

        def step(self, val_loss):
            if val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.should_stop = True

    def get_visualization(self, viz_name: str, inputs, labels, outputs):
        assert viz_name in ['input_output', 'density_scatter_plot',
                            'boxplot'], f"Visualization {viz_name} not implemented."

        # Detach and copy the labels and outputs, then undo the rescaling
        labels, outputs = labels.detach().clone(), outputs.detach().clone()

        # Undo the rescaling
        labels, outputs = labels * self.label_rescaling_factor, outputs * self.label_rescaling_factor


        def remove_sub_track_vis(inputs, labels, outputs):
            return inputs, labels.sum(
                axis=1), outputs  # Same as remove_sub_track, but for visualization (i.e. has outputs as well)

        if viz_name == 'input_output':
            viz_fn = visualization.get_input_output_visualization(rgb_channels=[6, 5, 4],
                                                                  process_variables=remove_sub_track_vis)
        elif viz_name == 'density_scatter_plot':
            viz_fn = visualization.get_density_scatter_plot_visualization(ignore_value=0,
                                                                          process_variables=remove_sub_track_vis)
        elif viz_name == 'boxplot':
            viz_fn = visualization.get_visualization_boxplots(ignore_value=0, process_variables=remove_sub_track_vis)
        return viz_fn(inputs=inputs, labels=labels, outputs=outputs)

    def get_optimizer(self, initial_lr: float) -> torch.optim.Optimizer:
        """
        Returns the optimizer.
        :param initial_lr: The initial learning rate
        :type initial_lr: float
        :return: The optimizer.
        :rtype: torch.optim.Optimizer
        """
        wd = self.config['weight_decay'] or 0.
        optim_name = self.config.optim or 'AdamW'
        if optim_name == 'SGD':
            optimizer = torch.optim.SGD(params=self.model.parameters(), lr=initial_lr,
                                        momentum=0.9,
                                        weight_decay=wd,
                                        nesterov=wd > 0.)
        elif optim_name == 'AdamW':
            optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=initial_lr,
                                          weight_decay=wd)
        else:
            raise NotImplementedError

        return optimizer

    def save_model(self, model_identifier: str, sync: bool = False) -> str:
        """
        Saves the model's state_dict to a file.
        :param model_identifier: Name of the file type.
        :type model_identifier: str
        :param sync: Whether to sync the file to wandb.
        :type sync: bool
        :return: Path to the saved model.
        :rtype: str
        """
        fName = f"{model_identifier}_model.pt"
        fPath = os.path.join(self.tmp_dir, fName)

            # Handle wrapped models properly
        if isinstance(self.model, CoordInjectionModelWrapper):
            # Save the base model state dict
            try:
                if hasattr(self.model.base_model, 'module'):
                    model_state_dict = self.model.base_model.module.state_dict()
                else:
                    model_state_dict = self.model.base_model.state_dict()
            except AttributeError:
                model_state_dict = self.model.base_model.state_dict()
        else:
            # Original logic for non-wrapped models
            try:
                model_state_dict = self.model.module.state_dict()
            except AttributeError:
                model_state_dict = self.model.state_dict()

        torch.save(model_state_dict, fPath)  # Save the state_dict

        if sync:
            wandb.save(fPath)
        return fPath

    def log(self, step: int, phase_runtime: float):
        """
        Logs the current training status.
        :param phase_runtime: The wall-clock time of the current phase.
        :type phase_runtime: float
        """
        loggingDict = self.get_metrics()
        loggingDict.update({
            'phase_runtime': phase_runtime,
            'iteration': step,
            'samples_seen': step * self.config.batch_size,
        })

        # Log and push to Wandb
        for metric_type, val in loggingDict.items():
            wandb.run.summary[f"{metric_type}"] = val

        wandb.log(loggingDict)

    def define_optimizer_scheduler(self):
        # Define the optimizer
        initial_lr = self.config.initial_lr
        self.optimizer = self.get_optimizer(initial_lr=initial_lr)

        # We define a scheduler. All schedulers work on a per-iteration basis
        n_total_iterations = self.config.n_iterations
        n_lr_cycles = self.config.n_lr_cycles or 0
        cyclic_mode = self.config.cyclic_mode or 'triangular2'
        n_warmup_iterations = int(0.1 * n_total_iterations) if n_lr_cycles == 0 else 0

        # Set the initial learning rate
        for param_group in self.optimizer.param_groups: param_group['lr'] = initial_lr

        # Define the warmup scheduler if needed
        warmup_scheduler, milestone = None, None
        if n_warmup_iterations > 0:
            # As a start factor we use 1e-20, to avoid division by zero when putting 0.
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer,
                                                                 start_factor=1e-20, end_factor=1.,
                                                                 total_iters=n_warmup_iterations)
            milestone = n_warmup_iterations + 1

        n_remaining_iterations = n_total_iterations - n_warmup_iterations
        if n_lr_cycles > 0:
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=self.optimizer, base_lr=0.,
                                                            max_lr=initial_lr, step_size_up=n_remaining_iterations // (2*n_lr_cycles),
                                                            mode=cyclic_mode,
                                                            cycle_momentum=False)
        else:
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer,
                                                      start_factor=1.0, end_factor=0.,
                                                      total_iters=n_remaining_iterations)

        # Reset base lrs to make this work
        scheduler.base_lrs = [initial_lr if warmup_scheduler else 0. for _ in self.optimizer.param_groups]

        # Define the Sequential Scheduler
        if warmup_scheduler is None:
            self.scheduler = scheduler
        else:
            self.scheduler = SequentialSchedulers(optimizer=self.optimizer, schedulers=[warmup_scheduler, scheduler],
                                                  milestones=[milestone])

    @torch.no_grad()
    def eval(self, data: str):
        """
        Evaluates the model on the given data set.
        :param data: string indicating the data set to evaluate on. Can be 'train' or 'val'.
        :type data: str
        """
        sys.stdout.write(f"Evaluating on {data} split.\n")
        for step, batch_data in enumerate(tqdm(self.loader[data]), 1):

            if self.use_coord_encoding and self.coord_injection_mode == "feature_maps":
                x_input, y_target, batch_coords = batch_data
                batch_coords = batch_coords.to(self.device, non_blocking=True)
            else:
                x_input, y_target = batch_data
                batch_coords = None

            x_input = x_input.to(self.device, non_blocking=True)
            y_target = y_target.to(self.device, non_blocking=True)

              # Set coordinates for feature map injection
            if (self.use_coord_encoding and 
                self.coord_injection_mode == "feature_maps" and
                isinstance(self.model, CoordInjectionModelWrapper)):
                self.model.set_coordinates(batch_coords)

            with autocast(enabled=self.use_amp):
                output = self.model.eval()(x_input)
                loss = self.loss_criteria[self.loss_name](output, y_target)
                self.metrics[data]['loss'](value=loss, weight=len(y_target))
                for loss_type in self.loss_criteria.keys():
                    metric_loss = self.loss_criteria[loss_type](output, y_target)
                    # Check if the metric_loss is nan
                    if not torch.isnan(metric_loss):
                        self.metrics[data][loss_type](value=metric_loss, weight=len(y_target))

            if step <= 4:
                # Create the visualizations for the first 4 batches
                for viz_func in ['input_output', 'density_scatter_plot', 'boxplot']:
                    viz = self.get_visualization(viz_name=viz_func, inputs=x_input, labels=y_target, outputs=output)
                    wandb.log({data + '/' + viz_func + "_" + str(step): wandb.Image(viz)}, commit=False)

    """
    @torch.no_grad()
    def eval_fixval(self):
        """"""Creates the fixval plots and logs them to wandb.""""""
        sys.stdout.write(f"Creating fixval plots.\n")
        def remove_sub_track_vis_wout_labels(inputs, labels, outputs):
            return inputs, None, outputs  # Same as remove_sub_track, but for visualization (i.e. has outputs as well)

        viz_fn = visualization.get_input_output_visualization(rgb_channels=[6, 5, 4],
                                                                  process_variables=remove_sub_track_vis_wout_labels)
        loggingDict = dict()
        for x_input, fileNames in tqdm(self.loader['fix_val']):
            x_input = x_input.to(self.device, non_blocking=True)
            with autocast(enabled=self.use_amp):
                output = self.model.eval()(x_input)
            viz = viz_fn(inputs=x_input, labels=None, outputs=output)
            jointName = "__".join(fileNames)
            wandb.log({'fixval' + '/' + 'input_output' + '/' + jointName: wandb.Image(viz)}, commit=False)

            # Get the min and max prediction for each image
            flattened_output = output.flatten(start_dim=1)
            min_values = flattened_output.min(dim=1).values
            max_values = flattened_output.max(dim=1).values

            for i, (min_val, max_val) in enumerate(zip(min_values, max_values)):
                # Log the min and max values as numeric values with the appropriate filename
                name = fileNames[i]
                loggingDict[f"fixval/min_{name}"] = min_val.item()
                loggingDict[f"fixval/max_{name}"] = max_val.item()
        wandb.log(loggingDict, commit=False)
    """

    def train(self):
        """Modified training loop to handle coordinate injection."""
        log_freq, n_iterations = self.config.log_freq, self.config.n_iterations
        ampGradScaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.reset_averaged_metrics()
        phase_start = time.time()
        
        early_stopper = self.EarlyStopping(patience=10, min_delta=0.001)

        # Define the Stochastic Weight Averaging model
        swa_model = None
        if self.config.use_swa:
            # Get the base model for SWA (unwrap if needed)
            base_model_for_swa = self.model
            if isinstance(self.model, CoordInjectionModelWrapper):
                base_model_for_swa = self.model.base_model
            
            swa_model = torch.optim.swa_utils.AveragedModel(base_model_for_swa)
            swa_start = max(1, int(0.75 * n_iterations))
            sys.stdout.write(f"SWA model will be updated from iteration {swa_start} onwards.\n")

        # Define the distribution of mixup
        if self.config.use_mixup:
            alpha = 0.2
            beta_distribution = torch.distributions.beta.Beta(alpha, alpha)
            sys.stdout.write(f"Using mixup with alpha={alpha}. Example sample: {beta_distribution.sample()}\n")

        for step in tqdm(range(1, n_iterations + 1, 1)):
            # Reinitialize the train iterator if it reaches the end
            if step == 1 or (step - 1) % len(self.loader['train']) == 0:
                train_iterator = iter(self.loader['train'])

            # Handle different data formats based on coordinate injection mode
            if self.use_coord_encoding and self.coord_injection_mode == "feature_maps":
                x_input, y_target, batch_coords = next(train_iterator)
                batch_coords = batch_coords.to(device=self.device, non_blocking=True)
            else:
                x_input, y_target = next(train_iterator)
                batch_coords = None

            # Move to CUDA if possible
            x_input = x_input.to(device=self.device, non_blocking=True)
            y_target = y_target.to(device=self.device, non_blocking=True)

            if self.config.use_mixup:
                # Sample from the beta distribution with alpha=0.2
                with torch.no_grad():
                    lam = beta_distribution.sample().to(self.device)
                    index = torch.randperm(x_input.size(0)).to(self.device)
                    x_input = lam * x_input + (1 - lam) * x_input[index, :]
                    y_target = lam * y_target + (1 - lam) * y_target[index, :]
                    
                    # Also mix coordinates if using feature map injection
                    if batch_coords is not None:
                        batch_coords = lam * batch_coords + (1 - lam) * batch_coords[index, :]

            # Set coordinates for feature map injection
            if (self.use_coord_encoding and 
                self.coord_injection_mode == "feature_maps" and
                isinstance(self.model, CoordInjectionModelWrapper)):
                self.model.set_coordinates(batch_coords)

            self.optimizer.zero_grad()
            itStartTime = time.time()

            with autocast(enabled=self.use_amp):
                output = self.model.train()(x_input)
                loss = self.loss_criteria[self.loss_name](output, y_target)
                ampGradScaler.scale(loss).backward()
                ampGradScaler.unscale_(self.optimizer)
                if self.config.use_grad_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                ampGradScaler.step(self.optimizer)
                ampGradScaler.update()
                self.scheduler.step()
                self.metrics['train']['loss'](value=loss, weight=len(y_target))

                with torch.no_grad():
                    for loss_type in self.loss_criteria.keys():
                        metric_loss = self.loss_criteria[loss_type](output, y_target)
                        if not torch.isnan(metric_loss):
                            self.metrics['train'][loss_type](value=metric_loss, weight=len(y_target))
                    itEndTime = time.time()
                    n_img_in_iteration = int(self.config.batch_size)
                    ips = n_img_in_iteration / (itEndTime - itStartTime)
                    self.metrics['train']['ips_throughput'](ips)

            if swa_model is not None:
                if step >= swa_start:
                    # Update SWA with the base model
                    if isinstance(self.model, CoordInjectionModelWrapper):
                        swa_model.update_parameters(self.model.base_model)
                    else:
                        swa_model.update_parameters(self.model)
                        
                    if step == n_iterations:
                        sys.stdout.write(f"Last step reached. Setting model to SWA model.\n")
                        # Set the model to the SWA model by copying the parameters
                        with torch.no_grad():
                            if isinstance(self.model, CoordInjectionModelWrapper):
                                for param1, param2 in zip(self.model.base_model.parameters(), swa_model.parameters()):
                                    param1.copy_(param2)
                            else:
                                for param1, param2 in zip(self.model.parameters(), swa_model.parameters()):
                                    param1.copy_(param2)
                            swa_model = None

            if step % log_freq == 0 or step == n_iterations:
                phase_runtime = time.time() - phase_start
                # Create the visualizations
                for viz_func in ['input_output', 'density_scatter_plot', 'boxplot']:
                    viz = self.get_visualization(viz_name=viz_func, inputs=x_input, labels=y_target, outputs=output)
                    wandb.log({'train/' + viz_func: wandb.Image(viz)}, commit=False)

                # Evaluate the validation dataset
                if not self.debug:
                    self.eval(data='val')

                    # Check early stopping after validation
                    val_loss = self.metrics['val']['loss'].compute()
                    early_stopper.step(val_loss)
                    if early_stopper.should_stop:
                        print(f"Early stopping at iteration {step} with best val loss {early_stopper.best_loss:.4f}")
                        break

                self.log(step=step, phase_runtime=phase_runtime)
                self.reset_averaged_metrics()
                phase_start = time.time()
    

    def run(self):
        """Controls the execution of the script."""
        # We start training from scratch
        self.set_seed(seed=self.seed)  # Set the seed
        loaders = self.get_dataloaders()
        self.loader['train'], self.loader['val'] = loaders #, self.loader['fix_val']
        self.model = self.get_model(reinit=True, model_path=self.model_paths['initial'])  # Load the model

        self.define_optimizer_scheduler()  # This was moved before define_strategy to have the optimizer available

        self.train()  # Train the model
        # Save the trained model and upload it to wandb
        self.save_model(model_identifier='trained', sync=True)
