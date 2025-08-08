from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import glob
import os
import torch
import numpy as np
import pandas as pd
import pdb
from torch.utils.data.dataloader import default_collate
import sys
import re
from pyproj import Transformer
from scipy.special import lpmv
import math
from encoder import (
    transform_coordinates,
    extract_epsg,
    ENCODER_MAP
)

means = {
    'icml_2024_global_rh100': (7377.7822, 8270.8516, 5199.0840, 4672.5337,  736.6648, 1055.1697,
        1316.8953, 1647.3466, 2188.7932, 2410.7446, 2469.4700, 2569.3171,
        2586.2542, 2025.0236),    
} # values for icml_2024_global_rh100

stds = {
    'icml_2024_global_rh100': (845.4938, 897.9979, 929.9027, 874.2733, 177.0846, 212.3855, 280.5904,
        280.2600, 336.5061, 374.1639, 398.3364, 390.1537, 372.1020, 340.5369),  
}   # values for icml_2024_global_rh100

percentiles = {
    'icml_2024_global_rh100': {
        1: (-8823.0, -8060.0, -13725.0, -13814.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        2: (-7813.0, -7198.0, -12623.0, -12896.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        5: (-6358.0, -6088.0, -11297.0, -11662.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        95: (26007.0, 25361.0, 21409.0, 22433.0, 16201.0, 16068.0, 15912.0, 15808.0, 15794.0, 15766.0, 15751.0, 15702.0, 12399.0, 13293.0),
        98: (27466.0, 26870.0, 22726.0, 24060.0, 16451.0, 16151.0, 15944.0, 15867.0, 15820.0, 15800.0, 15840.0, 15726.0, 13560.0, 14389.0),
        99: (28442.0, 28040.0, 23937.0, 25203.0, 16544.0, 16200.0, 16120.0, 15988.0, 15870.0, 15976.0, 16000.0, 15811.0, 13856.0, 15094.0),
    }  # values for icml_2024_global_rh100
}

class FixValDataset(Dataset):
    """
    Dataset class to load the fixval dataset.
    """
    def __init__(self, data_path, dataframe, image_transforms=None, use_coord_encoding=False, coord_encoder='raw'):
        self.data_path = data_path
        self.df = pd.read_csv(dataframe, index_col=False)
        self.files = list(self.df["paths"].apply(lambda x: os.path.join(data_path, x)))
        self.image_transforms = image_transforms
        self.use_coord_encoding = use_coord_encoding
        self.coord_encoder = coord_encoder

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index].replace(r"'", "")
        fileName = file[file.rfind('data_')+5: file.rfind('.npz')]
        data = np.load(file)

        image = data["data"].astype(np.float32)
        # Move the channel axis to the last position (required for torchvision transforms)
        image = np.moveaxis(image, 0, -1)
        if self.image_transforms:
            image = self.image_transforms(image)

        return image, fileName



class PreprocessedSatelliteDataset(Dataset):
    """
    Dataset class for preprocessed satellite imagery.
    """

    def __init__(self, data_path, dataframe=None, image_transforms=None, label_transforms=None, use_coord_encoding=False, coord_encoder: str = "raw", coord_injection_mode="input", joint_transforms=None, use_weighted_sampler=False,
                  use_weighting_quantile=None, use_memmap=False, remove_corrupt=True, load_labels=True, patch_size=512):
        
        self.use_memmap = use_memmap
        self.patch_size = patch_size
        self.load_labels = load_labels  # If False, we only load the images and not the labels
        self.df = pd.read_csv(dataframe)
        """
        if remove_corrupt:
            old_len = len(df)
            #df = df[df["missing_s2_flag"] == False] # Use only the rows that are not corrupt, i.e. those where df["missing_s2_flag"] == False

            # Use only the rows that are not corrupt, i.e. those where df["has_corrupt_s2_channel_flag"] == False
            df = df[df["has_corrupt_s2_channel_flag"] == False]
            sys.stdout.write(f"Removed {old_len - len(df)} corrupt rows.\n")
        """
        self.files = list(self.df["paths"].apply(lambda x: os.path.join(data_path, x)))
        
        self.use_coord_encoding = use_coord_encoding
        self.coord_encoder = coord_encoder
        self.coord_injection_mode = coord_injection_mode


        if use_weighted_sampler not in [False, None]:
            assert use_weighted_sampler in ['g5', 'g10', 'g15', 'g20', 'g25', 'g30']
            weighting_quantile = use_weighting_quantile
            assert weighting_quantile in [None, 'None'] or int(weighting_quantile) == weighting_quantile, "weighting_quantile must be an integer."
            if weighting_quantile in [None, 'None']:
                self.weights = (self.df[use_weighted_sampler] / self.df["totals"]).values.clip(0., 1.)
            else:
                # We do not clip between 0 and 1, but rather between the weighting_quantile and 1.
                weighting_quantile = float(weighting_quantile)
                self.weights = (self.df[use_weighted_sampler] / self.df["totals"]).values

                # Compute the quantiles, ignoring nan values and zero values
                tmp_weights = self.weights.copy()
                tmp_weights[np.isnan(tmp_weights)] = 0.
                tmp_weights = tmp_weights[tmp_weights > 0.]

                quantile_min = np.nanquantile(tmp_weights, weighting_quantile / 100)
                sys.stdout.write(f"Computed weighting {weighting_quantile}-quantile-lower bound: {quantile_min}.\n")

                # Clip the weights
                self.weights = self.weights.clip(quantile_min, 1.0)

            # Set the nan values to 0.
            self.weights[np.isnan(self.weights)] = 0.

        else:
            self.weights = None
        self.image_transforms, self.label_transforms, self.joint_transforms = image_transforms, label_transforms, joint_transforms

    def __len__(self):
        return len(self.files)

        
    def __getitem__(self, index):
        if self.use_memmap:
            image, label = self.getitem_memmap(index)
        else:
            image, label = self.getitem_classic(index)
           
        if self.use_coord_encoding:
            filename = self.df.iloc[index]['paths']
            utm_x = self.df.iloc[index]['longitudes']
            utm_y = self.df.iloc[index]['latitudes']
            epsg = extract_epsg(filename)
            lat, lon = transform_coordinates(utm_x, utm_y, src_epsg=epsg, dst_epsg=4326)

            encoder_fn = ENCODER_MAP.get(getattr(self, "coord_encoder")) #, "raw"
            coord_vec = encoder_fn(lat, lon)

            if self.coord_injection_mode == "input":
                # Original behavior: concatenate to input
                H, W = image.shape[1:]
                coord_tensor = torch.tensor(coord_vec).view(-1, 1, 1).repeat(1, H, W)
                image = torch.cat([image, coord_tensor], dim=0)
                return image, label
            elif self.coord_injection_mode == "feature_maps":
                # New behavior: return coordinates separately
                return image, label, torch.tensor(coord_vec, dtype=torch.float32)
        return image, label
    
        

    def getitem_memmap(self, index):
        file = self.files[index]
        with np.load(file, mmap_mode='r') as npz_file:
            image = npz_file['data'].astype(np.float32)
            # Move the channel axis to the last position (required for torchvision transforms)
            image = np.moveaxis(image, 0, -1)
            if self.image_transforms:
                image = self.image_transforms(image)
            if self.load_labels:
                label = npz_file['labels'].astype(np.float32)

                # Process label
                label = label[:3]  # Everything after index/granule 3 is irrelevant
                label = label / 100  # Convert from cm to m
                label = np.moveaxis(label, 0, -1)

                if self.label_transforms:
                    label = self.label_transforms(label)
                if self.joint_transforms:
                    image, label = self.joint_transforms(image, label)
                return image, label

        return image

    def getitem_classic(self, index):
        file = self.files[index]
        data = np.load(file)

        image = data["data"].astype(np.float32)
        # Move the channel axis to the last position (required for torchvision transforms)
        image = np.moveaxis(image, 0, -1)[:self.patch_size,:self.patch_size]
        if self.image_transforms:
            image = self.image_transforms(image)
        if self.load_labels:
            label = data["labels"].astype(np.float32)

            # Process label
            label = label[:3]  # Everything after index 3 is irrelevant
            label = label[:,:self.patch_size, :self.patch_size]
            label = label / 100  # Convert from cm to m
            label = np.moveaxis(label, 0, -1)

            if self.label_transforms:
                label = self.label_transforms(label)
            if self.joint_transforms:
                image, label = self.joint_transforms(image, label)
            return image, label

        return image

