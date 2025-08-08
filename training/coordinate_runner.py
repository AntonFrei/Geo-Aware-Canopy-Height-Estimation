"""
Coordinate-based prediction runner using FcNet architecture
Standalone implementation for testing coordinate encoding effectiveness
"""

import os
import sys
import time
import socket
import tempfile
import shutil
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import wandb
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from tqdm.auto import tqdm
from torchmetrics import MeanMetric

# Import from your existing files
from config import PreprocessedSatelliteDataset
from config import means as meanDict, stds as stdDict, percentiles as percentileDict
from encoder import ENCODER_MAP, transform_coordinates, extract_epsg
from utilities import GeneralUtility

class FcNet(nn.Module):
    """Simple fully connected network for coordinate-based prediction"""
    
    def __init__(self, input_dim=2, hidden_dim=512, num_layers=4, output_dim=1, dropout=0.1):
        super(FcNet, self).__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, coordinates):
        return self.network(coordinates)

class CoordinateRunner:
    """Runner for coordinate-only predictions"""
    
    def __init__(self, config, tmp_dir: str, debug: bool = False):
        self.config = config
        self.debug = debug
        self.tmp_dir = tmp_dir
        
        # Device setup
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Coordinate encoding setup
        self.coord_encoder = getattr(config, 'coord_encoder', 'raw')
        self.encoder_fn = ENCODER_MAP[self.coord_encoder]
        coord_dim = self.encoder_fn.num_output_channels
        
        # Model setup
        self.model = FcNet(
            input_dim=coord_dim,
            hidden_dim=getattr(config, 'hidden_dim', 512),
            num_layers=getattr(config, 'num_layers', 4),
            output_dim=1,
            dropout=getattr(config, 'dropout', 0.1)
        ).to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=getattr(config, 'lr', 1e-3),
            weight_decay=getattr(config, 'weight_decay', 1e-4)
        )
        
        # Metrics
        self.metrics = {
            'train': {'loss': MeanMetric().to(self.device)},
            'val': {'loss': MeanMetric().to(self.device)},
            'test': {'loss': MeanMetric().to(self.device)}
        }
        
        # Data loaders
        self.loaders = {}
        
        # Label rescaling (same as your original runner)
        self.label_rescaling_factor = 60. if getattr(config, 'use_label_rescaling', False) else 1.
        
    def get_dataset_root(self, dataset_name: str) -> str:
        """Get dataset root path"""
        rootPath = f"/home/ubuntu/work/satellite_data/{dataset_name}/"
        if not os.path.isdir(rootPath):
            raise FileNotFoundError(f"Dataset path does not exist: {rootPath}")
        return rootPath
    
    def build_transforms(self):
        """Build data transforms (same as your original runner)"""
        base_transform = transforms.ToTensor()
        transforms_list = [base_transform]
        
        if getattr(self.config, 'use_standardization', False):
            mean, std = meanDict[self.config.dataset], stdDict[self.config.dataset]
            transforms_list.append(transforms.Normalize(mean=mean, std=std))
        elif getattr(self.config, 'use_input_clipping', False):
            clip = int(self.config.use_input_clipping)
            lower = torch.tensor(percentileDict[self.config.dataset][clip]).view(-1, 1, 1)
            upper = torch.tensor(percentileDict[self.config.dataset][100 - clip]).view(-1, 1, 1)
            transforms_list.append(transforms.Lambda(lambda x: torch.clamp(x, min=lower, max=upper)))
        
        image_transform = transforms.Compose(transforms_list)
        label_transform = transforms.Compose([
            base_transform, 
            lambda x: x * (1. / self.label_rescaling_factor)
        ])
        
        return image_transform, label_transform
    
    def get_dataloaders(self):
        """Create train/val/test dataloaders with 80/10/10 split"""
        rootPath = self.get_dataset_root(self.config.dataset)
        print(f"Loading {self.config.dataset} dataset from {rootPath}.")
        
        # Use the same CSV files as your original runner
        train_csv = os.path.join(rootPath, 'train.csv')
        val_csv = os.path.join(rootPath, 'val.csv')
        
        image_transform, label_transform = self.build_transforms()
        
        # Load full training dataset
        full_train_dataset = PreprocessedSatelliteDataset(
            data_path=rootPath,
            dataframe=train_csv,
            image_transforms=image_transform,
            label_transforms=label_transform,
            use_coord_encoding=False,  # We'll handle coordinates separately
            use_memmap=getattr(self.config, 'use_memmap', False),
            remove_corrupt=not self.debug
        )
        
        # Load validation dataset (will be split into val/test)
        full_val_dataset = PreprocessedSatelliteDataset(
            data_path=rootPath,
            dataframe=val_csv,
            image_transforms=image_transform,
            label_transforms=label_transform,
            use_coord_encoding=False,
            use_memmap=getattr(self.config, 'use_memmap', False),
            remove_corrupt=not self.debug
        )
        
        # Split validation dataset into val (50%) and test (50%) to get 10/10 split overall
        val_size = len(full_val_dataset) // 2
        test_size = len(full_val_dataset) - val_size
        
        generator = torch.Generator().manual_seed(getattr(self.config, 'seed', 42))
        val_dataset, test_dataset = random_split(
            full_val_dataset, 
            [val_size, test_size], 
            generator=generator
        )
        
        print(f"Dataset splits - Train: {len(full_train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        # Create data loaders
        batch_size = getattr(self.config, 'batch_size', 32)
        num_workers = getattr(self.config, 'num_workers', 4)
        
        self.loaders['train'] = DataLoader(
            full_train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.loaders['val'] = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.loaders['test'] = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
    
    def get_coordinates_for_batch(self, dataset, indices):
        """Extract coordinates for a batch of samples"""
        # Get base dataset (handle random_split wrapper)
        base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
        
        coord_list = []
        for idx in indices:
            # Handle random_split indices
            if hasattr(dataset, 'indices'):
                actual_idx = dataset.indices[idx]
            else:
                actual_idx = idx
                
            row = base_dataset.df.iloc[actual_idx]
            utm_x = row['longitudes']
            utm_y = row['latitudes']
            epsg = extract_epsg(row['paths'])
            lat, lon = transform_coordinates(utm_x, utm_y, src_epsg=epsg, dst_epsg=4326)
            coord_vec = self.encoder_fn(lat, lon)
            coord_list.append(coord_vec)
        
        # Convert to numpy array first, then to tensor (much faster)
        coord_array = np.array(coord_list, dtype=np.float32)
        return torch.from_numpy(coord_array).to(self.device)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        for batch_idx, (images, targets) in enumerate(tqdm(self.loaders['train'], desc="Training")):
            targets = targets.to(self.device)
            
            # Get coordinates - use same improved approach as eval
            batch_size = targets.shape[0]
            coord_list = []
            
            dataset = self.loaders['train'].dataset
            base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
            
            for i in range(batch_size):
                idx = batch_idx * self.config.batch_size + i
                if hasattr(dataset, 'indices'):
                    if idx < len(dataset.indices):
                        actual_idx = dataset.indices[idx]
                    else:
                        actual_idx = dataset.indices[0]  # Fallback
                else:
                    actual_idx = idx % len(base_dataset)
                
                row = base_dataset.df.iloc[actual_idx]
                utm_x = row['longitudes']
                utm_y = row['latitudes']
                epsg = extract_epsg(row['paths'])
                lat, lon = transform_coordinates(utm_x, utm_y, src_epsg=epsg, dst_epsg=4326)
                coord_vec = self.encoder_fn(lat, lon)
                coord_list.append(coord_vec)
            
            coord_array = np.array(coord_list, dtype=np.float32)
            coordinates = torch.from_numpy(coord_array).to(self.device)
            
            # Convert spatial targets to per-image targets (mean over spatial dimensions)
            target_means = targets.mean(dim=[1, 2, 3])  # [B]
            
            # Forward pass
            self.optimizer.zero_grad()
            pred = self.model(coordinates).squeeze()  # [B]
            
            # Handle single sample case
            if pred.dim() == 0:
                pred = pred.unsqueeze(0)
            if target_means.dim() == 0:
                target_means = target_means.unsqueeze(0)
            
            # Loss and backward
            loss = self.criterion(pred, target_means)
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            self.metrics['train']['loss'](loss.item())
    
    @torch.no_grad()
    def evaluate(self, split='val'):
        """Evaluate on validation or test set"""
        self.model.eval()
        
        for batch_idx, (images, targets) in enumerate(tqdm(self.loaders[split], desc=f"Evaluating {split}")):
            targets = targets.to(self.device)
            
            # DEBUG: Print shapes and values
            if batch_idx < 3:  # Only debug first few batches
                print(f"\\nBatch {batch_idx}: targets.shape = {targets.shape}")
                print(f"Target mean range: {targets.mean(dim=[1,2,3]).min():.4f} to {targets.mean(dim=[1,2,3]).max():.4f}")
            
            # Get coordinates - improved approach
            batch_size = targets.shape[0]
            coord_list = []
            
            dataset = self.loaders[split].dataset
            base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
            
            for i in range(batch_size):
                # More robust index calculation
                idx = batch_idx * self.config.batch_size + i
                if hasattr(dataset, 'indices'):
                    # Handle random_split indices properly
                    if idx < len(dataset.indices):
                        actual_idx = dataset.indices[idx]
                    else:
                        actual_idx = dataset.indices[0]  # Fallback
                else:
                    actual_idx = idx % len(base_dataset)
                
                row = base_dataset.df.iloc[actual_idx]
                utm_x = row['longitudes']
                utm_y = row['latitudes']
                epsg = extract_epsg(row['paths'])
                lat, lon = transform_coordinates(utm_x, utm_y, src_epsg=epsg, dst_epsg=4326)
                coord_vec = self.encoder_fn(lat, lon)
                coord_list.append(coord_vec)
            
            coord_array = np.array(coord_list, dtype=np.float32)
            coordinates = torch.from_numpy(coord_array).to(self.device)
            
            # Convert spatial targets to per-image targets
            target_means = targets.mean(dim=[1, 2, 3])
            
            # Forward pass
            pred = self.model(coordinates).squeeze()
            
            # Handle single sample case
            if pred.dim() == 0:
                pred = pred.unsqueeze(0)
            if target_means.dim() == 0:
                target_means = target_means.unsqueeze(0)
            
            loss = self.criterion(pred, target_means)
            
            # DEBUG: Print predictions vs targets
            if batch_idx < 3:  # Only debug first few batches
                print(f"Pred range: {pred.min():.4f} to {pred.max():.4f}")
                print(f"Target means: {target_means[:3]}")  # First 3 targets
                print(f"Predictions: {pred[:3]}")  # First 3 predictions
                print(f"Loss: {loss.item():.6f}")
            
            # Update metrics
            self.metrics[split]['loss'](loss.item())
    
    def train(self):
        """Main training loop"""
        n_epochs = getattr(self.config, 'n_epochs', 50)
        best_val_loss = float('inf')
        
        for epoch in range(n_epochs):
            print(f"\\nEpoch {epoch + 1}/{n_epochs}")
            
            # Reset metrics
            for split in ['train', 'val']:  # Only reset train/val, not test
                for metric in self.metrics[split].values():
                    metric.reset()
            
            # Train
            self.train_epoch()
            
            # Evaluate
            self.evaluate('val')
            
            # Log metrics
            train_loss = self.metrics['train']['loss'].compute()
            val_loss = self.metrics['val']['loss'].compute()
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best_coordinate_model.pt')
                print(f"New best model saved! Val Loss: {val_loss:.4f}")
            
            # Log to wandb
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'val/loss': val_loss,
                'best_val_loss': best_val_loss
            })
        
        # Final test evaluation (only if enabled)
        if getattr(self.config, 'run_final_test', False):
            print("\\n" + "="*50)
            print("RUNNING FINAL TEST EVALUATION")
            print("="*50)
            self.metrics['test']['loss'].reset()
            self.evaluate('test')
            test_loss = self.metrics['test']['loss'].compute()
            print(f"Final Test Loss: {test_loss:.4f}")
            wandb.log({'final_test/loss': test_loss})
        else:
            print("\\nSkipping test evaluation (run_final_test=False)")
            print("Set run_final_test=True when ready for final evaluation")
    
    def save_model(self, filename='coordinate_model.pt'):
        """Save model"""
        filepath = os.path.join(self.tmp_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': dict(self.config),
            'coord_encoder': self.coord_encoder,
            'model_architecture': {
                'input_dim': self.encoder_fn.num_output_channels,
                'hidden_dim': getattr(self.config, 'hidden_dim', 512),
                'num_layers': getattr(self.config, 'num_layers', 4),
                'dropout': getattr(self.config, 'dropout', 0.1)
            }
        }, filepath)
        wandb.save(filepath)
        print(f"Model saved to: {filepath}")
        return filepath
    
    def load_model(self, filepath):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from: {filepath}")
        return checkpoint
    
    def run(self):
        """Main run function"""
        print("Setting up coordinate-only prediction...")
        
        # Setup data
        self.get_dataloaders()
        
        # Train
        self.train()
        
        # Save model
        self.save_model()
        
        print("Coordinate-only training completed!")

# Configuration and main execution
@contextmanager
def tempdir():
    """Same tempdir context as your original runner"""
    try:
        path = tempfile.mkdtemp()
        yield path
    finally:
        try:
            shutil.rmtree(path)
        except IOError:
            pass

def main():
    # Configuration (similar to your main.py)
    defaults = dict(
        # System
        seed=42,
        
        # Data
        dataset='icml_2024_global_rh100',
        batch_size=20,  # Larger batch size since no images processed
        
        # Model
        hidden_dim=512,
        num_layers=4,
        dropout=0.1,
        
        # Training
        n_epochs=5,
        lr=1e-3,
        weight_decay=1e-4,
        
        # Data processing
        use_standardization=False,
        use_label_rescaling=False,
        use_input_clipping=False,
        use_memmap=False,
        num_workers=8,
        
        # Coordinates
        coord_encoder="raw",  # Change this to test different encoders
        
        # Evaluation
        run_final_test=False,  # Set to True only for final evaluation
        
        # Other
        computer=socket.gethostname()
    )
    
    # Initialize wandb
    wandb.init(
        project='coordinate-prediction',
        name=f'fcnet-{defaults["coord_encoder"]}',
        config=defaults
    )
    
    config = wandb.config
    config = GeneralUtility.update_config_with_default(config, defaults)
    
    # Run training
    with tempdir() as tmp_dir:
        runner = CoordinateRunner(config=config, tmp_dir=tmp_dir, debug=False)
        runner.run()
    
    wandb.finish()

if __name__ == "__main__":
    main()