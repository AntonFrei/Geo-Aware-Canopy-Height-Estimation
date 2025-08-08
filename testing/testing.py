import os
import sys
import socket
import torch
import wandb
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from torchvision.transforms import transforms
import pandas as pd
from bootstrap_eval import bootstrap_ci

# --- Setup ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../training')))
from runner import Runner
from config import PreprocessedSatelliteDataset, means, stds, percentiles
from coord_injection_model import CoordInjectionModelWrapper

# --- Configuration ---
defaults = dict(
    # System
    seed=1,

    #Data
    dataset='icml_2024_global_rh100',
    batch_size=15,
    arch='unet',
    backbone='resnet50',
    use_pretrained_model=False,

    # Optimization
    optim='AdamW',
    loss_name='shift_huber',
    n_iterations=70,
    log_freq=10,
    initial_lr=1e-3,
    weight_decay=1e-2,
    use_standardization=True,          
    use_augmentation=False,
    use_label_rescaling=False,

    #Coordinates
    use_coord_encoding=True,                       
    coord_encoder='satclip',           # Options: raw, wrap, wrap_lon_only,fourier, satclip
    coord_injection_mode='feature_maps',      # Options: "input", "feature_maps"

    # Efficiency
    fp16=False,
    use_memmap=False,
    num_workers_per_gpu=8,

    # Other
    use_grad_clipping=True,
    use_weighted_sampler=False,
    use_weighting_quantile=10,
    use_mixup=False,
    use_swa=False,
    use_input_clipping=False,
    n_lr_cycles=0,
    cyclic_mode='triangular2',
    computer=socket.gethostname()
)
cfg = SimpleNamespace(**defaults)
cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


MODEL_PATH = "Updated-Global-Canopy-Height-Map-Coordinates/Global-Canopy-Height-Map/testing/model/tster_satclip_longrun/trained_model.pt"  # <------ CHANGE THIS!  
DATASET_ROOT = "/home/ubuntu/work/satellite_data/icml_2024_global_rh100/"
TEST_CSV = os.path.join(DATASET_ROOT, "test.csv")

wandb.init(project='test-000', name='model-test-eval', config=defaults)

# --- Transform builders ---
def build_input_transforms(cfg):
    transform_list = [transforms.ToTensor()]
    if cfg.use_standardization:
        mean, std = means[cfg.dataset], stds[cfg.dataset]
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    elif cfg.use_input_clipping not in [False, None, 'None']:
        from torch import tensor
        clip = int(cfg.use_input_clipping)
        lower = tensor(percentiles[cfg.dataset][clip]).view(-1, 1, 1)
        upper = tensor(percentiles[cfg.dataset][100 - clip]).view(-1, 1, 1)
        transform_list.append(transforms.Lambda(lambda x: torch.clamp(x, min=lower, max=upper)))
    return transforms.Compose(transform_list)

def build_label_transforms(cfg):
    if cfg.use_label_rescaling:
        return transforms.Compose([transforms.ToTensor(), lambda x: x * (1./60.)])
    return transforms.ToTensor()

# --- Custom collate function for coordinate handling ---
def collate_fn(batch):
    if cfg.use_coord_encoding and cfg.coord_injection_mode == "feature_maps":
        # batch contains (image, label, coordinates) tuples
        images, labels, coords = zip(*batch)
        images = torch.stack(images)
        labels = torch.stack(labels)
        coords = torch.stack(coords)
        return images, labels, coords
    else:
        # Standard collation for (image, label) tuples
        return torch.utils.data.dataloader.default_collate(batch)

# --- Dataset and loader ---
test_dataset = PreprocessedSatelliteDataset(
    data_path=DATASET_ROOT,
    dataframe=TEST_CSV,
    image_transforms=build_input_transforms(cfg),
    label_transforms=build_label_transforms(cfg),
    use_weighted_sampler=False,
    use_memmap=cfg.use_memmap,
    remove_corrupt=True,
    load_labels=True,
    use_coord_encoding=cfg.use_coord_encoding,
    coord_encoder=cfg.coord_encoder,
    coord_injection_mode=cfg.coord_injection_mode 
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    num_workers=cfg.num_workers_per_gpu,
    collate_fn=collate_fn 
)

# --- Model ---
runner = Runner(config=cfg, tmp_dir=".", debug=False)
model = runner.get_model(reinit=True, model_path=MODEL_PATH)
model.eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# --- Losses ---
loss_names = ['shift_l1', 'shift_l2', 'shift_huber', 'l1', 'l2', 'huber'] + [f"l1_{t}" for t in [5, 10, 15, 20, 25, 30]]
loss_fns = {name: runner.get_loss(loss_name='l1', threshold=int(name.split('_')[1])) if name.startswith('l1_')
            else runner.get_loss(name) for name in loss_names}
all_metrics = {k: [] for k in loss_names}

# --- Saving mean-patch-prediction for map ---
os.makedirs("/home/ubuntu/work/saved_data/map_tables/", exist_ok=True)
prediction_records = []

# Load the metadata CSV for test set
test_df = pd.read_csv(TEST_CSV)

# --- Evaluation loop ---
all_preds = [] # 4bootstrap
all_targets = [] # 4bootstrap

with torch.no_grad():
    for i, batch_data in enumerate(tqdm(test_loader, desc="Testing")):
        
        # Handle different data formats based on coordinate injection mode
        if cfg.use_coord_encoding and cfg.coord_injection_mode == "feature_maps":
            x, y, batch_coords = batch_data
            batch_coords = batch_coords.cuda()
        else:
            x, y = batch_data
            batch_coords = None
        
        x, y = x.cuda(), y.cuda()
        
        # Set coordinates for feature map injection
        if (cfg.use_coord_encoding and 
            cfg.coord_injection_mode == "feature_maps" and
            isinstance(model, CoordInjectionModelWrapper)):
            model.set_coordinates(batch_coords)
        
        preds = model(x)

        all_preds.append(preds.cpu()) # 4bootstrap
        all_targets.append(y.cpu()) # 4bootstrap

        # --- Log metrics ---
        for name, fn in loss_fns.items():
            all_metrics[name].append(fn(preds, y).item())

        # --- Extract mean predictions per patch (batch-wise) ---
        mean_preds = preds.mean(dim=[1, 2, 3]).cpu().numpy()

        # --- Collect lat/lon and prediction ---
        for j in range(len(mean_preds)):
            sample_idx = i * cfg.batch_size + j
            if sample_idx >= len(test_df):
                break

            row = test_df.iloc[sample_idx]
            prediction_records.append({
                "lat": row["latitudes"],
                "lon": row["longitudes"],
                "mean_prediction": mean_preds[j]
            })
        
        # Clear GPU cache periodically
        if i % 50 == 0:
            torch.cuda.empty_cache()
print("Evaluation loop complete. Processing bootstrap...")

# 4bootstrap
mean_metrics = {k: float(np.nanmean(v)) for k, v in all_metrics.items()}
print("\n=== Regular Test Results (batch-wise averaging) ===")
for k, v in mean_metrics.items():
    print(f"{k}: {v:.4f}")

metrics_to_bootstrap = ['shift_l1', 'shift_l2', 'l1_30']

# Compute and print bootstrapped confidence intervals
print("\n--- Bootstrap Confidence Intervals (Selected Metrics)  ---")
bootstrap_results = bootstrap_ci(
    all_preds=all_preds,
    all_targets=all_targets, 
    loss_fns=loss_fns,
    metrics_to_bootstrap=metrics_to_bootstrap,
    n_boot=1000,
    device=cfg.device
)

# Log bootstrap results
for metric_name, results in bootstrap_results.items():
    wandb.log({
        f"CI/{metric_name}_mean": results['mean'],
        f"CI/{metric_name}_low": results['ci_low'],
        f"CI/{metric_name}_high": results['ci_high']
    })


# --- Save to CSV ---
'''
df_preds = pd.DataFrame(prediction_records)
csv_path = "/home/ubuntu/work/saved_data/map_tables/a_tster_upd-std_raw_feature_maps.csv"  # <---- CHANGE THIS
df_preds.to_csv(csv_path, index=False)
print(f"Saved mean predictions to: {csv_path}")
'''
# --- Results ---
mean_metrics = {k: float(np.nanmean(v)) for k, v in all_metrics.items()}
wandb.log({f"test/{k}": v for k, v in mean_metrics.items()})
wandb.finish()

print("\n=== Test Results ===")
for k, v in mean_metrics.items():
    print(f"{k}: {v:.4f}")

print(f"\nModel type: {type(model)}")
print(f"Coordinate injection mode: {cfg.coord_injection_mode}")
print(f"Coordinate encoder: {cfg.coord_encoder}")
if isinstance(model, CoordInjectionModelWrapper):
    print(f"Coordinate channels: {model.coord_channels}")
    print(f"Original decoder channels: {model.original_in_channels}")
    print(f"Final layer input channels: {model.original_in_channels + model.coord_channels}")