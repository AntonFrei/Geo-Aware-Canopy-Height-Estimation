import numpy as np
import torch
from tqdm import tqdm

def bootstrap_ci(all_preds, all_targets, loss_fns, metrics_to_bootstrap, n_boot=1000, device="cuda"):
    """
    Efficient bootstrap implementation that pre-computes all batch losses once.
    Handles NaN values for batches with no >30.
    Args:
        all_preds: List of prediction tensors from each batch
        all_targets: List of target tensors from each batch  
        loss_fns: Dictionary of loss functions from runner
        metrics_to_bootstrap: List of metric names to compute bootstrap for
        n_boot: Number of bootstrap samples
        device: Device to use
        
    Returns:
        Dictionary with bootstrap results
    """
    print(f"\nStarting efficient bootstrap with {n_boot} resamples...")
    
    # PRE-COMPUTE: Calculate loss for each batch once
    print("Pre-computing losses for all batches...")
    n_batches = len(all_preds)
    batch_losses = {metric: [] for metric in metrics_to_bootstrap}
    
    for batch_idx, (preds_batch, targets_batch) in enumerate(tqdm(zip(all_preds, all_targets), 
                                                                  desc="Computing batch losses", 
                                                                  total=n_batches)):
        preds_batch = preds_batch.to(device)
        targets_batch = targets_batch.to(device)
        
        for metric_name in metrics_to_bootstrap:
            if metric_name in loss_fns:
                try:
                    with torch.no_grad():
                        loss_val = loss_fns[metric_name](preds_batch, targets_batch).item()
                        # Check for NaN values
                        if np.isnan(loss_val) or np.isinf(loss_val):
                            print(f"Warning: NaN/Inf loss for {metric_name} in batch {batch_idx}")
                            loss_val = np.nan
                        batch_losses[metric_name].append(loss_val)
                except Exception as e:
                    print(f"Error computing {metric_name} for batch {batch_idx}: {e}")
                    batch_losses[metric_name].append(np.nan)
            else:
                print(f"Warning: {metric_name} not found in loss_fns")
                batch_losses[metric_name].append(np.nan)
        
        # Clear cache periodically
        if batch_idx % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Convert to numpy arrays and check for valid data
    valid_metrics = []
    for metric_name in metrics_to_bootstrap:
        batch_losses[metric_name] = np.array(batch_losses[metric_name])
        n_valid = np.sum(~np.isnan(batch_losses[metric_name]))
        print(f"{metric_name}: {n_valid}/{n_batches} valid batches")
        if n_valid > 0:
            valid_metrics.append(metric_name)
        else:
            print(f"Warning: No valid losses for {metric_name} - skipping bootstrap")
            
    
    if not valid_metrics:
        print("Error: No valid metrics for bootstrap!")
        return {}
    
    # Calculate original metrics (average across all valid batches)
    original_metrics = {}
    for metric_name in valid_metrics:
        # Use nanmean to handle NaN values
        original_metrics[metric_name] = np.nanmean(batch_losses[metric_name])
        print(f"Original {metric_name}: {original_metrics[metric_name]:.4f}")
    
    # BOOTSTRAP: Resample the pre-computed losses
    print(f"Running bootstrap resampling for {len(valid_metrics)} valid metrics...")
    bootstrap_results = {metric: [] for metric in valid_metrics}
    
    for boot_i in tqdm(range(n_boot), desc="Bootstrap samples"):
        # Sample batch indices with replacement
        batch_indices = np.random.choice(n_batches, size=n_batches, replace=True)
        
        # For each metric, average the selected batch losses
        for metric_name in valid_metrics:
            # This is just array indexing - very fast!
            selected_losses = batch_losses[metric_name][batch_indices]
            # Use nanmean to handle any NaN values in the bootstrap sample
            bootstrap_avg = np.nanmean(selected_losses)
            bootstrap_results[metric_name].append(bootstrap_avg)
    
    # Compute confidence intervals
    results = {}
    for metric_name in valid_metrics:
        boot_values = np.array(bootstrap_results[metric_name])
        # Remove any NaN bootstrap samples
        boot_values_clean = boot_values[~np.isnan(boot_values)]
        
        if len(boot_values_clean) < n_boot * 0.5:  # Less than 50% valid samples
            print(f"Warning: Only {len(boot_values_clean)}/{n_boot} valid bootstrap samples for {metric_name}")
        
        if len(boot_values_clean) > 10:  # Need at least 10 samples for CI
            ci_low = np.percentile(boot_values_clean, 2.5)
            ci_high = np.percentile(boot_values_clean, 97.5)
            bootstrap_std = np.std(boot_values_clean)
        else:
            print(f"Error: Too few valid bootstrap samples for {metric_name}")
            ci_low = ci_high = bootstrap_std = np.nan
        
        results[metric_name] = {
            'mean': original_metrics[metric_name],
            'ci_low': ci_low,
            'ci_high': ci_high,
            'bootstrap_std': bootstrap_std,
            'n_valid_samples': len(boot_values_clean)
        }
        
        print(f"{metric_name.upper()} - Mean: {results[metric_name]['mean']:.4f}, "
              f"95% CI: [{ci_low:.4f}, {ci_high:.4f}], "
              f"Bootstrap Std: {bootstrap_std:.4f}, "
              f"Valid samples: {len(boot_values_clean)}/{n_boot}")
    
    return results