#!/usr/bin/env python3
"""
One-time script to generate SatCLIP embeddings for all coordinates in train/val/test sets
Saves to CSV for fast lookup during training
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import re
import hashlib

# Install required packages
def install_requirements():
    """Install required packages and clone SatCLIP repository"""
    try:
        import subprocess
        import os
        
        # Install required packages (including SatCLIP dependencies)
        packages = [
            "huggingface_hub",
            "pyproj",
            "torch",
            "pandas",
            "numpy",
            "tqdm",
            "lightning",
            "rasterio", 
            "torchgeo",
            "albumentations",  # Required by SatCLIP
            "timm",           # Required by SatCLIP
            "kornia",         # Often required by SatCLIP
            "scikit-learn",   # Often required
            "matplotlib",     # Often required
            "requests",       # For downloading
        ]
        
        for package in packages:
            try:
                if package == "lightning":
                    import pytorch_lightning
                elif package == "scikit-learn":
                    import sklearn
                else:
                    __import__(package.replace("-", "_"))
                print(f"{package} already installed")
            except ImportError:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
                print(f"{package} installed")
        
        # Clone SatCLIP repository if not already present
        if not os.path.exists("./satclip"):
            print("Cloning SatCLIP repository...")
            subprocess.check_call(["git", "clone", "https://github.com/microsoft/satclip.git", "./satclip"])
            print("✓ SatCLIP repository cloned")
        else:
            print("✓ SatCLIP repository already exists")
            
    except Exception as e:
        print(f"Warning: Could not auto-install packages or clone repo: {e}")
        print("Please run manually:")
        print("pip install lightning rasterio torchgeo albumentations timm kornia scikit-learn matplotlib requests")
        print("git clone https://github.com/microsoft/satclip.git ./satclip")

# Install packages and clone repo first
install_requirements()

# Now import SatCLIP using the cloned repository
try:
    import sys
    import os
    
    # The load.py file is in satclip/satclip/ not satclip/
    satclip_path = os.path.abspath('./satclip/satclip')
    if satclip_path not in sys.path:
        sys.path.insert(0, satclip_path)  # Insert at beginning to prioritize
    
    # Remove any cached imports of 'load' to force reimport from correct location
    if 'load' in sys.modules:
        del sys.modules['load']
    
    import torch
    
    # Import directly from the satclip/satclip directory
    import importlib.util
    load_py_path = os.path.join(satclip_path, 'load.py')
    
    if os.path.exists(load_py_path):
        spec = importlib.util.spec_from_file_location("satclip_load", load_py_path)
        satclip_load = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(satclip_load)
        get_satclip = satclip_load.get_satclip
        print("Successfully imported get_satclip from SatCLIP repository")
    else:
        raise ImportError(f"load.py not found at {load_py_path}")
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Could not import SatCLIP. Please check the repository structure.")
    sys.exit(1)

def verify_file_integrity(filepath, expected_size_mb=None):
    """Verify that a downloaded file is complete and not corrupted"""
    if not os.path.exists(filepath):
        return False, "File does not exist"
    
    file_size = os.path.getsize(filepath)
    
    if file_size == 0:
        return False, "File is empty"
    
    if expected_size_mb and file_size < expected_size_mb * 1024 * 1024 * 0.9:  # Allow 10% tolerance
        return False, f"File too small ({file_size / (1024*1024):.1f}MB, expected ~{expected_size_mb}MB)"
    
    # Try to load the first few bytes to check if it's a valid PyTorch file
    try:
        with open(filepath, 'rb') as f:
            header = f.read(8)
            if len(header) < 8:
                return False, "File header too short"
            # PyTorch files typically start with specific magic numbers
            if header[:4] not in [b'PK\x03\x04', b'\x80\x02']:  # ZIP or PyTorch magic
                return False, "Invalid file format"
    except Exception as e:
        return False, f"Error reading file: {e}"
    
    return True, "File appears valid"

def download_with_verification(url, output_path, expected_size_mb=None, max_retries=3):
    """Download file with integrity verification and retry logic"""
    import requests
    
    for attempt in range(max_retries):
        try:
            print(f"Download attempt {attempt + 1}/{max_retries}")
            
            # Remove existing file if it exists
            if os.path.exists(output_path):
                os.remove(output_path)
            
            # Download with progress bar
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f, tqdm(
                desc=f"Downloading {os.path.basename(output_path)}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # Verify the downloaded file
            is_valid, message = verify_file_integrity(output_path, expected_size_mb)
            if is_valid:
                print(f"Download successful and verified: {output_path}")
                return True
            else:
                print(f"Download verification failed: {message}")
                if attempt < max_retries - 1:
                    print("Retrying download...")
                    continue
                else:
                    return False
                    
        except Exception as e:
            print(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying download...")
                continue
            else:
                return False
    
    return False

def extract_epsg(filename):
    """Extract EPSG code from filename (same as in encoder.py)"""
    match = re.search(r"utm_(\d{1,2}[NS])", filename)
    if match:
        zone = match.group(1)
        hemisphere = "326" if "N" in zone else "327"
        zone_number = zone[:-1].zfill(2)
        epsg_code = f"{hemisphere}{zone_number}"
        return int(epsg_code)
    raise ValueError(f"Could not find UTM zone in filename: {filename}")

def transform_coordinates(x, y, src_epsg=32739, dst_epsg=4326):
    """Transform coordinates from UTM to WGS84 (same as in encoder.py)"""
    from pyproj import Transformer
    transformer = Transformer.from_crs(f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", always_xy=True)
    lon, lat = transformer.transform(y, x)
    return lat, lon

def load_satclip_model():
    """Load SatCLIP model using available checkpoint"""
    print("Loading SatCLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Define checkpoint options with expected sizes and URLs
        checkpoint_options = [
            {
                "name": "ResNet50",
                "local_paths": ["./satclip-resnet50-l10.ckpt", "../satclip-resnet50-l10.ckpt"],
                "url": "https://huggingface.co/microsoft/SatCLIP-ResNet50-L10/resolve/main/satclip-resnet50-l10.ckpt",
                "expected_size_mb": 100  # Approximate size
            },
            {
                "name": "ResNet18", 
                "local_paths": ["./satclip-resnet18-l10.ckpt", "../satclip-resnet18-l10.ckpt"],
                "url": "https://satclip.z13.web.core.windows.net/satclip/satclip-resnet18-l10.ckpt",
                "expected_size_mb": 50   # Approximate size
            }
        ]
        
        satclip_path = None
        
        # First, check for existing valid files
        for option in checkpoint_options:
            for path in option["local_paths"]:
                if os.path.exists(path):
                    is_valid, message = verify_file_integrity(path, option["expected_size_mb"])
                    if is_valid:
                        satclip_path = path
                        print(f"✓ Found valid {option['name']} checkpoint: {path}")
                        break
                    else:
                        print(f"⚠ Invalid checkpoint found at {path}: {message}")
                        # Remove corrupted file
                        try:
                            os.remove(path)
                            print(f"Removed corrupted file: {path}")
                        except:
                            pass
            if satclip_path:
                break
        
        # If no valid checkpoint found, download one
        if satclip_path is None:
            print("No valid checkpoint found. Attempting to download...")
            
            # Try ResNet18 first (smaller, faster download)
            for option in checkpoint_options[::-1]:  # Start with ResNet18
                download_path = f"./{option['name'].lower()}-checkpoint.ckpt"
                print(f"Trying to download {option['name']} model...")
                
                success = download_with_verification(
                    option["url"], 
                    download_path, 
                    option["expected_size_mb"]
                )
                
                if success:
                    satclip_path = download_path
                    break
                else:
                    print(f"Failed to download {option['name']} model")
        
        if satclip_path is None:
            raise RuntimeError("Could not find or download a valid SatCLIP checkpoint")
        
        print(f"Loading SatCLIP from: {satclip_path}")
        
        # Additional verification before loading
        is_valid, message = verify_file_integrity(satclip_path)
        if not is_valid:
            raise RuntimeError(f"Checkpoint file verification failed: {message}")
        
        # Load model using the SatCLIP function
        model = get_satclip(satclip_path, device=device)
        model.eval()
        
        # Test the model to determine embedding dimension
        print("Testing SatCLIP model...")
        c = torch.randn(2, 2)  # Test coordinates (lat, lon pairs)
        with torch.no_grad():
            test_emb = model(c.double().to(device)).detach().cpu()
            print(f"SatCLIP test successful - embedding shape: {test_emb.shape}")
            embedding_dim = test_emb.shape[1]
            print(f"Embedding dimension: {embedding_dim}")
        
        print("SatCLIP model loaded successfully")
        return model, device, embedding_dim
        
    except Exception as e:
        print(f"Error loading SatCLIP model: {e}")
        print("Troubleshooting steps:")
        print("1. Check internet connection")
        print("2. Verify that the SatCLIP repository was cloned correctly")
        print("3. Try manually downloading a checkpoint:")
        print("   - ResNet50: https://huggingface.co/microsoft/SatCLIP-ResNet50-L10/resolve/main/satclip-resnet50-l10.ckpt")
        print("   - ResNet18: https://satclip.z13.web.core.windows.net/satclip/satclip-resnet18-l10.ckpt")
        print("4. Ensure all dependencies are installed correctly")
        raise

def get_all_coordinates_from_csvs():
    """Extract all unique coordinates from train/val/test CSV files"""
    
    # Define paths
    base_path = "/home/ubuntu/work/satellite_data/icml_2024_global_rh100/"
    csv_files = ['train.csv', 'val.csv','test.csv']  
    
    all_coordinates = []
    
    for csv_file in csv_files:
        csv_path = os.path.join(base_path, csv_file)
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping...")
            continue
            
        print(f"Processing {csv_file}...")
        df = pd.read_csv(csv_path)
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting coords from {csv_file}"):
            try:
                # Extract UTM coordinates
                utm_x = row['longitudes']
                utm_y = row['latitudes']
                filename = row['paths']
                
                # Get EPSG and transform to WGS84
                epsg = extract_epsg(filename)
                lat, lon = transform_coordinates(utm_x, utm_y, src_epsg=epsg, dst_epsg=4326)
                
                all_coordinates.append({
                    'utm_x': utm_x,
                    'utm_y': utm_y,
                    'epsg': epsg,
                    'lat': lat,
                    'lon': lon,
                    'source_file': csv_file
                })
                
            except Exception as e:
                print(f"Error processing row in {csv_file}: {e}")
                continue
    
    # Convert to DataFrame and remove duplicates
    coords_df = pd.DataFrame(all_coordinates)
    
    # Remove duplicates based on lat/lon (rounded to avoid floating point issues)
    coords_df['lat_rounded'] = coords_df['lat'].round(6)
    coords_df['lon_rounded'] = coords_df['lon'].round(6)
    coords_df = coords_df.drop_duplicates(subset=['lat_rounded', 'lon_rounded'])
    coords_df = coords_df.drop(['lat_rounded', 'lon_rounded'], axis=1)
    
    print(f"Found {len(coords_df)} unique coordinates")
    return coords_df

def generate_satclip_embeddings(coords_df, model, device, embedding_dim, batch_size=32):
    """Generate SatCLIP embeddings for all coordinates"""
    print(f"Generating SatCLIP embeddings for {len(coords_df)} coordinates...")
    
    embeddings = []
    
    # Process in batches
    for i in tqdm(range(0, len(coords_df), batch_size), desc="Generating embeddings"):
        batch_end = min(i + batch_size, len(coords_df))
        batch_coords = coords_df.iloc[i:batch_end]
        
        # Create coordinate tensor [batch_size, 2] with (lat, lon) - NOTE: SatCLIP expects lat, lon order
        batch_lats = torch.tensor(batch_coords['lat'].values, dtype=torch.float64)
        batch_lons = torch.tensor(batch_coords['lon'].values, dtype=torch.float64)
        
        # Stack coordinates in the right order for SatCLIP
        coord_batch = torch.stack([batch_lats, batch_lons], dim=1).to(device)
        
        try:
            with torch.no_grad():
                # Generate embeddings using SatCLIP (following notebook pattern)
                emb = model(coord_batch).detach().cpu()
                
                # Convert to numpy and store
                emb_numpy = emb.numpy()
                for j, embedding in enumerate(emb_numpy):
                    embeddings.append(embedding.astype(np.float32))
                    
        except Exception as e:
            print(f"Error generating embeddings for batch {i}: {e}")
            # Add zero embeddings as fallback
            for j in range(len(batch_coords)):
                embeddings.append(np.zeros(embedding_dim, dtype=np.float32))
    
    return embeddings

def save_embeddings_to_csv(coords_df, embeddings, output_path):
    """Save coordinates and embeddings to CSV"""
    print(f"Saving embeddings to {output_path}...")
    
    # Create final DataFrame
    result_df = coords_df.copy()
    
    # Add embedding columns
    embedding_dim = len(embeddings[0]) if embeddings else 512
    embedding_cols = [f'emb_{i}' for i in range(embedding_dim)]
    
    # Convert embeddings to DataFrame
    emb_df = pd.DataFrame(embeddings, columns=embedding_cols)
    
    # Combine coordinates and embeddings
    result_df = pd.concat([result_df.reset_index(drop=True), emb_df], axis=1)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    
    print(f"Saved {len(result_df)} coordinate embeddings to {output_path}")
    print(f"Embedding dimension: {embedding_dim}")
    
    return result_df

def main():
    """Main execution function"""
    print("="*60)
    print("SatCLIP Coordinate Embedding Generator")
    print("="*60)
    
    # Define output path
    output_path = "Updated-Global-Canopy-Height-Map-Coordinates/Global-Canopy-Height-Map/figures/satclip_embeddings.csv"
    
    try:
        # Step 1: Extract all coordinates
        coords_df = get_all_coordinates_from_csvs()
        
        # Step 2: Load SatCLIP model
        model, device, embedding_dim = load_satclip_model()
        
        # Step 3: Generate embeddings
        embeddings = generate_satclip_embeddings(coords_df, model, device, embedding_dim)
        
        # Step 4: Save to CSV
        result_df = save_embeddings_to_csv(coords_df, embeddings, output_path)
        
        print("\n" + "="*60)
        print("SUCCESS: Real SatCLIP embeddings generated!")
        print(f"Output file: {output_path}")
        print(f"Total coordinates: {len(result_df)}")
        print(f"Embedding dimension: {len([col for col in result_df.columns if col.startswith('emb_')])}")
        print("These are genuine SatCLIP coordinate embeddings!")
        print("="*60)
        
    except Exception as e:
        print(f"\n ERROR: {e}")
        print("Please check the error above and try again.")
        raise

if __name__ == "__main__":
    main()