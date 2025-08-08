import numpy as np
import math
from scipy.special import lpmv
from pyproj import Transformer
import re
from scipy.spatial.distance import cdist
import pandas as pd



# --- Coordinate Transformation ---
def transform_coordinates(x, y, src_epsg=32739, dst_epsg=4326):
    """
    Transform coordinates from a source EPSG to a destination EPSG.
    Args:
    x (float): X coordinate (easting in UTM).
    y (float): Y coordinate (northing in UTM).
    src_epsg (int): Source EPSG code.
    dst_epsg (int): Destination EPSG code.
    Returns:
    tuple: Transformed coordinates as (latitude, longitude).
    """
    transformer = Transformer.from_crs(f"EPSG:{src_epsg}", f"EPSG:{dst_epsg}", always_xy=True)
    lon, lat = transformer.transform(y, x) # BEWARE COULD COMMON MISMATCH RECHECK IN DATAINSPECTION!! (y,x) 
    return lat, lon

def extract_epsg(filename):
    match = re.search(r"utm_(\d{1,2}[NS])", filename)
    if match:
        zone = match.group(1)
        hemisphere = "326" if "N" in zone else "327"
        zone_number = zone[:-1].zfill(2)
        epsg_code = f"{hemisphere}{zone_number}"
        return int(epsg_code)
    raise ValueError(f"Could not find UTM zone in filename: {filename}")

# --- Encoders ---
def encode_raw(lat, lon):
    return np.array([(lat) / 90, (lon) / 180], dtype=np.float32)
encode_raw.num_output_channels = 2  

def encode_wrap(lat, lon):
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)
    return np.array([np.sin(lat_rad), np.cos(lat_rad), np.sin(lon_rad), np.cos(lon_rad)], dtype=np.float32)
encode_wrap.num_output_channels = 4

def encode_wrap_lon_only(lat, lon):
    lon_rad = np.radians(lon)
    lat_norm = (lat) / 90
    return np.array([lat_norm, np.sin(lon_rad), np.cos(lon_rad)], dtype=np.float32)
encode_wrap_lon_only.num_output_channels = 3 


def encode_fourier(lat, lon, num_freqs=4):
#Fourier positional baseline - NeRF
    lat_norm = (lat + 90) / 180
    lon_norm = (lon + 180) / 360
    freqs = 2 ** np.arange(num_freqs) * np.pi
    encs = []
    for f in freqs:
        encs += [np.sin(f * lat_norm), np.cos(f * lat_norm), np.sin(f * lon_norm), np.cos(f * lon_norm)]
    return np.array(encs, dtype=np.float32)
encode_fourier.num_output_channels = 4 * 4 # num_freqs * 4

def encode_satclip(lat, lon, embedding_csv="Updated-Global-Canopy-Height-Map-Coordinates/Global-Canopy-Height-Map/figures/satclip_embeddings.csv", cache={}):
    # Load embeddings once and cache
    if 'lookup' not in cache:
        df = pd.read_csv(embedding_csv)
        # Create lookup dictionary: (lat, lon) -> embedding
        cache['lookup'] = {}
        embedding_cols = [col for col in df.columns if col.startswith('emb_')]
        for _, row in df.iterrows():
            key = (round(row['lat'], 6), round(row['lon'], 6))  # Round to avoid float precision issues
            cache['lookup'][key] = row[embedding_cols].values.astype(np.float32)
    
    # Direct lookup
    key = (round(lat, 6), round(lon, 6))
    return cache['lookup'][key]

encode_satclip.num_output_channels = 256



'''
def encode_grid(lat, lon, r_min=1, r_max=1000, S=4):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    encoding = []

    for s in range(S):
        alpha_s = r_min * ((r_max / r_min) ** (s / max(S - 1, 1)))
        encoding.extend([
            np.cos(lon_rad / alpha_s), np.sin(lon_rad / alpha_s),
            np.cos(lat_rad / alpha_s), np.sin(lat_rad / alpha_s),
        ])
    return np.array(encoding, dtype=np.float32)


def encode_sh(lat, lon, L=3):
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)
    result = []
    for l in range(L + 1):
        for m in range(-l, l + 1):
            P_lm = lpmv(abs(m), l, np.sin(lat_rad))
            norm = math.sqrt((2 * l + 1) / (4 * np.pi) * math.factorial(l - abs(m)) / math.factorial(l + abs(m)))
            if m < 0:
                result.append(norm * P_lm * np.sin(abs(m) * lon_rad))
            elif m == 0:
                result.append(norm * P_lm)
            else:
                result.append(norm * P_lm * np.cos(m * lon_rad))
    return np.array(result, dtype=np.float32)
encode_sh.num_output_channels = (3 + 1) ** 2 #(L + 1) ** 2 
'''

# --- Dispatcher ---
ENCODER_MAP = {
    "raw": encode_raw,
    "wrap": encode_wrap,
    "wrap_lon_only": encode_wrap_lon_only,
    "fourier": encode_fourier,
    "satclip": encode_satclip,
}
