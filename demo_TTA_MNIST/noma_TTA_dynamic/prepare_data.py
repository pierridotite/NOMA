#!/usr/bin/env python3
"""Prepare data for Dynamic TTA demonstration."""

import numpy as np
import struct
import json
from pathlib import Path
import shutil

def load_safetensors(filepath):
    with open(filepath, 'rb') as f:
        header_size = struct.unpack('<Q', f.read(8))[0]
        header_json = f.read(header_size).decode('utf-8')
        header = json.loads(header_json)
        data_start = 8 + header_size
        
        result = {}
        for name, info in header.items():
            shape = info['shape']
            start, end = info['data_offsets']
            f.seek(data_start + start)
            arr = np.frombuffer(f.read(end - start), dtype=np.float64)
            result[name] = arr.reshape(shape) if shape else arr
        
        return result

def save_safetensors(data_dict, filepath):
    header = {}
    data_bytes = b""
    offset = 0
    
    for name, arr in data_dict.items():
        arr = np.ascontiguousarray(arr, dtype=np.float64)
        arr_bytes = arr.tobytes()
        header[name] = {
            "dtype": "F64",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(arr_bytes)]
        }
        data_bytes += arr_bytes
        offset += len(arr_bytes)
    
    header_json = json.dumps(header).encode('utf-8')
    header_size = len(header_json)
    
    with open(filepath, 'wb') as f:
        f.write(struct.pack('<Q', header_size))
        f.write(header_json)
        f.write(data_bytes)

def main():
    print("=" * 50)
    print("Dynamic TTA - Data Preparation")
    print("=" * 50)
    
    Path("data").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    
    # Copy data from noma_TTA folder
    src_dir = Path("../noma_TTA/data")
    dst_dir = Path("data")
    
    if src_dir.exists():
        print("\nCopying data from noma_TTA...")
        for f in ["mnist_train.safetensors", "stream_clean.safetensors", 
                  "stream_drift.safetensors", "mnist_stream_full.safetensors"]:
            src = src_dir / f
            dst = dst_dir / f
            if src.exists():
                shutil.copy(src, dst)
                print(f"  Copied {f}")
        
        # Rename full stream file
        if (dst_dir / "mnist_stream_full.safetensors").exists():
            shutil.move(dst_dir / "mnist_stream_full.safetensors", 
                       dst_dir / "stream_full.safetensors")
            print("  Renamed mnist_stream_full.safetensors -> stream_full.safetensors")
        
        # Copy metadata
        src_meta = src_dir / "stream_metadata.json"
        if src_meta.exists():
            shutil.copy(src_meta, dst_dir / "metadata.json")
            print("  Copied metadata")
        
        print("\nData preparation complete.")
    else:
        print("ERROR: Source data not found in ../noma_TTA/data")
        print("Please run the noma_TTA pipeline first.")
        exit(1)

if __name__ == "__main__":
    main()
