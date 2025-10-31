#!/usr/bin/env python3
"""
Download Quick Draw Dataset for Training
"""

import os
import requests
import numpy as np
from tqdm import tqdm
import json
from config import DATASET_CONFIG, PATHS, USE_QUICK_CONFIG, QUICK_CONFIG

def download_file(url, filename):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=os.path.basename(filename),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                pbar.update(len(chunk))

def download_category(category, base_url, data_dir, samples_limit=None):
    """Download data for a specific category"""
    filename = f"{category.replace(' ', '_')}.npy"
    url = f"{base_url}/{filename}"
    filepath = os.path.join(data_dir, filename)
    
    if os.path.exists(filepath):
        print(f"✅ {category} already downloaded")
        return filepath
    
    try:
        print(f"📥 Downloading {category}...")
        download_file(url, filepath)
        
        # Verify the download and optionally limit samples
        if samples_limit:
            data = np.load(filepath)
            if len(data) > samples_limit:
                # Keep only the first N samples to save space
                limited_data = data[:samples_limit]
                np.save(filepath, limited_data)
                print(f"   Limited to {samples_limit} samples")
        
        print(f"✅ {category} downloaded successfully")
        return filepath
        
    except Exception as e:
        print(f"❌ Failed to download {category}: {e}")
        return None

def verify_downloads(categories, data_dir):
    """Verify all downloads are complete and valid"""
    print("\n🔍 Verifying downloads...")
    
    valid_categories = []
    total_samples = 0
    
    for category in categories:
        filename = f"{category.replace(' ', '_')}.npy"
        filepath = os.path.join(data_dir, filename)
        
        if os.path.exists(filepath):
            try:
                data = np.load(filepath)
                samples = len(data)
                total_samples += samples
                valid_categories.append(category)
                print(f"✅ {category}: {samples:,} samples")
            except Exception as e:
                print(f"❌ {category}: Invalid file - {e}")
        else:
            print(f"❌ {category}: File not found")
    
    print(f"\n📊 Summary:")
    print(f"   Valid categories: {len(valid_categories)}/{len(categories)}")
    print(f"   Total samples: {total_samples:,}")
    
    return valid_categories

def create_class_mapping(categories):
    """Create class name to index mapping"""
    class_to_idx = {category: idx for idx, category in enumerate(categories)}
    idx_to_class = {idx: category for category, idx in class_to_idx.items()}
    
    mapping = {
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "num_classes": len(categories)
    }
    
    # Save mapping
    mapping_file = os.path.join(PATHS["data_dir"], "class_mapping.json")
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"💾 Class mapping saved to {mapping_file}")
    return mapping

def main():
    """Main download function"""
    print("🎨 Quick Draw Dataset Downloader")
    print("=" * 50)
    
    # Use quick config if enabled
    if USE_QUICK_CONFIG:
        print("⚡ Using quick configuration for testing")
        categories = QUICK_CONFIG["categories"]
        samples_limit = QUICK_CONFIG["samples_per_category"]
    else:
        categories = DATASET_CONFIG["categories"]
        samples_limit = DATASET_CONFIG["samples_per_category"]
    
    base_url = DATASET_CONFIG["base_url"]
    data_dir = DATASET_CONFIG["raw_data_dir"]
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"📁 Data directory: {data_dir}")
    print(f"🎯 Categories to download: {len(categories)}")
    print(f"📊 Samples per category: {samples_limit:,}")
    print(f"💾 Estimated total size: ~{len(categories) * 50}MB")
    
    # Confirm download
    response = input("\nProceed with download? (y/n): ").lower().strip()
    if response != 'y':
        print("Download cancelled.")
        return
    
    # Download each category
    print(f"\n📥 Starting download of {len(categories)} categories...")
    
    successful_downloads = []
    failed_downloads = []
    
    for i, category in enumerate(categories, 1):
        print(f"\n[{i}/{len(categories)}] Processing {category}...")
        
        filepath = download_category(
            category, 
            base_url, 
            data_dir, 
            samples_limit
        )
        
        if filepath:
            successful_downloads.append(category)
        else:
            failed_downloads.append(category)
    
    # Verify downloads
    valid_categories = verify_downloads(successful_downloads, data_dir)
    
    # Create class mapping
    if valid_categories:
        create_class_mapping(valid_categories)
    
    # Final summary
    print(f"\n🎉 Download Complete!")
    print(f"✅ Successful: {len(successful_downloads)}")
    print(f"❌ Failed: {len(failed_downloads)}")
    
    if failed_downloads:
        print(f"\nFailed categories: {', '.join(failed_downloads)}")
        print("You can retry downloading failed categories by running this script again.")
    
    print(f"\n🚀 Next steps:")
    print(f"   1. Run: python train_model.py")
    print(f"   2. Monitor training progress")
    print(f"   3. Evaluate model performance")

if __name__ == "__main__":
    main()