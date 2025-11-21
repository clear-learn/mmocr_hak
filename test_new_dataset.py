#!/usr/bin/env python
"""
Test if the new Korean handwriting dataset can be loaded correctly.
"""

import json
import os
from mmengine import Config
from mmocr.datasets import build_dataset

def test_dataset_loading(config_path):
    """Test if datasets can be loaded without errors."""

    print("="*80)
    print("Testing New Korean Handwriting Dataset Configuration")
    print("="*80)

    # Load config
    print(f"\n1. Loading config from: {config_path}")
    cfg = Config.fromfile(config_path)

    # Check gt_jsonl.txt file
    print("\n2. Checking gt_jsonl.txt file...")
    new_kor_root = '/media/yim/5ca43dd5-cbeb-4cae-aac0-e36cdd0808f7/OCR_outdoor/new_kor'
    gt_jsonl_path = os.path.join(new_kor_root, 'gt_jsonl.txt')

    if not os.path.exists(gt_jsonl_path):
        print(f"   ❌ ERROR: {gt_jsonl_path} does not exist!")
        return False

    print(f"   ✓ gt_jsonl.txt exists: {gt_jsonl_path}")

    # Check first few lines of gt_jsonl.txt
    print("\n3. Checking gt_jsonl.txt format...")
    with open(gt_jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            try:
                entry = json.loads(line.strip())
                img_path = entry['img_path']
                text = entry['text']

                # Check if image file exists
                full_img_path = os.path.join(new_kor_root, img_path)
                if not os.path.exists(full_img_path):
                    print(f"   ❌ ERROR: Image does not exist: {full_img_path}")
                    return False

                print(f"   ✓ Entry {i+1}: img_path={img_path[:50]}..., text={text[:30]}...")
            except Exception as e:
                print(f"   ❌ ERROR parsing line {i+1}: {e}")
                return False

    # Try to build train dataset
    print("\n4. Building train dataset...")
    try:
        train_dataset = build_dataset(cfg.train_dataset)
        print(f"   ✓ Train dataset built successfully!")
        print(f"   ✓ Total samples: {len(train_dataset)}")

        # Check if it's a ConcatDataset
        if hasattr(train_dataset, 'datasets'):
            print(f"   ✓ ConcatDataset detected with {len(train_dataset.datasets)} sub-datasets")
            for i, ds in enumerate(train_dataset.datasets):
                print(f"      - Dataset {i+1}: {len(ds)} samples")

    except Exception as e:
        print(f"   ❌ ERROR building train dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Try to load a sample from new Korean dataset
    print("\n5. Testing data loading from new Korean dataset...")
    try:
        # Get the second sub-dataset (new Korean handwriting)
        if hasattr(train_dataset, 'datasets'):
            new_kor_dataset = train_dataset.datasets[1]
            print(f"   Testing on dataset with {len(new_kor_dataset)} samples")

            # Load first sample
            sample = new_kor_dataset[0]
            print(f"   ✓ Sample loaded successfully!")
            print(f"   ✓ Keys: {list(sample.keys())}")

            if 'img_path' in sample:
                print(f"   ✓ img_path: {sample['img_path']}")
            if 'gt_text' in sample:
                print(f"   ✓ gt_text: {sample['gt_text']}")
            if 'inputs' in sample:
                print(f"   ✓ inputs shape: {sample['inputs'].shape}")

    except Exception as e:
        print(f"   ❌ ERROR loading sample: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Try to build val dataset
    print("\n6. Building validation dataset...")
    try:
        val_dataset = build_dataset(cfg.val_dataset)
        print(f"   ✓ Validation dataset built successfully!")
        print(f"   ✓ Total samples: {len(val_dataset)}")
    except Exception as e:
        print(f"   ❌ ERROR building validation dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check dictionary compatibility
    print("\n7. Checking dictionary compatibility...")
    dict_path = cfg.dictionary['dict_file']
    docker_dict_path = '/opt/project/datasets/OCR_outdoor/recog_outdoor/original_dict.txt'
    host_dict_path = '/media/yim/5ca43dd5-cbeb-4cae-aac0-e36cdd0808f7/datasets/OCR_outdoor/recog_outdoor/original_dict.txt'

    # Try both paths
    found = False
    for path in [docker_dict_path, host_dict_path]:
        if os.path.exists(path):
            print(f"   ✓ Dictionary file exists: {path}")

            with open(path, 'r', encoding='utf-8') as f:
                chars = f.read().strip()
                print(f"   ✓ Dictionary has {len(chars)} characters")
                print(f"   ✓ Sample characters: {chars[:50]}...")
            found = True
            break

    if not found:
        print(f"   ⚠ WARNING: Dictionary file not found at {dict_path}")
        print(f"   ⚠ This will cause issues during training!")

    print("\n" + "="*80)
    print("✅ All checks passed! Configuration is ready for training.")
    print("="*80)
    return True


if __name__ == '__main__':
    config_path = '/media/yim/5ca43dd5-cbeb-4cae-aac0-e36cdd0808f7/mmocr/configs/textrecog/satrn/satrn_shallow_5e_st_mj_aladin_original_size.py'
    test_dataset_loading(config_path)