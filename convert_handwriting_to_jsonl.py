#!/usr/bin/env python
"""
Convert handwriting_data_info_clean.json to MMOCR gt_jsonl.txt format.
Output format: {"img_path": "relative/path/to/image.png", "text": "ground truth text"}
"""

import json
import os
from pathlib import Path
from tqdm import tqdm

def build_image_path_index(base_dir: str):
    """
    Build a mapping from filename to relative path by scanning all subdirectories.
    """
    print(f"Building image path index from: {base_dir}")
    image_map = {}

    # Scan all subdirectories for PNG files
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.png'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_dir)
                image_map[file] = rel_path

    print(f"Found {len(image_map)} unique image files")
    return image_map


def convert_handwriting_json_to_jsonl(
    json_path: str,
    output_dir: str,
    image_base_dir: str
):
    """
    Convert COCO-style handwriting JSON to MMOCR JSONL format.

    Args:
        json_path: Path to handwriting_data_info_clean.json
        output_dir: Directory to save gt_jsonl.txt
        image_base_dir: Base directory containing actual image files
    """
    print(f"Loading JSON from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total images: {len(data['images'])}")
    print(f"Total annotations: {len(data['annotations'])}")

    # Build image path index
    image_map = build_image_path_index(image_base_dir)

    # Create image_id -> annotation mapping
    print("Building annotation index...")
    anno_dict = {}
    for anno in tqdm(data['annotations']):
        anno_dict[anno['image_id']] = anno

    # Process images and create JSONL entries
    print("Converting to JSONL format...")
    output_lines = []
    missing_images = []

    for img_info in tqdm(data['images']):
        img_id = img_info['id']
        file_name = img_info['file_name']

        # Get corresponding annotation
        if img_id not in anno_dict:
            print(f"Warning: No annotation for image_id={img_id}")
            continue

        anno = anno_dict[img_id]
        text = anno['text']

        # Look up the relative path from our index
        if file_name not in image_map:
            missing_images.append(file_name)
            continue

        rel_path = image_map[file_name]

        # Create JSONL entry (use relative path from image_base_dir)
        # ★ MMOCR expects 'filename' key, not 'img_path'
        jsonl_entry = {
            "filename": rel_path,
            "text": text
        }
        output_lines.append(json.dumps(jsonl_entry, ensure_ascii=False))

    # Write to output file
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'gt_jsonl.txt')

    print(f"\nWriting {len(output_lines)} entries to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f"\nConversion complete!")
    print(f"Total entries written: {len(output_lines)}")
    print(f"Missing images: {len(missing_images)}")

    if missing_images and len(missing_images) <= 10:
        print("\nSample missing images:")
        for fname in missing_images[:10]:
            print(f"  - {fname}")


if __name__ == '__main__':
    # Paths
    json_path = '/media/yim/5ca43dd5-cbeb-4cae-aac0-e36cdd0808f7/OCR_outdoor/new_kor/handwriting_data_info_clean.json'
    output_dir = '/media/yim/5ca43dd5-cbeb-4cae-aac0-e36cdd0808f7/OCR_outdoor/new_kor'
    image_base_dir = '/media/yim/5ca43dd5-cbeb-4cae-aac0-e36cdd0808f7/OCR_outdoor/new_kor'

    convert_handwriting_json_to_jsonl(
        json_path=json_path,
        output_dir=output_dir,
        image_base_dir=image_base_dir
    )