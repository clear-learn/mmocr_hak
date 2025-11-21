"""
학습 파이프라인에 들어가는 이미지를 저장하는 디버그 스크립트

사용법:
python debug_pipeline.py --config configs/textrecog/satrn/satrn_shallow_5e_st_mj_aladin_original_size.py \
                         --num-samples 20 \
                         --output-dir debug_images

"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from mmengine.config import Config

# MMOCR 모듈 import (레지스트리 초기화)
import mmocr  # noqa: F401
from mmocr.registry import DATASETS


def denormalize_image(img_tensor, mean, std):
    """
    정규화된 텐서를 원본 이미지로 복원

    Args:
        img_tensor: (C, H, W) 텐서
        mean: 정규화에 사용된 평균값 [R, G, B]
        std: 정규화에 사용된 표준편차 [R, G, B]

    Returns:
        PIL Image
    """
    # 텐서를 numpy로 변환
    img = img_tensor.cpu().numpy()

    # (C, H, W) -> (H, W, C)
    img = np.transpose(img, (1, 2, 0))

    # denormalize
    mean = np.array(mean)
    std = np.array(std)
    img = img * std + mean

    # [0, 255] 범위로 클리핑
    img = np.clip(img, 0, 255).astype(np.uint8)

    # PIL Image로 변환
    return Image.fromarray(img)


def save_pipeline_images(config_path, num_samples=10, output_dir='debug_images', mode='train'):
    """
    파이프라인을 거친 이미지들을 저장

    Args:
        config_path: config 파일 경로
        num_samples: 저장할 샘플 수
        output_dir: 저장할 디렉토리
        mode: 'train' 또는 'val'
    """
    # Config 로드
    cfg = Config.fromfile(config_path)

    # 출력 디렉토리 생성
    output_path = Path(output_dir) / mode
    output_path.mkdir(parents=True, exist_ok=True)

    # Dataset 생성
    if mode == 'train':
        dataset_cfg = cfg.train_dataset
    else:
        dataset_cfg = cfg.val_dataset

    dataset = DATASETS.build(dataset_cfg)

    # data_preprocessor의 mean, std 가져오기
    mean = cfg.model.data_preprocessor.mean
    std = cfg.model.data_preprocessor.std

    print(f"\n{'='*60}")
    print(f"디버그 모드: {mode.upper()} 데이터셋")
    print(f"{'='*60}")
    print(f"총 데이터 개수: {len(dataset)}")
    print(f"저장할 샘플 수: {num_samples}")
    print(f"출력 디렉토리: {output_path}")
    print(f"정규화 파라미터 - mean: {mean}, std: {std}")
    print(f"{'='*60}\n")

    # 샘플 저장
    for i in range(min(num_samples, len(dataset))):
        try:
            # 데이터 가져오기
            data = dataset[i]

            # 이미지 텐서 추출
            img_tensor = data['inputs']

            # 메타 정보 추출
            data_sample = data['data_samples']
            img_path = data_sample.img_path if hasattr(data_sample, 'img_path') else 'unknown'
            ori_shape = data_sample.ori_shape if hasattr(data_sample, 'ori_shape') else 'unknown'
            img_shape = data_sample.img_shape if hasattr(data_sample, 'img_shape') else 'unknown'
            gt_text = data_sample.gt_text.item if hasattr(data_sample, 'gt_text') else 'unknown'

            # 정규화 해제 및 저장
            img_pil = denormalize_image(img_tensor, mean, std)

            # 파일명 생성
            original_name = Path(img_path).stem if img_path != 'unknown' else f'sample_{i}'
            save_path = output_path / f"{i:04d}_{original_name}.jpg"

            # 이미지 저장
            img_pil.save(save_path)

            # 메타 정보 저장
            meta_path = output_path / f"{i:04d}_{original_name}.txt"
            with open(meta_path, 'w', encoding='utf-8') as f:
                f.write(f"Original Path: {img_path}\n")
                f.write(f"Original Shape (H, W): {ori_shape}\n")
                f.write(f"Processed Shape (H, W, C): {img_shape}\n")
                f.write(f"Tensor Shape (C, H, W): {img_tensor.shape}\n")
                f.write(f"Ground Truth Text: {gt_text}\n")

            print(f"[{i+1}/{num_samples}] 저장 완료: {save_path.name}")
            print(f"  - 원본 경로: {img_path}")
            print(f"  - 원본 크기: {ori_shape}")
            print(f"  - 처리된 크기: {img_tensor.shape}")
            print(f"  - GT 텍스트: {gt_text}\n")

        except Exception as e:
            print(f"[{i+1}/{num_samples}] 에러 발생: {str(e)}\n")
            continue

    print(f"\n{'='*60}")
    print(f"✅ 완료! {output_path} 에 이미지가 저장되었습니다.")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='학습 파이프라인 디버그: 처리된 이미지를 저장합니다.')

    parser.add_argument(
        '--config',
        type=str,
        default='configs/textrecog/satrn/satrn_shallow_5e_st_mj_aladin_original_size.py',
        help='config 파일 경로')

    parser.add_argument(
        '--num-samples',
        type=int,
        default=20,
        help='저장할 샘플 수 (default: 20)')

    parser.add_argument(
        '--output-dir',
        type=str,
        default='debug_images',
        help='출력 디렉토리 (default: debug_images)')

    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'val', 'both'],
        default='both',
        help='디버그할 데이터셋 (train/val/both, default: both)')

    args = parser.parse_args()

    # Train 데이터셋 디버그
    if args.mode in ['train', 'both']:
        save_pipeline_images(
            config_path=args.config,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            mode='train'
        )

    # Val 데이터셋 디버그
    if args.mode in ['val', 'both']:
        save_pipeline_images(
            config_path=args.config,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            mode='val'
        )


if __name__ == '__main__':
    main()