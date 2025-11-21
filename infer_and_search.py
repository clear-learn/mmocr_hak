#!/usr/bin/env python
# ==========================================================
#  infer_and_search.py - 이미지에서 텍스트 추론 후 Milvus 검색
# ==========================================================
"""
이미지를 입력받아:
1. MMOCR 모델로 텍스트 추론
2. 추론된 텍스트로 Milvus에서 유사 책 검색
"""
import argparse
from pathlib import Path
import sys
import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.registry import init_default_scope, MODELS
from mmocr.utils import register_all_modules
from pymilvus import MilvusClient

# Compose import (구버전 우선)
try:
    from mmocr.datasets.pipelines.compose import Compose    # 1.0.0rc0~rc2
except ImportError:
    from mmcv.transforms import Compose                     # 정식 1.x


# ============================================================
# 1. MMOCR 모델 로드 및 추론
# ============================================================
def build_model_and_pipeline(cfg_path: Path,
                             ckpt_path: Path,
                             device: str = 'cpu'):
    """MMOCR 모델과 추론 파이프라인을 빌드합니다."""
    cfg = Config.fromfile(str(cfg_path))

    # pretrained / init_cfg 제거
    for k in ('pretrained', 'init_cfg'):
        cfg.model.pop(k, None)
        if isinstance(cfg.model.get('backbone'), dict):
            cfg.model.backbone.pop(k, None)

    # 레지스트리 초기화
    init_default_scope(cfg.default_scope)
    register_all_modules()

    # 모델 생성 + checkpoint 로드
    model = MODELS.build(cfg.model)
    load_checkpoint(model, str(ckpt_path), map_location=device)
    model.to(device).eval()

    # inference 전용 파이프라인 만들기
    orig_pipeline = (cfg.get('test_pipeline')
                     or cfg.test_dataloader.dataset.pipeline)

    # annotation을 요구하는 변환 제거
    inference_pipeline = [
        t for t in orig_pipeline
        if 'Annotation' not in t['type'] and 'Label' not in t['type']
    ]

    return model, inference_pipeline


@torch.inference_mode()
def infer_text_from_image(model, pipeline_cfg, img_path: Path) -> str:
    """이미지에서 텍스트를 추론합니다."""
    pipeline = Compose(pipeline_cfg)
    data = dict(img_path=str(img_path))
    data = pipeline(data)
    pred_sample = model.test_step([data])[0]
    return str(pred_sample.pred_text)


# ============================================================
# 2. Milvus 검색
# ============================================================
def search_in_milvus(
    query_text: str,
    collection_name: str = "domestic_book_meta_embedding",
    milvus_uri: str = "http://10.10.13.129:19530",
    search_field: str = "itemTitle_embedding",
    limit: int = 10
):
    """
    Milvus에서 유사도 검색을 수행합니다.

    Args:
        query_text: 검색할 텍스트
        collection_name: Milvus 컬렉션 이름
        milvus_uri: Milvus 서버 URI
        search_field: 검색할 임베딩 필드 (itemTitle_embedding 또는 authorName_embedding)
        limit: 반환할 결과 수

    Returns:
        검색 결과 리스트
    """
    client = MilvusClient(uri=milvus_uri)

    print("\n" + "=" * 80)
    print(f"🔎 Milvus 검색")
    print(f"   쿼리: '{query_text}'")
    print(f"   컬렉션: {collection_name}")
    print(f"   검색 필드: {search_field}")
    print(f"   결과 수: {limit}")
    print("=" * 80 + "\n")

    # Milvus 검색
    results = client.search(
        collection_name=collection_name,
        data=[query_text],
        anns_field=search_field,
        limit=limit,
        output_fields=[
            "itemId",
            "itemTitle",
            "itemSubTitle",
            "authorName",
            "publisherName",
            "price",
            "custReviewRank",
            "custReviewCount"
        ]
    )

    return results


def print_search_results(results, limit: int):
    """검색 결과를 출력합니다."""
    print(f"📚 검색 결과 (상위 {limit}개):\n")

    if not results or not results[0]:
        print("   검색 결과가 없습니다.")
        return

    for i, hit in enumerate(results[0], 1):
        entity = hit['entity']
        distance = hit['distance']

        print(f"   {i}. {entity.get('itemTitle', 'N/A')}")
        if entity.get('itemSubTitle'):
            print(f"      부제: {entity.get('itemSubTitle')}")
        print(f"      저자: {entity.get('authorName', 'N/A')}")
        print(f"      출판사: {entity.get('publisherName', 'N/A')}")
        print(f"      가격: {entity.get('price', 0):,}원")

        if entity.get('custReviewRank'):
            print(f"      평점: {entity.get('custReviewRank'):.1f}/5.0 ({entity.get('custReviewCount', 0)}개 리뷰)")

        print(f"      유사도 점수: {distance:.4f}")
        print(f"      Item ID: {entity.get('itemId')}")
        print()


# ============================================================
# 3. 메인 함수
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='이미지에서 텍스트 추론 후 Milvus에서 유사 책 검색'
    )

    # MMOCR 관련 인자
    parser.add_argument('--img', type=str,
                        default='/opt/project/datasets/k1.jpg',
                        help='입력 이미지 경로')
    parser.add_argument('--config', type=str,
                        default='/opt/project/datasets/mmocr/work_dirs/SATRN/satrn_shallow_5e_st_mj_infer.py',
                        help='MMOCR 설정 파일 경로')
    parser.add_argument('--checkpoint', type=str,
                        default='/opt/project/datasets/mmocr/work_dirs/SATRN/epoch_20.pth',
                        help='MMOCR 체크포인트 파일 경로')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='사용할 디바이스 (default: cuda:0)')

    # Milvus 관련 인자
    parser.add_argument('--collection', type=str,
                        default='domestic_book_meta_embedding',
                        help='Milvus 컬렉션 이름')
    parser.add_argument('--milvus-uri', type=str,
                        default='http://10.10.13.129:19530',
                        help='Milvus 서버 URI')
    parser.add_argument('--search-field', type=str,
                        default='itemTitle_embedding',
                        choices=['itemTitle_embedding', 'authorName_embedding'],
                        help='검색할 필드 (제목 또는 저자)')
    parser.add_argument('--limit', type=int, default=5,
                        help='반환할 검색 결과 수')

    args = parser.parse_args()

    # 파일 존재 확인
    img_path = Path(args.img)
    cfg_path = Path(args.config)
    ckpt_path = Path(args.checkpoint)

    for p, name in [(img_path, '이미지'), (cfg_path, '설정 파일'), (ckpt_path, '체크포인트')]:
        if not p.exists():
            sys.exit(f'❌  {name}이 존재하지 않습니다: {p}')

    # GPU 설정
    device = args.device
    if 'cuda' in device and not torch.cuda.is_available():
        print('⚠️  CUDA를 사용할 수 없습니다. CPU로 전환합니다.')
        device = 'cpu'

    print("\n" + "=" * 80)
    print("📖 이미지 텍스트 인식 및 Milvus 검색 시스템")
    print("=" * 80)

    # Step 1: 이미지에서 텍스트 추론
    print("\n🔍 Step 1: 이미지에서 텍스트 추론 중...")
    print(f"   이미지: {img_path}")
    print(f"   디바이스: {device}")

    model, pipeline_cfg = build_model_and_pipeline(cfg_path, ckpt_path, device)
    predicted_text = infer_text_from_image(model, pipeline_cfg, img_path)

    print(f"\n✅ 추론 완료!")
    print(f"📝 인식된 텍스트: '{predicted_text}'")

    # Step 2: Milvus에서 검색
    print(f"\n🔍 Step 2: Milvus에서 유사 책 검색 중...")

    results = search_in_milvus(
        query_text=predicted_text,
        collection_name=args.collection,
        milvus_uri=args.milvus_uri,
        search_field=args.search_field,
        limit=args.limit
    )

    # Step 3: 결과 출력
    print_search_results(results, args.limit)

    print("=" * 80)
    print("✅ 검색 완료!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()