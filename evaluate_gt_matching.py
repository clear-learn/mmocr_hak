#!/usr/bin/env python
# ==========================================================
#  evaluate_gt_matching.py - GT 데이터로 추론 정확도 평가
# ==========================================================
"""
GT 폴더의 이미지와 JSON을 사용하여:
1. JSON에서 좌표로 이미지 크롭
2. 크롭된 이미지에서 텍스트 추론 (MMOCR 파이프라인 사용)
3. Milvus에서 유사 책 검색
4. 추론 결과와 GT label을 비교하여 정확도 평가
"""
import argparse
import json
from pathlib import Path
import sys
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmengine.registry import init_default_scope, MODELS
from mmocr.utils import register_all_modules
from pymilvus import MilvusClient
import re # re 임포트 상단으로 이동

# Compose import (구버전 우선)
try:
    from mmocr.datasets.pipelines.compose import Compose
except ImportError:
    from mmcv.transforms import Compose


# ============================================================
# 1. 이미지 크롭 관련
# ============================================================
def get_bbox_from_points(points: List[List[float]]) -> Tuple[int, int, int, int]:
    """
    points에서 bounding box 좌표를 추출합니다.
    rectangle은 2개 포인트, polygon은 4개 포인트

    Returns:
        (x_min, y_min, x_max, y_max)
    """
    points_array = np.array(points)
    x_min = int(points_array[:, 0].min())
    y_min = int(points_array[:, 1].min())
    x_max = int(points_array[:, 0].max())
    y_max = int(points_array[:, 1].max())

    return x_min, y_min, x_max, y_max


def crop_image_from_shape(image_path: Path, shape: Dict, padding: int = 5) -> Image.Image:
    """
    JSON shape 정보로 이미지를 크롭합니다. (원본 사이즈 유지)

    Args:
        image_path: 이미지 경로
        shape: shape 정보 (좌표 포함)
        padding: 크롭 영역 주변에 추가할 패딩 (픽셀)
    """
    try:
        img = Image.open(image_path)
        img_width, img_height = img.size

        x_min, y_min, x_max, y_max = get_bbox_from_points(shape['points'])

        # 패딩 추가 (이미지 경계를 벗어나지 않도록)
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(img_width, x_max + padding)
        y_max = min(img_height, y_max + padding)

        # 이미지 크롭 (원본 사이즈)
        cropped = img.crop((x_min, y_min, x_max, y_max))
        return cropped
    except Exception as e:
        print(f"⚠️ 이미지 크롭 오류 ({image_path.name}): {e}")
        return None

# ============================================================
# 2. MMOCR 모델 로드 및 추론
# ============================================================
def build_model_and_pipeline(cfg_path: Path,
                             ckpt_path: Path,
                             device: str = 'cpu'):
    """MMOCR 모델과 추론 파이프라인 설정을 빌드합니다."""
    try:
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

        # inference 전용 파이프라인 설정 가져오기
        orig_pipeline = (cfg.get('test_pipeline')
                         or cfg.test_dataloader.dataset.pipeline)

        # annotation을 요구하는 변환 제거
        inference_pipeline_cfg = [
            t for t in orig_pipeline
            if 'Annotation' not in t['type'] and 'Label' not in t['type']
        ]

        # (!!!) 중요: 파이프라인 설정 자체를 반환 (Compose 객체 X)
        return model, inference_pipeline_cfg
    except Exception as e:
        sys.exit(f"❌ 모델 또는 파이프라인 빌드 실패: {e}")


@torch.inference_mode()
def infer_text_from_image(model, pipeline_cfg: List[Dict], img_path: Path) -> str:
    """
    이미지 경로를 받아 MMOCR 파이프라인을 적용하고 텍스트를 추론합니다.
    pipeline_cfg는 Compose 객체가 아닌 설정 리스트여야 합니다.
    """
    try:
        # (!!!) 수정: Compose 객체를 함수 내에서 생성
        pipeline = Compose(pipeline_cfg)
        # 이미지 경로만 전달, 파이프라인이 LoadImageFromFile부터 처리
        data = dict(img_path=str(img_path), img_shape=Image.open(img_path).size[::-1]) # img_shape 추가 (ConditionApply용)
        data = pipeline(data)

        # 배치 형태로 감싸기
        # MMOCR 최신 버전 호환성을 위해 `inputs` 대신 `imgs` 사용 고려
        # (단, 사용 중인 버전에 따라 inputs가 맞을 수 있음)
        inputs_key = 'imgs' if 'imgs' in data else 'inputs'
        batch_data = {
            inputs_key: [data[inputs_key]],
            'data_samples': [data['data_samples']]
        }

        pred_sample = model.test_step(batch_data)[0]

        # pred_text 추출 (여러 가능한 형식 처리)
        pred_text_obj = pred_sample.pred_text

        # MMOCR 버전에 따른 pred_text 추출 방식 분기
        if hasattr(pred_text_obj, 'item'): # 예: LabelData 객체
             pred_text = pred_text_obj.item
        elif isinstance(pred_text_obj, dict) and 'text' in pred_text_obj: # 예: {'text': '...', 'score': ...} 형태
             pred_text = pred_text_obj['text']
        elif isinstance(pred_text_obj, str):
             pred_text = pred_text_obj
        else: # 기타 경우 (예상치 못한 형식)
             pred_text = str(pred_text_obj)

        # OCR 결과 후처리 (evaluate_openai.py와 동일한 로직)
        pred_text = normalize_text_for_ocr(parse_item_text(pred_text))

        return pred_text
    except FileNotFoundError:
        print(f"    ⚠️ OCR 오류: 파일을 찾을 수 없음 - {img_path}")
        return ""
    except Exception as e:
        print(f"    ⚠️ OCR 추론 중 예외 발생 ({img_path.name}): {e}")
        # traceback.print_exc() # 상세 에러 확인 필요 시 주석 해제
        return ""


# ============================================================
# 3. Milvus 검색 (변경 없음)
# ============================================================
def search_in_milvus(
    query_text: str,
    collection_name: str = "domestic_book_meta_embedding",
    milvus_uri: str = "http://10.10.13.129:19530",
    search_field: str = "itemTitle_embedding",
    limit: int = 5
):
    """Milvus에서 유사도 검색을 수행합니다."""
    # Milvus 검색은 추론 텍스트가 비어있으면 의미가 없으므로 체크
    if not query_text:
        print("    ⚠️ Milvus 검색 건너뜀 (OCR 결과 없음)")
        return None
    try:
        client = MilvusClient(uri=milvus_uri)

        results = client.search(
            collection_name=collection_name,
            data=[query_text], # 임베딩 모델이 내장되어 있지 않으므로 텍스트 직접 전달
            anns_field=search_field,
            limit=limit,
            output_fields=[
                "itemId",
                "itemTitle",
                "itemSubTitle",
                "authorName",
            ]
        )
        # 결과 구조 확인 (결과가 없을 수 있음)
        if results and results[0]:
             # score 대신 distance 사용 확인
             if isinstance(results[0][0].get('score'), float):
                  dist_key = 'score'
             else:
                  dist_key = 'distance'

             # entity 구조 확인
             if 'entity' in results[0][0]:
                  entity_key = 'entity'
             else:
                  entity_key = None # entity 없이 바로 필드들이 나올 경우

             parsed_results = []
             for hit in results[0]:
                  entity_data = hit.get(entity_key) if entity_key else hit
                  parsed_results.append({
                       'id': hit['id'],
                       'distance': float(hit[dist_key]),
                       'entity': entity_data if entity_data else {}
                  })
             return [parsed_results] # 원본 함수와 동일한 형태로 반환 [[]]
        else:
             return None # 검색 결과 없음

    except Exception as e:
        print(f"    ⚠️ Milvus 검색 오류: {e}")
        return None

# ============================================================
# 4. OCR 결과 후처리 (evaluate_openai.py와 동일)
# ============================================================
def parse_item_text(text_str):
    """Parse dict-string format: "{'item': 'text', 'score': [...]}" """
    text_str = str(text_str)
    if text_str.startswith('{') and "'item':" in text_str:
        try:
            import ast
            parsed = ast.literal_eval(text_str)
            if isinstance(parsed, dict) and 'item' in parsed:
                return str(parsed['item'])
        except:
            pass
    return text_str


def normalize_text_for_ocr(text_str):
    """Remove special tokens and normalize whitespace for OCR output."""
    import re
    text_str = str(text_str)

    # Remove special tokens
    special_tokens = ['<UNK>', '<BOS>', '<EOS>', '<PAD>', '<unk>', '<bos>', '<eos>', '<pad>']
    for token in special_tokens:
        text_str = text_str.replace(token, ' ')

    # Normalize whitespace
    text_str = re.sub(r'\s+', ' ', text_str).strip()

    return text_str


# ============================================================
# 5. 평가용 텍스트 정규화 (변경 없음)
# ============================================================
def normalize_text(text: str) -> str:
    """텍스트 정규화 (공백, 특수문자 제거) - 평가용"""
    # 공백 제거
    text = re.sub(r'\s+', '', text)
    # 추가: 기본적인 특수문자 제거 (옵션)
    # text = re.sub(r'[^\w]', '', text) # 알파벳, 숫자, 밑줄(_)만 남김
    return text.lower()


def check_match(predicted: str, ground_truth: str, top_results: list) -> Dict:
    """
    예측 결과와 GT를 비교하고, Milvus 검색 결과도 평가합니다.
    """
    pred_norm = normalize_text(predicted)
    gt_norm = normalize_text(ground_truth)

    # OCR 정확도
    ocr_exact_match = (pred_norm == gt_norm)
    # 부분 일치 조건 완화: GT가 예측값 안에 포함되는지만 확인
    # (예: 예측='토지1권', GT='토지' -> 부분 일치 O)
    ocr_partial_match = (gt_norm in pred_norm) if pred_norm else False # 예측값이 비어있으면 부분일치 X

    # Milvus 검색 결과에서 정답 찾기
    milvus_match = False
    milvus_rank = None
    milvus_score = None
    search_results_info = []

    if top_results and top_results[0]:
        for rank, hit in enumerate(top_results[0], 1):
            entity = hit.get('entity', {}) # entity가 없을 경우 대비
            distance = hit.get('distance') # score/distance 통일
            title = entity.get('itemTitle', '')
            title_norm = normalize_text(title)

            is_match = (gt_norm in title_norm) if title_norm else False

            # 검색 결과 정보 저장
            search_results_info.append({
                'rank': rank,
                'title': title,
                'score': distance, # score/distance 통일
                'is_match': is_match
            })

            # 정답과 일치하는지 확인 (이미 매칭된 경우 순위만 기록)
            if is_match and not milvus_match:
                milvus_match = True
                milvus_rank = rank
                milvus_score = distance

    return {
        'ocr_exact_match': ocr_exact_match,
        'ocr_partial_match': ocr_partial_match,
        'milvus_match': milvus_match,
        'milvus_rank': milvus_rank,
        'milvus_score': milvus_score,
        'top_results': search_results_info,
    }


# ============================================================
# 6. 메인 평가 함수 (수정됨)
# ============================================================
def evaluate_gt_data(
    gt_folder: Path,
    model,
    pipeline_cfg: List[Dict], # (!!!) 수정: Compose 객체 대신 설정 리스트 받음
    milvus_collection: str,
    milvus_uri: str,
    search_field: str,
    top_k: int = 5,
    temp_dir: Path = None
):
    """
    GT 폴더의 모든 데이터를 평가합니다.
    이미지 처리는 infer_text_from_image 내부의 파이프라인에 맡깁니다.
    """

    if temp_dir is None:
        temp_dir = gt_folder / "temp_crops"
    temp_dir.mkdir(exist_ok=True)

    # JSON 파일 목록
    json_files = sorted(gt_folder.glob("*.json"))

    if not json_files:
        print("❌ JSON 파일을 찾을 수 없습니다.")
        return

    print(f"📊 총 {len(json_files)}개의 이미지 평가 시작 (MMOCR 파이프라인 적용)\n") # 로그 수정

    total_shapes = 0
    ocr_exact_correct = 0
    ocr_partial_correct = 0
    milvus_correct = 0

    detailed_results = []

    for json_file in json_files:
        # JSON 로드
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ JSON 파싱 오류: {json_file.name}")
            continue
        except Exception as e:
            print(f"⚠️ JSON 로드 오류 ({json_file.name}): {e}")
            continue

        # 이미지 파일 경로
        image_path = gt_folder / data.get('imagePath', '') # 키 존재 확인
        if not data.get('imagePath') or not image_path.exists():
            print(f"⚠️ 이미지 파일 경로 누락 또는 파일 없음: {image_path}")
            continue

        print(f"\n{'='*80}")
        print(f"📷 처리 중: {image_path.name}")
        print(f"{'='*80}\n")

        # 각 shape 처리
        shapes = data.get('shapes', [])
        if not shapes:
            print(f"    ⚠️ 'shapes' 정보 없음: {json_file.name}")
            continue

        for idx, shape in enumerate(shapes):
            ground_truth = shape.get('label')
            if not ground_truth:
                print(f"    ⚠️ shape {idx}: 'label' 정보 없음")
                continue

            total_shapes += 1

            # 1. 이미지 크롭 (패딩 포함, 원본 해상도 유지)
            cropped_img = crop_image_from_shape(image_path, shape)
            if cropped_img is None:
                continue # 크롭 실패 시 다음 shape으로

            # 2. 크롭된 이미지를 임시 파일로 저장 (MMOCR 파이프라인 입력용)
            crop_path = temp_dir / f"{image_path.stem}_{idx}.jpg"
            try:
                # RGBA -> RGB 변환 (저장 전)
                if cropped_img.mode == 'RGBA':
                    background = Image.new('RGB', cropped_img.size, (255, 255, 255))
                    background.paste(cropped_img, mask=cropped_img.split()[3])
                    cropped_img = background
                elif cropped_img.mode != 'RGB':
                    cropped_img = cropped_img.convert('RGB')

                cropped_img.save(crop_path)
            except Exception as e:
                 print(f"    ⚠️ 임시 파일 저장 오류 ({crop_path.name}): {e}")
                 continue # 이 shape 처리 불가

            # (!!!) 제거: 수동 회전 및 리사이즈 로직 삭제됨

            # 크롭된 원본 이미지 크기 정보 출력 (디버깅용)
            if total_shapes <= 3:  # 처음 3개만 상세 정보 출력
                print(f"      [DEBUG] 크롭된 원본 이미지: {crop_path.name}, 크기: {cropped_img.size}")

            # 3. OCR 추론 (MMOCR 파이프라인이 회전/리사이즈 처리)
            predicted_text = infer_text_from_image(model, pipeline_cfg, crop_path)

            # 4. Milvus 검색
            search_results = search_in_milvus(
                predicted_text,
                collection_name=milvus_collection,
                milvus_uri=milvus_uri,
                search_field=search_field,
                limit=top_k
            )

            # 5. 정확도 평가
            match_result = check_match(predicted_text, ground_truth, search_results)

            # 통계 업데이트
            if match_result['ocr_exact_match']:
                ocr_exact_correct += 1
            if match_result['ocr_partial_match']:
                ocr_partial_correct += 1
            if match_result['milvus_match']:
                milvus_correct += 1

            # 결과 저장
            result_entry = {
                'image': image_path.name,
                'shape_idx': idx,
                'ground_truth': ground_truth,
                'predicted': predicted_text,
                'ocr_exact': match_result['ocr_exact_match'],
                'ocr_partial': match_result['ocr_partial_match'],
                'milvus_match': match_result['milvus_match'],
                'milvus_rank': match_result['milvus_rank'],
                'milvus_score': match_result['milvus_score'],
                'milvus_in_top_k': match_result['milvus_rank'] is not None and match_result['milvus_rank'] <= top_k, # top-k 사용
                'top_search_results': match_result['top_results'],
                # 'ocr_input_size': cropped_img.size, # 삭제 또는 파이프라인 후 크기 기록 필요 시 수정
            }
            detailed_results.append(result_entry)

            # 개별 결과 출력
            print(f"  [{total_shapes}] GT: {ground_truth}")
            print(f"        예측: {predicted_text}")
            print(f"        OCR 정확 일치: {'✅' if match_result['ocr_exact_match'] else '❌'}")
            print(f"        OCR 부분 일치: {'✅' if match_result['ocr_partial_match'] else '❌'}")

            if match_result['milvus_match']:
                print(f"        Milvus 매칭: ✅ (순위: {match_result['milvus_rank']}, 점수: {match_result['milvus_score']:.4f})")
                print(f"        Top-{top_k} 포함: {'✅' if match_result['milvus_rank'] <= top_k else '❌'}")
            else:
                print(f"        Milvus 매칭: ❌")

            # Top 검색 결과 출력
            if match_result['top_results']:
                print(f"        Top-{len(match_result['top_results'])} 검색 결과:")
                for res in match_result['top_results']:
                    match_marker = " ✅" if res['is_match'] else ""
                    print(f"         {res['rank']}. {res['title']} (점수: {res['score']:.4f}){match_marker}")

            print()

    # ============================================================
    # 6. 최종 통계 출력
    # ============================================================
    print("\n" + "="*80)
    print("📊 최종 평가 결과")
    print("="*80)
    if total_shapes == 0:
        print("\n평가할 샘플이 없습니다.")
    else:
        ocr_exact_acc = ocr_exact_correct / total_shapes * 100
        ocr_partial_acc = ocr_partial_correct / total_shapes * 100
        milvus_acc = milvus_correct / total_shapes * 100

        print(f"\n총 샘플 수: {total_shapes}")
        print(f"\nOCR 정확도:")
        print(f"  - 정확 일치 (Exact Match): {ocr_exact_correct}/{total_shapes} ({ocr_exact_acc:.2f}%)")
        print(f"  - 부분 일치 (Partial Match): {ocr_partial_correct}/{total_shapes} ({ocr_partial_acc:.2f}%)")
        print(f"\nMilvus 매칭 성공률 (Recall@{top_k}):")
        print(f"  - Top-{top_k} 내 정답 포함: {milvus_correct}/{total_shapes} ({milvus_acc:.2f}%)")
    print("="*80 + "\n")

    # 결과를 JSON으로 저장
    result_file = gt_folder / "evaluation_results.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_samples': total_shapes,
                'ocr_exact_correct': ocr_exact_correct,
                'ocr_partial_correct': ocr_partial_correct,
                'milvus_correct': milvus_correct,
                'ocr_exact_accuracy': f"{ocr_exact_acc:.2f}%" if total_shapes else "N/A",
                'ocr_partial_accuracy': f"{ocr_partial_acc:.2f}%" if total_shapes else "N/A",
                f'milvus_recall_at_{top_k}': f"{milvus_acc:.2f}%" if total_shapes else "N/A",
                'top_k': top_k,
                # 'target_height': target_height, # 삭제됨
            },
            'detailed_results': detailed_results
        }, f, ensure_ascii=False, indent=2)

    print(f"✅ 상세 결과 저장: {result_file}")


# ============================================================
# 7. 메인 함수
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='GT 데이터로 OCR 및 Milvus 매칭 정확도 평가'
    )

    # 경로 관련 인자
    # (!!!) 수정: required=True 제거, default 경로 설정
    parser.add_argument('--gt-folder', type=str,
                        default='/opt/project/datasets/mmocr/GT', # 예시 기본 경로
                        help='GT 폴더 경로 (이미지 및 JSON 파일 포함)')

    # MMOCR 관련 인자
    # (!!!) 수정: required=True 제거, default 경로 설정
    parser.add_argument('--config', type=str,
                        default='/opt/project/datasets/mmocr/work_dirs/SATRN_original_size/satrn_shallow_5e_st_mj_aladin_original_size.py', # 예시 기본 경로
                        help='MMOCR 설정 파일 경로 (*.py)')
    # (!!!) 수정: required=True 제거, default 경로 설정
    parser.add_argument('--checkpoint', type=str,
                        default='/opt/project/datasets/mmocr/work_dirs/SATRN_original_size_/epoch_14.pth', # 예시 기본 경로
                        help='MMOCR 체크포인트 파일 경로 (*.pth)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='사용할 디바이스 (e.g., cuda:0, cpu)')

    # Milvus 관련 인자 (기존과 동일)
    parser.add_argument('--collection', type=str,
                        default='domestic_book_meta_embedding',
                        help='Milvus 컬렉션 이름')
    parser.add_argument('--milvus-uri', type=str,
                        default='http://10.10.13.129:19530', # 기본값 유지 또는 변경
                        help='Milvus 서버 URI')
    parser.add_argument('--search-field', type=str,
                        default='itemTitle_embedding',
                        choices=['itemTitle_embedding', 'authorName_embedding'],
                        help='검색할 임베딩 필드')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Milvus Top-K 검색')
    parser.add_argument('--temp-dir', type=str, default=None,
                        help='크롭된 이미지 임시 저장 경로 (기본값: GT폴더/temp_crops)')

    args = parser.parse_args()

    # 경로 확인 (이제 default 값이 있으므로 존재 여부만 확인)
    gt_folder = Path(args.gt_folder).resolve()
    cfg_path = Path(args.config).resolve()
    ckpt_path = Path(args.checkpoint).resolve()

    if not gt_folder.is_dir():
        sys.exit(f'❌ GT 폴더가 존재하지 않거나 디렉토리가 아닙니다: {gt_folder}')
    if not cfg_path.is_file():
        sys.exit(f'❌ 설정 파일이 존재하지 않거나 파일이 아닙니다: {cfg_path}')
    if not ckpt_path.is_file():
        sys.exit(f'❌ 체크포인트가 존재하지 않거나 파일이 아닙니다: {ckpt_path}')

    # 임시 폴더 경로 설정
    temp_dir = Path(args.temp_dir).resolve() if args.temp_dir else gt_folder / "temp_crops"

    # GPU 설정
    device = args.device
    if 'cuda' in device:
        if not torch.cuda.is_available():
            print('⚠️ CUDA를 사용할 수 없습니다. CPU로 전환합니다.')
            device = 'cpu'
        else:
            try:
                gpu_id = int(device.split(':')[-1])
                if gpu_id >= torch.cuda.device_count():
                     print(f'⚠️ 지정된 GPU ID({gpu_id})가 사용 가능한 GPU 수({torch.cuda.device_count()})보다 큽니다. cuda:0으로 설정합니다.')
                     device = 'cuda:0'
            except (ValueError, IndexError):
                 print(f'⚠️ 잘못된 device 형식입니다 ({device}). cuda:0으로 설정합니다.')
                 device = 'cuda:0'

    print("\n" + "="*80)
    print("📖 GT 데이터 평가 시스템")
    print("="*80)
    print(f"\n설정:")
    print(f"  - GT 폴더: {gt_folder}")
    print(f"  - MMOCR 설정: {cfg_path.name}")
    print(f"  - MMOCR 체크포인트: {ckpt_path.name}")
    print(f"  - 디바이스: {device}")
    print(f"  - Milvus URI: {args.milvus_uri}")
    print(f"  - Milvus 컬렉션: {args.collection}")
    print(f"  - Milvus 검색 필드: {args.search_field}")
    print(f"  - Milvus Top-K: {args.top_k}")
    print(f"  - 임시 폴더: {temp_dir}")

    # 모델 로드
    print(f"\n🔧 모델 로딩 중...")
    model, pipeline_cfg = build_model_and_pipeline(cfg_path, ckpt_path, device)
    print(f"✅ 모델 로딩 완료!")

    # 평가 시작
    evaluate_gt_data(
        gt_folder=gt_folder,
        model=model,
        pipeline_cfg=pipeline_cfg,
        milvus_collection=args.collection,
        milvus_uri=args.milvus_uri,
        search_field=args.search_field,
        top_k=args.top_k,
        temp_dir=temp_dir
    )

if __name__ == '__main__':
    main()