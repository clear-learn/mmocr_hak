# True 2D Attention Encoder-Decoder

완전한 2D Attention 기반 OCR 인코더-디코더 모델입니다.

## 🎯 핵심 특징

### **SATRN과의 차이점**

| 특징 | SATRN (기존) | True 2D (신규) |
|-----|-------------|---------------|
| **Encoder Attention** | 1D flatten + 2D pos encoding | ⭐ 진짜 2D Self-Attention with relative position bias |
| **Cross-Attention** | 1D flatten + 2D pos encoding | ⭐ Deformable Cross-Attention (grid_sample) |
| **공간 구조** | 부분적으로 보존 (유사 2D) | **완전히 보존** |
| **Spatial Sampling** | 모든 위치에 동일하게 attend | **학습 가능한 offset으로 adaptive sampling** |
| **Inductive Bias** | 약함 | **강함** (2D relative position bias) |

---

## 📂 파일 구조

```
mmocr/models/textrecog/
├── encoders/
│   └── true2d_encoder.py          # 완전한 2D Encoder
└── decoders/
    └── true2d_decoder.py          # Deformable Cross-Attention Decoder
```

---

## 🔬 구조 상세

### **1. True2DEncoder**

#### **특징:**
- **2D Relative Position Bias**: 각 pixel pair의 2D 거리를 attention score에 반영
- **Window Attention (Optional)**: local window 내에서만 attend 가능 (효율성)
- **공간 구조 완전 보존**: flatten하지만 2D bias로 spatial structure 유지

#### **핵심 코드:**
```python
class True2DSelfAttention(nn.Module):
    def forward(self, x, H, W):
        # (B, H*W, D) → (B, n_head, H*W, H*W)
        attn = (q @ k.T) * scale

        # ⭐ 2D relative position bias 추가
        attn = attn + self._get_relative_position_bias(H, W)

        return output
```

**2D Relative Position Bias란?**
- 모든 pixel pair `(i, j)`에 대해 2D 거리 `(Δh, Δw)`를 계산
- Learnable bias table에서 lookup: `bias[Δh, Δw]`
- Attention score에 더해짐 → **가까운 pixel끼리 더 강하게 attend**

---

### **2. True2DDecoder**

#### **특징:**
- **Deformable Cross-Attention**: 각 query가 2D feature map에서 K개의 point를 학습하여 sampling
- **Adaptive Spatial Sampling**: offset을 학습해서 중요한 위치만 attend
- **Grid Sample**: `F.grid_sample`로 진짜 bilinear interpolation

#### **핵심 코드:**
```python
class DeformableCrossAttention2D(nn.Module):
    def forward(self, query, reference_points, value_2d, H, W):
        # 1. Offset 예측
        offsets = self.sampling_offsets(query)  # (B, T, n_head, n_points, 2)

        # 2. Sampling locations 계산
        locations = reference_points + offsets / [W, H]

        # 3. ⭐ grid_sample로 2D spatial sampling
        for head in range(n_head):
            sampled = F.grid_sample(value_2d, locations[head])
            output[head] = weighted_sum(sampled, attention_weights)

        return output
```

**Deformable Attention이란?**
- 기존: 모든 H×W 위치에 attend (비효율적)
- 신규: **각 query마다 K개 (예: 4개)만 sampling**
  - K개 위치는 학습됨 (offset network)
  - `grid_sample`로 정확한 2D interpolation
  - 계산량 `O(T×H×W)` → `O(T×K)` 대폭 감소

---

## 🚀 사용 방법

### **Config 예시**

```python
# configs/textrecog/true2d/true2d_outdoor.py

model = dict(
    type='SATRN',  # 기존 wrapper 재사용

    backbone=dict(
        type='ResNet',
        depth=50,
        ...
    ),

    encoder=dict(
        type='True2DEncoder',
        in_channels=2048,
        d_model=512,
        n_layers=6,
        n_head=8,
        d_inner=2048,
        dropout=0.1,
        window_size=None,  # None = global attention
    ),

    decoder=dict(
        type='True2DDecoder',
        n_layers=6,
        d_embedding=512,
        d_model=512,
        n_head=8,
        d_inner=2048,
        n_points=4,  # 각 query마다 4개 point sampling
        dropout=0.1,

        dictionary=dictionary,
        max_seq_len=25,
        enc_channels=512,

        module_loss=dict(type='CEModuleLoss', ...),
        postprocessor=dict(type='AttentionPostprocessor'),
    ),

    data_preprocessor=dict(...),
)
```

### **학습**

```bash
python tools/train.py configs/textrecog/true2d/true2d_outdoor.py
```

---

## 🎓 기대 효과

### **장점**
1. ✅ **더 강력한 Spatial Modeling**
   - 2D relative bias → 공간 구조 완전 반영
   - Deformable attention → adaptive sampling

2. ✅ **계산 효율성**
   - Deformable: `O(T×K)` vs Full Attention: `O(T×H×W)`
   - K=4일 때 약 50-100배 감소

3. ✅ **더 나은 일반화**
   - 2D inductive bias → 다양한 layout에 강건
   - Curved text, rotation, perspective에 유리

### **단점**
1. ❌ **구조 복잡도**
   - SATRN 대비 구현 복잡
   - Debugging 어려움

2. ❌ **학습 불안정 가능성**
   - Deformable offset 학습 초기에 불안정할 수 있음
   - Proper initialization 중요

---

## 📊 권장 하이퍼파라미터

| 파라미터 | 권장값 | 설명 |
|---------|-------|------|
| `n_layers` | 6 | Encoder/Decoder layer 수 |
| `d_model` | 512 | Model dimension |
| `n_head` | 8 | Attention head 수 |
| `n_points` | 4 or 8 | Deformable sampling points (4가 일반적) |
| `dropout` | 0.1 | Dropout rate |
| `learning_rate` | 3e-4 | AdamW 기준 |
| `warmup_steps` | 10000 | LR warmup |

---

## 🔧 디버깅 팁

### **1. Offset 시각화**
```python
# Decoder forward에서
sampling_locations = reference_points + offsets / [W, H]
print(f"Sampling locations range: {sampling_locations.min()}, {sampling_locations.max()}")

# 범위가 [0, 1] 벗어나면 문제
```

### **2. Attention Weight 확인**
```python
# Layer forward에서
print(f"Attention weights: {attention_weights.mean()}, std: {attention_weights.std()}")

# Mean ~= 1/n_points, std가 너무 크면 학습 불안정
```

### **3. Gradient 체크**
```python
# Training loop에서
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm = {param.grad.norm()}")

# sampling_offsets.bias의 gradient가 0이면 offset 학습 안 됨
```

---

## 📚 참고 논문

1. **SATRN (기존)**
   - "On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention"
   - https://arxiv.org/abs/1910.04396

2. **Deformable Attention**
   - "Deformable DETR: Deformable Transformers for End-to-End Object Detection"
   - https://arxiv.org/abs/2010.04159

3. **Swin Transformer (Relative Position Bias)**
   - "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
   - https://arxiv.org/abs/2103.14030

---

## ⚠️ 주의사항

1. **현재는 코드만 작성됨** - 실제 적용 전에 충분한 테스트 필요
2. **MMOCR registry 등록 필요** - `__init__.py`에 import 추가해야 함
3. **사전 학습 없음** - 처음부터 학습해야 함 (SATRN pretrained weights 사용 불가)
4. **메모리 사용량** - Deformable은 SATRN보다 약간 더 많은 메모리 필요

---

## 🎯 다음 단계

1. **Registry 등록**
   ```python
   # mmocr/models/textrecog/encoders/__init__.py
   from .true2d_encoder import True2DEncoder

   # mmocr/models/textrecog/decoders/__init__.py
   from .true2d_decoder import True2DDecoder
   ```

2. **Config 파일 작성**
   - `configs/textrecog/true2d/true2d_outdoor.py`

3. **학습 시작**
   - 소규모 데이터로 먼저 테스트
   - Offset visualization으로 학습 확인

4. **SATRN과 비교**
   - 같은 데이터셋에서 정확도/속도 비교
   - Ablation study (2D bias, deformable 각각의 효과)

---

**작성자 노트**: 이 코드는 실험적 구현입니다. 실제 프로덕션 사용 전에 충분한 검증이 필요합니다.