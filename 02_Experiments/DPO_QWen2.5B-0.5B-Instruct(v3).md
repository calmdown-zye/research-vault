---
tags:
  - experiment
  - DPO
  - PRISM
  - ordered-logit
date: "2026-02-24"
model: Qwen/Qwen2.5-0.5B-Instruct
dataset: PRISM (within-model pairs)
status: v3-complete
---

# DPO on PRISM: Ordered Logit Split Experiment (v3)

## Goal
- v2에서는 W2 binary logit (Z_within = I(score_gap > 14))의 유의한 변수로 유저 그룹을 나눔
- v3에서는 **ordered logit** (Y ∈ {1,2,3,4}, score_gap quartile 기반)의 유의한 변수(p<0.05)로 확장
- Ordered logit은 binary → 4-level ordinal로 정보를 더 많이 보존 → 더 정밀한 변수 선별 가능
- v2에서 유의하지 않았던 S4_familiarity, 순환 논리인 S5_behavioral를 제거하고, 새로운 split 4개 추가
- 질문: ordered logit에서 유의한 변수로 나눈 그룹들이 DPO 학습 패턴 차이를 보이는가?

## v2 → v3 변경점

### 방법론 변경
| 항목 | v2 (W2 binary logit) | v3 (Ordered logit) |
|------|---------------------|-------------------|
| 종속변수 | Z_within ∈ {0,1} | Y ∈ {1,2,3,4} (quartile) |
| 모델 | logit(P(Z=1)) = Xβ | logit(P(Y≤j)) = θⱼ - Xβ |
| Thresholds | 14 (single cutoff) | 6, 14, 29 (quartile boundaries) |
| 정보 보존 | score_gap → binary | score_gap → 4-level ordinal |
| 유의 기준 | p < 0.1 (liberal) | p < 0.05 (strict) |

### Split 변경
| v2 Split | v3 Split | 변경 |
|----------|----------|------|
| S1_age | S1_age | 유지 (OL에서도 유의) |
| S2_personalisation | S3_personalisation | 유지 (OL p=0.0008, 가장 강한 stated pref) |
| S3_safety | S4_safety | 유지 (OL p=0.011) |
| S4_familiarity | — | **삭제** (OL에서 유의하지 않음) |
| S5_behavioral | — | **삭제** (순환 논리: score_gap으로 분류 → DPO도 score_gap 기반) |
| S6_region | S8_region | 유지 (null control) |
| — | **S2_education** | **NEW** (OL: +0.47~+0.57, p<0.001) |
| — | **S5_factuality** | **NEW** (OL: -0.00300, p=0.008) |
| — | **S6_helpfulness** | **NEW** (OL: -0.00312, p=0.017) |
| — | **S7_diversity** | **NEW** (OL: +0.00170, p=0.025) |

## Setup

### Model & Infra
- **Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Infra**: SLURM gpu3 (A6000Ada 48GB), bf16
- **Code**: `DPO-mini/260223_v3/` (GitHub: calmdown-zye/DPO-mini)

### Data
- **Source**: HannahRoseKirk/prism-alignment (within-model pairs)
- **Total**: 18,064 pairs, 1,393 users (score_gap ≥ 1)
- mean score_gap=21.5, median=15.0

### Hyperparams (v2와 동일)
| Param | Value |
|-------|-------|
| steps | 2000 |
| batch_size | 8 |
| beta | 0.1 |
| lr | 5e-6 |
| dtype | bf16 |
| max_pairs | 전체 |
| log_every | 25 |

### 8 User Group Splits

| Split | Group A | Group B | OL beta | p-value | 예측 방향 |
|-------|---------|---------|---------|---------|----------|
| S1_age | Young 18-34 (706u, 9376p) | Older 35+ (687u, 8688p) | -0.18~-0.36 | <0.005 | A > B |
| S2_education | Bachelor's+ (830u, 10818p) | Below (556u, 7177p) | +0.47~+0.57 | <0.001 | A > B |
| S3_personalisation | High (705u, 9078p) | Low (688u, 8986p) | +0.00218 | 0.0008 | A > B |
| S4_safety | High (704u, 9251p) | Low (689u, 8813p) | -0.00152 | 0.011 | A < B |
| S5_factuality | High (704u, 9239p) | Low (689u, 8825p) | -0.00300 | 0.008 | A < B |
| S6_helpfulness | High (705u, 9056p) | Low (688u, 9008p) | -0.00312 | 0.017 | A < B |
| S7_diversity | High (713u, 9250p) | Low (680u, 8814p) | +0.00170 | 0.025 | A > B |
| S8_region | Europe (570u, 7542p) | Americas (531u, 6840p) | null | — | A ≈ B |

Total: **17 runs** (1 baseline + 8 splits × 2 groups)

---

## Results

### Baseline
| Metric | v2 | v3 |
|--------|----|----|
| Final loss | 0.6176 | 0.7117 |
| Final margin | 1.6695 | -0.3328 |
| Final pref_acc | 75.00% | 50.00% |
| Mean loss (tail 20%) | 0.7022 | 0.7017 |
| Mean pref_acc (tail 20%) | 51.56% | 51.56% |

> Baseline tail pref_acc 동일 (51.56%). Final snapshot은 변동 크지만 tail average는 안정적.

### Group Comparison Summary

| Split | pref_acc A | pref_acc B | diff | Note | OL 예측 | Result |
|-------|-----------|-----------|------|------|---------|--------|
| S1_age | 89.84% | 87.50% | **+2.34%** | | A > B | **MATCH** |
| S2_education | 92.19% | 96.09% | -3.91% | mild | A > B | MISMATCH |
| S3_personalisation | 90.62% | 89.06% | +1.56% | | A > B | MISMATCH |
| S4_safety | 94.53% | 88.28% | **+6.25%** | **notable** | A < B | MISMATCH |
| S5_factuality | 92.97% | 93.75% | -0.78% | | A < B | MISMATCH |
| **S6_helpfulness** | **87.50%** | **90.62%** | **-3.12%** | mild | **A < B** | **MATCH** |
| S7_diversity | 90.62% | 94.53% | -3.91% | mild | A > B | MISMATCH |
| S8_region | 93.75% | 92.19% | +1.56% | | A ≈ B | **OK** |

> **2/7 MATCH + null OK** (S1_age, S6_helpfulness MATCH)

### Ordered Logit Correspondence Check

| Split | OL Expected | Observed | diff | Result |
|-------|-------------|----------|------|--------|
| **S1_age** | **A > B** | **A > B** | **+2.34%** | **MATCH** |
| S2_education | A > B | A < B | -3.91% | MISMATCH |
| S3_personalisation | A > B | A ≈ B | +1.56% | MISMATCH |
| S4_safety | A < B | A > B | +6.25% | MISMATCH |
| S5_factuality | A < B | A ≈ B | -0.78% | MISMATCH |
| **S6_helpfulness** | **A < B** | **A < B** | **-3.12%** | **MATCH** |
| S7_diversity | A > B | A < B | -3.91% | MISMATCH |
| **S8_region** | **A ≈ B** | **A ≈ B** | **+1.56%** | **OK** |

### Detailed Per-Split Results

**S1_age (Young vs Older) — MATCH**

| Metric | A (Young) | B (Older) | A-B |
|--------|----------|----------|-----|
| N pairs | 9376 | 8688 | |
| final_loss | 0.5183 | 0.5248 | -0.0066 |
| final_margin | 4.6803 | 3.7932 | +0.8871 |
| mean_loss_tail | 0.5202 | 0.5264 | -0.0061 |
| mean_margin_tail | 4.3016 | 4.2695 | +0.0321 |
| mean_pref_acc_tail | 89.84% | 87.50% | +2.34% |
| steps_to_60pct | 50 | 250 | -200 |

> v2에서는 MISMATCH (-3.91%) → v3에서 **MATCH (+2.34%)**로 방향 전환.
> steps_to_60pct에서 Young이 200 step 빠름 — 학습 속도 차이 뚜렷.

**S2_education (Bachelor's+ vs Below) — MISMATCH**

| Metric | A (Bach+) | B (Below) | A-B |
|--------|----------|----------|-----|
| N pairs | 10818 | 7177 | |
| final_loss | 0.4586 | 0.4858 | -0.0272 |
| final_margin | 5.7510 | 4.9517 | +0.7993 |
| mean_loss_tail | 0.5236 | 0.4476 | +0.0760 |
| mean_margin_tail | 4.0678 | 6.8516 | -2.7838 |
| mean_pref_acc_tail | 92.19% | 96.09% | -3.91% |
| steps_to_60pct | 25 | 25 | +0 |

> OL에서 가장 강한 효과 (p<0.001)이었으나 DPO에서 반대 방향.
> Below 그룹이 적은 데이터(7K vs 11K)로 margin이 더 크게 성장 — N 불균형 영향 가능.

**S3_personalisation (High vs Low) — MISMATCH**

| Metric | A (High) | B (Low) | A-B |
|--------|---------|--------|-----|
| N pairs | 9078 | 8986 | |
| final_loss | 0.6765 | 0.4428 | +0.2337 |
| final_margin | 0.3834 | 5.9936 | -5.6101 |
| mean_loss_tail | 0.5200 | 0.5224 | -0.0025 |
| mean_margin_tail | 4.2789 | 4.4248 | -0.1459 |
| mean_pref_acc_tail | 90.62% | 89.06% | +1.56% |
| steps_to_60pct | 25 | 25 | +0 |

> v2에서는 **MATCH (+7.81%)** — v3에서 +1.56%로 축소되어 MISMATCH.
> OL에서 가장 유의한 stated pref (p=0.0008)이지만 DPO 차이는 null 수준.

**S4_safety (High vs Low) — MISMATCH**

| Metric | A (High) | B (Low) | A-B |
|--------|---------|--------|-----|
| N pairs | 9251 | 8813 | |
| final_loss | 0.4536 | 0.4732 | -0.0196 |
| final_margin | 7.0080 | 5.9422 | +1.0658 |
| mean_loss_tail | 0.5139 | 0.5103 | +0.0036 |
| mean_margin_tail | 4.5194 | 4.7407 | -0.2212 |
| mean_pref_acc_tail | 94.53% | 88.28% | +6.25% |
| steps_to_60pct | 100 | 150 | -50 |

> v2에서는 **MATCH (-4.69%)** → v3에서 **+6.25% 방향 반전**.
> 동일한 split 정의인데 다른 결과 → seed 불안정성이 원인일 가능성.

**S5_factuality (High vs Low) — MISMATCH**

| Metric | A (High) | B (Low) | A-B |
|--------|---------|--------|-----|
| N pairs | 9239 | 8825 | |
| final_loss | 0.4774 | 0.5023 | -0.0248 |
| final_margin | 9.9997 | 4.4280 | +5.5717 |
| mean_loss_tail | 0.5123 | 0.5311 | -0.0188 |
| mean_margin_tail | 4.9782 | 3.8326 | +1.1456 |
| mean_pref_acc_tail | 92.97% | 93.75% | -0.78% |
| steps_to_60pct | 25 | 25 | +0 |

> 차이 -0.78%로 사실상 무차별. OL p=0.008이지만 DPO에서는 구분 안 됨.

**S6_helpfulness (High vs Low) — MATCH**

| Metric | A (High) | B (Low) | A-B |
|--------|---------|--------|-----|
| N pairs | 9056 | 9008 | |
| final_loss | 0.5181 | 0.5736 | -0.0555 |
| final_margin | 4.2825 | 3.0292 | +1.2533 |
| mean_loss_tail | 0.5339 | 0.4991 | +0.0348 |
| mean_margin_tail | 3.8978 | 5.0954 | -1.1976 |
| mean_pref_acc_tail | 87.50% | 90.62% | -3.12% |
| steps_to_60pct | 25 | 25 | +0 |

> OL: helpfulness beta=-0.00312 (p=0.017) → high helpfulness = weaker discrimination
> DPO: High helpfulness 87.50% < Low helpfulness 90.62% → 방향 일치!
> margin에서도 Low 그룹이 더 큰 margin으로 수렴 (5.10 vs 3.90).

**S7_diversity (High vs Low) — MISMATCH**

| Metric | A (High) | B (Low) | A-B |
|--------|---------|--------|-----|
| N pairs | 9250 | 8814 | |
| final_loss | 0.4954 | 0.5163 | -0.0209 |
| final_margin | 5.5504 | 4.1169 | +1.4335 |
| mean_loss_tail | 0.4924 | 0.5025 | -0.0100 |
| mean_margin_tail | 5.1942 | 4.7352 | +0.4591 |
| mean_pref_acc_tail | 90.62% | 94.53% | -3.91% |
| steps_to_60pct | 25 | 25 | +0 |

> OL: diversity beta=+0.00170 (positive) → A > B 예측이었으나 실제 A < B.

**S8_region (Europe vs Americas) — Null Control OK**

| Metric | A (Europe) | B (Americas) | A-B |
|--------|----------|------------|-----|
| N pairs | 7542 | 6840 | |
| final_loss | 0.2864 | 0.4694 | -0.1830 |
| final_margin | 12.7347 | 6.5979 | +6.1368 |
| mean_loss_tail | 0.4776 | 0.4447 | +0.0329 |
| mean_margin_tail | 5.7784 | 6.6228 | -0.8444 |
| mean_pref_acc_tail | 93.75% | 92.19% | +1.56% |
| steps_to_60pct | 175 | 125 | +50 |

> pref_acc 차이 1.56%p → 5% 이내, null control 통과. v2 (2.34%)보다 더 작아짐.

---

## Observations

### v2 → v3 비교 요약

| 지표 | v2 (W2 splits) | v3 (OL splits) | 비교 |
|------|---------------|---------------|------|
| Split 수 | 6 | 8 | +2 (새 split 4개, 삭제 2개) |
| 총 run 수 | 13 | 17 | +4 |
| Match 수 | 2/5 | 2/7 | 비율 하락 (40% → 29%) |
| Null control | OK (2.34%) | OK (1.56%) | 더 타이트해짐 |
| Baseline tail pref_acc | 51.56% | 51.56% | 동일 |
| 공통 split 비교 | — | 아래 표 참조 | — |

### 공통 Split에서의 v2 vs v3 비교

| Split | v2 diff | v2 result | v3 diff | v3 result | 안정성 |
|-------|---------|-----------|---------|-----------|-------|
| S1_age | -3.91% | MISMATCH | +2.34% | MATCH | 불안정 (방향 반전) |
| Personalisation | +7.81% | **MATCH** | +1.56% | MISMATCH | 불안정 (크기 축소) |
| Safety | -4.69% | **MATCH** | +6.25% | MISMATCH | 불안정 (방향 반전) |
| Region (null) | -2.34% | OK | +1.56% | OK | 안정 (둘 다 null 수준) |

> **핵심 문제: v2 MATCH였던 personalisation, safety가 v3에서 MISMATCH로 전환.**
> 동일한 split 정의 + 동일한 hyperparams인데 결과가 다름 → **seed 불안정성**이 주요 원인.

### 핵심 발견

1. **S6_helpfulness: 새로운 MATCH (-3.12%)**
   - OL에서 가장 큰 |beta| (-0.00312)를 가진 stated pref
   - DPO에서도 High helpfulness 그룹의 pref_acc가 낮음 → 방향 일치
   - margin에서도 일관된 패턴 (Low > High)

2. **S1_age: v2에서 MISMATCH → v3에서 MATCH**
   - v2: -3.91% → v3: +2.34%로 방향 전환
   - 단일 seed라서 어느 결과가 진짜인지 판단 불가

3. **S4_safety: v2에서 MATCH → v3에서 방향 반전**
   - v2: -4.69% (MATCH) → v3: +6.25% (MISMATCH)
   - 가장 우려되는 불안정성 — 동일 split에서 반대 결과

4. **Null control 안정적 (1.56%)**
   - v2 (2.34%) → v3 (1.56%) → 방법론 자체는 건전
   - 문제는 방법론이 아니라 signal의 크기가 noise 수준이라는 점

5. **새 split (S2, S5, S7) 모두 MISMATCH**
   - S2_education: OL에서 가장 강한 demographic 효과 (p<0.001)이었으나 DPO 반대 방향
   - S5_factuality: 사실상 무차별 (-0.78%)
   - S7_diversity: 반대 방향 (-3.91%)

### 종합 진단

| 진단 | 설명 |
|------|------|
| **Seed 불안정성** | 동일 split의 v2↔v3 결과가 방향까지 바뀜. 단일 seed로는 match/mismatch 판정이 신뢰할 수 없음 |
| **Signal-to-noise** | 대부분의 차이가 null control (1.5~2.3%) 수준. 실질적 차이라 보기 어려움 |
| **OL→DPO 전이 한계** | OL p-value가 작아도 DPO 학습 패턴 차이로 반드시 나타나지 않음. 두 분석의 메커니즘이 다름 |
| **Tail 수렴** | 모든 그룹이 87~96% pref_acc로 수렴 → 차이가 나려면 학습 중반부(step 200~800)에서 봐야 할 수 있음 |

---

## Next Steps
- [ ] **Multi-seed 실험 (최우선)**: 3~5개 seed로 v2 and v3 모두 반복 → 안정적으로 재현되는 패턴만 보고
- [ ] **Learning curve 중반부 분석**: step 200~800에서의 그룹 차이 (tail 수렴 전)
- [ ] **Effect size 분석**: pref_acc diff의 seed별 분포 → null control 대비 유의한 split 식별
- [ ] **모델 크기 확대**: 1.5B or 3B 모델에서 효과 크기가 커지는지 확인
- [ ] v3 plots 확인: `260223_v3/results/*.png`

---

## Reference
- [[DPO_QWen2.5B-0.5B-Instruct|v1 결과 (500 steps, batch 2, 1K pairs)]]
- [[DPO_QWen2.5B-0.5B-Instruct(v2)|v2 결과 (2000 steps, batch 8, W2 splits)]]
- Ordered logit 분석: `prism-alignment/results/260211/case1/repair/derived/out/withinmodel_advanced/method_ordered_logit/`
