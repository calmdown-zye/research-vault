---
tags:
  - experiment
  - DPO
  - PRISM
date: "2026-02-23"
model: Qwen/Qwen2.5-0.5B-Instruct
dataset: PRISM (within-model pairs)
status: v2-complete
---

# DPO on PRISM: User Group Split Experiment (v2)

## Goal
- W2 (logit on Z_within, cluster SE)에서 유저 특성이 선호 강도에 영향을 준다는 통계적 증거를 발견
- 이를 DPO 학습 패턴 차이로도 보일 수 있는지 검증 (computational evidence)
- 6가지 유저 그룹 분할로 DPO를 따로 돌려서, 그룹 간 학습 패턴(loss, margin, pref_acc) 차이를 관찰
- v1의 한계 (학습 부족, pref_acc 해상도 부족, null control 실패)를 해결한 재실험

## Setup

### Model & Infra
- **Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Infra**: SLURM gpu3 (A6000Ada 48GB), bf16
- **Code**: `DPO-mini/260223/` (GitHub: calmdown-zye/DPO-mini)

### Data: PRISM within-model pairs
- **Source**: HannahRoseKirk/prism-alignment
- **Pair 추출**: Turn 1+에서 같은 user, 같은 model, 같은 prompt에 대한 2개 응답
- **선호 기준**: user가 부여한 score (0-100) 기반, 높은 score = chosen
- **Total**: ~18,000 within-model pairs, ~1,400 users
- W2 분석 설계와 정확히 대응 (within-model, same prompt)

### Hyperparams: v1 vs v2

| Param | v1 | v2 | 변경 이유 |
|-------|----|----|----------|
| steps | 500 | **2000** | v1에서 학습 부족 (baseline pref_acc 50% 정체) |
| batch_size | 2 | **8** | v1 pref_acc {0, 0.5, 1.0}만 가능 → v2는 9단계 해상도 |
| max_pairs | 1000 | **전체** | 그룹당 ~9K pairs 전부 활용 |
| log_every | 10 | **25** | smoother curves |
| beta | 0.1 | 0.1 | 유지 |
| lr | 5e-6 | 5e-6 | 유지 |
| dtype | bf16 | bf16 | 유지 |

### 6 User Group Splits

| Split | Group A | Group B | W2 예측 | 분류 기준 |
|-------|---------|---------|---------|----------|
| S1_age | Young (18-34) | Older (35+) | A > B | survey |
| S2_personalisation | High | Low | A > B | survey (median split) |
| S3_safety | High | Low | A < B | survey (median split) |
| S4_familiarity | Very familiar | Not/Somewhat | 탐색적 | survey |
| S5_behavioral | High discriminators | Low discriminators | A > B | 행동 데이터 (순환 논리 주의) |
| S6_region | Europe | Americas | A = B | survey (null control) |

### Baseline comparison
- [[DPO_QWen2.5B-0.5B-Instruct|v1 결과 (500 steps, batch 2, 1K pairs)]]

---

## Results (v2 — 2000 steps, batch 8, all pairs)

### Baseline
| Metric | v1 | v2 |
|--------|----|----|
| Final loss | 0.7109 | **0.6176** |
| Final margin | -0.3483 | **1.6695** |
| Final pref_acc | 50.00% | **75.00%** |
| Mean loss (tail 20%) | 0.7091 | **0.7022** |
| Mean pref_acc (tail 20%) | 50.00% | **51.56%** |

### Group Comparison Summary

| Split | pref_acc A | pref_acc B | diff | Note |
|-------|-----------|-----------|------|------|
| S1_age | 89.84% | 93.75% | -3.91% | mild |
| **S2_personalisation** | **92.19%** | **84.38%** | **+7.81%** | **notable** |
| S3_safety | 91.41% | 96.09% | -4.69% | mild |
| S4_familiarity | 94.53% | 92.19% | +2.34% | small |
| S5_behavioral | 92.19% | 93.75% | -1.56% | negligible |
| S6_region | 94.53% | 96.88% | -2.34% | **null control OK** |

### W2 Correspondence Check

| Split | W2 Expected | Observed | diff | Result |
|-------|-------------|----------|------|--------|
| S1_age | A > B | A < B | -3.91% | MISMATCH |
| **S2_personalisation** | **A > B** | **A > B** | **+7.81%** | **MATCH** |
| **S3_safety** | **A < B** | **A < B** | **-4.69%** | **MATCH** |
| S4_familiarity | Exploratory | A > B | +2.34% | exploratory |
| S5_behavioral | A > B | A = B | -1.56% | MISMATCH (순환 논리) |
| S6_region | A = B | A = B | -2.34% | **OK** |

> v1: 1/5 MATCH, null FAILED → **v2: 2/5 MATCH, null OK**

### Detailed Per-Split Results

**S1_age (Young vs Older)**

| Metric | A (Young) | B (Older) | A-B |
|--------|----------|----------|-----|
| N pairs | 9376 | 8688 | |
| final_loss | 0.5066 | 0.5079 | -0.0013 |
| final_margin | 5.1167 | 4.6636 | +0.4531 |
| mean_loss_tail | 0.5223 | 0.5042 | +0.0181 |
| mean_margin_tail | 4.4569 | 4.8792 | -0.4222 |
| mean_pref_acc_tail | 89.84% | 93.75% | -3.91% |
| steps_to_60pct | 75 | 50 | +25 |

**S2_personalisation (High vs Low) — W2 MATCH**

| Metric | A (High) | B (Low) | A-B |
|--------|---------|--------|-----|
| N pairs | 9078 | 8986 | |
| final_loss | 0.6261 | 0.4326 | +0.1936 |
| final_margin | 1.5376 | 8.0284 | -6.4908 |
| mean_loss_tail | 0.5308 | 0.5317 | -0.0009 |
| mean_margin_tail | 3.8730 | 4.2534 | -0.3804 |
| mean_pref_acc_tail | 92.19% | 84.38% | +7.81% |
| steps_to_60pct | 50 | 50 | +0 |

> W2: personalisation coef=+0.0022, p=0.085 → DPO에서도 High pers 그룹의 pref_acc가 +7.81%p 높음

**S3_safety (High vs Low) — W2 MATCH**

| Metric | A (High) | B (Low) | A-B |
|--------|---------|--------|-----|
| N pairs | 9251 | 8813 | |
| final_loss | 0.4782 | 0.4601 | +0.0182 |
| final_margin | 5.8454 | 5.8487 | -0.0032 |
| mean_loss_tail | 0.5159 | 0.4708 | +0.0451 |
| mean_margin_tail | 4.5829 | 5.8952 | -1.3122 |
| mean_pref_acc_tail | 91.41% | 96.09% | -4.69% |
| steps_to_60pct | 25 | 25 | +0 |

> W2: safety coef=-0.0019, p=0.086 → DPO에서도 High safety 그룹의 pref_acc가 -4.69%p 낮음

**S4_familiarity (Expert vs Novice)**

| Metric | A (Very familiar) | B (Not/Somewhat) | A-B |
|--------|------------------|-----------------|-----|
| N pairs | 5049 | 13015 | |
| final_loss | 0.2929 | 0.5365 | -0.2437 |
| final_margin | 13.3764 | 3.5106 | +9.8658 |
| mean_loss_tail | 0.3865 | 0.5459 | -0.1594 |
| mean_margin_tail | 8.9179 | 3.7810 | +5.1369 |
| mean_pref_acc_tail | 94.53% | 92.19% | +2.34% |
| steps_to_60pct | 50 | 50 | +0 |

> N pairs 불균형 (5K vs 13K) 주의. Expert 그룹이 적은 데이터로 margin이 빠르게 커짐.

**S5_behavioral (High vs Low Discriminators)**

| Metric | A (High disc) | B (Low disc) | A-B |
|--------|-------------|------------|-----|
| N pairs | 8073 | 9991 | |
| final_loss | 0.5407 | 0.4384 | +0.1023 |
| final_margin | 3.3857 | 6.2691 | -2.8834 |
| mean_loss_tail | 0.4951 | 0.5179 | -0.0227 |
| mean_margin_tail | 5.1925 | 4.2046 | +0.9879 |
| mean_pref_acc_tail | 92.19% | 93.75% | -1.56% |
| steps_to_60pct | 50 | 25 | +25 |

> 순환 논리 주의: score_gap 패턴으로 유저를 분류 → DPO 학습 시그널도 score_gap 기반.
> 결과적으로 차이가 거의 없음 (-1.56%p) — 2000 steps에서 두 그룹 모두 수렴.

**S6_region (Europe vs Americas) — Null Control OK**

| Metric | A (Europe) | B (Americas) | A-B |
|--------|----------|------------|-----|
| N pairs | 7542 | 6840 | |
| final_loss | 0.3921 | 0.5149 | -0.1228 |
| final_margin | 9.3971 | 4.4136 | +4.9835 |
| mean_loss_tail | 0.4933 | 0.4266 | +0.0667 |
| mean_margin_tail | 5.1391 | 7.4861 | -2.3470 |
| mean_pref_acc_tail | 94.53% | 96.88% | -2.34% |
| steps_to_60pct | 25 | 50 | -25 |

> pref_acc 차이 2.34%p로 5% 이내 → 방법론 노이즈 수준 낮음, null control 통과.

---

## Observations

### v1 → v2 개선 요약

| 지표 | v1 | v2 | 개선 |
|------|----|----|------|
| Baseline pref_acc | 50% (학습 안됨) | 75% final | 학습 진행 |
| pref_acc 해상도 | 3단계 | 9단계 | 세밀한 비교 가능 |
| Null control (S6) | 10%p (FAILED) | 2.34%p (OK) | 방법론 검증 |
| W2 match | 1/5 | 2/5 + null OK | 개선 |
| Margin 안정성 | -2.6 ~ 11.7 | 3.3 ~ 9.4 | 안정 |

### 핵심 발견

1. **S2_personalisation: MATCH (+7.81%p)**
   - W2 예측: 개인화 선호 높은 유저 → 선호가 더 뚜렷 → DPO 학습 잘 됨
   - DPO 결과: High pers 92.19% > Low pers 84.38%
   - 이 차이는 null control 수준(2.34%)의 3배 이상 → 실질적 차이일 가능성
   - W2 coef=+0.0022 (p=0.085)와 방향 일치

2. **S3_safety: MATCH (-4.69%p)**
   - W2 예측: 안전 선호 높은 유저 → 선호가 덜 뚜렷 → DPO 학습 더 어려움
   - DPO 결과: High safety 91.41% < Low safety 96.09%
   - v1에서는 MISMATCH → v2에서 방향 전환. 데이터 충분성이 핵심
   - W2 coef=-0.0019 (p=0.086)와 방향 일치

3. **S6_region: Null control 통과 (2.34%p)**
   - S2(+7.81%), S3(-4.69%) 차이가 null 수준보다 큼 → 실질적 차이 가능성

4. **S1_age: MISMATCH (-3.91%p)**
   - null 수준(2.34%)에 근접 → 유의미한 차이라 보기 어려움

5. **S5_behavioral: 차이 없음 (-1.56%p)**
   - 순환 논리 문제에도 불구하고 차이 최소 → 충분한 학습에서는 두 그룹 모두 수렴

### 남아있는 한계
1. **Prompt-user confound**: 유저마다 다른 프롬프트 → 유저 효과와 프롬프트 효과 분리 불가
2. **단일 seed**: 결과 안정성 미검증
3. **Tail pref_acc 수렴**: 대부분 90%+로 수렴 → 중반부 학습 곡선에서 차이가 더 클 수 있음
4. **모델 크기**: 0.5B는 매우 작아서 현실적 DPO와 거리 있음

---

## Next Steps
- [ ] **Seed 실험**: 3~5개 seed로 반복 → S2, S3 차이가 안정적인지 확인
- [ ] **Learning curve 중반부 분석**: step 200~800 구간의 그룹 차이가 더 클 수 있음
- [ ] **Ordered logit 연결**: DPO margin을 ordered logit으로 분석 → W2와 직접 연결
- [ ] **Step 조정**: 1000 steps에서 그룹 차이가 더 드러나는지 확인 (수렴 전 구간)
- [ ] v2 plots 확인: `260223/results/*.png`
