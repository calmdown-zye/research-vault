---
tags:
  - experiment
  - DPO
  - WPO
  - PRISM
date: "2026-02-27"
status: completed
---

# v5b: WPO on v3 OL Splits (8 splits 확장)

## 목적

v5 WPO가 v2의 6 splits에서 personalisation/safety를 FIXED한 것을 확인했으나, v3의 OL-significant 8개 변수 전체에 대한 검증이 필요했다. 특히 v3에서 새로 추가된 education, factuality, helpfulness, diversity에서도 WPO가 OL→DPO 전이를 개선하는지 확인.

## 실험 설정

| 항목 | 값 |
|------|-----|
| Method | WPO (sampled alignment), v5와 동일 |
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Seeds | 42 (단일 seed, 시간 제약) |
| Splits | v3 OL 8개 중 **7개** (S8_region 제외 — SLURM job 실패) |
| Steps | 2,000 |
| Batch | 8 |
| Beta | 0.1, LR 5e-6 |
| DPO 비교 기준 | v3 DPO 5-seed reference [42, 123, 456, 789, 999] |

코드: `DPO-mini/260226_v3/`

### v5 → v5b 차이

| | v5 (260226/) | v5b (260226_v3/) |
|--|-------------|------------------|
| Split 기준 | v2 W2 binary logit (6개) | v3 Ordered logit p<0.05 (8개) |
| 공통 변수 | age, personalisation, safety, region | 동일 정의 |
| v3 추가 변수 | — | education, factuality, helpfulness, diversity |
| v2 전용 변수 | familiarity, behavioral | — |
| Seeds | 3 (42, 456, 789) | 1 (42) |

---

## 결과

### 1. Weight 분석

| Split | w_mean (A) | w_mean (B) | w_min (A) | w_min (B) |
|-------|-----------|-----------|----------|----------|
| S1_age | 0.2075 | 0.2233 | 0.0574 | 0.0438 |
| S2_education | 0.2036 | 0.2114 | 0.0397 | 0.0430 |
| S3_personalisation | 0.2411 | 0.2015 | 0.0633 | 0.0418 |
| S4_safety | 0.2108 | 0.2180 | 0.0442 | 0.0430 |
| S5_factuality | 0.2138 | 0.2067 | 0.0562 | 0.0427 |
| S6_helpfulness | 0.2260 | 0.2138 | 0.0548 | 0.0483 |
| S7_diversity | 0.2162 | 0.2196 | 0.0435 | 0.0620 |

v5와 일관: 평균 weight ~0.20-0.24, **80% off-policy**. 새로운 splits에서도 동일 패턴.

주목할 점:
- **S3_personalisation A (High pers)**: w=0.2411로 가장 높음 — "personalisation 중시" 유저의 선호 응답이 상대적으로 on-policy에 가까움
- 전반적으로 그룹 간 weight 차이 미미 (0.01~0.04) — weight 자체가 split 차이의 원인은 아님

### 2. Baseline pref_acc

| Method | Acc |
|--------|-----|
| WPO v5b | 47.66% |
| WPO v5 (3-seed mean) | 49.74% |
| DPO v4 (3-seed mean) | 54.95% |

Seed 42 단독이라 v5 mean보다 낮지만, WPO baseline이 DPO보다 낮은 패턴은 동일.

### 3. Split별 WPO vs DPO 비교

| Split | OL 예측 | WPO diff | WPO | DPO diff (5-seed) | DPO | Change |
|-------|--------|----------|-----|----------|-----|--------|
| **S1_age** | **A > B** | **+3.91%** | **MATCH** | **-0.31%** | **MISMATCH** | **FIXED** |
| S2_education | A > B | -5.47% | MISMATCH | -1.88% | MISMATCH | UNCHANGED |
| **S3_personalisation** | **A > B** | **+2.34%** | **MATCH** | **-1.88%** | **MISMATCH** | **FIXED** |
| S4_safety | A < B | +0.00% | MISMATCH | +2.34% | MISMATCH | UNCHANGED |
| S5_factuality | A < B | +0.78% | MISMATCH | +3.28% | MISMATCH | UNCHANGED |
| S6_helpfulness | A < B | -0.78% | WEAK | +5.31% | MISMATCH | IMPROVED |
| S7_diversity | A > B | +3.12% | MATCH | +1.41% | MATCH | STABLE |

**요약 (testable 7개)**:
- FIXED: **2/7** (age, personalisation)
- IMPROVED: **1/7** (helpfulness)
- UNCHANGED: **3/7** (education, safety, factuality)
- STABLE: **1/7** (diversity — 이미 DPO에서 MATCH)

### 4. Per-Seed Diffs

**WPO v5b (seed 42):**

| Split | s42 |
|-------|-----|
| S1_age | +3.91% |
| S2_education | -5.47% |
| S3_personalisation | +2.34% |
| S4_safety | +0.00% |
| S5_factuality | +0.78% |
| S6_helpfulness | -0.78% |
| S7_diversity | +3.12% |

**DPO v3 (5 seeds reference):**

| Split | s42 | s123 | s456 | s789 | s999 |
|-------|-----|------|------|------|------|
| S1_age | +1.56% | -5.47% | -0.78% | +3.12% | +0.00% |
| S2_education | -2.34% | -4.69% | -3.91% | -0.78% | +2.34% |
| S3_personalisation | -1.56% | -2.34% | +0.00% | -1.56% | -3.91% |
| S4_safety | +4.69% | +10.16% | -3.91% | +0.78% | +0.00% |
| S5_factuality | -0.78% | +4.69% | +4.69% | +7.03% | +0.78% |
| S6_helpfulness | +3.91% | +0.00% | +4.69% | +7.81% | +10.16% |
| S7_diversity | +0.78% | +3.12% | +3.12% | -4.69% | +4.69% |

---

## Cross-check: v2 WPO와 공통 splits 비교

v2와 v3에서 동일 정의인 변수들의 WPO 결과가 일치하는지 확인:

| Variable | v2 WPO (3-seed) | v2 verdict | v3 WPO (seed42) | v3 verdict | Agree? |
|----------|----------------|------------|-----------------|------------|--------|
| personalisation | +2.34% | MATCH | +2.34% | MATCH | **YES** |
| safety | -2.34% | MATCH | +0.00% | MISMATCH | NO |
| age | +3.39% | LEAN | +3.91% | MATCH | NO |

- **personalisation**: 두 버전 모두 +2.34% MATCH — **가장 robust한 WPO 결과**
- **safety**: v2 WPO 3-seed에서 3/3 음수(-1.56%, -3.12%, -2.34%)로 MATCH였으나, v3 seed42에서 정확히 0.00%. **seed variance일 가능성 높음** — multi-seed 필요
- **age**: v2 LEAN → v3 MATCH로 오히려 개선. seed 42 단독이라 해석에 주의 필요

---

## Weight 분포 시각화 분석

`analyze_weights.py`로 전체 ~18K pairs의 WPO weight 분포를 분석:

### Plot 1: Weight Histogram
- 분포가 **오른쪽으로 치우침** (right-skewed)
- 대부분의 pair가 weight 0.10-0.30 구간에 집중
- 소수의 pair만 weight > 0.5 (상대적 on-policy)
- Weight = 1.0 (완전 on-policy)인 pair는 거의 없음

### Plot 2: Weight vs Score Gap
- Score gap (선호 강도)과 weight 사이에 **뚜렷한 상관관계 없음**
- Binned means가 score gap 전 구간에서 ~0.2 수준으로 flat
- 해석: off-policy 정도는 선호 강도와 무관. 높은 score_gap pair도 낮은 score_gap pair도 비슷하게 off-policy

### Plot 3: Weight by Source Model
- **모델별로 weight가 다름** — 이것이 가장 흥미로운 발견
- Qwen 계열/작은 모델의 응답이 상대적으로 높은 weight (더 on-policy)
- GPT-4, Claude 등 대형 모델의 응답은 낮은 weight (더 off-policy)
- Source model이 support mismatch의 직접적 원인임을 시각적으로 확인

### Plot 4: Per-Split A vs B Weight Distribution
- 7개 split 모두에서 A/B 그룹 간 weight 분포가 거의 동일
- 그룹 간 weight 차이가 결과 차이의 원인이 아님을 확인
- WPO의 MATCH/MISMATCH 차이는 weight가 아닌 **선호 신호의 일관성**에서 옴

---

## 핵심 발견

### 1. WPO가 OL→DPO 전이를 부분적으로 개선 (3/7)

- **FIXED** (age, personalisation): off-policy noise 제거 후 OL 예측과 일치하는 신호 복원
- **IMPROVED** (helpfulness): DPO에서 +5.31%로 강하게 반대 방향이었는데, WPO에서 -0.78%로 올바른 방향 전환. 아직 WEAK이지만 방향이 바뀐 것이 의미 있음
- **UNCHANGED** (education, safety, factuality): support mismatch 교정만으로는 부족 — 다른 원인 존재

### 2. personalisation이 가장 robust한 WPO 신호

- v2 WPO (3-seed): +2.34% MATCH
- v3 WPO (seed42): +2.34% MATCH
- **두 버전, 다른 split 기준, 다른 seed에서 동일 값** — 재현성 매우 높음

### 3. safety의 version 간 불일치는 seed variance 가능성

- v2 WPO: 3/3 음수 → MATCH
- v3 WPO: seed 42에서 0.00% → MISMATCH
- v2 WPO seed 42에서도 -1.56%였으므로, v3의 0.00%는 split 정의 차이 또는 v3 데이터 구성 차이에서 올 수 있음
- **Multi-seed 확장이 필요한 핵심 변수**

### 4. education, factuality는 support 문제가 아닌 다른 원인

- S2_education: -5.47%로 **강하게** 반대 방향. DPO(-1.88%)보다 오히려 더 나빠짐
- S5_factuality: +0.78% (DPO +3.28%보다 개선되었으나 여전히 반대 방향)
- 가설: **preference heterogeneity** — 같은 교육 수준 그룹 내에서도 선호 방향이 다양하여, off-policy 교정만으로는 신호를 추출할 수 없음

### 5. Source model이 weight의 주요 결정 요인

Weight 시각화에서 확인: weight는 score_gap이 아닌 **source model**에 의해 결정됨. 이는 source model confound가 여전히 미해결 문제임을 강조.

---

## v5 vs v5b 종합 비교

| 공통 Split | v5 (v2, 3-seed) | v5b (v3, 1-seed) | 비고 |
|-----------|----------------|-----------------|------|
| age | +3.39% LEAN | +3.91% MATCH | v5b에서 개선 |
| personalisation | +2.34% MATCH | +2.34% MATCH | **완벽 일치** |
| safety | -2.34% MATCH | +0.00% MISMATCH | seed variance 의심 |

v5b 전용 (v3 추가 변수):

| Split | WPO | DPO | Change | 해석 |
|-------|-----|-----|--------|------|
| education | MISMATCH | MISMATCH | UNCHANGED | heterogeneity 의심 |
| factuality | MISMATCH | MISMATCH | UNCHANGED | 효과 크기 감소는 긍정적 |
| helpfulness | WEAK | MISMATCH | IMPROVED | 방향 전환 성공 |
| diversity | MATCH | MATCH | STABLE | WPO 전부터 OK |

---

## 해석 & 시사점

### Support mismatch는 "부분적" 원인

v5에서 2/4 FIXED → v5b에서 3/7 FIXED/IMPROVED. WPO가 일부 splits에서 OL→DPO 전이를 복원하지만, **전체 splits에 대해 universal한 해결책은 아님**.

### 원인 복합성 확인

OL→DPO 전이 실패의 원인이 단일하지 않음:
1. **Support mismatch**: WPO로 교정 가능 (personalisation, age, helpfulness에서 효과)
2. **Preference heterogeneity**: WPO로 교정 불가 (education, factuality에서 UNCHANGED)
3. **Source model confound**: weight 시각화에서 model별 차이 확인, 아직 미통제
4. **Seed variance**: safety처럼 단일 seed에서 판정이 불안정한 경우 존재

### 단일 seed 한계

v5b는 seed 42 하나만 사용. safety처럼 v2 WPO와 불일치하는 결과는 multi-seed 없이 결론 내리기 어려움. **최소 3-seed 확장 필요**.

---

## 다음 단계

- [ ] Safety multi-seed 확장 (seed 456, 789) — v2 WPO와의 불일치 해소
- [ ] Source model 고정 + WPO — confound 제거 후 education, factuality 재검증
- [ ] education의 -5.47%가 seed 특이적인지 multi-seed로 확인
- [ ] Preference heterogeneity 정량화 — 그룹 내 선호 방향 일관성 지표 개발

---

## Reference

- [[DPO_QWen2.5B-0.5B-Instruct(v5_WPO)|v5 WPO (v2 splits, 3-seed)]]
- [[DPO_QWen2.5B-0.5B-Instruct(v4)|v4 (multi-seed DPO)]]
- [[DPO_PRISM_연구정리|전체 연구 정리]]
- [WPO (EMNLP 2024)](https://arxiv.org/abs/2406.11827)
- 코드: `DPO-mini/260226_v3/`
