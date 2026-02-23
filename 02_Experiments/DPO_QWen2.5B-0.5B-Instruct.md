---
tags:
  - experiment
  - DPO
  - PRISM
date: "2026-02-23"
model: Qwen/Qwen2.5-0.5B-Instruct
dataset: PRISM (within-model pairs)
status: needs-rerun
---

# DPO on PRISM: User Group Split Experiment (v1)

## Goal
- W2 (logit on Z_within, cluster SE)에서 유저 특성이 선호 강도에 영향을 준다는 통계적 증거를 발견
- 이를 DPO 학습 패턴 차이로도 보일 수 있는지 검증 (computational evidence)
- 6가지 유저 그룹 분할로 DPO를 따로 돌려서, 그룹 간 학습 패턴(loss, margin, pref_acc) 차이를 관찰

## Setup

### Model & Infra
- **Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Infra**: SLURM gpu3 (A6000Ada), bf16
- **Code**: `DPO-mini/260223/` (GitHub: calmdown-zye/DPO-mini)

### Data: PRISM within-model pairs
- **Source**: HannahRoseKirk/prism-alignment
- **Pair 추출**: Turn 1+에서 같은 user, 같은 model, 같은 prompt에 대한 2개 응답
- **선호 기준**: user가 부여한 score (0-100) 기반, 높은 score = chosen
- **Total**: ~18,000 within-model pairs, ~1,400 users
- W2 분석 설계와 정확히 대응 (within-model, same prompt)

### Hyperparams (v1)
- **steps**: 500
- **beta**: 0.1
- **max_pairs**: 1000
- **batch_size**: 2
- **log_every**: 10
- **dtype**: bf16
- **lr**: 5e-7 (default)

### 6 User Group Splits

| Split | Group A | Group B | W2 예측 |
|-------|---------|---------|---------|
| S1_age | Young (18-34) | Older (35+) | A > B |
| S2_personalisation | High | Low | A > B |
| S3_safety | High | Low | A < B |
| S4_familiarity | Very familiar | Not/Somewhat | 탐색적 |
| S5_behavioral | High discriminators | Low discriminators | A > B |
| S6_region | Europe | Americas | A ≈ B (null control) |

### 실험 구성: 13 runs
1. Baseline (전체 pairs)
2-13. 6 splits x 2 groups (A, B)

## Results (v1 - 500 steps)

### Baseline
| Metric | Value |
|--------|-------|
| Final loss | 0.7109 |
| Final margin | -0.3483 |
| Final pref_acc | 50.00% |
| Mean loss (tail 20%) | 0.7091 |
| Mean pref_acc (tail 20%) | 50.00% |

### Group Comparison Summary

| Split | pref_acc A | pref_acc B | diff | Note |
|-------|-----------|-----------|------|------|
| S1_age | 50.00% | 70.00% | -20.00% | LARGE |
| S2_personalisation | 60.00% | 50.00% | +10.00% | notable |
| S3_safety | 45.00% | 40.00% | +5.00% | mild |
| S4_familiarity | 70.00% | 60.00% | +10.00% | notable |
| S5_behavioral | 35.00% | 55.00% | -20.00% | LARGE |
| S6_region | 55.00% | 45.00% | +10.00% | null control FAILED |

### W2 Correspondence Check

| Split | W2 Expected | Observed | Result |
|-------|-------------|----------|--------|
| S1_age | A > B (young > older) | A < B (-20%) | MISMATCH |
| S2_personalisation | A > B (high > low) | A > B (+10%) | MATCH |
| S3_safety | A < B (high safety = lower disc) | A > B (+5%) | MISMATCH |
| S4_familiarity | Exploratory | A > B (+10%) | exploratory |
| S5_behavioral | A > B (high disc > low disc) | A < B (-20%) | MISMATCH |
| S6_region | A ≈ B (null) | A > B (+10%) | UNEXPECTED |

### Detailed Per-Split Results

**S1_age (Young vs Older)**

| Metric | A (Young) | B (Older) | A-B |
|--------|----------|----------|-----|
| N pairs | 1000 | 1000 | |
| final_loss | 0.6843 | 0.6720 | +0.0123 |
| final_margin | 0.1789 | 0.4918 | -0.3129 |
| mean_loss_tail | 0.7062 | 0.6439 | +0.0623 |
| mean_margin_tail | -0.1788 | 1.3193 | -1.4982 |
| mean_pref_acc_tail | 50.00% | 70.00% | -20.00% |
| steps_to_60pct | 10 | 110 | -100 |

**S2_personalisation (High vs Low)**

| Metric | A (High) | B (Low) | A-B |
|--------|---------|--------|-----|
| N pairs | 1000 | 1000 | |
| final_loss | 0.6747 | 0.6979 | -0.0232 |
| final_margin | 0.3795 | -0.0927 | +0.4722 |
| mean_loss_tail | 0.6719 | 0.6671 | +0.0048 |
| mean_margin_tail | 0.5171 | 0.7186 | -0.2015 |
| mean_pref_acc_tail | 60.00% | 50.00% | +10.00% |
| steps_to_60pct | 60 | 30 | +30 |

**S3_safety (High vs Low)**

| Metric | A (High) | B (Low) | A-B |
|--------|---------|--------|-----|
| N pairs | 1000 | 1000 | |
| final_loss | 0.3937 | 0.7936 | -0.4000 |
| final_margin | 11.7479 | -1.9127 | +13.6605 |
| mean_loss_tail | 0.6762 | 0.7015 | -0.0253 |
| mean_margin_tail | 0.9823 | -0.1008 | +1.0831 |
| mean_pref_acc_tail | 45.00% | 40.00% | +5.00% |
| steps_to_60pct | 40 | 10 | +30 |

**S4_familiarity (Expert vs Novice)**

| Metric | A (Very familiar) | B (Not/Somewhat) | A-B |
|--------|------------------|-----------------|-----|
| N pairs | 1000 | 1000 | |
| final_loss | 0.7914 | 0.4959 | +0.2956 |
| final_margin | -1.6778 | 5.2090 | -6.8869 |
| mean_loss_tail | 0.6454 | 0.6641 | -0.0187 |
| mean_margin_tail | 1.7092 | 0.7488 | +0.9605 |
| mean_pref_acc_tail | 70.00% | 60.00% | +10.00% |
| steps_to_60pct | 20 | 10 | +10 |

**S5_behavioral (High vs Low Discriminators)**

| Metric | A (High disc) | B (Low disc) | A-B |
|--------|-------------|------------|-----|
| N pairs | 1000 | 1000 | |
| final_loss | 0.8397 | 0.6462 | +0.1935 |
| final_margin | -2.6782 | 0.9621 | -3.6403 |
| mean_loss_tail | 0.7309 | 0.6799 | +0.0510 |
| mean_margin_tail | -0.6832 | 0.3338 | -1.0170 |
| mean_pref_acc_tail | 35.00% | 55.00% | -20.00% |
| steps_to_60pct | 40 | 60 | -20 |

**S6_region (Europe vs Americas) - Null Control**

| Metric | A (Europe) | B (Americas) | A-B |
|--------|----------|------------|-----|
| N pairs | 1000 | 1000 | |
| final_loss | 0.7060 | 0.4446 | +0.2614 |
| final_margin | -0.2509 | 6.0090 | -6.2599 |
| mean_loss_tail | 0.6761 | 0.6774 | -0.0013 |
| mean_margin_tail | 0.3809 | 0.4768 | -0.0959 |
| mean_pref_acc_tail | 55.00% | 45.00% | +10.00% |
| steps_to_60pct | 40 | 50 | -10 |

### Notable Examples per Split

각 split에서 score_gap이 가장 큰 pair(=학습 시그널 강)와 가장 작은 pair(=모호한 선호)를 추출.

**S1_age**
> **[A: Young] Largest gap (99)** — Wiccan items 질문에 두 응답 모두 비슷한 구조이지만 유저가 100 vs 1점 부여. 구체적 가게명(Mystic Moments) 포함 여부로 갈림.
> **[B: Older] Smallest gap (1)** — 사적 공간 발언 영향력 질문. chosen(92) vs rejected(91), 거의 동일한 내용. DPO가 구분하기 가장 어려운 유형.

**S2_personalisation**
> **[B: Low pers] Largest gap (99)** — "innocent people have to die" 토픽에서 rejected가 "EARLY BIRD BOOKS FRESH EBOOK DEALS" 광고 텍스트를 포함 (모델 오류). 선호 시그널 명확.
> **[A: High pers] Smallest gap (1)** — 종교 참여 이유 질문. GPT-4의 두 응답이 거의 동일한 구조, 미세한 표현 차이만 존재.

**S3_safety**
> **[A: High safety] Largest gap (99)** — 지출 절약 질문에 rejected가 "EMPTY STRING" (빈 응답). 안전 중시 유저도 빈 응답은 확실히 거부.
> **[B: Low safety] Smallest gap (1)** — 정직에 대한 토론. chosen(87) vs rejected(86), 동일한 논점을 약간 다른 어조로 전달.

**S4_familiarity**
> **[A: Expert] Largest gap (99)** — 무고한 사람 관련 토픽. rejected에 광고 텍스트 혼입. LLM 전문가도 이런 확실한 오류는 쉽게 판별.
> **[B: Novice] Smallest gap (1)** — 종교 관련 질문. 두 GPT-4 응답의 차이가 "purpose, community, moral structure" vs "comfort, guidance, answers to life's mysteries" 정도.

**S5_behavioral**
> **[A: High disc] Largest gap (99)** — Wiccan items 질문. 높은 판별자는 유사 응답에서도 99점 차이를 부여하는 유저.
> **[A: High disc] Smallest gap (1)** — 전과자 대통령 출마 질문. chosen(90) vs rejected(89), 두 claude-2.1 응답이 거의 동일한 문장으로 시작.

**S6_region (Null Control)**
> **[A: Europe] Largest gap (99)** — "Buddhism is religion?" 질문. 유럽 유저가 100 vs 1점 부여. 두 응답 모두 합리적이나 유저 선호가 극단적.
> **[B: Americas] Smallest gap (1)** — 원자력 안전 토론. GPT-4의 두 응답이 동일한 논점, 거의 구분 불가.

## Observations

### 결과 신뢰성 문제 (v1의 한계)
1. **pref_acc 해상도 부족**: batch_size=2이므로 pref_acc가 {0, 0.5, 1.0}으로만 나옴. tail 20% 평균도 이산적 → 그룹 간 차이가 artifact일 가능성 높음
2. **학습 부족**: 500 steps x batch 2 = 1,000 examples만 학습. Baseline pref_acc가 50%에서 거의 안 움직임 → 모델이 충분히 학습하지 못함
3. **Null control 실패**: S6_region에서 10%p 차이 → 방법론 자체의 노이즈가 크다는 뜻. 현재 차이들이 진짜 유저 특성 효과인지 랜덤 분산인지 구분 불가
4. **final_loss/margin 불안정**: S3, S4, S6에서 final_loss와 final_margin이 극단적 값 (예: margin 11.7, -2.6) → 학습이 수렴하지 않고 oscillation

### W2 대응 결과
- 6개 중 1개만 MATCH (S2_personalisation)
- 하지만 위 한계들 때문에 이 결과로 W2를 confirm/deny 할 수 없음

## Next Steps
- [ ] **v2 re-run** with 강화된 hyperparams:
  - steps: 500 → **2000~3000**
  - batch_size: 2 → **8** (pref_acc 해상도: 0, 0.125, 0.25, ... 1.0)
  - max_pairs: 1000 → **전체 사용 또는 5000+**
  - log_every: 10 → **20~50** (smoother curves)
- [ ] Baseline pref_acc가 70%+ 넘어가는지 먼저 확인 (학습 충분성 기준)
- [ ] Null control (S6) 차이가 5% 이내로 줄어드는지 확인 후 나머지 해석
- [ ] 결과 안정적이면 W2 correspondence 재검증
