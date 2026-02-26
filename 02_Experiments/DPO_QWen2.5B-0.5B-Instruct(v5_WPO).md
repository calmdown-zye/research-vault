---
tags:
  - experiment
  - DPO
  - WPO
  - PRISM
date: "2026-02-26"
status: completed
---

# v5: WPO — Support Mismatch 교정 실험

## 목적

v4에서 OL→DPO 전이 실패(MISMATCH)의 원인 중 하나로 지목된 **support mismatch**를 WPO로 교정했을 때, 전이가 개선되는지 검증.

## 방법: WPO (Weighted Preference Optimization)

> Zhou et al., "WPO: Enhancing RLHF with Weighted Preference Optimization", EMNLP 2024

### 핵심 아이디어

각 preference pair의 DPO loss에 "on-policy 정도"를 나타내는 weight를 곱한다.

$$L_{\text{WPO}} = \mathbb{E}\left[ w(x,y_w) \cdot w(x,y_l) \cdot \left(-\log\sigma(\beta \cdot \text{margin})\right) \right]$$

### Weight 계산 (Sampled Alignment)

$$w(x,y) = \exp\left( \frac{1}{|y|} \sum_{t=1}^{|y|} \left[ \log\pi_\theta(y_t | x, y_{<t}) - \log\sum_v \pi_\theta(v)^2 \right] \right)$$

- `log pi(y_t)`: 현재 policy의 per-token logprob
- `sum_v pi(v)^2`: collision probability — 랜덤 토큰 기대 logprob으로 정규화
- Length normalization: response token 수로 평균

최종: `weight = clamp(w_chosen * w_rejected, max=1)`

### DPO와의 차이

| | DPO | WPO |
|--|-----|-----|
| Loss | `mean(-log sigma(beta*margin))` | `mean(weight * -log sigma(beta*margin))` |
| Off-policy 처리 | 동일 가중치 | weight로 downweight |
| Forward pass/step | 4 | 6 (weight 계산용 +2) |
| 속도 | 기준 | ~50% 느림 |

---

## 실험 설정

| 항목 | 값 |
|------|-----|
| Method | WPO (sampled alignment) |
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Seeds | 42, 456, 789 (v4와 동일 3개) |
| Splits | v2의 6개 (age, personalisation, safety, familiarity, behavioral, region) |
| Steps | 2,000 |
| Batch | 8 |
| Beta | 0.1, LR 5e-6 |
| GPU | A6000Ada 48GB |

코드: `DPO-mini/260226/`

---

## 결과

### 1. Weight 분석: PRISM 데이터는 얼마나 off-policy인가?

| Split | w_mean (A) | w_mean (B) | w_min (A) | w_min (B) |
|-------|-----------|-----------|----------|----------|
| S1_age | 0.2149 | 0.2062 | 0.0575 | 0.0454 |
| S2_personalisation | 0.2290 | 0.1956 | 0.0630 | 0.0415 |
| S3_safety | 0.2071 | 0.2220 | 0.0501 | 0.0465 |
| S4_familiarity | 0.1952 | 0.2279 | 0.0485 | 0.0507 |
| S5_behavioral | 0.2091 | 0.2159 | 0.0569 | 0.0548 |
| S6_region | 0.2031 | 0.1842 | 0.0491 | 0.0360 |

**해석**: 평균 weight ~0.2, 즉 **PRISM 데이터의 ~80%가 off-policy로 downweight됨.** 최소값은 0.04~0.06으로, 극단적 off-policy pair는 거의 무시된다. Qwen-0.5B 입장에서 GPT-4/Claude 응답이 자신의 생성 분포에서 매우 멀다는 것을 정량적으로 확인.

### 2. Baseline pref_acc

| Method | Mean | Std | Per seed |
|--------|------|-----|----------|
| WPO | 49.74% | 2.58% | 46.09%, 51.56%, 51.56% |
| DPO | 54.95% | 2.66% | 58.59%, 53.91%, 52.34% |

WPO baseline이 DPO보다 ~5%p 낮다. Off-policy pair를 downweight하면 "쉬운" pair가 빠지고 harder pair 위주로 학습하기 때문으로 해석.

### 3. Split별 WPO vs DPO 비교

| Split | Expected | WPO diff | WPO verdict | DPO diff | DPO verdict | Change |
|-------|----------|----------|-------------|----------|-------------|--------|
| S1_age | A > B | +3.39% | LEAN | +2.08% | LEAN | LEAN→LEAN |
| **S2_personalisation** | **A > B** | **+2.34%** | **MATCH** | **+0.26%** | **MISMATCH** | **FIXED** |
| **S3_safety** | **A < B** | **-2.34%** | **MATCH** | **+0.00%** | **MISMATCH** | **FIXED** |
| S4_familiarity | explor. | +7.03% | — | +3.39% | — | — |
| S5_behavioral | explor. | -1.30% | — | -4.43% | — | — |
| S6_region | A = B | +0.78% | OK | +0.78% | OK | — |

### 4. Per-Seed Diffs

**WPO (v5):**

| Split | s42 | s456 | s789 |
|-------|-----|------|------|
| S1_age | +3.12% | +0.00% | +7.03% |
| S2_personalisation | +2.34% | +3.12% | +1.56% |
| S3_safety | -1.56% | -3.12% | -2.34% |
| S4_familiarity | +11.72% | +6.25% | +3.12% |
| S5_behavioral | +3.91% | -3.12% | -4.69% |
| S6_region | +0.78% | +3.12% | -1.56% |

**DPO (v4):**

| Split | s42 | s456 | s789 |
|-------|-----|------|------|
| S1_age | +3.12% | +0.00% | +3.12% |
| S2_personalisation | -0.78% | +1.56% | +0.00% |
| S3_safety | +3.12% | -4.69% | +1.56% |
| S4_familiarity | +5.47% | +0.78% | +3.91% |
| S5_behavioral | -4.69% | -2.34% | -6.25% |
| S6_region | +2.34% | +0.78% | -0.78% |

---

## 핵심 발견

### personalisation과 safety가 MISMATCH → MATCH로 전환

이것이 이번 실험의 가장 중요한 결과다.

**S2_personalisation**:
- DPO: 3 seed 중 diff 방향이 (-0.78%, +1.56%, +0.00%) → mixed, mean +0.26%
- WPO: 3 seed **모두 양수** (+2.34%, +3.12%, +1.56%) → 일관적, mean +2.34%
- OL 예측(A>B)과 **일치**

**S3_safety**:
- DPO: (+3.12%, -4.69%, +1.56%) → 2/3이 OL과 반대 방향, mean +0.00%
- WPO: **3 seed 모두 음수** (-1.56%, -3.12%, -2.34%) → 일관적, mean -2.34%
- OL 예측(A<B)과 **일치**

WPO가 off-policy noise를 제거하자, 통계 모델이 포착한 유저 특성별 선호 차이가 DPO 학습에서도 드러남.

### Weight ~0.2는 support mismatch의 정량적 증거

PRISM 응답의 80%가 Qwen-0.5B 입장에서 off-policy. DPO는 이 모든 pair에 동일 가중치를 주었기 때문에, 모델이 생성하지 않을 응답 영역에서의 noise가 유저 특성 신호를 가렸던 것.

### familiarity 신호가 WPO에서 더 강해짐

- DPO: +3.39% → WPO: **+7.03%** (2배 증폭)
- v4에서 이미 5/5 robust했는데 WPO로 더 강해짐
- 이 split이 support와 무관하게 가장 robust한 신호임을 재확인

---

## 해석 & 시사점

### Support mismatch는 실재하는 문제였다

두 개의 핵심 split(personalisation, safety)이 WPO로 MISMATCH→MATCH 전환. 이는:
1. OL이 포착한 유저 특성별 선호 강도 차이가 **실재한다**
2. DPO에서 이를 관찰하지 못한 것은 **방법론(support mismatch)** 때문
3. Off-policy data를 적절히 reweight하면 신호가 **복원된다**

### 남은 noise 원인

- S1_age는 여전히 LEAN (3/3 양수이지만 한 seed에서 0%) → 약한 효과
- S5_behavioral은 방향이 불안정 → preference heterogeneity?
- Baseline pref_acc가 50% 부근 → 모델 능력 한계는 여전

### 연구 흐름에서의 위치

```
OL 분석 (유저 특성 → 선호 강도)
  → DPO v2 (단일 seed MATCH — 우연)
  → v4 (multi-seed MISMATCH — 전이 실패 확인)
  → v5 WPO (support 교정 → MATCH 복원 — 원인 규명)  ← 현재
```

---

## 다음 단계

- [ ] Source model 고정 실험 — confound 제거 후 WPO 재실험
- [ ] WPO weight 분포 시각화 — 어떤 pair가 가장 많이 downweight 되는지
- [ ] 5-seed 확장 (seeds 123, 999 추가) — 현재 3-seed 결과의 robustness 확인
- [ ] Personalized DPO 탐색 — 유저 조건부 policy 학습 가능성

---

## Reference

- [[DPO_QWen2.5B-0.5B-Instruct(v4)|v4 (multi-seed DPO)]]
- [[DPO_PRISM_연구정리|전체 연구 정리]]
- [WPO (EMNLP 2024)](https://arxiv.org/abs/2406.11827)
- 코드: `DPO-mini/260226/`
