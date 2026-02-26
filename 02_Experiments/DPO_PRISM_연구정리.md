---
tags:
  - experiment
  - DPO
  - PRISM
  - overview
date: "2026-02-26"
status: in-progress
---

# DPO on PRISM: 연구 여정 전체 정리

## 큰 연구 질문

> 생성모형이 만든 y_1, y_2 모두 사람이 보기에는 괜찮지만, 모델(policy) 입장에서는 생성할 확률이 극히 낮은 응답일 때 — 그런 응답에 대한 인간 선호를 DPO로 반영하면 어떤 일이 생기는가?

이 질문은 세 가지 하위 문제로 분해된다:

1. **Support mismatch**: policy가 생성하지 않을 영역의 선호를 학습하는 것이 의미있는가?
2. **Preference heterogeneity**: 유저마다 선호 방향이 다를 때, 하나의 policy로 수렴시키는 것이 가능한가?
3. **OL→DPO 전이**: 통계 모델(OL)이 포착한 유저 특성별 선호 강도 차이가 DPO 학습 패턴에도 나타나는가?

---

## 실험 여정 타임라인

```
OL/W2 분석 → v1 (pilot) → v2 (개선) → v3 (확장) → v4 (multi-seed) → 진단 → v5 WPO (원인 규명)
```

### Phase 0: 통계 분석 (OL/W2)

**발견**: PRISM 데이터에서 유저 특성(age, safety, personalisation 등)이 선호 강도(score_gap)에 유의미한 영향을 미친다.
- W2 (binary logit): score_gap > 14 여부를 종속변수로, 유저 특성이 설명변수
- Ordered logit: score_gap quartile을 종속변수로, 더 정밀한 변수 선별 (p<0.05)

**질문**: 이 통계적 발견이 DPO 학습에서도 관찰 가능한가?

---

### Phase 1: v1 — Pilot 실험

> [[DPO_QWen2.5B-0.5B-Instruct|v1 노트]]

| 항목 | 설정 |
|------|------|
| 모델 | Qwen/Qwen2.5-0.5B-Instruct |
| 데이터 | PRISM within-model pairs, **1K pairs/group** |
| Steps | 500, batch 2 |
| Splits | 6 (W2 기반) |

**결과**: 실패
- Baseline pref_acc 50%에서 정체 (학습 부족)
- pref_acc 해상도 3단계뿐 (batch 2 → {0, 0.5, 1.0})
- Null control(region) 10%p 차이 → 방법론 자체가 noisy
- W2 match: 1/5

**교훈**: steps, batch_size, data 양 모두 부족. 실험 설계를 대폭 강화해야 함.

---

### Phase 2: v2 — W2 Binary Logit Splits (단일 seed)

> [[DPO_QWen2.5B-0.5B-Instruct(v2)|v2 노트]]

| 항목 | v1 → v2 변경 |
|------|-------------|
| Steps | 500 → **2000** |
| Batch | 2 → **8** (9단계 해상도) |
| Data | 1K → **전체 (~9K/group)** |
| Splits | 6 (W2 기반, 동일) |

**결과**: 개선됨
- Baseline pref_acc 75% (학습 진행 확인)
- Null control 2.34% (OK)
- **personalisation +7.81% MATCH**, **safety -4.69% MATCH** → W2 예측과 일치!
- W2 match: 2/5 + null OK

**당시 해석**: "W2 통계 결과가 DPO에서도 부분적으로 재현된다"

---

### Phase 3: v3 — Ordered Logit Splits (단일 seed)

> [[DPO_QWen2.5B-0.5B-Instruct(v3)|v3 노트]]

| 항목 | 변경점 |
|------|--------|
| Split 기준 | W2 binary logit → **Ordered logit (p<0.05)** |
| Splits | 6 → **8** (familiarity, behavioral 삭제 / education, factuality, helpfulness, diversity 추가) |
| Hyperparams | 동일 |

**결과**: v2와 불일치 발견
- **personalisation**: v2 +7.81% MATCH → v3 +1.56% MISMATCH (크기 축소)
- **safety**: v2 -4.69% MATCH → v3 **+6.25% MISMATCH** (방향 반전!)
- 동일 split 정의, 동일 hyperparams인데 결과가 다름

**핵심 문제 인식**: **단일 seed에서의 match/mismatch 판정을 신뢰할 수 없다.** v2의 "성공"이 우연이었을 가능성.

---

### Phase 4: v4 — Multi-Seed Robustness (5 seeds)

> [[DPO_QWen2.5B-0.5B-Instruct(v4)|v4 노트]]

| 항목 | 설정 |
|------|------|
| Seeds | [42, 123, 456, 789, 999] |
| v2 runs | 5 seeds × 13 = 65 runs (완료) |
| v3 runs | 5 seeds × 17 = 85 runs (seed 42 missing) |

**결과 — v2 (5 seeds)**:

| 신뢰도 | Split | Mean diff | x/N |
|--------|-------|-----------|-----|
| 강함 | **familiarity** | **+5.16%** | **5/5** |
| 강함 | **behavioral** | **-3.44%** | **5/5** |
| 약함 | safety | +2.19% | 4/5 (std=4.49%) |
| noise | personalisation | -0.94% | 3/5 |
| noise | age | +1.25% | 3/5 |
| OK | region (null) | -1.72% | 3/5 |

**결과 — v3 (4 seeds)**:

| 신뢰도 | Split | Mean diff | x/N |
|--------|-------|-----------|-----|
| 강함 | **helpfulness** | **+5.31%** | **4/5** |
| 중간 | factuality | +3.28% | 4/5 |
| 중간 | personalisation | -1.88% | 4/5 (std=1.27% tight) |
| noise | safety | +2.34% | 3/5 (std=4.76%) |
| 유일 MATCH | diversity | +0.78% | 4/5 (효과 작음) |

**결과 — Cross-version (공통 4 splits)**:
- **3/4 방향 일치** (personalisation, safety, region)
- age만 불일치 (v3에서 mixed)
- safety: v2 +2.19%, v3 +2.34% — 크기까지 유사하지만 **둘 다 Expected와 반대**

**핵심 발견**:
1. **v2 단일 seed의 MATCH는 우연이었다** — personalisation(+7.81%→-0.94%), safety(-4.69%→+2.19%) 모두 multi-seed에서 반전
2. Cross-version 방향 일치는 있지만 **OL 예측과는 대부분 반대 방향**
3. 가장 강한 신호(familiarity 5/5, behavioral 5/5)는 **exploratory split** — OL에서 유의하지 않았던 변수

---

### Phase 5: 진단 — 왜 OL→DPO 전이가 실패하는가?

multi-seed 결과를 놓고 원인을 분석한 결과, 네 가지 구조적 문제가 식별됨:

#### A. 모델 능력 한계
- 0.5B 모델이 GPT-4/Claude 수준 응답의 미묘한 quality 차이를 logprob으로 구분하기 어려움

#### B. Support mismatch (Off-policy 문제)
- PRISM의 응답은 21개 모델이 생성 → 학습 대상 Qwen-0.5B가 생성할 확률이 극히 낮은 텍스트
- logprob이 -300대 → 두 응답 모두 "내가 거의 생성하지 않을 텍스트"
- DPO 원논문의 가정(preference data가 policy의 support 내) 위반

#### C. Source model confound
- 그룹별로 GPT-4/Claude/Llama pair의 비율이 다를 수 있음
- pref_acc 차이가 유저 특성이 아니라 source model 분포 차이에서 올 가능성
- **통제 방법**: source model을 고정하여 실험 (아직 미실시)

#### D. Preference heterogeneity
- 같은 그룹 내에서도 유저마다 선호 방향이 다름
- score_gap이 크더라도 방향이 제각각이면 → DPO가 하나의 policy로 수렴하기 어려움
- pref_acc tail이 50~55%에 머무르는 근본 원인일 수 있음

---

### Phase 6: 데이터셋 구조 점검

#### PRISM vs HH-RLHF 비교

| | HH-RLHF | PRISM |
|--|---------|-------|
| 생성 모델 | **1개** (Anthropic 자체) | **21개** (GPT-4, Claude, Llama 등) |
| 학습 대상과의 관계 | 생성 모델 = 학습 대상 | 생성 모델 ≠ 학습 대상 |
| Support mismatch | 없음 | 있음 |
| 선호 판단 | Binary choice (crowd worker) | 0-100 score (개별 유저) |
| 유저 정보 | 없음 | 풍부 (survey) |
| Pair 수 | 160K | 18K (단일 모델 고정 시 ~1K) |
| DPO 적합성 | 설계 목적 | 설계 목적 아님 |

#### PRISM 모델별 pair 분포

| 모델 | Pairs | 오픈소스 |
|------|-------|---------|
| Cohere command | 1,584 | ✗ |
| claude-instant-1 | 1,291 | ✗ |
| **zephyr-7b-beta** | **1,274** | **✓** |
| PaLM 2 | 1,233 | ✗ |
| **Llama-2-7b-chat** | **1,198** | **✓** |
| gpt-4-turbo | 1,043 | ✗ |
| **Llama-2-70b-chat** | **995** | **✓** |

- 오픈소스 합계: 4,082 pairs (22.6%)
- 단일 오픈소스 모델 최대: zephyr-7b (1,274 pairs)

**문제**: 오픈소스 모델 하나로 고정하면 ~1,200 pairs → 유저 그룹당 ~600 pairs (v1보다 적음)

---

### Phase 7: 관련 연구 & 해결 방향 탐색

#### 유사 구조의 데이터셋

- **UltraFeedback**: 17개 LLM에서 prompt당 4개 응답 생성, GPT-4가 평가. 64K prompts. Zephyr-7B가 이걸로 DPO 학습. **support mismatch를 명시적으로 해결하지 않고 사용.**
- **Chatbot Arena (LMSYS)**: 20+ 모델의 cross-model pairwise 비교. 33K+ pairs.

#### Support mismatch 해결 방법들

| 방법 | 핵심 | 네 상황 적합성 |
|------|------|---------------|
| **On-policy DPO** | policy가 직접 생성한 응답으로 학습 | 비용 큼, 현실적 어려움 |
| **WPO** | off-policy data를 policy 확률로 reweight | **가장 적합** — 코드 수정 최소, 기존 데이터 활용 |
| **MPO** | importance sampling으로 KL term 교정 | 이론적으로 엄밀하나 구현 복잡 |
| **InCo-DPO** | on-policy + off-policy 동적 혼합 | on-policy 생성 필요 |
| **데이터 필터링** | policy logprob 너무 낮은 pair 제외 | 간단하지만 데이터 손실 |

---

### Phase 8: v5 — WPO (Support Mismatch 교정)

> [[DPO_QWen2.5B-0.5B-Instruct(v5_WPO)|v5 WPO 노트]]

| 항목 | 설정 |
|------|------|
| Method | WPO (sampled alignment) |
| Seeds | [42, 456, 789] (v4와 동일 3개) |
| Splits | 6 (v2와 동일) |
| Hyperparams | v4와 동일 (2000 steps, batch 8, beta 0.1) |

**핵심 변경**: DPO loss에 per-example weight를 곱함. weight = on-policy 정도. off-policy pair는 downweight.

**Weight 분석 — PRISM 데이터가 얼마나 off-policy인가?**:
- 평균 weight ~**0.20** (80%가 downweight됨)
- 최소값 ~0.04 (극단적 off-policy pair는 거의 무시)
- **정량적 증거**: Qwen-0.5B에게 GPT-4/Claude 응답은 자신의 생성 분포에서 매우 멀다

**결과 — WPO vs DPO 비교 (3 seeds)**:

| Split | Expected | WPO diff | WPO | DPO diff | DPO | Change |
|-------|----------|----------|-----|----------|-----|--------|
| S1_age | A > B | +3.39% | LEAN | +2.08% | LEAN | LEAN→LEAN |
| **S2_pers** | **A > B** | **+2.34%** | **MATCH** | **+0.26%** | **MISMATCH** | **FIXED** |
| **S3_safety** | **A < B** | **-2.34%** | **MATCH** | **+0.00%** | **MISMATCH** | **FIXED** |
| S4_fam | explor. | +7.03% | — | +3.39% | — | — |
| S5_behav | explor. | -1.30% | — | -4.43% | — | — |
| S6_region | A = B | +0.78% | OK | +0.78% | OK | — |

**핵심 발견**:

1. **personalisation과 safety가 MISMATCH→MATCH로 전환** — WPO가 off-policy noise를 제거하자 OL 예측과 일치하는 신호가 복원됨
2. **S2_personalisation**: DPO (-0.78%, +1.56%, +0.00%) mixed → WPO (+2.34%, +3.12%, +1.56%) 3/3 양수
3. **S3_safety**: DPO (+3.12%, -4.69%, +1.56%) 2/3 OL 반대 → WPO (-1.56%, -3.12%, -2.34%) 3/3 음수
4. **familiarity**: DPO +3.39% → WPO +7.03% (2배 증폭) — support와 무관한 가장 robust한 신호

**결론**: **Support mismatch는 OL→DPO 전이 실패의 실재하는 원인이었다.** WPO로 off-policy pair를 downweight하면 통계 모델의 예측이 DPO에서 복원된다.

---

## 현재 위치 요약

### 확립된 것
1. OL에서 유저 특성별 score_gap 차이는 통계적으로 유의하다
2. 표준 DPO에서는 이 전이가 대부분 실패한다 (MISMATCH)
3. 이 실패는 단일 seed의 문제가 아니다 — multi-seed에서도 일관적으로 MISMATCH
4. **v5 WPO**: support mismatch를 교정하면 personalisation, safety에서 MATCH 복원 (FIXED 2/4)
5. **PRISM 데이터의 ~80%가 off-policy** (weight ~0.20) — support mismatch의 정량적 증거

### 해결된 / 진행 중인 구조적 문제
1. ~~**Support mismatch**~~: **WPO로 교정 가능 확인** (v5에서 2 splits FIXED)
2. **Source model confound**: 21개 모델 응답이 혼재, 그룹별 분포 미통제 (미해결)
3. **Preference heterogeneity**: 그룹 내 선호 방향 비일관성 (미해결)
4. **측정의 차이**: OL(유저의 주관적 score_gap) vs DPO pref_acc(모델의 logprob margin)

### 열린 질문
1. WPO를 5-seed로 확장하면 현재 3-seed 결과가 robust한가?
2. Source model을 통제 + WPO 조합이면 더 많은 split이 MATCH 되는가?
3. age split이 LEAN에 머무르는 이유는? (효과 자체가 약한가, 다른 confound?)
4. 궁극적으로 P(y|x,u) 유저 조건부 학습이 필요한가? (personalized DPO)

---

## 다음 단계 (우선순위)

- [x] ~~**WPO 구현**~~ — 완료, personalisation/safety FIXED (v5)
- [ ] **5-seed 확장** — seeds 123, 999 추가하여 3-seed 결과 robustness 확인
- [ ] **Source model 분포 확인** — 각 유저 그룹별 모델 비율이 균일한지 점검
- [ ] **Source model 고정 + WPO** — zephyr-7b 고정 후 WPO 재실험
- [ ] **Weight 분포 시각화** — 어떤 pair가 가장 많이 downweight 되는지 분석
- [ ] WPO + source model 통제 후에도 남은 MISMATCH → heterogeneity 논의로 연결

---

## Reference
- [[DPO_QWen2.5B-0.5B-Instruct|v1 (pilot, 500 steps)]]
- [[DPO_QWen2.5B-0.5B-Instruct(v2)|v2 (W2 splits, single seed)]]
- [[DPO_QWen2.5B-0.5B-Instruct(v3)|v3 (OL splits, single seed)]]
- [[DPO_QWen2.5B-0.5B-Instruct(v4)|v4 (multi-seed robustness)]]
- [[DPO_QWen2.5B-0.5B-Instruct(v5_WPO)|v5 (WPO, support mismatch fix)]]
- [WPO: Weighted Preference Optimization (EMNLP 2024)](https://arxiv.org/abs/2406.11827)
- [UltraFeedback (ICML 2024)](https://arxiv.org/abs/2310.01377)
- [MPO: Maximum Preference Optimization](https://arxiv.org/abs/2312.16430)
- [InCo-DPO (2025)](https://arxiv.org/abs/2503.15880)
- [Zephyr: Direct Distillation of LM Alignment (ICLR 2024)](https://arxiv.org/abs/2310.16944)
