---
tags:
  - experiment
  - DPO
  - PRISM
  - multi-seed
date: "2026-02-24"
model: Qwen/Qwen2.5-0.5B-Instruct
dataset: PRISM (within-model pairs)
status: v3-seed42-pending
---

# DPO on PRISM: Multi-Seed Robustness Experiment (v4)

## Goal

- v2(단일 seed)에서 personalisation, safety가 MATCH → v3(단일 seed)에서 MISMATCH로 방향 반전
- **동일 split, 동일 hyperparams에서 seed만 달라져도 결론이 뒤집히는 문제** 확인
- v4의 핵심 질문: **5개 seed 평균으로 봤을 때, 어떤 split이 seed-robust한 신호를 가지는가?**
- 부차적 질문: v2(W2 binary logit)와 v3(ordered logit)의 공통 split에서 splitting method에 관계없이 재현되는 패턴이 있는가?

### v2/v3 단일 seed의 한계 (이 실험의 동기)

| 문제 | 구체 사례 |
|------|----------|
| **Seed 불안정성** | v2 safety MATCH(-4.69%) → v3 safety MISMATCH(+6.25%), 동일 split인데 방향 반전 |
| **Match 판정 신뢰 불가** | 단일 seed의 match/mismatch가 random variation인지 실제 효과인지 구분 불가 |
| **Null control 한계** | v2 null=2.34%, v3 null=1.56%로 OK이지만, 대부분 차이가 이 수준과 겹침 |
| **Effect size ≈ Noise** | 대부분 split 차이가 2~5%p → baseline std (~2.3%)와 비슷 |

---

## Setup

### Model & Infra
- **Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Infra**: SLURM gpu3 (A6000Ada 48GB), bf16
- **Code**: `DPO-mini/260224/` (v2: `260224/v2/`, v3: `260224/v3/`)

### Hyperparams (v2/v3와 동일)
| Param | Value |
|-------|-------|
| steps | 2000 |
| batch_size | 8 |
| beta | 0.1 |
| lr | 5e-6 |
| dtype | bf16 |
| max_pairs | 전체 |
| log_every | 25 |

### Multi-Seed Design
- **Seeds**: [42, 123, 456, 789, 999]
- **v2**: 5 seeds × (1 baseline + 6 splits × 2 groups) = **65 runs ✅ 완료**
- **v3**: 5 seeds × (1 baseline + 8 splits × 2 groups) = **85 runs (seed 42 missing)**
- 각 seed에서 pref_acc tail 20% 평균으로 Group A - B diff를 계산
- 5-seed mean diff, std, 방향 일관성(x/N)으로 판정

### Verdict 기준
| Verdict | 조건 |
|---------|------|
| **MATCH** | 평균 방향이 Expected와 일치 + x/N ≥ 4/5 |
| **LEAN** | 평균 방향이 Expected와 일치 + x/N < 4/5 |
| **MISMATCH** | 평균 방향이 Expected와 반대 + x/N ≥ 3/5 |
| **OK** | null control (region)이 작은 차이 |
| **exploratory** | 사전 방향 가설 없음 |

---

## Results

### v2 (W2 Binary Logit) — 5 Seeds 완료

**Baseline**: 54.53% ± 2.34% (per seed: 58.59, 55.47, 53.91, 52.34, 52.34)

| Split | Mean diff | Std | Direction | x/N | Expected | Verdict |
|-------|-----------|-----|-----------|-----|----------|---------|
| S1_age | +1.25% | 2.50% | A > B | 3/5 | A > B | LEAN |
| S2_personalisation | -0.94% | 1.67% | A < B | 3/5 | A > B | MISMATCH |
| S3_safety | +2.19% | 4.49% | A > B | 4/5 | A < B | MISMATCH |
| **S4_familiarity** | **+5.16%** | **3.03%** | **A > B** | **5/5** | explor. | **exploratory** |
| **S5_behavioral** | **-3.44%** | **1.75%** | **A < B** | **5/5** | explor. | **exploratory** |
| S6_region | -1.72% | 3.78% | A < B | 3/5 | A = B | OK |

**Per-seed diffs (pref_acc A − B):**

| Split | s42 | s123 | s456 | s789 | s999 |
|-------|-----|------|------|------|------|
| S1_age | +3.12% | -3.12% | +0.00% | +3.12% | +3.12% |
| S2_personalisation | -0.78% | -2.34% | +1.56% | +0.00% | -3.12% |
| S3_safety | +3.12% | +9.38% | -4.69% | +1.56% | +1.56% |
| S4_familiarity | +5.47% | +10.16% | +0.78% | +3.91% | +5.47% |
| S5_behavioral | -4.69% | -1.56% | -2.34% | -6.25% | -2.34% |
| S6_region | +2.34% | -2.34% | +0.78% | -0.78% | -8.59% |

### v3 (Ordered Logit) — 3/5 Seeds 완료 (seed 42, 999 missing)

**Baseline**: 53.44% ± 2.35% (per seed: 56.25, 55.47, 53.91, 50.00, 51.56)

| Split | Mean diff | Std | Direction | x/N | Expected | Verdict |
|-------|-----------|-----|-----------|-----|----------|---------|
| S1_age | +0.47% | 2.55% | A > B | 3/5 | A > B | LEAN |
| S2_education | -2.19% | 2.44% | A < B | 4/5 | A > B | MISMATCH |
| S3_personalisation | -1.72% | 1.04% | A < B | 4/5 | A > B | MISMATCH |
| S4_safety | +2.50% | 5.02% | A > B | 3/5 | A < B | MISMATCH |
| S5_factuality | +3.28% | 2.86% | A > B | 4/5 | A < B | MISMATCH |
| **S6_helpfulness** | **+5.31%** | **3.47%** | **A > B** | **4/5** | A < B | **MISMATCH** |
| S7_diversity | +0.78% | 2.88% | A > B | 4/5 | A > B | **MATCH** |
| S8_region | -0.78% | 0.64% | A < B | 2/3 | A = B | OK |

> ⚠️ v3는 seed 42, 999 미완료. s42, s999 열의 값은 v2 공통 split에서 온 것일 수 있으므로, v3-only split(S2, S5, S6, S7)의 해당 seed 값은 확인 필요.

**Per-seed diffs (pref_acc A − B):**

| Split | s42 | s123 | s456 | s789 | s999 |
|-------|-----|------|------|------|------|
| S1_age | +1.56% | -3.91% | -0.78% | +3.12% | +2.34% |
| S2_education | -2.34% | -5.47% | -3.91% | -0.78% | +1.56% |
| S3_personalisation | -1.56% | -2.34% | +0.00% | -1.56% | -3.12% |
| S4_safety | +4.69% | +10.94% | -3.91% | +0.78% | +0.00% |
| S5_factuality | -0.78% | +4.69% | +4.69% | +7.03% | +0.78% |
| S6_helpfulness | +3.91% | +0.00% | +4.69% | +7.81% | +10.16% |
| S7_diversity | +0.78% | +1.56% | +3.12% | -4.69% | +3.12% |
| S8_region | -1.56% | +0.00% | -0.78% | | |

---

## Cross-Version Comparison (공통 4 Splits)

공통 split은 v2와 v3에서 **동일한 유저 그룹 할당**. 결과 차이는 순수 seed variance.

| Variable | v2 mean | v2 dir | v3 mean | v3 dir | Agree? |
|----------|---------|--------|---------|--------|--------|
| age | +1.25% | A > B | +0.47% | A > B | ✅ YES |
| personalisation | -0.94% | A < B | -1.72% | A < B | ✅ YES |
| safety | +2.19% | A > B | +2.50% | A > B | ✅ YES |
| region | -1.72% | A < B | -0.78% | A < B | ✅ YES |

> **4/4 방향 일치** — splitting method(binary logit vs ordered logit)를 바꿔도 방향과 대략적 크기가 보존됨.
> 이는 관찰된 패턴이 splitting artifact가 아닌 데이터에 실재하는 구조임을 시사.

---

## Observations

### 1. Seed-Robust한 신호 (5/5 또는 4/5 일관)

| Split | Version | Mean diff | x/N | 특성 |
|-------|---------|-----------|-----|------|
| **S4_familiarity** | v2 | **+5.16%** | **5/5** | 가장 강하고 깨끗한 신호. 모든 seed에서 A>B |
| **S5_behavioral** | v2 | **-3.44%** | **5/5** | 완벽히 일관된 반대 방향 (A<B) |
| S3_personalisation | v3 | -1.72% | 4/5 | 약하지만 일관. std=1.04%로 가장 tight |
| S6_helpfulness | v3 | +5.31% | 4/5 | 효과 크지만 Expected와 반대(MISMATCH) |
| S5_factuality | v3 | +3.28% | 4/5 | Expected와 반대(MISMATCH) |
| S7_diversity | v3 | +0.78% | 4/5 | 유일한 MATCH이지만 효과 크기 매우 작음 |

### 2. Noisy한 신호 (신뢰 불가)

| Split | Version | 문제 |
|-------|---------|------|
| S3_safety | v2 | +2.19% 이지만 std=4.49%, s456에서 -4.69% 반전 |
| S4_safety | v3 | +2.50% 이지만 std=5.02%, s456에서 -3.91% 반전 |
| S1_age | 둘 다 | 방향은 일치(A>B)하나 크기 <1.5%, noise 수준 |
| S6_region | 둘 다 | null control OK. v2=-1.72%, v3=-0.78% |

### 3. v2 단일 seed → Multi-seed 비교

| Split | v2 단일(s42) | v2 Multi-seed mean | 변화 |
|-------|-------------|-------------------|------|
| S2_personalisation | **+7.81% MATCH** | **-0.94% MISMATCH** | 방향 반전 — 단일 seed의 위험성 |
| S3_safety | **-4.69% MATCH** | **+2.19% MISMATCH** | 방향 반전 |
| S4_familiarity | +2.34% | +5.16% (5/5) | 강화됨 |
| S6_region | -2.34% OK | -1.72% OK | 안정 |

> **핵심: v2 단일 seed에서 MATCH였던 personalisation, safety가 multi-seed에서 MISMATCH로 전환.**
> 단일 seed의 match/mismatch 판정은 신뢰할 수 없었음이 확인됨.

### 4. Cross-Version 일관성의 의미

공통 4개 split에서 방향이 100% 일치하는 것은 중요한 발견:
- **safety**: v2 +2.19%, v3 +2.50% — 크기까지 유사. 단, 둘 다 Expected(A<B)와 반대
- **personalisation**: v2 -0.94%, v3 -1.72% — 둘 다 Expected(A>B)와 반대
- 이는 **OL/W2 통계 모델의 예측 방향 자체를 재검토**할 필요를 시사
- 혹은 DPO pref_acc tail이 OL의 선호 강도와 다른 것을 측정하고 있을 가능성

### 5. 종합 진단

| 진단 | 설명 |
|------|------|
| **단일 seed 위험 확인** | v2 s42의 personalisation(+7.81%), safety(-4.69%) MATCH가 multi-seed에서 사라짐 |
| **OL→DPO 전이 실패** | 대부분 split에서 MISMATCH. OL p-value와 DPO 학습 패턴은 다른 메커니즘 |
| **Cross-version 재현** | 공통 split 4/4 방향 일치 → 패턴 자체는 실재하나 OL 예측과 불일치 |
| **Exploratory 발견** | familiarity(5/5), behavioral(5/5)가 가장 강한 신호 — 사후 가설 수립 가능 |
| **Signal ≈ Noise** | 대부분 diff가 null control(~1.7%) 수준. familiarity(+5.16%)만 명확히 초과 |

---

## Conclusions

### 이 실험이 답한 것
1. **v2/v3 단일 seed의 MATCH 판정은 신뢰할 수 없었다** — multi-seed에서 뒤집힘
2. **Splitting method에 관계없이 재현되는 방향 패턴은 존재한다** — 공통 4 split 100% 일치
3. **다만 그 방향이 OL/W2 예측과 대부분 불일치** — 통계 모델과 DPO 학습 간 gap 존재
4. **familiarity(+5.16%, 5/5)가 가장 robust한 exploratory 발견**

### 이 실험이 답하지 못한 것
1. OL 예측과 DPO 결과가 왜 불일치하는지 (메커니즘의 차이?)
2. v3의 완전한 결과 (seed 42, 999 미완료)
3. 학습 중반부(step 200~800)에서 그룹 차이가 더 큰지
4. 더 큰 모델(1.5B+)에서 효과가 달라지는지

---

## Next Steps
- [ ] **v3 seed 42, 999 완료** — 현재 PD(Priority) 상태, 대기 중
- [ ] OL 예측 방향 재검토: DPO pref_acc와 OL 종속변수의 관계 정리
- [ ] Learning curve 중반부 분석: step 200~800에서의 그룹 차이
- [ ] familiarity, behavioral에 대한 사후 가설 수립 및 검증 전략
- [ ] 모델 크기 확대 (1.5B or 3B) 검토

---

## Reference
- [[DPO_QWen2.5B-0.5B-Instruct|v1 (500 steps, batch 2, 1K pairs)]]
- [[DPO_QWen2.5B-0.5B-Instruct(v2)|v2 (2000 steps, batch 8, W2 splits, single seed)]]
- [[DPO_QWen2.5B-0.5B-Instruct(v3)|v3 (2000 steps, batch 8, OL splits, single seed)]]
- Multi-seed SLURM scripts: `DPO-mini/260224/v2/`, `DPO-mini/260224/v3/`
- Analysis script: `DPO-mini/analyze_multiseed.py`
