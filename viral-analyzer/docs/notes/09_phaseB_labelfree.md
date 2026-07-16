# Phase B — 라벨-프리 b0·b1 (통계 프록시)와 DMS-free 평가

**질문:** escape 라벨 없이, 통계 데이터만으로 b0·b1을 정할 수 있는가? 그리고 DMS 정답 없이 모델을 정직하게 평가할 수 있는가?

모델: `score = b0·S1 + b1·S2 + κ·CLIB(t=0.01)`, κ=1 고정(offset 연구 결론). backbone = ESM2cov·L1. 후보 = WT 스파이크 단일변이 24,187.

## B — 빈도 프록시로 라벨-프리 적합 (`phaseB_labelfree.py`)
- 타깃: CoV-Spectrum(LAPIS open, ~930만 서열) 관측 치환 빈도 → `y=1[proportion≥θ]`. escape 라벨 미사용.
- `logistic(y_freq ~ b0·S1+b1·S2, offset=1·CLIB)` → **b0(sem)≈0.19, b1(gram)≈1.0–2.1** (θ 무관 안정). 빈도는 문법성+CLIB을 크게, semantic은 거의 안 씀 — 생물학적으로 타당.
- escape 평가: 변이-계통 task(Omicron/Alpha/Beta/Delta)에서 등가중 대비 큰 향상, **그러나 빈도-양성과 escape 정답의 중첩이 100%(누출)**. 누출 없는 DMS에서는 등가중 대비 개선이 유의하지 않음(CI가 0 포함).

## B2 — DMS 없는 평가: 시계열(A) + Bloom 적합도(B) (`phaseB2_dmsfree.py`)
- **A 시계열 창발:** 초기창(≤2021-12) vs 후기창(≥2022-06). 창발 양성 = 초기 희귀·후기 우세. `proxy_early`(초기 빈도로만 적합)가 미래 창발을 가장 잘 예측: **AUROC 0.899 > 지도학습(DMS) 0.894 > 등가중 0.860**. 미래는 진짜 held-out이라 누출 없음.
- **B Bloom 계통수 적합도:** 독립발생 기반 delta_fitness(7,472 후보). 상위 고적합 식별 AUROC ~0.85, 단 연속 순위상관은 낮음(0.08–0.15) — escape ≠ 일반 적합도.

## B3 — 누출제거(LOVO) + 다중 컷오프 (`phaseB3_dmsfree_plus.py`)
- **(2) leave-one-variant-out:** 평가 변이 V의 정의변이를 빈도프록시 적합에서 **제외**한 뒤 V로 평가. 핵심 결과 — 누출을 제거해도 라벨-프리 가중치가 **5/5 변이에서 등가중 대비 유의하게 우수**(ΔAUROC +0.033~+0.060, 모든 CI가 0 배제)이고 proxy_full과 거의 동일(예: Omicron 0.982 vs 0.983). → **빈도프록시의 이득은 순환 암기가 아니라 일반화되는 실제 escape 신호.**
- **(3) 다중 컷오프 시계열:** 3개 컷오프(2021-06/2021-12/2022-06). proxy_early가 2/3에서 등가중 대비 유의 우세(C1 +0.036 CI[0.025,0.046], C2 +0.038 CI[0.017,0.061]), C3만 비유의(+0.018). 가중치는 컷오프 무관 안정(b0≈0.18, b1≈1.5–1.8).

## 결론
1. **라벨-프리 적합 가능** — 통계(빈도)만으로 안정적이고 생물학적으로 타당한 b0·b1 획득.
2. **LOVO가 누출 반증** — 변이 task 향상은 순환이 아니라 일반화. 라벨-프리 가중치는 미학습 변이의 escape도 잘 순위매김.
3. **DMS 없는 정직한 평가 성립** — 시계열 창발 예측에서 라벨-프리 CLIMB가 DMS 지도학습에 필적/우위. 시계열이 최선의 DMS-free 벤치마크.
4. 미해결: 최신기(C3)에서 이득 축소 — 후속에서 교차신호/앙상블/멀티백본으로 추적.

## 데이터/재현
`scripts/fetch_observed_freq.py`(LAPIS 스냅샷 → `data/observed/*.tsv`), Bloom `data/observed/bloom_spike_fitness.tsv`.
```bash
conda activate vanalyzer
python scripts/phaseB_labelfree.py      # 빈도프록시 적합 + escape 평가(+누출 진단)
python scripts/phaseB2_dmsfree.py       # 시계열(A) + Bloom(B)
python scripts/phaseB3_dmsfree_plus.py  # LOVO 누출제거 + 다중 컷오프
```

## B4 — 교차신호 일반화 (`phaseB4_crosssignal.py`)
3개 독립 통계신호(빈도·Bloom 적합도·시계열-초기)로 각각 b0·b1을 라벨-프리 적합 후 교차평가.
- **가중치 수렴:** 셋 다 grammar≫semantic으로 수렴(b0 0.19–0.29, b1 0.94–1.68). CLIB offset 고정.
- **핵심:** **Bloom로 적합한 가중치(b0=0.285, b1=0.942)가 DMS 지도학습 가중치(0.268, 0.955)를 거의 복원**하고, 누출 없는 **DMS AUROC 0.899로 최고**(등가중 0.884, freq-fit 0.883). → 계통수 적합도가 기능적 escape의 최적 라벨-프리 프록시이며, escape 라벨 없이 지도학습급 가중치 도달.
- 세 신호 모두 variant_avg(0.97 vs 등가중 0.93)·emergent_avg(0.90 vs 0.87)에서 등가중 상회.
