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

## B5 — 멀티백본 강건성 (`gen_backbone_dumps.py`, `phaseB5_multibackbone.py`)
base ESM2·Hie·ESM2cov 세 backbone에서 라벨-프리(freq/bloom-fit) vs 등가중/CLIB-only/지도학습(DMS) 비교. 정직한 시험(DMS, 창발).
- **ESM2cov(강·도메인적응):** bloom-fit(0.285,0.942)이 지도학습(0.268,0.955) 복원, DMS 0.899·창발 0.898로 지도학습과 동등. ✓
- **base ESM2(약):** 단백질 S1/S2가 노이즈(등가중 DMS 0.786). 라벨-프리가 이를 올바르게 ~0/음수로 낮추고 **CLIB-only(DMS 0.883)가 지배**. 모델이 CLIB로 환원. ✓(무해)
- **Hie:** freq/bloom-fit이 grammar에 **음의 가중치**를 줘 DMS 0.837/0.861로 지도학습(0.907)보다 **악화**. 창발은 전 방법 ~0.5(무작위). ✗ 라벨-프리 실패.
- **[정정] 결론:** Hie는 **범용 PLM이 아니라 Hie et al.이 바이러스별로 학습시킨 LSTM**이므로 backbone-보편성의 데이터포인트에서 **제외**해야 한다 — 그 실패는 범용 PLM 보편성의 반례가 아니다. **범용 PLM(ESM2 계열)만 보면 반례가 없다**: base ESM2는 약한 신호를 무해하게 CLIB로 환원, ESM2cov는 지도학습 복원. 즉 **범용 PLM 한정 보편성은 열려 있고 그럴듯**하다(B8에서 실제 시험). 실무 권고는 여전히 **ESM2cov + bloom-fit**.

## B6 — 앙상블 타깃 = 무개선 (종점) (`phaseB6_ensemble.py`)
세 신호(빈도·Bloom·초기빈도) rank-sum 복합 타깃으로 적합 → 단일 최고를 못 넘음. DMS: ensemble 0.887 < bloom_fit 0.899. 최신기 C3는 전 방법 ~0.83 수렴(재가중 무관, 내재적 한계). → offset 연구와 동일하게 **복잡도 추가 무효 = 라벨-프리 최적화 종점**.

## 종점 결론
- **권장 라벨-프리 CLIMB = ESM2cov + bloom-fit (b0=0.285, b1=0.942), CLIB offset κ=1, t=0.01.** escape 라벨 0개로 지도학습급(DMS 0.899, 창발 0.898).
- 정직성 검증: LOVO 5/5, 시계열 창발 proxy_early 0.899(>지도 0.894), 3신호 가중치 수렴.
- 한계: backbone 의존(ESM2cov에서만 성립), 최신기(2022-06+) 이득 축소는 데이터/신호 한계.

## B7 — 강한 백본 cross-strain 전이 (flagship, `phaseB7_crossstrain.py`)
지금까지 변이 strain엔 ESM2cov가 없어 못 풀던 #1 질문. 5개 clean strain(WT/Alpha/Beta/Gamma/Omicron; Delta는 참조서열 'X'로 pos94 절단 → Phase6와 동일 제외)에 ESM2cov 단일변이 feature를 생성 후 leave-one-strain-out 전이.
- **protein_only 0.842** (Phase6 base ESM2 ~0.60) — 강한 백본은 단백질 신호가 **strain 간 전이됨**.
- **protein+offset 0.890** > **CLIB_only 0.864** — 강한 백본이 CLIB 위에 전이 가능한 escape 신호를 **추가**.
- CLIB_only 0.864 = Phase6와 동일(백본 무관, 재확인).
- **결론:** Phase6의 "CLIB이 전이신호 대부분을 담당"은 **약한 백본 한정**이었다. 도메인적응 백본(=CoVFit 백본)은 strain 배경을 넘어 일반화되는 escape 표현을 학습했다 — 강한 백본의 cross-strain 가치를 처음으로 실증.

## B8 — 범용 PLM 보편성 (ESM2 150M/650M/3B/cov) (`phaseB8_pluniversality.py`)
Hie 제외, 범용 PLM만 시험(SARS 스파이크, DMS·창발).
- **protein-only DMS AUROC**: 150M 0.563, 650M 0.497, 3B 0.505, **ESM2cov 0.817**. → **크기(150M→3B)는 escape 신호를 안 준다; 도메인 적응이 준다.**
- **CLIB-only = 0.883 (전 모델 동일)** — backbone 무관 강한 **보편 베이스라인**. 모든 백본을 protein-only ~0.50에서 ~0.88로 끌어올림.
- **label-free(bloom) ≈ supervised**: 150M gap +0.000, ESM2cov +0.001 성립 / 650M −0.017, 3B −0.053 미달(단백질 노이즈 → CLIB-only가 최선, 어떤 protein 가중도 그 이하).
- **결론:** (1) **CLIB offset의 보편성 = 성립**(모든 범용 PLM). (2) "단백질 신호가 escape 담음 & 라벨-프리 지도학습급"은 **모델 일반성/크기가 아니라 도메인 적응 여부**의 문제 — ESM2cov(도메인적응)에서만. (3) 라벨-프리는 어떤 범용 PLM에서도 CLIB-only 근처 이하로 안전(크게 해롭지 않음).

## B9 — 종간 CLIB 보편성 (Influenza HA, HIV Env) (`port_hie_species.py`, `phaseB9_species.py`)
Hie 데이터(Doud2018 flu HA 170변이/41부위, Dingens2019 HIV Env 161변이/69부위) 포팅, base ESM2-650M, 5-fold CV.
- **CLIB_only: flu 0.508, HIV 0.547 (거의 무작위)** — SARS의 0.883과 정반대. **CLIB offset 추가 시 악화**(flu 0.639→0.527, HIV 0.603→0.583). protein-only ESM2가 약하지만 최선(0.60~0.64).
- **결정적 confound — modal-codon 근사**: SARS는 실제 codon(seq_cov_wt.csv), flu/HIV는 표준 modal-codon 역번역(Hie 전체 데이터도 AA-only, CDS 부재 확인). CLIB의 단일-염기 접근성은 codon에 민감하므로, 이 near-random이 (a)근사 아티팩트인지 (b)진짜 종 차이인지 실제 CDS 없이 분리 불가.
- **[confound 제거] 실제 CDS로 재실행**: GenBank에서 WSN HA(J02176, 내 WT와 99.1% 일치)·BG505 Env(DQ208458, 99.9%) CDS를 받아 **실제 codon(99%+)**으로 codon table 재구성 후 재실행 → **결과 사실상 동일**(CLIB_only flu 0.500, HIV 0.581; +offset flu 0.513, HIV 0.618). 즉 near-random은 **modal-codon 아티팩트가 아님**.
- **최종 결론:** **CLIB의 escape 예측력은 종간 보편적이지 않다 — SARS-CoV-2 특이적**. flu HA는 완전 무작위(0.50), HIV Env는 약함(0.58). "보편적 메커니즘 prior"는 SARS에 국한. 반면 **protein LM 신호가 flu/HIV에서 (약하게라도) 최선** → 종-이전성은 CLIB보다 단백질 신호 쪽. 해석: SARS escape는 팬데믹 초기 저다양성에서 단일-염기 접근성에 강하게 제약(CLIB가 포착), flu/HIV(고다양성·항원 구조 제약)의 DMS escape는 접근성으로 설명 안 됨.
