## 요지
DMS 없이 특정 wild-type에 대한 **CAC 스코어의 합리적 계수(b0·b1)를 선정**하고, 그 방법의 **모델·종 보편성**을 확립한다. CLIMB: `score = b0·S1(semantic) + b1·S2(grammar) + κ·CLIB`, κ=1 offset 고정, b0·b1만 결정.

작업 브랜치: `research/climb-labelfree` (미merge). 리포트: `viral-analyzer/docs/research_report_labelfree_en.md`, 노트: `viral-analyzer/docs/notes/09_phaseB_labelfree.md`.

## 답 (현재까지)
**DMS 없이 계수 선정**: b0·b1을 escape가 아니라 **Bloom 계통수 적합도**(라벨-프리, 공개데이터)에 로지스틱 적합(CLIB offset 고정) → `(b0,b1)≈(0.285,0.942)`로 **DMS 지도학습 가중치(0.268,0.955)를 복원**, escape 라벨 0개로 지도학습급. 단순 fallback은 z-score 후 등가중(1,1).

## 검증된 결과 (Phase B–B8, 독립 적대적 검증 완료)
- [x] B: 빈도프록시 라벨-프리 적합 (누출 진단)
- [x] B2: DMS-free 평가 — 시계열 창발 예측(AUROC 0.899, 지도학습과 동률) + Bloom 교차검증
- [x] B3: LOVO 누출제거(5/5 변이 생존) + 다중 컷오프
- [x] B4: 3신호(빈도·Bloom·시계열) 가중치 수렴
- [x] B5→B8 정정: **Hie는 범용 PLM이 아니라 바이러스별 LSTM** → backbone-보편성 반례에서 제외
- [x] B7 (flagship): **강한 백본 cross-strain 전이** — protein-only 0.842 (base ESM2 0.60), protein+offset 0.890 > CLIB-only 0.864
- [x] B8: **범용 PLM 보편성** — CLIB-only=0.883 전 모델 동일(보편), protein escape 신호는 **크기 아닌 도메인 적응**에서 옴(150M/650M/3B protein-only ~0.50, ESM2cov 0.82)

## 핵심 결론
1. **CLIB offset의 보편성 = 성립** (모든 범용 PLM에서 ~0.88).
2. "단백질 신호가 escape 담음 & 라벨-프리=지도학습"은 **도메인 적응** 속성(ESM2cov). 모델 크기·일반성 아님.
3. 라벨-프리는 어떤 범용 PLM에서도 CLIB-only 이하로 안전.

## 남은 작업 (tracking)
- [ ] **종간 보편성**: Influenza HA·HIV Env로 확장. CLIB용 SBS 스펙트럼은 저장소에 존재(`sbs_freq.csv`), escape 정답은 Hie 코드(`viral-mutation/`)에 있어 viral-analyzer 포맷 포팅 필요. 라벨-프리 타깃은 종별 빈도(Nextstrain flu / LANL HIV)로 대체.
- [ ] 종별 CLIB(SBS+stationary dist) 등록 + base ESM2 feature 생성(HA~566aa, Env~860aa)
- [ ] 최신기(2022-06+) 라벨-프리 이득 축소 원인 규명
- [ ] clean(X-free) Delta 참조서열로 strain 패널 완성
- [ ] `research/climb-labelfree` → main merge 여부 결정

## 산출물
아티팩트: 📊 빈도프록시 · 🔭 DMS-free · 🧭 Label-Free 종합 · ⚖️ 등가중 작동점.
스크립트: `viral-analyzer/scripts/phaseB*.py`, `fetch_observed_freq.py`, `gen_backbone_dumps.py`.
