## 배경 — 관찰된 결과 (Phase B9)
CLIB(뉴클레오타이드 접근성 prior)의 escape 예측력은 **SARS-CoV-2 특이적**이고 종을 넘지 못한다. base ESM2-650M, 5-fold CV, CLIB-only escape AUROC:

| 종 | CLIB-only | protein-only | protein+offset |
|---|---|---|---|
| SARS-CoV-2 Spike | **0.883** | ~0.50 | ~0.88 |
| Influenza HA (Doud2018) | 0.500 | 0.639 | 0.513 |
| HIV Env (Dingens2019) | 0.581 | 0.603 | 0.618 |

- flu/HIV에선 CLIB가 거의 무작위이고 **offset을 더하면 오히려 악화**.
- **confound 제거됨**: 실제 CDS(WSN HA J02176 99.1%·BG505 Env DQ208458 99.9%)로 99%+ 실제 codon 재실행해도 동일 → modal-codon 근사 아티팩트 아님.

## 가설 (핵심)
다른 종(Influenza·HIV)에서 CLIB가 안 통하는 이유는 **이 종들이 역사가 길어(고다양성) 단일 point mutation만으로 escape 될 가능성이 낮기 때문**이다. escape가 여러 변이의 조합으로 일어나므로, **단일변이(point) 기반 방법론 자체가 이 종들엔 부적합**하다.

## 제안 방향 — point → multi 확장
다른 종을 제대로 타겟하려면 방법론을 **단일변이에서 다중변이(combinatorial)로 확장**해야 한다.

### 설계 노트
- **CLIB는 이미 다중변이로 자연 확장된다**: CTMC `P=exp(t·Q)` + 위치독립 가정 하에서, 다중변이 조합의 접근성 = 각 위치 전이확률의 곱. 즉 다중변이 변이체의 CLIB = ∏ᵢ P(codonᵢ → aaᵢ, t).
- **t 파라미터가 point↔multi 손잡이**: 작은 t(=0.01)는 단일-염기 regime(SARS, offset 연구가 여기 고정). **큰 t는 다중 치환을 허용 = 다중변이 regime**(오래된 종). → "point→multi 확장"은 사실상 "다중변이 변이체 스코어링 + CLIB의 큰 t".
- **Semantic/Grammar도 다중변이로**: 전체 다중변이 서열의 임베딩 이동(semantic), 위치별 확률의 결합(grammar).

### 필요 데이터 (다중변이 ground truth)
- **Wu2020** flu HA "site B" 조합 fitness (`data/influenza/fitness_wu2020/data_all.csv`, 6–7 부위 조합) — 저장소에 존재, 즉시 프로토타입 가능.
- Haddox2018 HIV Env prefs, 균주 간 항원 드리프트(H3 clusters) 등.

### 다음 단계 (tracking)
- [ ] 다중변이 CLIB/CAC 스코어러 구현(변이 집합 → semantic·grammar·multi-CLIB, t 스윕)
- [ ] Wu2020 조합 fitness로 프로토타입: 다중변이 CAC가 조합 적합도를 예측하는가, 큰 t가 필요한가
- [ ] flu/HIV 다중변이 escape/항원거리 ground truth 확보
- [ ] point vs multi 성능을 종별로 비교

## 관련
- 상위 연구: branch `research/climb-labelfree`, 리포트 `viral-analyzer/docs/research_report_labelfree_en.md`, 노트 `docs/notes/09_phaseB_labelfree.md` (B9), 아티팩트 🦠 교차종 리포트.
- 관련 이슈: #1 (라벨-프리 CAC 계수 & 보편성)
