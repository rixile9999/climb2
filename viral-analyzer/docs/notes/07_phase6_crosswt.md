# 연구노트 07 — Phase 6: cross-WT 일반화 (leave-one-strain-out)

**연구 단위:** 실전 목표(미래 escape 예측 = 전이 성능) 검증 — 한 strain에서 배운 가중치+offset이 다른 strain의 escape를 예측하는가?
**스크립트:** `scripts/phase6_crosswt.py` · **산출물:** `outputs/phase0/phase6_crosswt.csv`

## 방법
모든 strain에 존재하는 유일한 공통 backbone인 **base ESM2-650M** 사용. 6 strain(WildType·Alpha·Beta·Gamma·Delta·Omicron) 각자의 서열 feature·CLIB offset·`>DMS` escape(~18-19). leave-one-strain-out: held-out strain을 빼고 5개로 가중치 학습 → held-out 예측. 모델 A(protein-only, 전이) / B(protein+offset, 전이) / C(CLIB-only, 학습 전혀 없음).

## 데이터 이슈 (선행 count assertion)
- 5개 strain(WildType·Alpha·Beta·Gamma·Omicron): >DMS 완전 매칭 ✓.
- **Delta 제외**: 좌표 버그가 아니라 base-650M **feature 파일이 position 0-93만 담긴 잘린 파일**(escape는 RBD 400+ 위치라 존재 불가). escape 리스트 자체는 정상 → 데이터 생성 갭. 전체 Delta 서열 재추론이 필요(GPU, 범위 밖)하여 제외.

## 발견 (5개 held-out strain 평균)
| 모델 | mean-rank | AUROC | 설명 |
|---|---|---|---|
| A protein-only (전이) | 9,631 | **0.601** | base ESM2 단백질 feature는 거의 무작위 수준으로 전이 |
| B protein+offset (전이) | 3,428 | 0.858 | offset 추가 시 급등, **5/5 strain에서 CI가 0 배제** |
| **C CLIB-only (학습 0)** | **3,257** | **0.864** | 학습·단백질 전혀 없이 뉴클레오타이드 접근성만으로 최고 |

## 인사이트 (이 연구의 정점)
1. **offset은 strain 간 견고하게 전이된다** — 5/5 strain에서 유의(ΔB−A CI 전부 0 배제, ~−6,000 mean-rank).
2. **약한 backbone에서는 CLIB이 전이 신호를 거의 전부 담당한다.** base 단백질 feature는 전이 시 AUROC 0.60(거의 무작위)인 반면, **CLIB만으로 AUROC 0.864** — 학습도 단백질도 없는 이 항이 full 모델과 대등하거나 근소 우위. CLIB이 진짜로 전이 가능한 mechanistic prior임을 최강 형태로 입증.
3. **가중치(b0,b1)는 잘 전이된다**(2개 숫자라 과적합 없음) — 문제는 base 단백질 feature 자체가 어디서든 약하다는 것(Phase 3의 in-strain 0.53과 일관).

## 주의 (반드시 함께 읽을 것)
- **backbone 한정:** 강한 ESM2cov는 WildType만 있어 cross-strain 불가. 위 결론은 **약한 base ESM2** 조건. 강한 backbone에서는 단백질 feature가 in-strain에서 분명히 기여(Phase 3).
- **Ascertainment 가능성:** DMS escape 양성이 뉴클레오타이드-접근 가능한 변이 쪽으로 큐레이션 편향돼 있으면 CLIB 우위가 과대평가될 수 있다. Baum et al. DMS는 전(全) 변이를 기능적으로 측정하므로 편향은 제한적일 것으로 보이나, 큐레이션 상세 없이 완전 배제 불가 → **차기 검증의 핵심 열린 질문.**

## 결론
offset의 가치가 전이 조건에서 최대로 확인됨. 동시에 강한-backbone cross-strain feature 부재라는 **데이터 천장**에 도달. 모델 쪽 개선은 소진.
