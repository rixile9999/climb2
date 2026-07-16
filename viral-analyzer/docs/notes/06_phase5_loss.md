# 연구노트 06 — Phase 5: 순위 정렬 손실(pairwise) vs pointwise logistic

**연구 단위:** 리포트 지표가 순위(mean-rank)이므로, 계획서 권고대로 pairwise learning-to-rank 손실이 pointwise logistic보다 나은지 A/B.
**스크립트:** `scripts/phase5_loss.py` · **산출물:** `outputs/phase0/phase5_loss.csv`
**분기 여부:** 즉시 승자를 채택하는 튜닝 A/B(병렬 유지가 필요한 아키텍처 갈림길이 아님) → 연구 라인에서 인라인 비교.

## 방법
ESM2cov, DMS·Omicron, 고정 offset(t=0.01, κ=1). POINT=로지스틱 NLL, PAIR=RankNet식 pos–neg 쌍 로지스틱(음성 4,000개 subsample). 5 seeds × 5-fold OOF, paired bootstrap으로 mean-rank delta.

## 발견
| task | mr_point | mr_pair | AUROC p/pair | Δ(pair−point) | CI | pair 승? |
|---|---|---|---|---|---|---|
| DMS | 2,868 | 2,947 | 0.882 / 0.879 | +79 | [−71, +248] | ✗ |
| Omicron | 391 | 381 | 0.984 / 0.985 | −9 | [−42, +27] | ✗ |

## 인사이트
- **손실 종류는 유의한 차이를 주지 않는다**(두 CI 모두 0 포함). 학습 가중치가 (b0,b1) 2개뿐이라, 로지스틱이 찾는 선형결합 방향이 이미 사실상 순위-최적이다. 모델이 이렇게 단순하면 손실 함수가 순위에 영향을 못 준다.
- **단순 pointwise logistic 유지**(convex, 잘 이해됨).

## 누적 결론 — 모델 복잡도 plateau
Phase 3b(이산화)·4(스태킹)·5(손실)에서 **3연속 "유의 개선 없음".** 현재 라벨·신호 예산에서 모델 쪽 개선은 소진됐다. 남은 유의미한 축은 **모델이 아니라 데이터/일반화** — 즉 cross-WT 전이(base ESM2로 leave-one-strain-out)와 더 많은 escape 라벨이다. 다음: Phase 6 cross-WT 일반화.
