# 연구노트 05 — Phase 4 (분기): 다중 backbone 스태킹은 이득이 있는가?

**연구 단위:** 사용자의 원래 "메타모델" 비전을 정직하게 시험 — 여러 단백질 LM의 신호를 결합하면 단일 최강 backbone을 넘어설 수 있는가?
**분기:** `fork/phase4-stacking` (단순 2-param offset 유지 vs 복잡한 다중-backbone 메타모델)
**스크립트:** `scripts/phase4_stack.py` · **산출물:** `outputs/phase0/phase4_stack.csv`

## 핵심 전제
CLIB은 protein 모델과 무관하게 코돈에만 의존 → **backbone 전체에 공통 offset** 하나. 정답이 적으므로(DMS 19, Omicron 30) 6개 feature 학습은 과적합 위험 → L2(ridge, 강도 inner-CV) 필수.

## 방법
모든 backbone을 mutation_code로 정렬(24,187개 전부 정렬됨). L2-로지스틱 + CLIB offset(t=0.01, κ=1), 5 seeds × 5-fold nested CV. 모델: M1(cov), M2(cov+base), M3(cov+base+hie). 판정: mean-rank delta(vs M1)의 부트스트랩 CI가 0 미만이면 스태킹 승리.

## 발견
| task | model | mean-rank | AUROC | Δ vs M1 | CI | 이김? |
|---|---|---|---|---|---|---|
| DMS | M1 cov | 2,868 | 0.882 | 0 | — | — |
| DMS | M3 cov+base+hie | 2,525 | 0.896 | −341 | [−812, +72] | ✗ (유의X) |
| Omicron | M1 cov | 391 | 0.985 | 0 | — | — |
| Omicron | M2 cov+base | 526 | 0.979 | +137 | [56, 232] | ✗ 악화 |
| Omicron | M3 cov+base+hie | 827 | 0.966 | +443 | [210, 742] | ✗ 악화 |

## 인사이트 (음성 결과 — 값짐)
- **스태킹은 단일 최강 backbone(ESM2cov)을 정직하게 이기지 못한다.** DMS에선 전체 스택이 점추정 개선(AUROC 0.88→0.896)을 보이나 정답 19개로는 유의성에 못 미침. Omicron에선 이미 near-perfect라 약한 backbone 혼합이 신호를 희석해 **유의하게 악화**.
- 이는 "645 라벨이 모델 복잡도를 제약한다"는 초기 결론의 **직접 실증**이다. 복잡한 메타모델이 정직한 조건에서 단순 모델을 못 이긴다.
- **단, 힌트:** DMS 같은 어려운 task에서 다양한 backbone(Hie 포함)의 점추정 개선은 실재한다. escape 정답이 더 많아지면 유의해질 여지 → 라벨이 늘면 재검토.

## 분기 결정
**스태킹 폐기, 단순 2-param offset(M1_cov) 유지.** 실험/노트는 연구 라인에 병합해 기록 보존. 모델 자체는 채택하지 않음.
