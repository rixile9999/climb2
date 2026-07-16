# 연구노트 02 — go/no-go: CLIB은 독립적인 OOS 신호를 갖는가?

**연구 단위:** 6일치 오프셋 계획에 착수하기 전, 가장 위험한 가정 하나를 값싸게 검증한다 — S3(CLIB)가 S1·S2 너머의 독립적 escape 신호를 정말 갖는가?
**스크립트:** `scripts/phaseD_gonogo.py`
**산출물:** `outputs/phase0/phaseD_gonogo.csv`

## 방법
정직한 out-of-sample 평가를 위해 5-fold 교차검증. 세 모델을 비교: A(κ=0, CLIB 없음), B(offset κ=1), C(free b3, S3 계수까지 학습). ESM2cov backbone, clean task(Omicron n=30, DMS n=19), t∈{0.1,1.0}. 판정: 정답 부트스트랩으로 얻은 mean-rank 개선폭 95% CI가 0을 배제해야 "도움". S3 독립성은 LRT(우도비검정)로 선별.

## 발견
| task | t | LRT p(S3) | mean-rank A→B | ΔB−A CI | AUROC A→B | 도움? |
|---|---|---|---|---|---|---|
| Omicron | 0.1 | 0.058 | 401→379 | [−72,+16] | 0.984→0.985 | ✗ |
| **DMS** | **0.1** | **0.0003** | 4,680→3,250 | **[−2492,−466]** | 0.807→0.866 | **✓** |
| DMS | 1.0 | 0.045 | 4,680→4,280 | [−1078,+294] | 0.807→0.823 | ✗ |

## 인사이트
- **조건부 GO.** CLIB offset은 **단백질 모델이 약한 곳(DMS, AUROC 0.807)에서 정직한 OOS로도 유의하게 도움**이 되고, 이미 near-ceiling인 곳(Omicron, 0.984)에서는 무효.
- **t는 nuisance가 아니라 신호가 사는 자리** — t=0.1에서만 효과, t=1.0에서 사라짐.
- 이 결과는 기존 CAC_RESULTS.md의 "WildType→DMS, t=0.1에서 CAC가 가장 개선" 주장이 **in-sample 인공물이 아니라 실재**했음을 정직한 조건에서 재확인한다. 다만 개선은 보편적이지 않고 특정 상황에 국한.
