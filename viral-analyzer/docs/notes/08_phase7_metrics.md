# 연구노트 08 — Phase 7: 강한 backbone 내 semantic 강화 (최종 확인)

**연구 단위:** 남은 가장 그럴듯한 레버 — 신호가 사는 강한 ESM2cov backbone의 표현을 L1+L2 거리로 강화하면 나아지는가? (Phase 4는 약한 backbone 추가였음.)
**스크립트:** `scripts/phase7_metrics.py` · **산출물:** `outputs/phase0/phase7_metrics.csv`

## 발견
| task | featset | mean-rank | AUROC | Δ vs base | CI | 이김? |
|---|---|---|---|---|---|---|
| DMS | base(L1+gram) | 2,868 | 0.882 | 0 | — | — |
| DMS | +L2 | 3,172 | 0.869 | +301 | [−179,772] | ✗ |
| DMS | L2only | 2,931 | 0.879 | +62 | [−4,141] | ✗ |
| Omicron | +L2 | 550 | 0.978 | +161 | [76,252] | ✗ 악화 |
| Omicron | L2only | 392 | 0.984 | +0.8 | [−4,6] | ✗ 동일 |

## 인사이트
- **더 풍부한 semantic도 도움 안 됨.** L2를 더하면 오히려 악화(상관 높은 feature 추가 → 과적합/희석), L1↔L2 교체는 동일.
- **4연속 무개선**(Phase 3b 이산화 · 4 스태킹 · 5 손실 · 7 metric강화). 엔드포인트 확정.

## 결론
현재 라벨·신호 예산에서 **어떤 모델 복잡화도 정직한 OOS로 단순 2-feature offset을 못 이긴다.** 최적화 종료.
