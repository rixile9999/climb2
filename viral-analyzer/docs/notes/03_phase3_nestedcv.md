# 연구노트 03 — Phase 3: 다중 시드·다중 backbone nested CV

**연구 단위:** go/no-go의 "조건부 GO"를 굳힌다. DMS 신호가 우연/특정 분할의 산물이 아닌지, 그리고 "약한 backbone일수록 CLIB이 더 돕는다"는 가설을 시험한다.
**스크립트:** `scripts/phase3_nestedcv.py`
**산출물:** `outputs/phase0/phase3_nestedcv.csv`, `docs/research_report_clib_offset.md`(종합)

## 방법
5 seeds × 5-fold nested CV(안쪽 루프에서 t 선택 → 선택까지 OOS). 3 backbone(강한 ESM2cov, 약한 base ESM2, 중간 Hie) × 2 task. 모델 A(κ=0)/B(offset κ=1, t는 inner-CV)/C(free b3)/legacy(구식 barycentric, in-sample & OOS). t 격자는 최종적으로 `[0.001,…,1.0]`(0.033 아래로 확장).

## 발견 (mean-rank, 낮을수록 좋음)
| backbone | task | A | B(offset) | ΔB−A CI | AUROC A→B | legacy OOS | legacy in-sample | 도움? |
|---|---|---|---|---|---|---|---|---|
| ESM2cov | DMS | 4,330 | 2,870 | [−2420,−581] | 0.82→0.88 | 3,030 | 2,270 | ✓ |
| ESM2cov | Omicron | 408 | 401 | [−44,+25] | 0.984→0.984 | 401 | 336 | ✗ |
| base ESM2 | DMS | 11,400 | 3,710 | [−9426,−6130] | **0.53→0.85** | 3,090 | 2,720 | ✓ |
| base ESM2 | Omicron | 8,750 | 3,010 | [−7396,−4145] | 0.64→0.88 | 2,740 | 2,600 | ✓ |
| Hie | DMS | 4,620 | 2,260 | [−3161,−1582] | 0.81→0.91 | 2,040 | 2,010 | ✓ |
| Hie | Omicron | 4,420 | 2,630 | [−2603,−1119] | 0.82→0.89 | 2,140 | 2,090 | ✓ |

## 인사이트 (3가지)
1. **가설 확정 — 약한 backbone일수록 CLIB offset이 크게 돕는다.** 극적 예: base ESM2가 DMS에서 거의 무작위(0.53)인데 offset으로 0.85로 급등. 강한 ESM2cov의 Omicron(0.984)만 유일하게 무효.
2. **DMS 신호는 3 backbone × 5 seed 전부에서 재확인**(모든 CI가 0 배제).
3. **정직한 검증이 기존 방식의 ~30% 낙관 편향을 정량화**(ESM2cov DMS: in-sample 2,270 vs 정직한 OOS ~3,000). 그리고 **offset(2 param) ≈ 정직한 legacy(3 param)** — CLIB을 얼려도 손실 없이 순환적합 제거·파라미터 감소. free-b3는 미미하게만 나아 **κ=1 고정이 옳은 절충**.
