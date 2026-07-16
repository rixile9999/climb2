# 연구노트 04 — Phase 3b: t 하이퍼파라미터를 버릴 수 있는가?

**연구 단위:** Phase 3에서 t\*가 항상 격자 최솟값으로 내려가고 이득이 포화된다는 관찰 → 진화시간 t를 아예 없앨 수 있는지 시험한다.
**스크립트:** `scripts/phase3b_simplify.py`
**산출물:** `outputs/phase0/phase3b_simplify.csv`

## 방법
동일한 5 seeds × 5-fold nested CV에서 세 오프셋을 비교:
- **B_full**: CLIB S3(t\*), t는 inner-CV 선택 (현 제안).
- **B_fix**: CLIB S3 at **t=0.01 고정** (선택 없음).
- **B_min**: **z(−최소 염기변화 수)** — t가 전혀 없는 이산 코돈-거리 feature.
"유지(retains)" = 단순화 모델의 mean-rank가 B_full 대비 유의하게 나쁘지 않음(paired bootstrap ΔCI가 0 포함).

## 발견
- **B_fix(t=0.01 고정): 6칸 전부 성능 유지** — mean-rank·AUROC가 B_full과 사실상 동일, ΔCI 모두 0 포함.
- **B_min(이산 코돈거리): 강한/중간 backbone(ESM2cov·Hie)에선 유지되나, 약한 base ESM2에선 유의하게 나빠짐**(DMS Δ+844 [137,1466], AUROC 0.847→0.813). CLIB이 무거운 짐을 질 때 이산 근사는 너무 거칠다.
- 코돈 거리 분포: 1-base 7,568 · 2-base 12,388 · 3-base 4,231.

## 인사이트
- **t는 0.01로 고정해도 무손실** → 하이퍼파라미터 하나 제거. 채택.
- **완전 이산화(최소 염기변화 수)는 backbone이 충분히 강할 때만 안전.** 약한 backbone에서는 within-k 세부 rate 구조(연속 CLIB이 담는)가 실제로 기여하므로 연속 CLIB을 유지해야 한다.
- 생물학적 함의: t→0 극한에서 CLIB의 지배적 성분은 "단일 뉴클레오타이드로 도달 가능한가"라는 이산 접근성이지만, 그 위의 미세 구조도 약한 신호원에서는 값을 한다.

**결정:** 기본 스코어러를 `score = b0·S1 + b1·S2 + κ·CLIB(t=0.01)`, κ=1로 확정(하이퍼파라미터 없음).
