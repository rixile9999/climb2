# 연구노트 01 — Phase 0: 데이터 위생과 좌표계 버그

**연구 단위:** 오프셋 실험을 시작하기 전, 특징(feature)과 정답 라벨이 서로 아귀가 맞는지 검증한다.
**스크립트:** `scripts/phase0_dump.py`, `scripts/fix_escape_coords.py`, `scripts/_diag_coords.py`
**산출물:** `outputs/phase0/tidy_esm2cov_l1.parquet`, `metadata/escape_mutants/cov-wt.json`(수정), `.bak`(백업)

## 질문
모델을 만들기 전에, 각 면역회피(escape) 정답 리스트의 돌연변이가 실제로 후보 24,187개 안에 존재하는가? (count assertion)

## 방법
`analyzer` 파이프라인 그대로 ESM2cov-L1 backbone의 tidy 데이터를 만들고, 각 WildType escape task에 대해 `mutation_code = wt+pos+mut` 매칭 수를 metadata 개수와 비교했다. off-by-one을 자동 진단하려고 pos, pos−1, pos+1 세 프레이밍을 함께 셌다.

## 발견
- **대부분(DMS·Alpha·Beta·Gamma·Omicron)은 0-based로 정확히 매칭** — 이 repo의 관례는 0-based(TSV pos 0 = 생물학적 1번 잔기).
- **`Actual` 리스트만 1-based로 작성된 좌표계 버그** — 32개 중 4개만 매칭, 위치를 −1하면 31개가 맞음. 즉 이 리스트만 생물학 교과서식 번호를 썼다.
- **`Delta`·`Combined`의 미매칭 4개는 좌표 문제가 아니라** `E155X` 같은 결실/와일드카드("X")로, 단일치환 후보에 존재 불가.
- 기존 파이프라인은 개수를 검증하지 않아 이 오염된 task들에서 **조용히 틀린 돌연변이를 채점**하고 있었다.

## 조치
- `Actual`: 전 위치 −1로 0-based 정규화(잔기 자체가 불일치하는 `D769H` 제외).
- `Delta`·`Combined`: 비표준 AA("X") 항목 제외.
- 수정 후 **모든 task에서 count assertion 통과.** 원본은 `.bak`로 백업.

## 인사이트
화려한 모델보다 먼저 "라벨이 후보와 맞는가"를 단순히 세어보는 점검이, 오래 숨어 있던 데이터 버그를 잡는다. 이 count assertion은 파이프라인에 상시 게이트로 넣어야 한다. 또한 repo가 비표준 0-based 인덱싱을 쓰므로, 향후 문헌에서 변이를 추가하는 사람은 같은 버그를 반복하기 쉽다 → 장기적으로 1-based로의 통일을 권고.
