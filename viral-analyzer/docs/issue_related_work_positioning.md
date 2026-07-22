## 논문의 대화상대 (Related-work positioning) — deep-research 결과

PRL/PRX 원고("Mutation-process priors calibrate PLMs …")를 **라벨-프리 self-calibration** 결과로 강화할 때, 반드시 인용·차별화해야 하는 선행연구 지도. 딥리서치(102 agents, 8 verified findings)로 정리.

### 핵심 포지셔닝 (한 줄)
현재 원고는 가중치를 벤치마크에 최적화(순환적합). 새 결과는 가중치를 **독립적 계통수-fitness(Bloom-Neher)로 라벨 없이 고정** → "mutation as stochastic supply × selection as filter"의 원리적 분해. 이 novelty는 아래 세 흐름과의 대비에서 가장 날카롭다.

### 🎯 킬러 훅 — 우리가 답하는 논쟁
**Allman/Johnson et al., *J. R. Soc. Interface* 22(225):20240598 (2025)** — zero-shot CSCS가 SARS-CoV-2 escape에서 **작동하지 않는다**고 체계적 반박: "semantic change가 escape/non-escape를 구분 못함", 개념적 결함(무방향 임베딩 이동은 ACE2 결합도 똑같이 파괴해야), **지도학습 fine-tuning으로 대체 권고**.
→ **우리 논문의 응답: 라벨 없이 codon-accessibility 보정으로 PLM escape 신호를 되살린다.** 이게 본 원고의 가장 시의적·강력한 프레이밍.

### 관련연구 지도 (인용·차별점)

| 연구 | 무엇 | 우리의 관계·차별점 |
|---|---|---|
| **Hie et al., Science 2021** (abd7331) | CSCS 원조 (zero-shot semantic+grammaticality, Spike AUC 0.85) | **우리의 베이스라인** — CAC가 보정·개선 |
| **Allman/Johnson, JRSI 2025** (20240598) | zero-shot CSCS 실패 반박, 지도학습 권고 | **우리가 답하는 위협** — 라벨 없이 신호 복원 |
| **Lamb et al., Nat Commun 2026** (s41467-026-69569-9) | 무수정 ESM-2가 이미 진화궤적·VOC 포착 | **강한 PLM 베이스라인** — 인용·비교 |
| **EVEscape, Nature 2023** (s41586-023-06617-0) | 학습 constraint(EVE) × **독립 물리 접근성**(WCN 구조) × 화학 비유사도, 라벨 없이 DMS 예측 | **최근접 선행** — 단 그들 prior는 **구조/생물물리**, 우리는 **codon-transition Markov(유전형)** |
| **CoVFit, Nat Commun 2025** (s41467-025-59422-w) | PLM+진화fitness 결합, 미래변이 예측(Spearman 0.862) | **최직접 경쟁자** — 단 **강한 지도학습**(Re+DMS fine-tune). 우리는 **라벨-프리·post-hoc·codon prior 동결** |
| **Bloom & Neher, Virus Evolution 2023** (vead055) | 계통수 fitness Δf=ln((n_act+P)/(n_exp+P)), spike DMS와 r=0.66 | **우리 라벨-프리 타깃의 원천** (DMS와 독립) |
| **PyR0, Science 2022** (abm1208) | 빈도 기반 계층 베이지안 per-mutation fitness | **차별 대상** — 우리는 fitness 추론이 아니라 그 신호로 **PLM을 보정** |
| **Rodrigue et al., PNAS 2010** (0910915107) | codon mutation-selection: 치환율 = 돌연변이 공급 × 선택 고정확률 | **물리 프레이밍의 population-genetic 전례** |

### 저널 함의 (중요)
논쟁(Allman)·최근접 경쟁자(EVEscape·CoVFit)·타깃 원천(Bloom-Neher)·차별대상(PyR0)이 **전부 Nature/Science/생물-진화 저널**에 있음. 물리저널(PRL/PRX)의 PLM/viral-evolution 전례는 못 찾음.
→ **근본적 fit: eLife / Nature Communications**(전문 리뷰어·경쟁자·데이터 출처가 그곳; "2025년 실패 판정 난 label-free PLM escape를 되살린다"가 가장 세게 먹힘). **PRX Life**는 물리 정체성 유지 옵션, **PRL**은 고임팩트 도박.

### 재작성 반영 항목 (checklist)
- [ ] Intro에 Allman 2025 위협 명시 + 라벨-프리 응답 프레이밍
- [ ] EVEscape(구조 prior) vs CLIMB(codon Markov prior) 차별 단락
- [ ] CoVFit(지도) vs 우리(라벨-프리) 차별 단락
- [ ] Bloom-Neher를 라벨-프리 타깃 근거로 인용 (DMS 독립성 명시)
- [ ] PyR0 차별(fitness 추론 vs PLM 보정)
- [ ] Rodrigue codon mutation-selection로 물리 프레이밍 근거
- [ ] 저널 방향 확정 (eLife/NatComm vs PRX Life vs PRL)

*출처: deep-research 리포트, 8 findings all high-confidence, 3-vote adversarial verified.*
