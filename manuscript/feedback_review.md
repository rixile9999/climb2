# Feedback Review and Revision Notes

## Overall Judgment

피드백은 대체로 타당합니다. 핵심 문제는 방법 자체보다 논문이 기존 진화생물학의 mutation-selection 관점을 충분히 인정하지 않고, "evolution proposes"처럼 생물학적으로 부정확한 표현을 쓰며, escape prediction을 운영적 예측처럼 읽히게 만든 점입니다. 따라서 본문을 바이러스 중심의 예측 논문이 아니라, 공개 데이터 기반의 retrospective genotype-phenotype evolutionary calibration 논문으로 재구성했습니다.

## 반영한 주요 수정

- 제목과 초록을 "viral escape prediction" 중심에서 genotype-phenotype evolutionary forecasting 중심으로 변경했습니다.
- semantic change와 grammaticality의 의미를 초반에 명시했습니다.
- "evolution proposes"와 "proposal-limited" 계열 표현을 제거하고, mutation arises stochastically and selection filters phenotypes라는 진화론적 표현으로 바꿨습니다.
- CAC/CLIMB를 새 진화 메커니즘처럼 보이지 않게 하고, 기존 codon-substitution 및 mutation-selection 모델의 PLM 보정 계층으로 설명했습니다.
- Markov time T를 실제 달력 시간이나 역학 예측 시간이 아니라 dimensionless Markov horizon/local mutational neighborhood로 정의했습니다.
- CLIMB-only 성능은 단독 예측기 우월성이 아니라 benchmark annotations가 genotype-space accessibility에 의해 구조화되어 있다는 해석으로 낮췄습니다.
- genomic/temporal models를 대체재가 아니라 비교 및 보완 대상이라고 Discussion에 반영했습니다.
- OpenAI 안전정책 관점에서 운영적 변이 예측, 실험 절차, candidate variant construction 권고로 읽힐 수 있는 문장을 피하고, retrospective/public-data/evolutionary interpretation임을 limitations에 명시했습니다.
- Figure 1의 "Constrain Score" 오타를 "Constrained Score"로 수정한 revised 그림을 만들고 DOCX 및 TeX가 revised 그림을 쓰도록 했습니다.

## 새 분석이 필요한 항목

- reviewer가 제안한 alternative rate matrix, 예를 들어 다른 실험적/문헌 기반 Q와의 비교는 현재 원고 수정만으로는 검증할 수 없어 limitation과 future baseline으로 처리했습니다.
- 개별 substitution 단위의 사례 분석은 새 계산이나 표 정리가 필요합니다. 본문에는 특정 mutation이 발생/확산될 것처럼 읽히지 않도록 domain-level/benchmark-level biological interpretation만 넣었습니다.
- Figure 내부 축 라벨과 heatmap legend의 세부 수정은 editable figure source가 없어 DOCX 내 embedded PNG의 명백한 오타만 고쳤고, 나머지는 caption에서 보강했습니다.

## 산출물

- Revised Word manuscript: `main_prl_revised.docx`
- Updated TeX source: `_Bio__Viral_Escape_1/main_prl.tex`
- Revised Figure 1 assets: `_Bio__Viral_Escape_1/Figure_Nature_1_revised.png`, `_Bio__Viral_Escape_1/Figure_Nature_1_revised.pdf`
