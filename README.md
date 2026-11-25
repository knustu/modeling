# OIBC 2026 태양광 발전량 예측 연구

여러 태양광 발전소를 두 그룹으로 배정하고, 그룹별 발전량 예측과 오차가 정산 결과에 주는 영향을 비교하는 독립 프로토타입입니다.

## 실행 가이드

가장 뒤 단계인 [Phase 4 브랜치](https://github.com/knustu/modeling/tree/phase4-independent-settlement)에 Phase 1–4 코드가 함께 있습니다. 각 단계의 README에서 입력 형식과 예제 실행 방법을 확인할 수 있습니다.

| 단계 | 목표 | 실행 가이드 |
| --- | --- | --- |
| 1 | 발전소를 두 그룹으로 배정 | [OIBC_NEW_GROUPING](https://github.com/knustu/modeling/blob/phase1-independent-grouping/OIBC_NEW_GROUPING/README.md) |
| 2 | 발전소·그룹별 발전량 예측 | [OIBC_NEW_FORECASTING](https://github.com/knustu/modeling/blob/phase2-independent-forecasting/OIBC_NEW_FORECASTING/README.md) |
| 3 | 발전량 예측 오차 분석 | [OIBC_NEW_ERROR_ANALYSIS](https://github.com/knustu/modeling/blob/phase3-independent-error-analysis/OIBC_NEW_ERROR_ANALYSIS/README.md) |
| 4 | 배정·예측 후보의 정산 목적함수 비교 | [OIBC_NEW_SETTLEMENT](https://github.com/knustu/modeling/blob/phase4-independent-settlement/OIBC_NEW_SETTLEMENT/README.md) |

## 평가 범위

Phase 4의 CLI는 합성 입력으로 정산 점수 비교와 배정 탐색을 시연합니다. 실제 Phase 1–3 산출물을 묶는 파일 어댑터와 공식 대회 데이터 검증은 후속 작업입니다. 재현 결과를 정리할 때는 사용한 입력, 평가 구간, 모델, 그룹 배정을 함께 기록합니다.
