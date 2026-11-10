# 시간 간격 검증 수정

사용자가 보고한 실행은 setUpClass 오류로 본 테스트 0개 실행입니다. 통과한 테스트는 없습니다.

기존 코드가 DatetimeIndex.asi8을 항상 나노초로 가정하여 3600*10**9와 비교했습니다. pandas 시간 저장 해상도가 마이크로초/초 등인 경우 정상 1시간 간격도 거절할 수 있습니다. 실제 로컬 해상도는 별도 조회하지 않았습니다.

수정: 인접 시각의 차를 pd.Timedelta(hours=1)와 직접 비교하고, 전체 시간축도 UTC로 통일한 Timestamp 값끼리 비교합니다. asi8 정수 단위 가정을 제거했습니다. 누락, 중복, 분 단위 시각, 1시간 이동 오류는 계속 거절합니다.

GitHub contracts.py를 직접 수정했으므로 로컬 패치 스크립트는 필요 없습니다. Phase 1/2는 수정하지 않습니다. 회귀 테스트 6개는 아직 실행되지 않았습니다.

CMD:
```bat
cd /d C:\Users\yhy42\modeling_phase3\OIBC_NEW_ERROR_ANALYSIS
git pull --ff-only origin phase3-independent-error-analysis
py -m unittest discover -s tests -p "test_*.py" -v
```

Phase 3 로컬 브랜치 이름은 phase3-local이어도 위 pull은 지정된 원격 브랜치를 가져옵니다. 충돌 또는 로컬 변경 경고 시 강제 덮어쓰기하지 마세요. 통과 여부는 재실행 결과로만 판단합니다.
