# 시간축 검증 수정 안내

사용자가 제공한 실행 결과: 기존 합성 테스트 14개 중 오류 1개, 실패 0개. test_official_format에서 Wrong official target grid 발생. 나머지 13개는 보고된 실행에서 오류/실패가 없었습니다.

원인: times()는 Asia/Seoul 시간대로 정규화하지만 official_grid()는 +09:00 고정 오프셋 시간대를 만듭니다. pandas DatetimeIndex.equals는 같은 순간을 나타내더라도 서로 다른 시간대 dtype을 동일하게 처리하지 않을 수 있습니다. 비교 양쪽을 UTC로 통일하되 외부 CSV의 +09:00 표기, 시각, 시간 수, 중복 검사 규칙은 유지합니다.

안전한 로컬 패치 스크립트는 예상 한 줄만 수정하며 원본을 .bak 파일로 보존합니다. 수정된 core.py를 강제 덮어쓰거나 실제 데이터를 읽지 않습니다. 재실행 시 이미 적용된 패치는 건너뜁니다.

CMD에서 프로젝트 폴더 기준:

```bat
git pull --ff-only origin phase2-independent-forecasting
py fix_time_grid.py
py -m unittest discover -s tests -p "test_*.py" -v
```

GitHub에는 패치 스크립트와 회귀 테스트를 추가했습니다. core.py 자체는 이 커밋에서 변경하지 않았으므로 로컬 패치 명령이 필요합니다. 새 회귀 테스트 5개는 아직 실행되지 않았습니다. 수정 후 전체 테스트 통과를 주장하지 않습니다. 로컬 core.py 변경은 검토 후 필요시 사용자가 커밋할 수 있습니다. .bak 파일은 업로드하지 마세요.
