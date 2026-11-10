# OIBC_NEW_GROUPING — 독립 Phase 1

## 상태
구현 초안입니다. 이 커밋의 합성 테스트는 아직 실행되지 않았습니다. 실제 대회 결과 및 테스트 통과 수치를 주장하지 않습니다. 루트 README와 M3는 수정하지 않습니다.

## 보안과 실행 원칙
- 소스 코드, 설정 예시, 합성 데이터 생성기, 테스트만 GitHub에 보관합니다.
- 실제 대회 데이터, 실제 ID, 특징, 편성 CSV, 보고서, HTML은 로컬에만 둡니다.
- `.gitignore`는 이미 추적 중인 파일을 보호하지 않습니다. 커밋 전 `git status`와 `git diff --cached`를 확인하세요.
- 외부 데이터/API, 익명 좌표, 기존 그룹, M3 코드/가중치/예측을 사용하지 않습니다.
- GitHub Actions, 패키지 설치, 실제 데이터 실행, 대회 제출은 자동으로 수행하지 않습니다.
- 실제 데이터는 가능하면 이 저장소 바깥에 보관하세요.

## 구성
- `src/grouping.py`: 입력 검증, 시간 규칙, 특징, 용량 초기화, 로컬 탐색, 평가, 출력 검증.
- `src/visualizer.py`: 네트워크 없는 후보 선택, ID/용량 조회, SVG 저장, 나란히 비교.
- `tests/`: 합성 자료와 unittest.
- `run_grouping.py`: 명시적인 로컬 실행 진입점.

## 방법
1. 용량 기준: 설비용량 절반 근처의 부분집합 빔 탐색 후 이동/교환 개선. 전역 최적 보장은 없습니다.
2. 프로파일 기준: kWh/kW 월별-시간대 평균을 발전소 간 표준화하여 두 그룹의 동일 가중 평균 차를 최소화합니다. 서로 다른 발전 프로파일을 양 그룹에 분산시키는 대리 목적입니다.
3. 최적화: 위 초기화와 여러 시드의 실현 가능 초기화에서 용량/프로파일 가중합을 개선합니다. 승인된 모든 이동은 용량 제약을 통과합니다.

목적함수: `wC * ((C1-C2)/Ctotal)^2 + wP * mean((mean(z_G1)-mean(z_G2))^2)`.
두 항은 무차원입니다. 가중치는 가설이며 실제 예측오차나 수익 최적성을 뜻하지 않습니다.
특징이 모두 일정하면 프로파일 항은 0이 됩니다. 결측은 0으로 바꾸지 않고 관측 임계값 및 발전소 간 중앙값을 사용합니다.
상관계수는 최소 중첩 관측을 사용하고 상수 발전소의 정의되지 않은 쌍은 제외합니다. 집계 변동성과 계절 비교는 전 발전소가 관측된 시간만 사용합니다.
앞/뒤 기간 진단은 전체 과거 구간에 맞춘 배정의 기술통계이며 홀드아웃 성능이 아닙니다.

## 시간과 단위
공식 규칙을 확인하여 `timestamp_convention`을 `interval_start` 또는 `interval_end`로 지정해야 실제 실행이 가능합니다. 구간 종료 표기는 1시간을 빼 구간 시작으로 통일합니다. 종료가 cutoff를 넘는 구간은 제외합니다. timezone 없는 시각은 설정한 KST로 해석합니다. 숫자형 epoch 시각은 공식 단위를 확인하여 사전에 명시적으로 변환하세요. `energy_to_kwh`도 공식 단위에 맞춰야 합니다. 열 이름은 config에서 지정하고 실제 입력 열을 검증합니다.

## Windows PowerShell: 합성 검증
저장소 복제 후 해당 브랜치에서 실행하세요. 설치 명령은 포함하지 않습니다.

```powershell
git clone --branch phase1-independent-grouping https://github.com/knustu/modeling.git
cd modeling\OIBC_NEW_GROUPING
py -c "import numpy,pandas; from zoneinfo import ZoneInfo; print(numpy.__version__,pandas.__version__,ZoneInfo('Asia/Seoul'))"
py -m unittest discover -s tests -p "test_*.py" -v
py run_grouping.py --synthetic --config .\config\grouping_config.json --output .\outputs\synthetic_run_001
Start-Process .\outputs\synthetic_run_001\grouping_visualization.html
```

기존 출력 경로는 거절합니다. 재실행에는 새 이름을 사용하세요. 브라우저에서 인터넷 연결을 끊고 후보 전환, 점 검사, SVG 저장, 나란히 표시를 확인하세요. 자원 검사 unittest는 실제 브라우저 검증을 대체하지 않습니다.

## 실제 실행: 개발 단계에는 실행하지 마세요
공식 데이터/특징 허용 정책, 시간대, 단위, 메타데이터 순서를 사용자가 확인한 뒤 로컬에서만 실행합니다. 공식 규칙 문서가 제공되지 않아 본 코드만으로 규정 준수를 인증할 수 없습니다.

```powershell
Copy-Item .\config\grouping_config.json .\local_config.json
# local_config.json의 시각 표기, 열 이름, 단위를 공식 규칙에 맞게 수정
py run_grouping.py --allow-real-data --config .\local_config.json --sites D:\LOCAL_DATA\sites.csv --generation D:\LOCAL_DATA\train\generation.parquet --output .\outputs\local_run_001
```

## 출력
composition_balanced.csv, composition_profile.csv, composition_optimized.csv,
grouping_comparison.csv, grouping_report.md, grouping_visualization.html,
validation_report.json, plant_features.csv, feature_manifest.json,
resolved_config.json, multistart_diagnostics.json, convergence_*.json, environment.json.
실제 실행 출력은 GitHub에 커밋하지 않습니다.

## 계산량과 수렴
N 발전소, D 특징, B 빔 폭, I 반복, S 표본 교환일 때 빔 탐색은 대략 O(N B log B), 로컬 탐색은 O(I (N+S)(D+N))입니다. 후보별 특징 합을 갱신하고 승인 전 전체 용량을 다시 검증합니다. 상관 진단은 대략 O(T N²)입니다. 실현 가능 초기화 탐색 실패는 불가능성 증명이 아니므로 빔 폭을 늘릴 수 있습니다. 표본 교환 종료는 전역/전체 이웃 최적성 보장이 아닙니다. 실제 벽시계 시간과 SVD 축 부호는 환경에 따라 달라질 수 있습니다.

## Phase 2 인계
세 후보를 모두 유지합니다. `site_id,group` CSV와 해시, 원본 메타데이터 순서, 용량 검증, 특징/특징 명세, cutoff 및 설정, 환경 버전, 이동 로그를 로컬에 보관합니다. 미래 예측기는 정확한 ID로 매핑하고 그룹별 예측/실측을 합산합니다. 후보 선택과 예측기 학습 기간을 분리한 합법적 시계열 외표본 평가 전에는 우승 후보를 확정하지 않습니다. Phase 2 구현은 포함하지 않습니다.
