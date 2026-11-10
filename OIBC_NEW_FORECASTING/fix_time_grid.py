"""시간대 이름 차이로 발생하는 시간축 비교 오류를 정확한 한 줄 패치로 수정합니다."""
from pathlib import Path


def main():
    path = Path(__file__).resolve().parent / 'src' / 'core.py'
    old = "if not hourly(frame.loc[frame.group.eq(g),'ts']).sort_values().equals(official_grid()):"
    new = "if not hourly(frame.loc[frame.group.eq(g),'ts']).sort_values().tz_convert('UTC').equals(official_grid().tz_convert('UTC')):"
    original = path.read_bytes()
    text = original.decode('utf-8')
    if text.count(new) == 1 and old not in text:
        print('이미 시간축 패치가 적용되어 있습니다.')
        return
    if text.count(old) != 1:
        raise SystemExit('예상 코드와 다릅니다. 파일을 변경하지 않았습니다.')
    backup = path.with_name('core.py.pre_time_grid_fix.bak')
    if backup.exists():
        raise SystemExit('백업 파일이 이미 존재합니다. 덮어쓰지 않고 중단합니다.')
    with backup.open('xb') as handle:
        handle.write(original)
    updated = text.replace(old,new,1).encode('utf-8')
    # 읽은 이후 다른 작업이 파일을 수정했으면 보존하고 중단합니다.
    if path.read_bytes() != original:
        raise SystemExit('동시 변경을 감지했습니다. 원본을 수정하지 않았습니다.')
    path.write_bytes(updated)
    print('UTC 시각 기준 비교 패치 적용 완료. 원본 백업:',backup)
    print('테스트는 아직 실행하지 않았습니다. 합성 테스트를 다시 실행하세요.')


if __name__ == '__main__':
    main()
