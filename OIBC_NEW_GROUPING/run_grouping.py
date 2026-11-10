import argparse
import json
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from src.grouping import InputError,run,dump
from tests.synthetic_data import make_synthetic


def main():
    parser=argparse.ArgumentParser(description='독립 태양광 그룹 편성: 기본적으로 합성 자료만 개발합니다.')
    parser.add_argument('--config',default='config/grouping_config.json')
    parser.add_argument('--output',required=True)
    parser.add_argument('--synthetic',action='store_true')
    parser.add_argument('--allow-real-data',action='store_true')
    parser.add_argument('--sites')
    parser.add_argument('--generation')
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    if Path(args.output).exists():
        parser.error('기존 출력 경로는 덮어쓰지 않습니다. 새 경로를 사용하세요.')
    if args.synthetic:
        if args.sites or args.generation or args.allow_real_data:
            parser.error('합성 실행에는 실제 데이터 인수를 사용할 수 없습니다.')
        cfg['timestamp_convention']='interval_start'
        sites,history=make_synthetic(cfg['seed'])
    else:
        if not args.allow_real_data or not args.sites or not args.generation:
            parser.error('실제 실행에는 --allow-real-data, --sites, --generation이 모두 필요합니다.')
        if cfg['timestamp_convention'] not in ('interval_start','interval_end'):
            parser.error('공식 시간 간격 규칙과 단위를 설정 파일에서 먼저 확인하세요.')
        # ID 문자열과 메타데이터 행 순서를 보존합니다.
        sites=pd.read_csv(args.sites,dtype={cfg['columns']['site_id']:str},keep_default_na=False)
        history=pd.read_parquet(args.generation)
        print('확인된 발전량 열:',list(history.columns))
    run(sites,history,cfg,args.output,synthetic=args.synthetic)
    dump(Path(args.output)/'environment.json',{'python':sys.version,'platform':platform.platform(),
         'numpy':np.__version__,'pandas':pd.__version__})
    print('로컬 결과 저장 완료:',Path(args.output).resolve())
    print('공식 점수 계산, 예측모델 학습, 외부 전송은 수행하지 않았습니다.')


if __name__=='__main__':
    try:
        main()
    except (InputError,FileExistsError) as exc:
        raise SystemExit('안전 중단: '+str(exc))
