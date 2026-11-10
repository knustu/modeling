import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description='Independent Phase 2; synthetic execution only until official rule verification.')
    parser.add_argument('--synthetic',action='store_true')
    parser.add_argument('--config',default='config/forecasting_config.json')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if not args.synthetic: parser.error('Real-data execution is locked; no competition files will be read.')
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    for key in ['batch_size','threads','candidate_iterations','expected_sites']:
        if not isinstance(cfg[key],int) or cfg[key]<1: parser.error(key+' must be a positive integer.')
    # Set before importing numerical libraries; no package installation or CUDA assumptions.
    for var in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        os.environ[var]=str(cfg['threads'])
    from src.synthetic import fixture
    from src.backtest import execute
    from src.core import ContractError,write_json
    import numpy as np
    import pandas as pd
    sites,targets,labels,weather,compositions=fixture(cfg)
    bounds=[('2025-01-06T00:00:00+09:00','2025-01-09T00:00:00+09:00')]
    holdout=('2025-01-11T00:00:00+09:00','2025-01-15T00:00:00+09:00')
    try:
        execute(sites,targets,labels,weather,compositions,cfg,args.output,bounds,holdout)
        write_json(Path(args.output)/'environment.json',{'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__})
    except (ContractError,FileExistsError) as exc:
        raise SystemExit(str(exc))
    print('SYNTHETIC artifacts saved locally. No real data read; tests/browser checks require separate execution.')


if __name__=='__main__': main()
