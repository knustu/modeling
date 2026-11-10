import argparse
import json
from pathlib import Path
import pandas as pd
from src.submission import validate


def main():
    p=argparse.ArgumentParser(description='Validate synthetic format fixtures only.')
    p.add_argument('--synthetic',action='store_true');p.add_argument('--directory',required=True)
    a=p.parse_args()
    if not a.synthetic:p.error('Real submission validation requires separately reviewed source manifest.')
    root=Path(a.directory)
    if not (root/'NON_SUBMITTABLE_SYNTHETIC.txt').is_file():p.error('Synthetic marker missing.')
    from src.synthetic import fixture
    cfg=json.loads(Path('config/optimization_config.json').read_text(encoding='utf-8'))
    sites,_,_=fixture(cfg)
    result=validate(pd.read_csv(root/'composition.csv',dtype=str),pd.read_csv(root/'forecast.csv',dtype={'ts':str}),sites,cfg)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
