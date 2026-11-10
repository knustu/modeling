import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description='Independent diagnostics: synthetic-only execution.')
    p.add_argument('--synthetic',action='store_true');p.add_argument('--config',default='config/analysis_config.json')
    p.add_argument('--output',required=True);args=p.parse_args()
    if not args.synthetic:p.error('Real analysis disabled until separate authorization and manifest/rules verification.')
    cfg=json.loads(Path(args.config).read_text())
    from src.synthetic import fixture
    from src.pipeline import run
    from src.contracts import DiagnosticError
    sites,plant,compositions,grid,weather=fixture(cfg)
    try:run(sites,plant,compositions,grid,weather,cfg,args.output,synthetic=True)
    except (DiagnosticError,FileExistsError) as exc:raise SystemExit('Stopped safely: '+str(exc))
    print('SYNTHETIC diagnostic artifacts saved locally. No models trained, forecasts changed, or Phase 4 optimization performed.')
    print('Unit tests and offline browser verification require separate execution.')


if __name__=='__main__':main()
