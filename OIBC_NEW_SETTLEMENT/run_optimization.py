import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description='Phase 4 synthetic-only experiment; no real inputs accepted.')
    p.add_argument('--synthetic',action='store_true');p.add_argument('--config',default='config/optimization_config.json');p.add_argument('--output',required=True)
    args=p.parse_args()
    if not args.synthetic:p.error('Real-data optimization/inference is locked pending separate authorization and official rules.')
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    for k in ['iterations','swap_trials','starts','bootstrap_replicates']:
        if not isinstance(cfg[k],int) or cfg[k]<1:p.error(k+' must be positive integer.')
    from src.synthetic import fixture
    from src.experiment import run
    from src.contracts import GateError
    sites,compositions,parts=fixture(cfg)
    try:run(sites,compositions,parts,cfg,args.output,synthetic=True)
    except (GateError,FileExistsError) as exc:raise SystemExit('Stopped safely: '+str(exc))
    print('SYNTHETIC Phase 4 artifacts saved locally. No real data read, upstream modified, or competition submission performed.')
    print('Tests/browser verification remain separate. Candidates remain unverified.')


if __name__=='__main__':main()
