"""독립 태양광 그룹 편성. 실제 데이터 실행은 CLI의 명시적 허가가 필요합니다."""
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd


class InputError(ValueError):
    pass


class SearchExhausted(InputError):
    pass


def dump(path, value):
    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float) and not math.isfinite(v):
            return None
        return v
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False,
                                    indent=2, allow_nan=False), encoding='utf-8')


def validate_config(cfg):
    for key in ('beam_width', 'max_iterations', 'swap_trials', 'expected_sites',
                'minimum_feature_observations', 'minimum_pair_overlap'):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise InputError(f'{key}: 양의 정수가 필요합니다.')
    if not isinstance(cfg['random_starts'], int) or cfg['random_starts'] < 0:
        raise InputError('random_starts: 음수가 아닌 정수가 필요합니다.')
    for key in ('minimum_capacity_kw', 'energy_to_kwh'):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise InputError(f'{key}: 유한한 양수가 필요합니다.')
    for key in ('capacity_weight', 'profile_weight', 'tolerance'):
        if not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise InputError(f'{key}: 유한한 음수 아닌 수가 필요합니다.')
    if cfg['capacity_weight'] + cfg['profile_weight'] <= 0:
        raise InputError('목적함수 가중치가 모두 0입니다.')
    if not 0 < cfg['minimum_feature_site_fraction'] <= 1:
        raise InputError('특징 관측 비율은 0 초과 1 이하입니다.')
    if not cfg['daytime_hours'] or any(h not in range(24) for h in cfg['daytime_hours']):
        raise InputError('주간 시간은 0~23 범위여야 합니다.')
    if cfg['duplicate_policy'] not in ('reject', 'mean'):
        raise InputError('중복 정책은 reject 또는 mean입니다.')


def sites_frame(raw, cfg):
    cols = cfg['columns']
    required = [cols['site_id'], cols['capacity']]
    if not set(required) <= set(raw.columns):
        raise InputError(f'메타데이터 열을 확인하세요: {list(raw.columns)}')
    s = raw[required].copy()
    s.columns = ['site_id', 'capacity_kw']
    if s.site_id.isna().any():
        raise InputError('발전소 ID 누락.')
    s['site_id'] = s.site_id.astype(str)
    if s.site_id.str.strip().eq('').any() or s.site_id.duplicated().any():
        raise InputError('발전소 ID 공백 또는 중복.')
    s['capacity_kw'] = pd.to_numeric(s.capacity_kw, errors='coerce')
    c = s.capacity_kw.to_numpy(float)
    if not np.isfinite(c).all() or (c <= 0).any():
        raise InputError('설비용량은 유한한 양수여야 합니다.')
    if len(s) != cfg['expected_sites']:
        raise InputError(f"발전소 수 불일치: {len(s)} / {cfg['expected_sites']}")
    total, m = math.fsum(c), cfg['minimum_capacity_kw']
    if total < 2*m or c.max() > total-m:
        raise InputError('용량 제약상 불가능한 입력입니다.')
    return s.reset_index(drop=True)


def stamp(value, cfg):
    t = pd.Timestamp(value)
    if pd.isna(t):
        raise InputError('시각 누락.')
    if t.tzinfo is None:
        t = t.tz_localize(cfg['naive_timezone'])
    return t.tz_convert('Asia/Seoul')


def history_frame(raw, sites, cfg):
    cols = cfg['columns']
    required = [cols['generation_site'], cols['timestamp'], cols['energy']]
    if not set(required) <= set(raw.columns):
        raise InputError(f'발전량 열을 확인하세요: {list(raw.columns)}')
    convention = cfg['timestamp_convention']
    if convention not in ('interval_start', 'interval_end'):
        raise InputError('공식 시각 규칙을 확인하여 timestamp_convention을 설정하세요.')
    h = raw[required].copy()
    h.columns = ['site_id', 'ts', 'energy']
    if h.site_id.isna().any():
        raise InputError('발전량 ID 누락.')
    h['site_id'] = h.site_id.astype(str)
    if not set(h.site_id) <= set(sites.site_id):
        raise InputError('메타데이터에 없는 발전소 ID.')
    h['ts'] = pd.DatetimeIndex([stamp(v, cfg) for v in h.ts])
    if any(t.minute or t.second or t.microsecond or t.nanosecond for t in h.ts):
        raise InputError('정시 1시간 간격 시각만 지원합니다.')
    if convention == 'interval_end':
        h['ts'] -= pd.Timedelta(hours=1)
    cutoff = stamp(cfg['cutoff_exclusive'], cfg)
    allowed = h.ts + pd.Timedelta(hours=1) <= cutoff
    excluded = int((~allowed).sum())
    h = h.loc[allowed].copy()
    if h.empty:
        raise InputError('허용된 과거 구간에 관측이 없습니다.')
    original = h.energy.copy()
    h['energy'] = pd.to_numeric(original, errors='coerce')
    if (original.notna() & h.energy.isna()).any():
        raise InputError('숫자가 아닌 발전량.')
    observed = h.energy.dropna().to_numpy(float)
    if not np.isfinite(observed).all() or (observed < 0).any():
        raise InputError('발전량은 유한한 음수 아닌 수 또는 결측이어야 합니다.')
    duplicates = int(h.duplicated(['site_id', 'ts'], keep=False).sum())
    if duplicates:
        if cfg['duplicate_policy'] == 'reject':
            raise InputError('발전소-시간 중복 관측.')
        h = h.groupby(['site_id', 'ts'], as_index=False).energy.mean()
    h['energy'] *= cfg['energy_to_kwh']
    if not np.isfinite(h.energy.dropna().to_numpy(float)).all():
        raise InputError('단위 변환 후 발전량 범위 초과.')
    p = h.pivot(index='ts', columns='site_id', values='energy')
    grid = pd.date_range(h.ts.min(), h.ts.max(), freq='h')
    p = p.reindex(index=grid, columns=sites.site_id)
    return p, {'excluded_records': excluded, 'duplicate_records': duplicates,
               'missing_cells': int(p.isna().sum().sum()),
               'start': str(p.index.min()), 'end': str(p.index.max())}


def features(p, sites, cfg):
    q = p.divide(sites.set_index('site_id').capacity_kw, axis=1)
    q = q.loc[q.index.hour.isin(cfg['daytime_hours'])]
    if q.empty:
        raise InputError('설정한 주간 시간에 관측이 없습니다.')
    keys = [q.index.month, q.index.hour]
    means, counts = q.groupby(keys).mean(), q.groupby(keys).count()
    means = means.where(counts >= cfg['minimum_feature_observations'])
    means = means.loc[means.notna().mean(axis=1) >= cfg['minimum_feature_site_fraction']]
    if means.empty:
        raise InputError('특징을 계산할 과거 관측이 부족합니다.')
    frame = means.T.reindex(sites.site_id)
    missing = frame.isna()
    filled = frame.fillna(frame.median())
    sd = filled.std(ddof=0)
    keep = sd > 1e-12
    if keep.any():
        f = filled.loc[:, keep]
        z = ((f-f.mean()) / sd[keep]).to_numpy(float)
        names = [f'month_{m:02d}_hour_{h:02d}' for m, h in f.columns]
    else:
        z, names = np.zeros((len(sites), 1)), ['uninformative_profile']
    return z, {'feature_names': names, 'imputed_cells': int(missing.sum().sum()),
               'zero_observation_sites': q.columns[q.count().eq(0)].tolist(),
               'observations_by_site': q.count().to_dict(),
               'scaling': '설비용량 정규화 후 발전소 간 표준화; 발전소 동일 가중치'}


def group_capacities(x, c):
    return math.fsum(c[x]), math.fsum(c[~x])


def feasible(x, c, minimum):
    a, b = group_capacities(x, c)
    return bool(x.any() and (~x).any() and a >= minimum and b >= minimum)


def validate_assignment(x, sites, cfg):
    x = np.asarray(x)
    if x.dtype != bool or x.shape != (len(sites),):
        raise InputError('메타데이터 행마다 Boolean 배정이 필요합니다.')
    c = sites.capacity_kw.to_numpy(float)
    if not feasible(x, c, cfg['minimum_capacity_kw']):
        raise InputError('그룹 용량 제약 위반.')
    a, b = group_capacities(x, c)
    total = math.fsum(c)
    preserved = math.isclose(a+b, total, rel_tol=1e-12, abs_tol=1e-8)
    if not preserved:
        raise InputError('총 설비용량 불일치.')
    return {'valid': True, 'site_count': len(c), 'g1_count': int(x.sum()),
            'g2_count': int((~x).sum()), 'g1_capacity_kw': a,
            'g2_capacity_kw': b, 'total_capacity_kw': total,
            'capacity_imbalance_kw': abs(a-b), 'capacity_preserved': preserved}


def initialize(c, cfg, seed=None):
    order = np.argsort(-c, kind='stable') if seed is None else np.random.default_rng(seed).permutation(len(c))
    total, m = math.fsum(c), cfg['minimum_capacity_kw']
    states = [(0.0, 0)]
    for position, i in enumerate(order):
        remaining = math.fsum(c[order[position+1:]])
        choices = {}
        for value, bits in states:
            for take in (False, True):
                v = value + (c[i] if take else 0)
                if v <= total-m+1e-8 and v+remaining >= m-1e-8:
                    choices.setdefault(v, bits | ((1 << int(i)) if take else 0))
        states = sorted(choices.items(), key=lambda it: (abs(it[0]-total/2), it[1]))[:cfg['beam_width']]
        if not states:
            raise SearchExhausted('탐색 후보 소진. beam_width를 늘리세요. 불가능성 증명은 아닙니다.')
    for _, bits in states:
        x = np.array([bool(bits & (1 << i)) for i in range(len(c))])
        if feasible(x, c, m):
            return x
    raise SearchExhausted('실현 가능한 배정을 찾지 못했습니다. 탐색 폭을 늘리세요.')


def loss(cap, count, sums, total, all_sums, n, weights):
    if not 0 < count < n:
        return float('inf')
    balance = ((2*cap-total)/total)**2
    profile = np.mean((sums/count-(all_sums-sums)/(n-count))**2)
    return float(weights[0]*balance + weights[1]*profile)


def objective(x, c, z, weights):
    return loss(c[x].sum(), int(x.sum()), z[x].sum(axis=0), c.sum(), z.sum(axis=0), len(c), weights)


def optimize(initial, c, z, cfg, seed, weights=None):
    weights = weights if weights is not None else (cfg['capacity_weight'], cfg['profile_weight'])
    x = initial.copy()
    m = cfg['minimum_capacity_kw']
    if not feasible(x, c, m):
        raise InputError('초기 배정이 용량 제약을 위반합니다.')
    total, all_sums, n = c.sum(), z.sum(axis=0), len(c)
    cap, count, sums = c[x].sum(), int(x.sum()), z[x].sum(axis=0)
    score = loss(cap, count, sums, total, all_sums, n, weights)
    log = [{'iteration': 0, 'objective': score, 'improvement': 0.0, 'affected_indices': []}]
    rng = np.random.default_rng(seed)
    stop = '반복 한도'
    for iteration in range(1, cfg['max_iterations']+1):
        moves = [(i,) for i in range(n)]
        left, right = np.flatnonzero(x), np.flatnonzero(~x)
        moves.extend((int(rng.choice(left)), int(rng.choice(right))) for _ in range(cfg['swap_trials']))
        best, best_score = None, score
        for indices in moves:
            idx = list(indices)
            delta = np.where(x[idx], -1, 1)
            next_cap = cap + np.dot(delta, c[idx])
            next_count = count + int(delta.sum())
            if not 0 < next_count < n or next_cap < m or total-next_cap < m:
                continue
            next_sums = sums + (z[idx]*delta[:, None]).sum(axis=0)
            value = loss(next_cap, next_count, next_sums, total, all_sums, n, weights)
            if value < best_score-cfg['tolerance']:
                proposal = x.copy()
                proposal[idx] = ~proposal[idx]
                if feasible(proposal, c, m):
                    best, best_score = (idx, proposal), value
        if best is None:
            stop = '표본 이웃에서 개선 없음'
            break
        idx, x = best
        # 승인 후 정확한 상태를 다시 계산하여 누적 오차를 방지합니다.
        cap, count, sums = c[x].sum(), int(x.sum()), z[x].sum(axis=0)
        new_score = loss(cap, count, sums, total, all_sums, n, weights)
        if new_score >= score-cfg['tolerance']:
            x[idx] = ~x[idx]
            stop = '수치 오차로 개선 불확실'
            break
        log.append({'iteration': iteration, 'objective': new_score,
                    'improvement': score-new_score, 'affected_indices': idx})
        score = new_score
    assert feasible(x, c, m)
    return x, log, stop


def diagnostics(p, sites, x, cfg):
    c = sites.capacity_kw.to_numpy(float)
    q = p.divide(c, axis=1)
    q = q.loc[q.index.hour.isin(cfg['daytime_hours'])]
    out = {'daytime_hours': len(q)}
    for label, mask in (('g1', x), ('g2', ~x)):
        cor = q.loc[:, mask].corr(min_periods=cfg['minimum_pair_overlap']).to_numpy()
        pairs = cor[np.triu_indices(len(cor), 1)]
        pairs = pairs[np.isfinite(pairs)]
        out[label+'_correlation'] = float(pairs.mean()) if len(pairs) else None
        out[label+'_valid_pairs'] = len(pairs)
    complete = q.dropna(how='any')
    out['complete_daytime_hours'] = len(complete)
    monthly = []
    for label, mask in (('g1', x), ('g2', ~x)):
        v = complete.loc[:, mask].mul(c[mask], axis=1).sum(axis=1)/c[mask].sum()
        out[label+'_normalized_std'] = float(v.std(ddof=0)) if len(v) >= 2 else None
        monthly.append(v.groupby(v.index.month).mean())
    out['seasonal_profile_rms'] = float(np.sqrt(np.mean((monthly[0]-monthly[1])**2))) if len(complete) else None
    return out


def read_composition(path, sites, cfg):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != ['site_id', 'group']:
        raise InputError('배정 CSV 열은 site_id,group이어야 합니다.')
    if frame.site_id.tolist() != sites.site_id.tolist():
        raise InputError('배정 CSV의 ID 또는 원본 순서 불일치.')
    if frame.site_id.duplicated().any() or not frame.group.isin(['G1','G2']).all():
        raise InputError('중복 ID 또는 허용되지 않은 그룹.')
    x = frame.group.eq('G1').to_numpy()
    validate_assignment(x, sites, cfg)
    return x


def run(raw_sites, raw_history, cfg, destination, synthetic=False):
    from .visualizer import write_visualization
    validate_config(cfg)
    # 입력 검증을 완료하기 전에는 출력 디렉터리를 생성하지 않습니다.
    sites = sites_frame(raw_sites, cfg)
    p, quality = history_frame(raw_history, sites, cfg)
    z, manifest = features(p, sites, cfg)
    c = sites.capacity_kw.to_numpy(float)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('기존 출력 경로는 사용하지 않습니다.')
    candidates, logs, stops, runtime = {}, {}, {}, {}
    seed = cfg['seed']
    start = time.perf_counter()
    initial = initialize(c, cfg)
    candidates['balanced'], logs['balanced'], stops['balanced'] = optimize(initial,c,z,cfg,seed,(1.,0.))
    runtime['balanced'] = time.perf_counter()-start
    start = time.perf_counter()
    candidates['profile'], logs['profile'], stops['profile'] = optimize(candidates['balanced'],c,z,cfg,seed+1,(0.,1.))
    runtime['profile'] = time.perf_counter()-start
    start = time.perf_counter()
    starts = [('balanced',candidates['balanced']),('profile',candidates['profile'])]
    failures = []
    for i in range(cfg['random_starts']):
        try:
            starts.append((f'random_{i}', initialize(c,cfg,seed+100+i)))
        except SearchExhausted as exc:
            failures.append({'seed': seed+100+i, 'reason': str(exc)})
    best, trials = None, []
    weights = (cfg['capacity_weight'],cfg['profile_weight'])
    for i,(name,assignment) in enumerate(starts):
        result, trace, stop = optimize(assignment,c,z,cfg,seed+1000+i)
        score = objective(result,c,z,weights)
        trials.append({'start': name,'objective': score,'stop': stop,'accepted_moves': len(trace)-1})
        if best is None or score < best[0]:
            best = score,result.copy(),trace,stop
    _,candidates['optimized'],logs['optimized'],stops['optimized'] = best
    runtime['optimized'] = time.perf_counter()-start
    validation, rows = {}, []
    midpoint = p.index.min()+(p.index.max()-p.index.min())/2
    for name,x in candidates.items():
        check = validate_assignment(x,sites,cfg)
        validation[name] = check
        row = dict(check,candidate=name,objective=objective(x,c,z,weights),runtime_seconds=runtime[name])
        row.update(diagnostics(p,sites,x,cfg))
        for label,window in [('earlier',p.loc[p.index<=midpoint]),('later',p.loc[p.index>midpoint])]:
            row.update({label+'_'+k:v for k,v in diagnostics(window,sites,x,cfg).items()})
        rows.append(row)
    destination.mkdir(parents=True,exist_ok=False)
    hashes = {}
    for name,x in candidates.items():
        path = destination/f'composition_{name}.csv'
        pd.DataFrame({'site_id':sites.site_id,'group':np.where(x,'G1','G2')}).to_csv(path,index=False)
        read_composition(path,sites,cfg)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        trace = [dict(entry,affected_sites=sites.iloc[entry['affected_indices']].site_id.tolist()) for entry in logs[name]]
        dump(destination/f'convergence_{name}.json',trace)
    pd.DataFrame(rows).to_csv(destination/'grouping_comparison.csv',index=False)
    pd.DataFrame(z,index=sites.site_id,columns=manifest['feature_names']).rename_axis('site_id').to_csv(destination/'plant_features.csv')
    dump(destination/'resolved_config.json',cfg)
    dump(destination/'feature_manifest.json',manifest)
    dump(destination/'multistart_diagnostics.json',{'starts':trials,'failed_initializations':failures})
    dump(destination/'validation_report.json',{'data_mode':'합성' if synthetic else '사용자 로컬 실행',
        'data_quality':quality,'assignments':validation,'composition_sha256':hashes,
        'stop_reasons':stops,'test_suite_executed':False,'official_score_calculated':False})
    write_visualization(destination/'grouping_visualization.html',sites,p,z,candidates,logs,rows)
    report = '# Phase 1 그룹 편성 보고서\n\n'
    report += '실제 예측오차나 정산 점수가 아닌 과거 대리 지표입니다.\n\n'
    report += '용량 기준, 과거 프로파일 기준, 다중 초기화 최적화 후보를 모두 보존합니다.\n'
    report += '목적함수는 전체 용량으로 정규화한 용량 차의 제곱과 표준화 특징 평균 차의 평균제곱 가중합입니다.\n'
    report += '그룹 간 평균 프로파일 균형은 그룹 내부의 상호보완성을 유도하는 휴리스틱이며 수익 최적성 증명이 아닙니다.\n'
    report += '상관계수는 실제 발전량 기준이며 예측오차 상관계수가 아닙니다. 공통 일주기 효과가 남습니다.\n'
    report += '집계 변동성과 계절 차는 전체 발전소가 관측된 시간만 사용하므로 결측이 많으면 계산되지 않습니다.\n'
    report += '특징 결측은 발전소 간 중앙값으로 대치합니다. 관측 부족 발전소의 다양성이 과소평가될 수 있습니다.\n'
    report += '앞/뒤 기간 진단은 전체 과거 구간으로 정한 배정의 기술통계이며 홀드아웃 검증이 아닙니다.\n'
    report += '탐색 종료는 전역 최적해를 뜻하지 않습니다. 무작위 초기화는 실현 가능한 배정의 균등 표본이 아닙니다.\n'
    report += '실행시간은 공통 로딩/특징 계산을 제외하며 프로파일 후보는 용량 후보를 재사용합니다.\n'
    report += '합성 테스트와 브라우저 오프라인 검사는 별도 실행해야 합니다. 이 파이프라인은 테스트 통과를 주장하지 않습니다.\n'
    (destination/'grouping_report.md').write_text(report,encoding='utf-8')
    return candidates,rows
