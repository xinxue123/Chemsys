"""Export prospective Note packets with full scalar KF arithmetic for audit."""
from pathlib import Path
import csv
import json
import math

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'paper/v13'


def close(a,b):
    if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12):raise ValueError((a,b))


def main():
    with (P/'data/metrics.csv').open(encoding='utf-8',newline='') as f:
        runs=[x for x in csv.DictReader(f) if x['method']=='v13']
    rows=[]
    for m in runs:
        sf=ROOT/m['state_path'];s=json.loads(sf.read_text(encoding='utf-8'))
        for d in s['decisions']:
            q=d['query'];c=json.loads((sf.parent/'rounds'/f'{q:02}_commit.json').read_text(encoding='utf-8'))
            packets={f['commit_sha256']:f for f in c['decision']['forecasts']}
            for e in d['feedback']:
                f=packets[e['commit_sha256']];v=e['calibration'];b=v['before'];a=v['after'];y=float(e['improved'])
                z=(y-f['p_GP'])**2-(y-f['p_LLM'])**2
                R=max(.0025,4*(f['p_LLM']-f['p_GP'])**2*f['p_GP']*(1-f['p_GP']))
                prior=b['variance']+.0025;K=prior/(prior+R)
                mean=b['mean']+K*(z-b['mean']);var=(1-K)**2*prior+K*K*R
                weight=.8*.5*math.erfc(-mean/math.sqrt(2*var))
                if z<=1e-12:weight=min(weight,b['reliability'])
                for aa,bb in ((z,v['Brier_gain']),(R,v['measurement_variance']),(K,v['kalman_gain']),
                              (mean,a['mean']),(var,a['variance']),(weight,a['reliability'])):close(aa,bb)
                rows.append(dict(regime=m['regime'],task=m['task'],seed=int(m['seed']),query=q,note_id=f['id'],version=f['version'],
                    candidate_id=f['candidate_id'],use=e['use'],improved=int(e['improved']),
                    current_incumbent=f['incumbent'],p_LLM=f['p_LLM'],p_GP=f['p_GP'],
                    mean_before=b['mean'],variance_before=b['variance'],reliability_before=b['reliability'],
                    Brier_gain=z,measurement_variance=R,predicted_variance=prior,kalman_gain=K,
                    mean_after=a['mean'],variance_after=a['variance'],reliability_after=a['reliability'],
                    permission_before=e['recommendation_before'],permission_after=e['recommendation_after'],
                    accurate_failure=int(e['accurate_failure_forecast']),packet_sha256=f['commit_sha256'],event_id=e['event_id']))
    assert len(rows)==338
    with (P/'data/forecast_calculations.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (P/'audit/forecast_arithmetic.json').write_text(json.dumps(dict(status='PASS',prospective_packets=338,
        relative_Brier_gain_and_scalar_KF_recomputed=True,GP_values_changed=False,new_LLM_calls=0),indent=2)+'\n',encoding='utf-8')
    print('Verified all338 prospective Brier/KF updates against native packets.')


if __name__=='__main__':main()
