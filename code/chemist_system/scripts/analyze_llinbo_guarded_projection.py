"""Post-hoc coordinate transport diagnostics; never generates study proposals."""
from collections import defaultdict
import statistics as st
import numpy as np
from chem.benchmarks import llinbo_guarded_study as s
from chem.benchmarks import llinbo_guarded_engine as e
from chem.benchmarks import guarded_suzuki_engine as shared
from chem.benchmarks import guarded_suzuki_study as parent


def main():
    result={}
    for regime in s.REGIMES:
        counts=defaultdict(int);distances=[]
        for file in (s.ROOT/'runs').glob('*/state.json'):
            row=s.read(file)
            if row['case']==1 or row['regime']!=regime:continue
            assert row['status']=='COMPLETE'
            folder=parent.ROOT/'inputs'/f"case{row['case']}"
            pool=s.read(folder/'pool.json');lookup={key:i for i,key in enumerate(pool['ids'])}
            advice=s.read(folder/'advice.json')[str(row['seed'])][regime]
            membership=shared.clause_membership(pool['z'],advice)
            for event in row['decisions']:
                raw=event['projection']['raw_point'];raw_z=event['projection']['raw_z']
                inside_raw=bool(shared.clause_membership([raw_z],advice).all())
                inside_projected=bool(membership[:,lookup[event['proposed_id']]].all())
                prefix=row['history'][:event['iteration']+3]
                queried={r['candidate_id'] for r in prefix}
                nearest_including_used=e.project_proposal(raw,pool,set())
                counts['proposals']+=1
                counts['raw_proposal_inside_preference']+=inside_raw
                counts['projected_proposal_inside_preference']+=inside_projected
                counts['projection_leaves_preference']+=inside_raw and not inside_projected
                counts['projection_enters_preference']+=inside_projected and not inside_raw
                counts['nearest_pool_point_already_queried']+=nearest_including_used['candidate_id'] in queried
                distances.append(event['projection']['normalized_numeric_distance'])
        result[regime]={**dict(counts),'distance_mean':st.mean(distances),'distance_median':st.median(distances),
            'distance_p90':float(np.quantile(distances,.9)),'distance_max':max(distances)}
    s.write(s.OUT/'projection_diagnostics.json',{'scope':'Case2-4 evaluator-only post-hoc coordinate audit',
        'distance_definition':'Euclidean distance in normalized residence,temperature,loading, same catalyst.',
        'interpretation':'Nearest unused projection changes physical advice and may leave a supplied soft preference. It is part of the matched transport, not a literal continuous LLINBO recommendation.',
        'by_regime':result})
    print(result)


if __name__=='__main__':main()
