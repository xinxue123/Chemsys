"""Factual post-hoc counts; no policy updates or new oracle queries."""
from chem.benchmarks.guarded_suzuki_study import read,write,ROOT,OUT

result={}
for method in ('LGBO_region','B4_base','B4_LGBO'):
    count={'decisions':0,'wrong_category_retained':0,'query_in_wrong_category':0,
        'retained_wrong_category_but_queried_elsewhere':0,
        'equally_large_admissible_subset_could_drop_wrong_category':0}
    for file in (ROOT/'test/runs').glob('*/state.json'):
        state=read(file)
        if state['method']!=method or state['regime']!='wrong_category':continue
        folder=ROOT/'inputs'/f"case{state['case']}"
        pool=read(folder/'pool.json');lookup={v:i for i,v in enumerate(pool['ids'])}
        advice=read(folder/'advice.json')[str(state['seed'])]['wrong_category']
        cat=next(h for h in advice['clauses'] if h['coordinate']==3)
        for event in state['decisions']:
            code=pool['z'][lookup[event['selected_id']]][3]
            inside=cat['interval'][0]<=code<=cat['interval'][1]
            retained=cat['id'] in event['retained_ids']
            count['decisions']+=1;count['wrong_category_retained']+=retained
            count['query_in_wrong_category']+=inside
            count['retained_wrong_category_but_queried_elsewhere']+=retained and not inside
            detail=event.get('minimal_revocation_receipt')
            if detail and retained:
                count['equally_large_admissible_subset_could_drop_wrong_category']+=any(
                    row['eligible_count']>0 and len(row['retained_ids'])==len(event['retained_ids'])
                    and cat['id'] not in row['retained_ids'] for row in detail['subset_checks'])
    result[method]=count
write(OUT/'wrong_category_mechanism.json',{'scope':'posthoc factual trace counts, not a causal re-run','counts':result})
print(result)
