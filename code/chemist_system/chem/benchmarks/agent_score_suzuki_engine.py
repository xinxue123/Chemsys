"""Execute current-assistant scorecards over the complete fixed candidate pool.

No fitting, BO acquisition, hidden outcome access or per-candidate LLM call.
The assistant authors every card after seeing only that trajectory's prefix.
"""
import numpy as np

FIELDS={'bias','target','curvature','novelty','anchors','reason','support'}


def validate_card(card,history):
    if set(card)!=FIELDS:raise ValueError('scorecard must have the exact public fields')
    for key,n,lo,hi in [('bias',8,-100,150),('target',3,0,1),('curvature',3,0,150)]:
        a=np.asarray(card[key],dtype=float)
        if a.shape!=(n,) or not np.isfinite(a).all() or not ((a>=lo)&(a<=hi)).all():raise ValueError(key)
    if not isinstance(card['novelty'],(int,float)) or not 0<=card['novelty']<=50:raise ValueError('novelty')
    if not isinstance(card['reason'],str) or not card['reason'].strip():raise ValueError('reason')
    if not card['support'] or any(type(i)!=int or not 1<=i<=len(history) for i in card['support']):raise ValueError('support')
    if not isinstance(card['anchors'],list) or len(card['anchors'])>8:raise ValueError('anchors')
    for anchor in card['anchors']:
        if not isinstance(anchor,list) or len(anchor)!=3:raise ValueError('anchor triple')
        i,amplitude,width=anchor
        if type(i)!=int or not 1<=i<=len(history) or not -50<=amplitude<=50 or not .03<=width<=2:raise ValueError('anchor')


def score_pool(z,categories,history_z,history_categories,card):
    """One fixed vectorized operator; coefficients are supplied by the assistant.

    s(x)=b_c - sum_j w_j*(x_j-target_j)^2 + eta*d_same_cat(x,H)
         + sum_a A_a*exp(-||x-h_a||^2/(2*width_a^2))*1[c=c_a].
    d_same_cat is normalized nearest observed numeric distance, capped at1;
    an unseen catalyst gets1. Scores are decision utilities, not calibrated yields.
    """
    x=np.asarray(z,dtype=float)[:,:3];c=np.asarray(categories,dtype=int)
    h=np.asarray(history_z,dtype=float)[:,:3];hc=np.asarray(history_categories,dtype=int)
    scores=np.asarray(card['bias'])[c]-(((x-np.asarray(card['target']))**2)*np.asarray(card['curvature'])).sum(1)
    distance=np.ones(len(x))
    for category in range(8):
        available=c==category;seen=hc==category
        if seen.any() and available.any():
            delta=x[available,None,:]-h[None,seen,:]
            distance[available]=np.minimum(np.sqrt((delta**2).mean(2)).min(1),1.)
    scores=scores+float(card['novelty'])*distance
    for index,amplitude,width in card['anchors']:
        i=index-1;delta=x-h[i];scores+=float(amplitude)*np.exp(-(delta**2).sum(1)/(2*float(width)**2))*(c==hc[i])
    if not np.isfinite(scores).all():raise ValueError('nonfinite agent scores')
    return scores


def select(scores,ids,queried):
    available=[i for i,key in enumerate(ids) if key not in queried]
    if not available:raise ValueError('no unused candidate')
    return min(available,key=lambda i:(-float(scores[i]),ids[i]))
