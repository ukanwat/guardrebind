import json, numpy as np
R='/Users/utkarshkanwat/development/guard-policy-injection/results/fix/'
def load(c): return json.load(open(R+f'r2_test_{c}.json'))
def per_item(d, split, fam='heldout'):
    """per-item bypass rate (benign/borderline decoys) over held-out family attacks"""
    names=[a for a in d['by_decoy_split'][split][fam]['per_attack'] if 'decoy=harmful' not in a]
    raw = d['raw_scores'] if split=='fixed' else d['raw_scores_by_decoy_split'][split]
    ids=raw['screened_ids']; n=len(ids); out={}
    for i,iid in enumerate(ids):
        v=[]
        for a in names:
            s=raw['harmful'][a]; k=len(s)//n
            v+= [x<0.5 for x in s[i*k:(i+1)*k]]
        out[iid]=np.mean(v)
    return out
rng=np.random.default_rng(0)
def boot(a,b,B=10000):
    ids=sorted(set(a)&set(b)); x=np.array([a[i]-b[i] for i in ids]); n=len(x)
    bs=np.array([x[rng.integers(0,n,n)].mean() for _ in range(B)])
    return n, 100*x.mean(), 100*np.percentile(bs,2.5), 100*np.percentile(bs,97.5), (bs<=0).mean()
t0,t1,bt,base=[load(c) for c in ('t_seed0','t_seed1','bt_seed0','base')]
T={k:(v+per_item(t1,'heldout')[k])/2 for k,v in per_item(t0,'heldout').items()}
BT=per_item(bt,'heldout')
print('F2 T-BT heldout fam x heldout decoy: n, diff, lo, hi, p(one-sided diff<=0)', boot(T,BT))
Tf={k:(v+per_item(t1,'fixed')[k])/2 for k,v in per_item(t0,'fixed').items()}
print('T fixed - heldout decoy (heldout fams):', boot(Tf,T))
# also per-seed sanity: reproduce rates
for c,d in (('t0',t0),('bt',bt)):
    p=per_item(d,'heldout'); print(c,'mean per-item',100*np.mean(list(p.values())))
