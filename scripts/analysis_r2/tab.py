import json, math
R=str(__import__('pathlib').Path(__file__).resolve().parents[2]/'results'/'fix')+'/'
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h
C=['base','B','detector','segmax_infer','t_seed0','t_seed1','bt_seed0','cc_seed0']
for c in C:
    d=json.load(open(R+f'r2_test_{c}.json'))
    f=d['by_decoy_split']['fixed']; h=d['by_decoy_split']['heldout']
    assert f['train']['bypass_benign_decoy']==d['attacks']['train']['bypass_benign_decoy']
    def bb(x): b=x['bypass_benign_decoy']; return f"{b['k']}/{b['n']}={100*b['rate']:.1f}"
    abf=[f[s]['attacked_benign_fpr_benign_decoy'] for s in ('train','heldout')]
    k=sum(a['k'] for a in abf); n=sum(a['n'] for a in abf)
    hd=[v for s in ('train','heldout') for a,v in f[s]['per_attack_benign_fpr'].items() if 'decoy=harmful' in a]
    hdh=[v for s in ('train','heldout') for a,v in h[s]['per_attack_benign_fpr'].items() if 'decoy=harmful' in a]
    o=d['overrefusal']
    print(c, 'scr',d['screened'],'| fixed tr',bb(f['train']),'ho',bb(f['heldout']),'| HDsplit tr',bb(h['train']),'ho',bb(h['heldout']),
      f"| auroc {d['auroc']:.3f} rec {d['recall']['k']}/{d['recall']['n']}={100*d['recall']['rate']:.1f} fpr {d['fpr_clean']['k']}/{d['fpr_clean']['n']}={100*d['fpr_clean']['rate']:.1f}",
      f"| orb {o['overrefusal_orbench']['k']}={100*o['overrefusal_orbench']['rate']:.1f} xst {o['overrefusal_xstest']['k']}={100*o['overrefusal_xstest']['rate']:.1f} sb {o['structured_benign']['k']}={100*o['structured_benign']['rate']:.1f}",
      f"| abfpr {k}/{n}={100*k/n:.1f} | harmdec fixed n={len(hd)} {100*sum(hd)/len(hd):.1f} heldout {100*sum(hdh)/len(hdh):.1f}")
