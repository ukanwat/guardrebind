import json, math
R=str(__import__('pathlib').Path(__file__).resolve().parents[2]/'results'/'fix')+'/'
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return c-h,c+h
def newcombe(k1,n1,k2,n2):  # p1-p2
    p1,p2=k1/n1,k2/n2; l1,u1=wilson(k1,n1); l2,u2=wilson(k2,n2)
    d=p1-p2
    return 100*d, 100*(d-math.sqrt((p1-l1)**2+(u2-p2)**2)), 100*(d+math.sqrt((u1-p1)**2+(p2-l2)**2))
def hm(A,n1,n2):
    q1=A/(2-A); q2=2*A*A/(1+A)
    return math.sqrt((A*(1-A)+(n1-1)*(q1-A*A)+(n2-1)*(q2-A*A))/(n1*n2))
b=json.load(open(R+'r2_test_base.json'))
for c in ('bt_seed0','t_seed0','t_seed1'):
    d=json.load(open(R+f'r2_test_{c}.json'))
    da=d['auroc']-b['auroc']; se=math.hypot(hm(d['auroc'],801,699),hm(b['auroc'],801,699))
    print(c,f"dAUROC {da:+.3f} [{da-1.96*se:+.3f},{da+1.96*se:+.3f}]")
    r=lambda x:(x['k'],x['n'])
    print('  drecall %+.1f [%+.1f,%+.1f]'%newcombe(*r(d['recall']),*r(b['recall'])))
    for s in ('overrefusal_xstest','overrefusal_orbench','structured_benign'):
        print('  d',s,'%+.1f [%+.1f,%+.1f]'%newcombe(*r(d['overrefusal'][s]),*r(b['overrefusal'][s])))
for c in ('base','B','t_seed0','t_seed1','bt_seed0','cc_seed0','detector','segmax_infer'):
    d=json.load(open(R+f'r2_test_{c}.json')); x=d['by_decoy_split']['heldout']['heldout']['bypass_benign_decoy']
    lo,hi=wilson(x['k'],x['n']); print(c,'HOxHO',x['k'],x['n'],'%.1f [%.1f,%.1f]'%(100*x['rate'],100*lo,100*hi))
for c in ('base','t_seed0','t_seed1'):
    d=json.load(open(R+f'r2_test_{c}.json')); x=d['by_decoy_split']['fixed']['heldout']['bypass_benign_decoy']
    lo,hi=wilson(x['k'],x['n']); print(c,'fixed HO',x['k'],x['n'],'%.1f [%.1f,%.1f]'%(100*x['rate'],100*lo,100*hi))
d=json.load(open(R+'r2_test_segmax_infer.json'));x=d['overrefusal']['structured_benign'];print('segmax sb',x['k'],x['n'],wilson(x['k'],x['n']))
d=json.load(open(R+'r2_test_detector.json'));x=d['overrefusal']['structured_benign'];print('det sb',x['k'],x['n'],wilson(x['k'],x['n']))
