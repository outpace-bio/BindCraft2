"""Primary analysis, per the user's choices:
   endpoint = final recorded ipTM/ipSAE from each trajectory's per-step losses CSV
   terminated trajectories contribute their last recorded value
   all 128 pairs per block-set, paired on trajectory index within block."""
import csv, glob, json, math, os, statistics as st

BLOCKS=[("arm_a","arm_b"),("block_20260924_arm_a","block_20260924_arm_b"),
        ("block_20260925_arm_a","block_20260925_arm_b"),("block_20260926_arm_a","block_20260926_arm_b")]

def arm(name):
    """trajectory index -> final recorded values, from the losses CSV (complete for every trajectory)."""
    base=f'results/ipsae_ab_powered/{name}'
    design={}
    for r in csv.DictReader(open(f'{base}/1_Trajectories/!_Trajectories.csv')):
        design[int(r['trajectory'])]=(r['design'], (r.get('terminated') or '').strip())
    out={}
    for t,(dn,term) in design.items():
        p=f'{base}/1_Trajectories/{dn}/{dn}_losses.csv'
        if not os.path.exists(p): continue
        rows=list(csv.DictReader(open(p)))
        if not rows: continue
        last=rows[-1]
        g=lambda k: float(last['hPDL1.'+k]) if last.get('hPDL1.'+k) else None
        out[t]={'iptm':g('iptm'),'ipsae':g('ipsae'),'steps':len(rows),
                'phase':last['phase'],'terminated':term,'completed':term==''}
    return out

pairs=[]
for bi,(A,B) in enumerate(BLOCKS):
    a,b=arm(A),arm(B)
    for t in sorted(set(a)&set(b)):
        if a[t]['iptm'] is None or b[t]['iptm'] is None: continue
        pairs.append({'block':bi,'traj':t,'A':a[t],'B':b[t]})

def wilcoxon(d):
    d=[x for x in d if x!=0]; n=len(d)
    if n<6: return None,n
    o=sorted(range(n),key=lambda i:abs(d[i])); rk=[0]*n; i=0
    while i<n:
        j=i
        while j+1<n and abs(d[o[j+1]])==abs(d[o[i]]): j+=1
        av=(i+j)/2+1
        for k in range(i,j+1): rk[o[k]]=av
        i=j+1
    W=sum(rk[i] for i in range(n) if d[i]>0); mu=n*(n+1)/4; sd=math.sqrt(n*(n+1)*(2*n+1)/24)
    z=(W-mu)/sd
    return (z,2*(1-0.5*(1+math.erf(abs(z)/math.sqrt(2))))),n

res={'n_pairs':len(pairs),'metrics':{}}
print(f"  pairs with a recoverable value in both arms: {len(pairs)} / 128\n")
for k,lab in (('ipsae','PRIMARY  ipSAE'),('iptm','SECONDARY ipTM')):
    av=[p['A'][k] for p in pairs]; bv=[p['B'][k] for p in pairs]
    d=[y-x for x,y in zip(av,bv)]
    r,nn=wilcoxon(d); sd=st.stdev(d); se=sd/math.sqrt(len(d))
    m=st.mean(d); lo,hi=m-1.96*se,m+1.96*se
    res['metrics'][k]={'A_mean':st.mean(av),'A_median':st.median(av),'A_sd':st.stdev(av),
                       'B_mean':st.mean(bv),'B_median':st.median(bv),'B_sd':st.stdev(bv),
                       'diff':m,'ci':[lo,hi],'dz':m/sd,'p':r[1] if r else None,'z':r[0] if r else None,
                       'B_higher':sum(1 for x in d if x>0),'n':len(d)}
    print(f"  {lab}")
    print(f"    arm A  mean {st.mean(av):.4f}  median {st.median(av):.4f}  sd {st.stdev(av):.4f}")
    print(f"    arm B  mean {st.mean(bv):.4f}  median {st.median(bv):.4f}  sd {st.stdev(bv):.4f}")
    print(f"    paired diff B-A  {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  dz {m/sd:+.3f}")
    print(f"    B higher in {sum(1 for x in d if x>0)}/{len(d)} pairs")
    if r: print(f"    Wilcoxon z={r[0]:+.3f}  p={r[1]:.4f}  -> {'significant' if r[1]<0.05 else 'NOT significant'} at 0.05\n")

sa=[p['A']['steps'] for p in pairs]; sb=[p['B']['steps'] for p in pairs]
ca=sum(1 for p in pairs if p['A']['completed']); cb=sum(1 for p in pairs if p['B']['completed'])
res['steps']={'A_mean':st.mean(sa),'B_mean':st.mean(sb),'A_completed':ca,'B_completed':cb,'n':len(pairs)}
print(f"  CONTEXT the endpoint choice carries:")
print(f"    mean steps reached  A {st.mean(sa):.1f}   B {st.mean(sb):.1f}   (of 140)")
print(f"    ran to completion   A {ca}/{len(pairs)}   B {cb}/{len(pairs)}")
print(f"    so arm B's values are read, on average, {st.mean(sa)-st.mean(sb):.1f} steps earlier")
json.dump({'pairs':pairs,'summary':res}, open('/tmp/claude-1000/plot/final.json','w'))
