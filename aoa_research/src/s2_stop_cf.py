"""2단계-5: 꼬리 손실 차단 문턱(k ATR) 반사실 분석과 일간 손익 분포(학습 구간만)."""
import pandas as pd, numpy as np
from config import CACHE, RESULTS
from prices import bitstamp_bars
e=pd.read_parquet(CACHE/'btc_episodes_atr.parquet')
panel=pd.read_parquet(CACHE/'panel_open.parquet')
h1=bitstamp_bars('1h'); b=h1.copy(); b.index=b.index+pd.Timedelta(hours=1)
p=panel[panel.ep>=0][['ep','avg_entry','Q','equity_btc']].join(b[['high','low','close']])
rows=[]
tr=e[e.period=='train'].set_index('ep')
pp=p[p.ep.isin(tr.index)]
grp={k:g for k,g in pp.groupby('ep')}
for k in [1,1.5,2,3,4,5,6,8,10,15,1e9]:
    tot=0.0; n_stop=0; tot_eq=0.0
    for ep,row in tr.iterrows():
        g=grp.get(ep)
        a0=row.atr14_0
        if g is None or len(g)==0 or not np.isfinite(a0):
            tot+=row.pnl_btc; tot_eq+=row.pnl_btc/row.equity_btc_0; continue
        d=np.sign(g.Q.to_numpy())
        adv=np.where(d>0, (g.avg_entry-g.low).to_numpy(), (g.high-g.avg_entry).to_numpy())
        hit=np.nonzero(adv>=k*a0)[0]
        if len(hit)==0:
            tot+=row.pnl_btc; tot_eq+=row.pnl_btc/row.equity_btc_0; continue
        i=hit[0]; Q=abs(g.Q.iloc[i]); A=g.avg_entry.iloc[i]
        stop_px=A-k*a0 if d[i]>0 else A+k*a0
        pnl=Q*(1/A-1/stop_px) if d[i]>0 else Q*(1/stop_px-1/A)
        pnl-= Q/stop_px*0.00075
        tot+=pnl; n_stop+=1; tot_eq+=pnl/row.equity_btc_0
    rows.append(dict(k_atr=k, n_stopped=n_stop, pnl_btc=round(tot,1), pnl_eq_sum=round(tot_eq,3)))
cf=pd.DataFrame(rows); print(cf.to_string()); cf.to_csv(RESULTS/'s2_stop_counterfactual_train.csv',index=False)
pt=panel[panel.period=='train']
d=pt[['equity_btc','net_dep_btc']].resample('1D').last()
r=(d.equity_btc.diff()-d.net_dep_btc.diff())/d.equity_btc.shift(1)
print('daily ret quantiles', r.quantile([0.01,0.05,0.1,0.5,0.9,0.95,0.99]).round(4).to_dict(), 'worst', round(r.min(),4), 'n', r.notna().sum())
