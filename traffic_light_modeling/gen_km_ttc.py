import pandas as pd, numpy as np, matplotlib.pyplot as plt, os, warnings
warnings.filterwarnings('ignore')
from lifelines import KaplanMeierFitter

BASE = os.path.dirname(os.path.abspath(__file__))
df   = pd.read_csv(os.path.join(BASE, 'output', 'windows_1s_clean.csv'))

def assign_segments(event_df):
    edf = event_df.sort_values('start_time').reset_index(drop=True)
    n = len(edf); seg = pd.Series([np.nan]*n, dtype=float)
    def is_junc(i): return edf.at[i,'current_phase'] in ('InsideJunction','LeavingJunction')
    def is_appr(i): return edf.at[i,'current_phase'] == 'Approaching'
    inside_rows = edf.index[edf['current_phase']=='InsideJunction'].tolist()
    fi = inside_rows[0] if inside_rows else None
    yellow_rows = edf.index[edf['yellow_transition']==1].tolist()
    if not yellow_rows:
        for i in range(n):
            if is_appr(i): seg[i]=1
            elif is_junc(i): seg[i]=3
        r=seg.copy(); r.index=event_df.index; return r
    yp=yellow_rows[0]
    for i in range(yp):
        if is_appr(i): seg[i]=1
    if fi is None:
        for i in range(yp,n):
            if is_appr(i): seg[i]=2
        r=seg.copy(); r.index=event_df.index; return r
    entry_light=edf.at[fi,'TrafficLight_current']
    if entry_light=='Green':
        red_rows=[i for i in range(yp,n) if edf.at[i,'TrafficLight_current']=='Red']
        seg2_end=red_rows[-1] if red_rows else yp
        for i in range(yp,seg2_end+1):
            if is_appr(i): seg[i]=2
        for i in range(seg2_end+1,n):
            if is_appr(i) or is_junc(i): seg[i]=3
    else:
        stopped=[i for i in range(fi,n) if is_junc(i) and edf.at[i,'is_stopped']==1]
        if stopped:
            sp=stopped[0]
            for i in range(yp,sp+1):
                if is_appr(i) or is_junc(i): seg[i]=2
            for i in range(sp+1,n):
                if is_appr(i) or is_junc(i): seg[i]=3
        else:
            for i in range(yp,n):
                if is_appr(i) or is_junc(i): seg[i]=2
    r=seg.copy(); r.index=event_df.index; return r

df['segment'] = df.groupby('event_id', group_keys=False).apply(
    lambda g: assign_segments(g), include_groups=False)
pre = df['segment'].isin([1.0, 2.0])
df['TTC_imputed'] = np.nan
df.loc[pre,'TTC_imputed'] = df.loc[pre,'TTC_min'].fillna(999.0).clip(upper=6.0)

seg1 = df[df['segment']==1.0].copy()
seg2 = df[df['segment']==2.0].copy()

# KM TTC Seg1
km1_rows = []
for ev_id, g in seg1.groupby('event_id'):
    g  = g.sort_values('start_time')
    fb = g[g['first_feedback_relavet_to_event']==1]
    ttc_val = fb['TTC_imputed'].iloc[0] if len(fb) else g['TTC_imputed'].min()
    km1_rows.append({'ttc': ttc_val, 'event': 1 if len(fb) else 0})
km1_ttc = pd.DataFrame(km1_rows).dropna(subset=['ttc'])

fig, ax = plt.subplots(figsize=(9, 6))
kmf = KaplanMeierFitter()
kmf.fit(km1_ttc['ttc'], event_observed=km1_ttc['event'],
        label=f'Seg 1 (n={len(km1_ttc)}, {int(km1_ttc["event"].sum())} feedback events)')
kmf.plot_survival_function(ax=ax, ci_show=True, color='#4878d0')
ax.invert_xaxis()
ax.set_title('Seg 1 — Kaplan-Meier by TTC\n(X decreasing = driver approaching traffic light)', fontsize=11)
ax.set_xlabel('TTC (s)   ←   approaching')
ax.set_ylabel('P(no feedback yet)')
ax.set_ylim(0, 1.05)
ax.grid(True, ls='--', alpha=0.4)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'output', 'fig_km_ttc_seg1.png'), dpi=140, bbox_inches='tight')
plt.close()
print('Saved fig_km_ttc_seg1.png')

# KM TTC Seg2
km2_rows = []
for ev_id, g in seg2.groupby('event_id'):
    g  = g.sort_values('start_time')
    fb = g[g['first_feedback_relavet_to_event']==1]
    ttc_val = fb['TTC_imputed'].iloc[0] if len(fb) else g['TTC_imputed'].min()
    km2_rows.append({'ttc': ttc_val, 'event': 1 if len(fb) else 0})
km2_ttc = pd.DataFrame(km2_rows).dropna(subset=['ttc'])

fig, ax = plt.subplots(figsize=(9, 6))
kmf = KaplanMeierFitter()
kmf.fit(km2_ttc['ttc'], event_observed=km2_ttc['event'],
        label=f'Seg 2 (n={len(km2_ttc)}, {int(km2_ttc["event"].sum())} feedback events)')
kmf.plot_survival_function(ax=ax, ci_show=True, color='#ee854a')
ax.invert_xaxis()
ax.set_title('Seg 2 — Kaplan-Meier by TTC\n(X decreasing = driver approaching traffic light)', fontsize=11)
ax.set_xlabel('TTC (s)   ←   approaching')
ax.set_ylabel('P(no feedback yet)')
ax.set_ylim(0, 1.05)
ax.grid(True, ls='--', alpha=0.4)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'output', 'fig_km_ttc_seg2.png'), dpi=140, bbox_inches='tight')
plt.close()
print('Saved fig_km_ttc_seg2.png')
