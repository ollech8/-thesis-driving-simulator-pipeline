import pandas as pd, numpy as np, matplotlib.pyplot as plt, matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import stats
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import warnings, os
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT      = os.path.join(BASE_DIR, 'output')
df = pd.read_csv(os.path.join(OUT, 'windows_1s_clean.csv'))

TTC_CEILING, TTC_NAN_FILL = 6.0, 999.0
SEG_C = {1:'#4878d0', 2:'#ee854a', 3:'#9467bd'}
LC    = {'Green':'#2ecc71','Yellow':'#f1c40f','Red':'#e74c3c'}
np.random.seed(0)

# ═══════════════════════════════════════════════════════════════
# STEP 1 — Segment assignment
# ═══════════════════════════════════════════════════════════════
def assign_segments(event_df):
    edf = event_df.sort_values('start_time').reset_index(drop=True)
    n   = len(edf)
    seg = pd.Series([np.nan]*n, dtype=float)
    def is_junc(i):   return edf.at[i,'current_phase'] in ('InsideJunction','LeavingJunction')
    def is_appr(i):   return edf.at[i,'current_phase'] == 'Approaching'
    inside_rows = edf.index[edf['current_phase']=='InsideJunction'].tolist()
    fi          = inside_rows[0] if inside_rows else None
    yellow_rows = edf.index[edf['yellow_transition']==1].tolist()
    if not yellow_rows:
        for i in range(n):
            if is_appr(i): seg[i] = 1
            elif is_junc(i): seg[i] = 3
        r = seg.copy(); r.index = event_df.index; return r
    yp = yellow_rows[0]
    for i in range(yp):
        if is_appr(i): seg[i] = 1
    if fi is None:
        for i in range(yp, n):
            if is_appr(i): seg[i] = 2
        r = seg.copy(); r.index = event_df.index; return r
    entry_light = edf.at[fi,'TrafficLight_current']
    if entry_light == 'Green':
        red_rows = [i for i in range(yp,n) if edf.at[i,'TrafficLight_current']=='Red']
        seg2_end = red_rows[-1] if red_rows else yp
        for i in range(yp, seg2_end+1):
            if is_appr(i): seg[i] = 2
        for i in range(seg2_end+1, n):
            if is_appr(i) or is_junc(i): seg[i] = 3
    else:
        stopped = [i for i in range(fi,n) if is_junc(i) and edf.at[i,'is_stopped']==1]
        if stopped:
            sp = stopped[0]
            for i in range(yp, sp+1):
                if is_appr(i) or is_junc(i): seg[i] = 2
            for i in range(sp+1, n):
                if is_appr(i) or is_junc(i): seg[i] = 3
        else:
            for i in range(yp, n):
                if is_appr(i) or is_junc(i): seg[i] = 2
    r = seg.copy(); r.index = event_df.index; return r

df['segment'] = df.groupby('event_id', group_keys=False).apply(
    lambda g: assign_segments(g), include_groups=False)
print(f"Segments: {df['segment'].value_counts(dropna=False).to_dict()}  NaN={df['segment'].isna().sum()}")

# ═══════════════════════════════════════════════════════════════
# STEP 2 — TTC imputation
# ═══════════════════════════════════════════════════════════════
pre = df['segment'].isin([1.0, 2.0])
df['TTC_imputed'] = np.nan
df.loc[pre,'TTC_imputed'] = df.loc[pre,'TTC_min'].fillna(TTC_NAN_FILL).clip(upper=TTC_CEILING)

# ═══════════════════════════════════════════════════════════════
# STEP 3 — Time variables
#   Seg 1: time from EVENT start
#   Seg 2, 3: time from SEGMENT start (within each event)
# ═══════════════════════════════════════════════════════════════
df['time_from_event_start'] = df.groupby('event_id')['start_time'].transform(lambda x: x - x.min())
df['time_from_seg_start']   = np.nan
for _sid in [1.0, 2.0, 3.0]:
    _mask = df['segment'] == _sid
    df.loc[_mask, 'time_from_seg_start'] = (
        df[_mask].groupby('event_id')['start_time'].transform(lambda x: x - x.min())
    )

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def fit_logistic(X, y, label=''):
    if y.sum() == 0 or (1-y).sum() == 0:
        print(f'  {label}: only one class — skip'); return None, np.nan
    m   = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000).fit(X, y)
    auc = roc_auc_score(y, m.predict_proba(X)[:,1])
    print(f'  {label}: n={len(y)}, fb={y.sum()}, AUC={auc:.3f}, coefs={m.coef_[0]}')
    return m, auc

def km_plot(ax, durations, events, group, title, xlabel='זמן (שניות)', palette=('#2166ac','#d6604d')):
    """Kaplan-Meier with two groups (0/1) and log-rank test."""
    kmf = KaplanMeierFitter()
    labels = {0:'נמוך', 1:'גבוה'}
    for g_val, color in zip([0,1], palette):
        mask = group == g_val
        if mask.sum() == 0: continue
        kmf.fit(durations[mask], event_observed=events[mask], label=labels[g_val])
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color)
    res = logrank_test(durations[group==0], durations[group==1],
                       event_observed_A=events[group==0], event_observed_B=events[group==1])
    ax.set_title(f'{title}\nLog-rank p={res.p_value:.3f}', fontsize=10)
    ax.set_xlabel(xlabel); ax.set_ylabel('P(טרם פידבק)')
    ax.set_ylim(0,1.05); ax.grid(True, ls='--', alpha=0.4)
    ax.legend(title='קבוצה', fontsize=9)
    return res.p_value

# ═══════════════════════════════════════════════════════════════
# FIG 1 — Event timelines
# ═══════════════════════════════════════════════════════════════
seg1_events = df[df['segment']==1.0]['event_id'].unique()
seg3g = df[(df['segment']==3.0)]['event_id'].unique()
sample_ids = list(seg1_events[:2]) + list(seg3g[:2])
sample_ids = list(dict.fromkeys(sample_ids))[:4]

fig, axes = plt.subplots(len(sample_ids), 1, figsize=(15, 3.2*len(sample_ids)))
if len(sample_ids)==1: axes=[axes]
fig.suptitle('ציר זמן אירועים — תיוג סגמנטים', fontsize=13, fontweight='bold')
SEG_L = {1:'Seg 1 — התקרבות',2:'Seg 2 — החלטת צהוב',3:'Seg 3 — צומת'}
for ax, ev_id in zip(axes, sample_ids):
    ev = df[df['event_id']==ev_id].sort_values('start_time')
    for _, row in ev.iterrows():
        ax.barh(0, 1, left=row['start_time'], color=SEG_C.get(row['segment'],'#d5d5d5'),
                height=0.5, edgecolor='white', lw=0.3)
        if row['first_feedback_relavet_to_event']==1:
            ax.plot(row['start_time']+0.5, 0, 'v', color='black', ms=9, zorder=5)
        ax.barh(-0.4, 1, left=row['start_time'],
                color=LC.get(row['TrafficLight_current'],'grey'), height=0.15, alpha=0.8)
    ax.set_title(ev_id, fontsize=8); ax.set_yticks([])
    ax.set_xlabel('זמן (שניות)'); ax.set_xlim(ev['start_time'].min()-0.5, ev['start_time'].max()+1.5)
patches = [mpatches.Patch(color=v, label=SEG_L[k]) for k,v in SEG_C.items()]
patches += [plt.Line2D([0],[0], marker='v', color='black', lw=0, ms=8, label='פידבק')]
fig.legend(handles=patches, loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT,'fig1_timelines.png'), dpi=140, bbox_inches='tight')
plt.close(); print('fig1 saved')

# ═══════════════════════════════════════════════════════════════
# FIG 2 — Seg 1: Logistic (TTC + time_from_event_start)
# ═══════════════════════════════════════════════════════════════
seg1 = df[df['segment']==1.0].copy()
X1   = seg1[['TTC_imputed','time_from_event_start']].values
y1   = seg1['first_feedback_relavet_to_event'].values
print('\nSeg 1:')
m1, auc1 = fit_logistic(X1, y1, 'Seg1')

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('Seg 1 — התקרבות באור ירוק\nמנבאים: TTC + זמן מתחילת האירוע', fontsize=12, fontweight='bold')

# subplot 1: TTC effect (hold time at median)
ax = axes[0]
if m1:
    t_med = np.median(seg1['time_from_event_start'])
    ttc_r = np.linspace(seg1['TTC_imputed'].min(), seg1['TTC_imputed'].max(), 300)
    grid  = np.column_stack([ttc_r, np.full(300, t_med)])
    ax.scatter(seg1['TTC_imputed'], y1+np.random.uniform(-0.03,0.03,len(y1)),
               alpha=0.2, s=15, color='#4878d0')
    ax.plot(ttc_r, m1.predict_proba(grid)[:,1], color='crimson', lw=2.5,
            label=f'AUC={auc1:.2f}')
    ax.set_xlabel('TTC (שניות, תקרה 6s)'); ax.set_ylabel('P(פידבק)')
    ax.set_title(f'השפעת TTC (זמן={t_med:.0f}s קבוע)\nCoef={m1.coef_[0][0]:.3f}')
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)

# subplot 2: time effect (hold TTC at median)
ax = axes[1]
if m1:
    ttc_med = np.median(seg1['TTC_imputed'])
    time_r  = np.linspace(seg1['time_from_event_start'].min(), seg1['time_from_event_start'].max(), 300)
    grid2   = np.column_stack([np.full(300, ttc_med), time_r])
    ax.scatter(seg1['time_from_event_start'], y1+np.random.uniform(-0.03,0.03,len(y1)),
               alpha=0.2, s=15, color='#4878d0')
    ax.plot(time_r, m1.predict_proba(grid2)[:,1], color='darkorange', lw=2.5,
            label=f'AUC={auc1:.2f}')
    ax.set_xlabel('זמן מתחילת האירוע (שניות)'); ax.set_ylabel('P(פידבק)')
    ax.set_title(f'השפעת זמן (TTC={ttc_med:.1f}s קבוע)\nCoef={m1.coef_[0][1]:.3f}')
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)

# subplot 3: distribution
ax = axes[2]
ax.hist(seg1.loc[y1==0,'TTC_imputed'], bins=30, alpha=0.6, color='#4878d0', density=True,
        label=f'ללא פידבק (n={(y1==0).sum()})')
ax.hist(seg1.loc[y1==1,'TTC_imputed'], bins=10, alpha=0.85, color='crimson', density=True,
        label=f'פידבק (n={y1.sum()})')
ax.set_xlabel('TTC (שניות)'); ax.set_ylabel('צפיפות')
ax.set_title('התפלגות TTC לפי קבוצה')
ax.legend(); ax.grid(True, ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT,'fig2_seg1_logistic.png'), dpi=140, bbox_inches='tight')
plt.close(); print('fig2 saved')

# ═══════════════════════════════════════════════════════════════
# FIG 3 — Seg 1: Kaplan-Meier (TTC median split)
# ═══════════════════════════════════════════════════════════════
# Per event: duration = time_from_event_start of feedback (or last window if censored)
# Use average TTC in seg1 of that event for grouping
km1_rows = []
for ev_id, g in seg1.groupby('event_id'):
    g = g.sort_values('start_time')
    fb_rows = g[g['first_feedback_relavet_to_event']==1]
    if len(fb_rows):
        dur   = fb_rows['time_from_event_start'].iloc[0]
        event = 1
    else:
        dur   = g['time_from_event_start'].max()
        event = 0
    ttc_mean = g['TTC_imputed'].mean()
    km1_rows.append({'duration': dur, 'event': event, 'TTC_mean': ttc_mean})
km1_df = pd.DataFrame(km1_rows)
med_ttc1 = km1_df['TTC_mean'].median()
km1_df['group'] = (km1_df['TTC_mean'] >= med_ttc1).astype(int)

fig, ax = plt.subplots(figsize=(9, 6))
fig.suptitle('Seg 1 — מבחן הישרדות (Kaplan-Meier)\nחלוקה: TTC ממוצע בסגמנט (חציון)', fontsize=12, fontweight='bold')
p1 = km_plot(ax, km1_df['duration'], km1_df['event'], km1_df['group'],
             title=f'TTC חציון={med_ttc1:.1f}s  |  n={len(km1_df)} אירועים')
plt.tight_layout()
plt.savefig(os.path.join(OUT,'fig3_seg1_km.png'), dpi=140, bbox_inches='tight')
plt.close(); print('fig3 saved')

# ═══════════════════════════════════════════════════════════════
# FIG 4 — Seg 2A: Logistic event-level (TTC at yellow)
# ═══════════════════════════════════════════════════════════════
ttc_yel  = (df[df['yellow_transition']==1].sort_values('start_time')
            .groupby('event_id').first()[['TTC_imputed']]
            .rename(columns={'TTC_imputed':'TTC_at_yellow'}))
seg2_fb  = (df[df['segment']==2.0].groupby('event_id')['first_feedback_relavet_to_event']
            .max().rename('feedback_in_seg2'))
ev2      = ttc_yel.join(seg2_fb, how='inner').dropna(subset=['TTC_at_yellow'])
X2a      = ev2[['TTC_at_yellow']].values
y2a      = ev2['feedback_in_seg2'].fillna(0).values
r2, p2   = stats.pointbiserialr(y2a, X2a.ravel())
print('\nSeg 2A:')
m2a, auc2a = fit_logistic(X2a, y2a, 'Seg2A')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Seg 2A — החלטת הצהוב (רמת אירוע)\nמנבא: TTC ברגע המעבר לצהוב', fontsize=12, fontweight='bold')
ax = axes[0]
if m2a:
    tl2 = np.linspace(ev2['TTC_at_yellow'].min(), ev2['TTC_at_yellow'].max(), 300).reshape(-1,1)
    ax.scatter(ev2['TTC_at_yellow'], y2a+np.random.uniform(-0.04,0.04,len(y2a)),
               alpha=0.55, s=40, color='#ee854a')
    ax.plot(tl2, m2a.predict_proba(tl2)[:,1], color='crimson', lw=2.5, label=f'AUC={auc2a:.2f}')
    ax.set_xlabel('TTC בצהוב (שניות)'); ax.set_ylabel('P(פידבק בסגמנט 2)')
    ax.set_title(f'n={len(ev2)} אירועים\nr={r2:.3f}  p={p2:.3f}')
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)
ax = axes[1]
ax.hist(ev2.loc[y2a==0,'TTC_at_yellow'], bins=20, alpha=0.6, color='#ee854a', density=True,
        label=f'ללא פידבק (n={(y2a==0).sum()})')
ax.hist(ev2.loc[y2a==1,'TTC_at_yellow'], bins=15, alpha=0.85, color='crimson', density=True,
        label=f'פידבק (n={y2a.sum()})')
ax.set_xlabel('TTC בצהוב (שניות)'); ax.set_ylabel('צפיפות')
ax.set_title('התפלגות TTC בצהוב לפי קבוצה')
ax.legend(); ax.grid(True, ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT,'fig4_seg2a_logistic.png'), dpi=140, bbox_inches='tight')
plt.close(); print('fig4 saved')

# ═══════════════════════════════════════════════════════════════
# FIG 5 — Seg 2B: Logistic window-level (Speed + time from yellow)
# ═══════════════════════════════════════════════════════════════
seg2 = df[df['segment']==2.0].copy()
X2b  = seg2[['Speed_mean','time_from_seg_start']].values
y2b  = seg2['first_feedback_relavet_to_event'].values
print('\nSeg 2B:')
m2b, auc2b = fit_logistic(X2b, y2b, 'Seg2B')

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('Seg 2B — חלון-חלון בסגמנט 2\nמנבאים: מהירות + זמן מהצהוב', fontsize=12, fontweight='bold')
ax = axes[0]
if m2b:
    t_med2 = np.median(seg2['time_from_seg_start'])
    spd_r  = np.linspace(seg2['Speed_mean'].min(), seg2['Speed_mean'].max(), 300)
    grid   = np.column_stack([spd_r, np.full(300, t_med2)])
    ax.scatter(seg2['Speed_mean'], y2b+np.random.uniform(-0.03,0.03,len(y2b)),
               alpha=0.2, s=15, color='#ee854a')
    ax.plot(spd_r, m2b.predict_proba(grid)[:,1], color='crimson', lw=2.5, label=f'AUC={auc2b:.2f}')
    ax.set_xlabel('מהירות ממוצעת (m/s)'); ax.set_ylabel('P(פידבק)')
    ax.set_title(f'השפעת מהירות (זמן={t_med2:.0f}s קבוע)\nCoef={m2b.coef_[0][0]:.3f}')
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)
ax = axes[1]
if m2b:
    spd_med2 = np.median(seg2['Speed_mean'])
    time_r2  = np.linspace(seg2['time_from_seg_start'].min(), seg2['time_from_seg_start'].max(), 300)
    grid2    = np.column_stack([np.full(300, spd_med2), time_r2])
    ax.scatter(seg2['time_from_seg_start'], y2b+np.random.uniform(-0.03,0.03,len(y2b)),
               alpha=0.2, s=15, color='#ee854a')
    ax.plot(time_r2, m2b.predict_proba(grid2)[:,1], color='darkorange', lw=2.5, label=f'AUC={auc2b:.2f}')
    ax.set_xlabel('זמן מהמעבר לצהוב (שניות)'); ax.set_ylabel('P(פידבק)')
    ax.set_title(f'השפעת זמן (מהירות={spd_med2:.1f} קבועה)\nCoef={m2b.coef_[0][1]:.3f}')
    ax.legend(); ax.grid(True, ls='--', alpha=0.4)
ax = axes[2]
ax.hist(seg2.loc[y2b==0,'Speed_mean'], bins=25, alpha=0.6, color='#ee854a', density=True,
        label=f'ללא פידבק (n={(y2b==0).sum()})')
ax.hist(seg2.loc[y2b==1,'Speed_mean'], bins=12, alpha=0.85, color='crimson', density=True,
        label=f'פידבק (n={y2b.sum()})')
ax.set_xlabel('מהירות ממוצעת (m/s)'); ax.set_ylabel('צפיפות')
ax.set_title('התפלגות מהירות לפי קבוצה')
ax.legend(); ax.grid(True, ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT,'fig5_seg2b_logistic.png'), dpi=140, bbox_inches='tight')
plt.close(); print('fig5 saved')

# ═══════════════════════════════════════════════════════════════
# FIG 6 — Seg 2: Kaplan-Meier (TTC at yellow, median split)
# ═══════════════════════════════════════════════════════════════
# Duration = time from yellow until feedback (or end of seg 2)
seg2_times = df[df['segment']==2.0].groupby('event_id')['time_from_seg_start'].max().rename('seg2_duration')
ev2_km = ev2.join(seg2_times)
ev2_km['duration'] = np.nan
for ev_id, g in df[df['segment']==2.0].groupby('event_id'):
    fb = g[g['first_feedback_relavet_to_event']==1]
    if len(fb):
        ev2_km.loc[ev_id,'duration'] = fb['time_from_seg_start'].iloc[0]
        ev2_km.loc[ev_id,'event_km'] = 1
    else:
        ev2_km.loc[ev_id,'duration'] = g['time_from_seg_start'].max()
        ev2_km.loc[ev_id,'event_km'] = 0
ev2_km = ev2_km.dropna(subset=['duration','TTC_at_yellow'])
med_ttc2 = ev2_km['TTC_at_yellow'].median()
ev2_km['group'] = (ev2_km['TTC_at_yellow'] >= med_ttc2).astype(int)

fig, ax = plt.subplots(figsize=(9, 6))
fig.suptitle('Seg 2 — מבחן הישרדות (Kaplan-Meier)\nחלוקה: TTC בצהוב (חציון)', fontsize=12, fontweight='bold')
km_plot(ax, ev2_km['duration'], ev2_km['event_km'], ev2_km['group'],
        title=f'TTC חציון={med_ttc2:.1f}s  |  n={len(ev2_km)} אירועים',
        xlabel='זמן מהמעבר לצהוב (שניות)')
plt.tight_layout()
plt.savefig(os.path.join(OUT,'fig6_seg2_km.png'), dpi=140, bbox_inches='tight')
plt.close(); print('fig6 saved')

# ═══════════════════════════════════════════════════════════════
# FIG 7 — Seg 3: Logistic (Speed + time, split by entry color)
# ═══════════════════════════════════════════════════════════════
seg3 = df[df['segment']==3.0].copy()
entry_col = (df[df['current_phase']=='InsideJunction'].sort_values('start_time')
             .groupby('event_id').first()[['TrafficLight_current']]
             .rename(columns={'TrafficLight_current':'entry_light'}))
seg3 = seg3.merge(entry_col, on='event_id', how='left')
seg3['entry_group'] = seg3['entry_light'].map(lambda x: 'Green' if x=='Green' else 'Yellow/Red')
seg3a = seg3[seg3['entry_group']=='Green']
seg3b = seg3[seg3['entry_group']=='Yellow/Red']

print('\nSeg 3:')
res3 = {}
for sub, label, col in [(seg3a,'3a-ירוק','#9467bd'),(seg3b,'3b-צהוב/אדום','#c44e52')]:
    X = sub[['Speed_mean','time_from_seg_start']].values
    y = sub['first_feedback_relavet_to_event'].values
    m, auc = fit_logistic(X, y, label)
    res3[label] = (sub, m, auc, y, col)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Seg 3 — חציית הצומת\nמנבאים: מהירות + זמן מתחילת הסגמנט', fontsize=12, fontweight='bold')
for row_idx, (label, (sub, model, auc, y_sub, color)) in enumerate(res3.items()):
    ax_spd  = axes[row_idx][0]
    ax_time = axes[row_idx][1]
    n_fb = int(y_sub.sum())
    if model is None:
        ax_spd.text(0.5,0.5,'נתונים לא מספיקים',ha='center',va='center',transform=ax_spd.transAxes)
        ax_time.text(0.5,0.5,'נתונים לא מספיקים',ha='center',va='center',transform=ax_time.transAxes)
        continue
    t_med = np.median(sub['time_from_seg_start'])
    spd_r = np.linspace(sub['Speed_mean'].min(), sub['Speed_mean'].max(), 300)
    grid  = np.column_stack([spd_r, np.full(300, t_med)])
    ax_spd.scatter(sub['Speed_mean'], y_sub+np.random.uniform(-0.03,0.03,len(y_sub)),
                   alpha=0.2, s=15, color=color)
    ax_spd.plot(spd_r, model.predict_proba(grid)[:,1], color='crimson', lw=2.5, label=f'AUC={auc:.2f}')
    ax_spd.set_xlabel('מהירות (m/s)'); ax_spd.set_ylabel('P(פידבק)')
    ax_spd.set_title(f'{label} — השפעת מהירות\n(n={len(sub)} חלונות, {n_fb} פידבק)')
    ax_spd.legend(); ax_spd.grid(True, ls='--', alpha=0.4)

    spd_med = np.median(sub['Speed_mean'])
    time_r  = np.linspace(sub['time_from_seg_start'].min(), sub['time_from_seg_start'].max(), 300)
    grid2   = np.column_stack([np.full(300, spd_med), time_r])
    ax_time.scatter(sub['time_from_seg_start'], y_sub+np.random.uniform(-0.03,0.03,len(y_sub)),
                    alpha=0.2, s=15, color=color)
    ax_time.plot(time_r, model.predict_proba(grid2)[:,1], color='darkorange', lw=2.5, label=f'AUC={auc:.2f}')
    ax_time.set_xlabel('זמן מתחילת הסגמנט (שניות)'); ax_time.set_ylabel('P(פידבק)')
    ax_time.set_title(f'{label} — השפעת זמן\n(מהירות={spd_med:.1f} קבועה)')
    ax_time.legend(); ax_time.grid(True, ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT,'fig7_seg3_logistic.png'), dpi=140, bbox_inches='tight')
plt.close(); print('fig7 saved')

# ═══════════════════════════════════════════════════════════════
# FIG 8 — Seg 3: Kaplan-Meier (Speed median split)
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Seg 3 — מבחן הישרדות (Kaplan-Meier)\nחלוקה: מהירות ממוצעת בסגמנט (חציון)', fontsize=12, fontweight='bold')

for ax, (sub, label) in zip(axes, [(seg3a,'3a — כניסה בירוק'),(seg3b,'3b — כניסה בצהוב/אדום')]):
    km_rows = []
    for ev_id, g in sub.groupby('event_id'):
        g     = g.sort_values('start_time')
        fb    = g[g['first_feedback_relavet_to_event']==1]
        dur   = fb['time_from_seg_start'].iloc[0] if len(fb) else g['time_from_seg_start'].max()
        evnt  = 1 if len(fb) else 0
        spd   = g['Speed_mean'].mean()
        km_rows.append({'duration': dur, 'event': evnt, 'Speed': spd})
    km_df = pd.DataFrame(km_rows)
    if len(km_df) < 3:
        ax.text(0.5,0.5,'נתונים לא מספיקים',ha='center',va='center',transform=ax.transAxes)
        continue
    med_spd = km_df['Speed'].median()
    km_df['group'] = (km_df['Speed'] >= med_spd).astype(int)
    km_plot(ax, km_df['duration'], km_df['event'], km_df['group'],
            title=f'{label}\nמהירות חציון={med_spd:.1f} m/s  |  n={len(km_df)} אירועים',
            xlabel='זמן מתחילת הסגמנט (שניות)')
plt.tight_layout()
plt.savefig(os.path.join(OUT,'fig8_seg3_km.png'), dpi=140, bbox_inches='tight')
plt.close(); print('fig8 saved')

# ═══════════════════════════════════════════════════════════════
# FIG 9 — AUC Summary
# ═══════════════════════════════════════════════════════════════
m3a_data = res3.get('3a-ירוק', (None,None,np.nan,None,None))
m3b_data = res3.get('3b-צהוב/אדום', (None,None,np.nan,None,None))

names  = ['Seg 1\nTTC + זמן\nמאירוע','Seg 2A\nTTC בצהוב\n(אירוע)','Seg 2B\nמהירות + זמן\nמצהוב',
          'Seg 3a\nמהירות + זמן\n(כניסה ירוק)','Seg 3b\nמהירות + זמן\n(כניסה צהוב/אדום)']
aucs   = [auc1, auc2a, auc2b, m3a_data[2], m3b_data[2]]
colors = ['#4878d0','#ee854a','#ee854a','#9467bd','#c44e52']
ns     = [len(seg1), len(ev2), len(seg2), len(seg3a), len(seg3b)]
fbs    = [int(y1.sum()), int(y2a.sum()), int(y2b.sum()),
          int(m3a_data[3].sum()) if m3a_data[3] is not None else 0,
          int(m3b_data[3].sum()) if m3b_data[3] is not None else 0]

fig, ax = plt.subplots(figsize=(13, 6))
valid  = [(n,a,c,ns_,fb) for n,a,c,ns_,fb in zip(names,aucs,colors,ns,fbs) if not np.isnan(a)]
bars   = ax.bar([v[0] for v in valid], [v[1] for v in valid],
                color=[v[2] for v in valid], alpha=0.85, edgecolor='white', lw=1.5, width=0.55)
ax.axhline(0.5, color='grey', ls='--', lw=1.5, label='סיכוי אקראי (AUC=0.5)')
for bar, (n,a,c,ns_,fb) in zip(bars, valid):
    ax.text(bar.get_x()+bar.get_width()/2, a+0.01, f'AUC={a:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(bar.get_x()+bar.get_width()/2, 0.28, f'n={ns_}\n{fb} פידבק',
            ha='center', va='center', fontsize=8.5, color='white', fontweight='bold')
ax.set_ylim(0, 0.9)
ax.set_ylabel('AUC-ROC', fontsize=12)
ax.set_title('סיכום ביצועי המודלים לפי סגמנט\n(Logistic Regression, class_weight=balanced)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, axis='y', ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUT,'fig9_auc_summary.png'), dpi=140, bbox_inches='tight')
plt.close(); print('fig9 saved')

print('\n=== ALL DONE ===')
