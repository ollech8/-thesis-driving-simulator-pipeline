import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = r'C:\Users\ASUS VIVOBOOK\Desktop\אוניברסיטה\תואר שני\תזה\קוד\claude code\event relavent feedback\univariate_feedback_prediction\traffic_light_segmented_models.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)

def code(src): return {'cell_type':'code','metadata':{},'source':[src],'outputs':[],'execution_count':None}
def md(src):   return {'cell_type':'markdown','metadata':{},'source':[src]}

cells = nb['cells']

# Cell 1: imports — add lifelines
cells[1]['source'] = [
"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy import stats
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('output/windows_1s_clean.csv')
TTC_CEILING, TTC_NAN_FILL = 6.0, 999.0
SEG_COLORS = {1: '#4878d0', 2: '#ee854a', 3: '#9467bd'}
print(f'Loaded {len(df)} windows, {df[\"event_id\"].nunique()} events')"""
]

# Cell 8: Model 1 markdown
cells[8]['source'] = [
"""## Model 1 — Seg 1: התקרבות באור ירוק

**רמה:** חלון (שנייה-שנייה)
**מנבאים:** `TTC_imputed` + `time_from_event_start`
**מטרה:** האם יתן פידבק בחלון הזה?
**מודל:** Logistic Regression (`class_weight='balanced'`)
**מדד:** AUC-ROC"""
]

# Cell 9: Model 1 code
cells[9]['source'] = [
"""seg1   = df[df['segment'] == 1.0].copy()
X1     = seg1[['TTC_imputed', 'time_from_event_start']].values
y1     = seg1['first_feedback_relavet_to_event'].values

model1 = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
model1.fit(X1, y1)
auc1   = roc_auc_score(y1, model1.predict_proba(X1)[:, 1])

print(f'Seg 1  |  n={len(seg1)} windows  |  feedback={y1.sum()}  |  AUC={auc1:.3f}')
print(f'Coef TTC={model1.coef_[0][0]:.3f}   OR={np.exp(model1.coef_[0][0]):.3f}')
print(f'Coef time={model1.coef_[0][1]:.3f}  OR={np.exp(model1.coef_[0][1]):.3f}')"""
]

# Cell 10: Model 1 plots (3 subplots)
cells[10]['source'] = [
"""fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('Seg 1 — התקרבות באור ירוק', fontsize=12, fontweight='bold')

t_med   = np.median(seg1['time_from_event_start'])
ttc_med = np.median(seg1['TTC_imputed'])

# TTC effect
ax = axes[0]
ttc_r = np.linspace(seg1['TTC_imputed'].min(), seg1['TTC_imputed'].max(), 300)
grid  = np.column_stack([ttc_r, np.full(300, t_med)])
ax.scatter(seg1['TTC_imputed'], y1 + np.random.uniform(-0.03,0.03,len(y1)), alpha=0.2, s=15, color='#4878d0')
ax.plot(ttc_r, model1.predict_proba(grid)[:, 1], color='crimson', lw=2.5, label=f'AUC={auc1:.2f}')
ax.set_xlabel('TTC (s, cap 6s)'); ax.set_ylabel('P(feedback)')
ax.set_title(f'TTC effect (time={t_med:.0f}s fixed)\\nCoef={model1.coef_[0][0]:.3f}')
ax.legend(); ax.grid(True, ls='--', alpha=0.4)

# Time effect
ax = axes[1]
time_r = np.linspace(seg1['time_from_event_start'].min(), seg1['time_from_event_start'].max(), 300)
grid2  = np.column_stack([np.full(300, ttc_med), time_r])
ax.scatter(seg1['time_from_event_start'], y1 + np.random.uniform(-0.03,0.03,len(y1)), alpha=0.2, s=15, color='#4878d0')
ax.plot(time_r, model1.predict_proba(grid2)[:, 1], color='darkorange', lw=2.5, label=f'AUC={auc1:.2f}')
ax.set_xlabel('Time from event start (s)'); ax.set_ylabel('P(feedback)')
ax.set_title(f'Time effect (TTC={ttc_med:.1f}s fixed)\\nCoef={model1.coef_[0][1]:.3f}')
ax.legend(); ax.grid(True, ls='--', alpha=0.4)

# Distribution
ax = axes[2]
ax.hist(seg1.loc[y1==0,'TTC_imputed'], bins=30, alpha=0.6, color='#4878d0', density=True, label=f'No feedback (n={(y1==0).sum()})')
ax.hist(seg1.loc[y1==1,'TTC_imputed'], bins=10, alpha=0.85, color='crimson', density=True, label=f'Feedback (n={y1.sum()})')
ax.set_xlabel('TTC (s)'); ax.set_ylabel('Density'); ax.set_title('TTC distribution by class')
ax.legend(); ax.grid(True, ls='--', alpha=0.4)
plt.tight_layout(); plt.show()"""
]

# Cell 11: Model 2 markdown
cells[11]['source'] = [
"""## Model 2A — Seg 2: החלטת הצהוב (רמת אירוע)

**רמה:** אירוע (ערך יחיד לכל אירוע)
**מנבא:** `TTC_at_yellow`
**מטרה:** האם יתן פידבק בסגמנט 2?

## Model 2B — Seg 2: חלון-חלון

**רמה:** חלון (שנייה-שנייה)
**מנבאים:** `Speed_mean` + `time_from_seg_start` (מהצהוב)
**הסבר:** TTC לא מוגדר לאחר מעבר הרמזור — משתמשים במהירות"""
]

# Cell 12: Model 2A + 2B code
cells[12]['source'] = [
"""# Model 2A - event level
ttc_yel  = (df[df['yellow_transition']==1].sort_values('start_time')
            .groupby('event_id').first()[['TTC_imputed']]
            .rename(columns={'TTC_imputed':'TTC_at_yellow'}))
seg2_fb  = (df[df['segment']==2.0].groupby('event_id')['first_feedback_relavet_to_event']
            .max().rename('feedback_in_seg2'))
ev2      = ttc_yel.join(seg2_fb, how='inner').dropna(subset=['TTC_at_yellow'])
X2a      = ev2[['TTC_at_yellow']].values
y2a      = ev2['feedback_in_seg2'].fillna(0).values
r2, p2   = stats.pointbiserialr(y2a, X2a.ravel())
model2a  = LogisticRegression(class_weight='balanced', random_state=42).fit(X2a, y2a)
auc2a    = roc_auc_score(y2a, model2a.predict_proba(X2a)[:,1])
print(f'Seg 2A  |  n={len(ev2)} events  |  fb={y2a.sum()}  |  AUC={auc2a:.3f}  |  r={r2:.3f} p={p2:.3f}')

# Model 2B - window level
seg2    = df[df['segment']==2.0].copy()
X2b     = seg2[['Speed_mean','time_from_seg_start']].values
y2b     = seg2['first_feedback_relavet_to_event'].values
model2b = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000).fit(X2b, y2b)
auc2b   = roc_auc_score(y2b, model2b.predict_proba(X2b)[:,1])
print(f'Seg 2B  |  n={len(seg2)} windows  |  fb={y2b.sum()}  |  AUC={auc2b:.3f}')
print(f'  Coef speed={model2b.coef_[0][0]:.3f}  Coef time={model2b.coef_[0][1]:.3f}')"""
]

# Cell 13: Model 2 plots
cells[13]['source'] = [
"""fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Seg 2 — החלטת הצהוב', fontsize=13, fontweight='bold')

# 2A probability curve
ax = axes[0][0]
tl2 = np.linspace(ev2['TTC_at_yellow'].min(), ev2['TTC_at_yellow'].max(), 300).reshape(-1,1)
ax.scatter(ev2['TTC_at_yellow'], y2a + np.random.uniform(-0.04,0.04,len(y2a)), alpha=0.55, s=40, color='#ee854a')
ax.plot(tl2, model2a.predict_proba(tl2)[:,1], color='crimson', lw=2.5, label=f'AUC={auc2a:.2f}')
ax.set_xlabel('TTC at yellow (s)'); ax.set_ylabel('P(feedback in Seg 2)')
ax.set_title(f'2A — Event level: TTC at yellow\\n(n={len(ev2)} events, r={r2:.3f}, p={p2:.3f})')
ax.legend(); ax.grid(True, ls='--', alpha=0.4)

# 2A distribution
ax = axes[0][1]
ax.hist(ev2.loc[y2a==0,'TTC_at_yellow'], bins=20, alpha=0.6, color='#ee854a', density=True, label=f'No fb (n={(y2a==0).sum()})')
ax.hist(ev2.loc[y2a==1,'TTC_at_yellow'], bins=15, alpha=0.85, color='crimson', density=True, label=f'Feedback (n={y2a.sum()})')
ax.set_xlabel('TTC at yellow (s)'); ax.set_ylabel('Density'); ax.set_title('2A — TTC distribution')
ax.legend(); ax.grid(True, ls='--', alpha=0.4)

# 2B speed effect
t_med2 = np.median(seg2['time_from_seg_start'])
ax = axes[1][0]
spd_r  = np.linspace(seg2['Speed_mean'].min(), seg2['Speed_mean'].max(), 300)
ax.scatter(seg2['Speed_mean'], y2b + np.random.uniform(-0.03,0.03,len(y2b)), alpha=0.2, s=15, color='#ee854a')
ax.plot(spd_r, model2b.predict_proba(np.column_stack([spd_r, np.full(300,t_med2)]))[:,1],
        color='crimson', lw=2.5, label=f'AUC={auc2b:.2f}')
ax.set_xlabel('Speed (m/s)'); ax.set_ylabel('P(feedback)')
ax.set_title(f'2B — Window level: speed effect\\n(time={t_med2:.0f}s fixed)')
ax.legend(); ax.grid(True, ls='--', alpha=0.4)

# 2B time effect
spd_med2 = np.median(seg2['Speed_mean'])
ax = axes[1][1]
time_r2  = np.linspace(seg2['time_from_seg_start'].min(), seg2['time_from_seg_start'].max(), 300)
ax.scatter(seg2['time_from_seg_start'], y2b + np.random.uniform(-0.03,0.03,len(y2b)), alpha=0.2, s=15, color='#ee854a')
ax.plot(time_r2, model2b.predict_proba(np.column_stack([np.full(300,spd_med2), time_r2]))[:,1],
        color='darkorange', lw=2.5, label=f'AUC={auc2b:.2f}')
ax.set_xlabel('Time from yellow (s)'); ax.set_ylabel('P(feedback)')
ax.set_title(f'2B — Window level: time effect\\n(speed={spd_med2:.1f} fixed)')
ax.legend(); ax.grid(True, ls='--', alpha=0.4)
plt.tight_layout(); plt.show()"""
]

# Cell 14: Model 3 markdown
cells[14]['source'] = [
"""## Model 3 — Seg 3: חציית הצומת

**רמה:** חלון (שנייה-שנייה)
**מנבאים:** `Speed_mean` + `time_from_seg_start`
**חלוקה:** נפרד לכניסה בירוק ולכניסה בצהוב/אדום"""
]

# Cell 15: Model 3 setup
cells[15]['source'] = [
"""seg3      = df[df['segment'] == 3.0].copy()
entry_col = (df[df['current_phase']=='InsideJunction'].sort_values('start_time')
             .groupby('event_id').first()[['TrafficLight_current']]
             .rename(columns={'TrafficLight_current':'entry_light'}))
seg3 = seg3.merge(entry_col, on='event_id', how='left')
seg3['entry_group'] = seg3['entry_light'].map(lambda x: 'Green' if x=='Green' else 'Yellow/Red')
seg3a = seg3[seg3['entry_group']=='Green']
seg3b = seg3[seg3['entry_group']=='Yellow/Red']
print(f'Seg 3a (Green entry):     {len(seg3a)} windows, {seg3a["first_feedback_relavet_to_event"].sum()} feedback')
print(f'Seg 3b (Yellow/Red entry): {len(seg3b)} windows, {seg3b["first_feedback_relavet_to_event"].sum()} feedback')"""
]

# Cell 16: Model 3 fit
cells[16]['source'] = [
"""def fit_seg3(data, label):
    X = data[['Speed_mean', 'time_from_seg_start']].values
    y = data['first_feedback_relavet_to_event'].values
    if y.sum() == 0 or (1-y).sum() == 0:
        print(f'{label}: single class — cannot fit'); return None, np.nan, y
    m   = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000).fit(X, y)
    auc = roc_auc_score(y, m.predict_proba(X)[:,1])
    print(f'{label}  |  n={len(data)}  |  fb={y.sum()}  |  AUC={auc:.3f}')
    print(f'  Coef speed={m.coef_[0][0]:.3f}  Coef time={m.coef_[0][1]:.3f}')
    return m, auc, y

model3a, auc3a, y3a = fit_seg3(seg3a, 'Seg 3a — Green entry')
model3b, auc3b, y3b = fit_seg3(seg3b, 'Seg 3b — Yellow/Red entry')"""
]

# Cell 17: Model 3 plots
cells[17]['source'] = [
"""fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Seg 3 — Junction traversal\\nPredictors: Speed + time from segment start', fontsize=12, fontweight='bold')

for row_idx, (sub, model, auc, y_sub, label, color) in enumerate([
    (seg3a, model3a, auc3a, y3a, '3a — Green entry',      '#9467bd'),
    (seg3b, model3b, auc3b, y3b, '3b — Yellow/Red entry', '#c44e52'),
]):
    ax_s = axes[row_idx][0]; ax_t = axes[row_idx][1]
    if model is None:
        for ax in [ax_s, ax_t]:
            ax.text(0.5,0.5,'Insufficient data',ha='center',va='center',transform=ax.transAxes)
        continue
    t_med   = np.median(sub['time_from_seg_start'])
    spd_med = np.median(sub['Speed_mean'])
    spd_r   = np.linspace(sub['Speed_mean'].min(), sub['Speed_mean'].max(), 300)
    time_r  = np.linspace(sub['time_from_seg_start'].min(), sub['time_from_seg_start'].max(), 300)

    ax_s.scatter(sub['Speed_mean'], y_sub + np.random.uniform(-0.03,0.03,len(y_sub)), alpha=0.2, s=15, color=color)
    ax_s.plot(spd_r, model.predict_proba(np.column_stack([spd_r, np.full(300,t_med)]))[:,1],
              color='crimson', lw=2.5, label=f'AUC={auc:.2f}')
    ax_s.set_xlabel('Speed (m/s)'); ax_s.set_ylabel('P(feedback)')
    ax_s.set_title(f'{label} — speed effect\\n(n={len(sub)}, {int(y_sub.sum())} feedback)')
    ax_s.legend(); ax_s.grid(True, ls='--', alpha=0.4)

    ax_t.scatter(sub['time_from_seg_start'], y_sub + np.random.uniform(-0.03,0.03,len(y_sub)), alpha=0.2, s=15, color=color)
    ax_t.plot(time_r, model.predict_proba(np.column_stack([np.full(300,spd_med), time_r]))[:,1],
              color='darkorange', lw=2.5, label=f'AUC={auc:.2f}')
    ax_t.set_xlabel('Time from seg start (s)'); ax_t.set_ylabel('P(feedback)')
    ax_t.set_title(f'{label} — time effect\\n(speed={spd_med:.1f} fixed)')
    ax_t.legend(); ax_t.grid(True, ls='--', alpha=0.4)
plt.tight_layout(); plt.show()"""
]

# Cell 18: Summary markdown
cells[18]['source'] = ["## סיכום — ביצועי כל המודלים"]

# Cell 19: Summary code
cells[19]['source'] = [
"""print('=' * 70)
print('  SEGMENTED MODEL SUMMARY')
print('=' * 70)
rows = [
    ('Seg 1',  'TTC + time (event start)', len(seg1),  int(y1.sum()),  auc1),
    ('Seg 2A', 'TTC at yellow (event)',    len(ev2),   int(y2a.sum()), auc2a),
    ('Seg 2B', 'Speed + time (yellow)',    len(seg2),  int(y2b.sum()), auc2b),
    ('Seg 3a', 'Speed + time (green)',     len(seg3a), int(y3a.sum() if y3a is not None else 0), auc3a),
    ('Seg 3b', 'Speed + time (yel/red)',   len(seg3b), int(y3b.sum() if y3b is not None else 0), auc3b),
]
print(f'  {"Model":<8}  {"Predictors":<28}  {"n":<7}  {"FB":<5}  AUC')
print('  ' + '-' * 60)
for name, pred, n, fb, auc in rows:
    print(f'  {name:<8}  {pred:<28}  {n:<7}  {fb:<5}  {auc:.3f}' if not np.isnan(auc) else f'  {name:<8}  {pred:<28}  {n:<7}  {fb:<5}  N/A')

# AUC bar chart
valid = [(nm, a, c, fb) for nm,a,c,fb in zip(
    ['Seg1','Seg2A','Seg2B','Seg3a','Seg3b'],
    [auc1,auc2a,auc2b,auc3a,auc3b],
    ['#4878d0','#ee854a','#ee854a','#9467bd','#c44e52'],
    [int(y1.sum()),int(y2a.sum()),int(y2b.sum()),
     int(y3a.sum() if y3a is not None else 0),
     int(y3b.sum() if y3b is not None else 0)]
) if not np.isnan(a)]
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar([v[0] for v in valid],[v[1] for v in valid],color=[v[2] for v in valid],
              alpha=0.85, edgecolor='white', lw=1.5, width=0.55)
ax.axhline(0.5, color='grey', ls='--', lw=1.5, label='Chance (AUC=0.5)')
for bar,(nm,a,c,fb) in zip(bars,valid):
    ax.text(bar.get_x()+bar.get_width()/2, a+0.01, f'{a:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(bar.get_x()+bar.get_width()/2, 0.28, f'{fb} fb', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
ax.set_ylim(0,0.9); ax.set_ylabel('AUC-ROC', fontsize=11)
ax.set_title('Model performance by segment', fontsize=12, fontweight='bold')
ax.legend(); ax.grid(True, axis='y', ls='--', alpha=0.4)
plt.tight_layout(); plt.show()"""
]

# ── New cells to insert ────────────────────────────────────────────────────
time_md   = md(
"""## Step 3 — משתני זמן

- **`time_from_event_start`** — שניות מתחילת האירוע (Seg 1: הנהג לא יודע מתי יהיה צהוב)
- **`time_from_seg_start`** — שניות מתחילת הסגמנט (Seg 2/3: הטריגר הוא המעבר לצהוב / כניסה לצומת)"""
)
time_code = code(
"""df['time_from_event_start'] = df.groupby('event_id')['start_time'].transform(lambda x: x - x.min())
df['time_from_seg_start']   = np.nan
for _sid in [1.0, 2.0, 3.0]:
    _mask = df['segment'] == _sid
    df.loc[_mask, 'time_from_seg_start'] = (
        df[_mask].groupby('event_id')['start_time'].transform(lambda x: x - x.min())
    )
print('time_from_event_start range:', df['time_from_event_start'].min(), '—', df['time_from_event_start'].max())
print('time_from_seg_start NaN:', df['time_from_seg_start'].isna().sum())"""
)

km1_md   = md("### Kaplan-Meier — Seg 1\nחלוקה לפי TTC ממוצע בסגמנט (חציון): נמוך vs גבוה")
km1_code = code(
"""km1_rows = []
for ev_id, g in seg1.groupby('event_id'):
    g    = g.sort_values('start_time')
    fb   = g[g['first_feedback_relavet_to_event']==1]
    km1_rows.append({
        'duration': fb['time_from_event_start'].iloc[0] if len(fb) else g['time_from_event_start'].max(),
        'event':    1 if len(fb) else 0,
        'TTC_mean': g['TTC_imputed'].mean()
    })
km1_df = pd.DataFrame(km1_rows)
med1   = km1_df['TTC_mean'].median()
km1_df['group'] = (km1_df['TTC_mean'] >= med1).astype(int)

fig, ax = plt.subplots(figsize=(9, 6))
kmf = KaplanMeierFitter()
for g_val, color, lbl in [
    (0, '#2166ac', f'TTC low  (<{med1:.1f}s)'),
    (1, '#d6604d', f'TTC high (>={med1:.1f}s)')
]:
    mask = km1_df['group'] == g_val
    kmf.fit(km1_df.loc[mask,'duration'], event_observed=km1_df.loc[mask,'event'], label=lbl)
    kmf.plot_survival_function(ax=ax, ci_show=True, color=color)
res1_km = logrank_test(
    km1_df.loc[km1_df['group']==0,'duration'], km1_df.loc[km1_df['group']==1,'duration'],
    event_observed_A=km1_df.loc[km1_df['group']==0,'event'],
    event_observed_B=km1_df.loc[km1_df['group']==1,'event'])
ax.set_title(f'Seg 1 — Kaplan-Meier\\nLog-rank p={res1_km.p_value:.3f}  |  n={len(km1_df)} events', fontsize=11)
ax.set_xlabel('Time from event start (s)'); ax.set_ylabel('P(no feedback yet)')
ax.set_ylim(0,1.05); ax.grid(True, ls='--', alpha=0.4); ax.legend(title='Group')
plt.tight_layout(); plt.show()"""
)

km2_md   = md("### Kaplan-Meier — Seg 2\nחלוקה לפי TTC בצהוב (חציון)")
km2_code = code(
"""ev2_km = ev2.copy()
ev2_km['duration'] = np.nan; ev2_km['event_km'] = 0
for ev_id, g in df[df['segment']==2.0].groupby('event_id'):
    fb = g[g['first_feedback_relavet_to_event']==1]
    ev2_km.loc[ev_id,'duration'] = fb['time_from_seg_start'].iloc[0] if len(fb) else g['time_from_seg_start'].max()
    ev2_km.loc[ev_id,'event_km'] = 1 if len(fb) else 0
ev2_km = ev2_km.dropna(subset=['duration','TTC_at_yellow'])
med2   = ev2_km['TTC_at_yellow'].median()
ev2_km['group'] = (ev2_km['TTC_at_yellow'] >= med2).astype(int)

fig, ax = plt.subplots(figsize=(9, 6))
kmf = KaplanMeierFitter()
for g_val, color, lbl in [
    (0, '#2166ac', f'TTC low  (<{med2:.1f}s)'),
    (1, '#d6604d', f'TTC high (>={med2:.1f}s)')
]:
    mask = ev2_km['group'] == g_val
    kmf.fit(ev2_km.loc[mask,'duration'], event_observed=ev2_km.loc[mask,'event_km'], label=lbl)
    kmf.plot_survival_function(ax=ax, ci_show=True, color=color)
res2_km = logrank_test(
    ev2_km.loc[ev2_km['group']==0,'duration'], ev2_km.loc[ev2_km['group']==1,'duration'],
    event_observed_A=ev2_km.loc[ev2_km['group']==0,'event_km'],
    event_observed_B=ev2_km.loc[ev2_km['group']==1,'event_km'])
ax.set_title(f'Seg 2 — Kaplan-Meier\\nLog-rank p={res2_km.p_value:.3f}  |  n={len(ev2_km)} events', fontsize=11)
ax.set_xlabel('Time from yellow transition (s)'); ax.set_ylabel('P(no feedback yet)')
ax.set_ylim(0,1.05); ax.grid(True, ls='--', alpha=0.4); ax.legend(title='Group')
plt.tight_layout(); plt.show()"""
)

km3_md   = md("### Kaplan-Meier — Seg 3\nחלוקה לפי מהירות ממוצעת (חציון)")
km3_code = code(
"""fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Seg 3 — Kaplan-Meier  |  Split: median speed', fontsize=12, fontweight='bold')
for ax, (sub, label) in zip(axes, [(seg3a,'3a — Green entry'),(seg3b,'3b — Yellow/Red entry')]):
    km_rows = []
    for ev_id, g in sub.groupby('event_id'):
        g  = g.sort_values('start_time')
        fb = g[g['first_feedback_relavet_to_event']==1]
        km_rows.append({'duration': fb['time_from_seg_start'].iloc[0] if len(fb) else g['time_from_seg_start'].max(),
                        'event': 1 if len(fb) else 0, 'Speed': g['Speed_mean'].mean()})
    km_df = pd.DataFrame(km_rows)
    if len(km_df) < 3:
        ax.text(0.5,0.5,'Insufficient data',ha='center',va='center',transform=ax.transAxes); continue
    med_spd = km_df['Speed'].median()
    km_df['group'] = (km_df['Speed'] >= med_spd).astype(int)
    kmf = KaplanMeierFitter()
    for g_val, color, lbl in [(0,'#2166ac',f'Speed low (<{med_spd:.1f})'),(1,'#d6604d',f'Speed high (>={med_spd:.1f})')]:
        mask = km_df['group']==g_val
        kmf.fit(km_df.loc[mask,'duration'], event_observed=km_df.loc[mask,'event'], label=lbl)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color)
    res3_km = logrank_test(km_df.loc[km_df['group']==0,'duration'], km_df.loc[km_df['group']==1,'duration'],
                           event_observed_A=km_df.loc[km_df['group']==0,'event'],
                           event_observed_B=km_df.loc[km_df['group']==1,'event'])
    ax.set_title(f'{label}\\nLog-rank p={res3_km.p_value:.3f}  |  n={len(km_df)} events', fontsize=10)
    ax.set_xlabel('Time from seg start (s)'); ax.set_ylabel('P(no feedback yet)')
    ax.set_ylim(0,1.05); ax.grid(True, ls='--', alpha=0.4); ax.legend(title='Group', fontsize=8)
plt.tight_layout(); plt.show()"""
)

# ── Rebuild cell list ─────────────────────────────────────────────────────
new_cells = (
    cells[0:6]                              # 0-5: intro, imports, seg desc, assign, TTC md, TTC code
    + [time_md, time_code]                  # time variables
    + [cells[6], cells[7]]                  # timeline md + code
    + [cells[8], cells[9], cells[10]]       # Model 1
    + [km1_md, km1_code]                    # KM Seg 1
    + [cells[11], cells[12], cells[13]]     # Model 2
    + [km2_md, km2_code]                    # KM Seg 2
    + [cells[14], cells[15], cells[16], cells[17]]  # Model 3
    + [km3_md, km3_code]                    # KM Seg 3
    + [cells[18], cells[19]]                # Summary
)

nb['cells'] = new_cells
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f'Done — {len(new_cells)} cells saved')
