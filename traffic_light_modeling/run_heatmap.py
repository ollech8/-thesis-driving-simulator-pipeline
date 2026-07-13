import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings, os
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
df   = pd.read_csv(os.path.join(BASE, 'output', 'windows_1s_clean.csv'))

# ── assign segments ───────────────────────────────────────────────────────
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

# ── build non-overlapping windows ─────────────────────────────────────────
def build_windows(df, win_sec, horizon_sec):
    rows = []
    for ev_id, g in df.groupby('event_id'):
        g = g.sort_values('start_time').reset_index(drop=True)
        t_start = g['start_time'].min()
        t_end   = g['start_time'].max() + 1
        t = t_start
        while t + win_sec <= t_end:
            win_mask = (g['start_time'] >= t) & (g['start_time'] < t + win_sec)
            hor_mask = (g['start_time'] >= t + win_sec) & (g['start_time'] < t + win_sec + horizon_sec)
            win_rows = g[win_mask]
            hor_rows = g[hor_mask]
            if len(win_rows) == 0:
                t += win_sec; continue
            ttc_min  = win_rows['TTC_imputed'].min()
            spd_mean = win_rows['Speed_mean'].mean()
            target   = 1 if (len(hor_rows) > 0 and hor_rows['first_feedback_relavet_to_event'].max() == 1) else 0
            rows.append({'TTC_min': ttc_min, 'Speed_mean': spd_mean, 'target': target})
            t += win_sec
    return pd.DataFrame(rows)

# ── 5x5 grid ─────────────────────────────────────────────────────────────
wins     = [1, 2, 3, 4, 5]
horizons = [1, 2, 3, 4, 5]
auc_grid = np.full((5, 5), np.nan)
n_grid   = np.zeros((5, 5), dtype=int)
fb_grid  = np.zeros((5, 5), dtype=int)

for i, w in enumerate(wins):
    for j, h in enumerate(horizons):
        wdf  = build_windows(df, w, h)
        y    = wdf['target'].values
        n_grid[i,j]  = len(wdf)
        fb_grid[i,j] = y.sum()
        if y.sum() < 3 or (1-y).sum() < 3:
            continue
        X   = wdf[['TTC_min','Speed_mean']].fillna(6.0).values
        m   = LogisticRegression(class_weight='balanced', random_state=42, max_iter=500).fit(X, y)
        auc = roc_auc_score(y, m.predict_proba(X)[:,1])
        auc_grid[i,j] = auc
        print(f'win={w}s  hor={h}s  n={len(wdf)}  fb={y.sum()}  AUC={auc:.3f}')

# ── heatmap ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Window Size x Prediction Horizon Grid\n(Predictors: TTC + Speed, no overlap)',
             fontsize=13, fontweight='bold')

# AUC heatmap
ax = axes[0]
vmax = np.nanmax(auc_grid)
im   = ax.imshow(auc_grid, vmin=0.5, vmax=min(vmax+0.02, 1.0), cmap='RdYlGn', aspect='auto')
plt.colorbar(im, ax=ax, label='AUC-ROC')
ax.set_xticks(range(5)); ax.set_xticklabels([f'{h}s' for h in horizons])
ax.set_yticks(range(5)); ax.set_yticklabels([f'{w}s' for w in wins])
ax.set_xlabel('Prediction horizon (seconds ahead)', fontsize=12)
ax.set_ylabel('Window size', fontsize=12)
ax.set_title('AUC-ROC', fontsize=12, fontweight='bold')
for i in range(5):
    for j in range(5):
        v = auc_grid[i,j]
        if not np.isnan(v):
            ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='white' if v > (0.5 + (vmax-0.5)*0.6) else 'black')
        else:
            ax.text(j, i, 'N/A', ha='center', va='center', fontsize=9, color='grey')

# Feedback count heatmap
ax = axes[1]
im2 = ax.imshow(fb_grid, cmap='Blues', aspect='auto')
plt.colorbar(im2, ax=ax, label='Feedback windows')
ax.set_xticks(range(5)); ax.set_xticklabels([f'{h}s' for h in horizons])
ax.set_yticks(range(5)); ax.set_yticklabels([f'{w}s' for w in wins])
ax.set_xlabel('Prediction horizon (seconds ahead)', fontsize=12)
ax.set_ylabel('Window size', fontsize=12)
ax.set_title('Number of windows with feedback in horizon', fontsize=12, fontweight='bold')
for i in range(5):
    for j in range(5):
        ax.text(j, i, f'{fb_grid[i,j]}\n(n={n_grid[i,j]})',
                ha='center', va='center', fontsize=9,
                color='white' if fb_grid[i,j] > fb_grid.max()*0.6 else 'black')

plt.tight_layout()
plt.savefig(os.path.join(BASE, 'output', 'fig_heatmap_window_horizon.png'), dpi=140, bbox_inches='tight')
plt.close()
print('Heatmap saved.')
