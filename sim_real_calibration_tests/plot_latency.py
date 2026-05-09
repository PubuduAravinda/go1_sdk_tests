#!/usr/bin/env python3
"""
Go1 Test 3 — Control Latency Plots (Academic Final)
=====================================================
Three figures from calib_latency_*.npz:

  Fig 1 — Spike response traces: 3×4 grid, all joints
           All trials overlaid (grey), mean trace (colour).
           Vertical line = detected latency per trial.

  Fig 2 — Latency distributions: 3×4 grid
           Box+strip plot per joint. RL_th shown with
           non-detection markers.

  Fig 3 — Summary: mean latency + alpha per joint
           Bar chart with group shading. RL_th annotated.

Usage:
  python3 go1_test3_latency_plots.py calib_latency_*.npz
  python3 go1_test3_latency_plots.py calib_latency_*.npz --save
"""
import os, sys, argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

JNAMES   = ['FL_hip','FR_hip','RL_hip','RR_hip',
            'FL_th', 'FR_th', 'RL_th', 'RR_th',
            'FL_kn', 'FR_kn', 'RL_kn', 'RR_kn']
GROUPS   = {'Hip':[0,1,2,3],'Thigh':[4,5,6,7],'Knee':[8,9,10,11]}
GCOLS    = {'Hip':'#2166ac','Thigh':'#4dac26','Knee':'#d6604d'}
GBG      = {'Hip':'#eef4fb','Thigh':'#f0f9f0','Knee':'#fdf0ef'}
FAULT_J  = 'RL_th'   # known fault joint
FAULT_COL= '#cc4400'

def gcol(i):
    if i<4: return GCOLS['Hip']
    if i<8: return GCOLS['Thigh']
    return GCOLS['Knee']

def gname(i):
    if i<4: return 'Hip'
    if i<8: return 'Thigh'
    return 'Knee'

def alpha_from_lat(lat_ms, dt_policy_s=0.02):
    return dt_policy_s / (dt_policy_s + lat_ms/1000.0)


# ── Figure 1: Spike response traces ──────────────────────────────────────────
def figure1_traces(d, save=False, prefix='latency'):
    SPIKE_STEPS = int(d['spike_steps'][0])
    DT_MS       = float(d['dt_ms'][0])
    SPIKE_AMP   = d['spike_amp']
    N_TOTAL     = d['FL_hip_q_trials'].shape[1]
    t_axis      = np.arange(N_TOTAL) * DT_MS   # ms

    fig, axes = plt.subplots(3, 4, figsize=(22,12), sharey=False)
    fig.suptitle(
        'Go1 Test 3 — Joint Step-Response Traces (All Trials)\n'
        'Position mean-subtracted per trial (baseline = 0). '
        'Grey lines = individual trials. Colour line = trial mean.\n'
        'Vertical dashed lines = detected first-motion latency per trial. '
        'Spike command active during shaded region (0\u201340\,ms).',
        fontsize=10.5, fontweight='bold')

    for row, gn in enumerate(['Hip','Thigh','Knee']):
        for col, ji in enumerate(GROUPS[gn]):
            ax  = axes[row][col]
            jn  = JNAMES[ji]
            c   = gcol(ji)
            is_fault = (jn == FAULT_J)
            col_use  = FAULT_COL if is_fault else c

            q_trials = d[f'{jn}_q_trials']    # (n_trials, n_steps)
            lats     = d[f'{jn}_lat_all_ms']
            amp      = float(SPIKE_AMP[ji])
            n_trials = q_trials.shape[0]

            # mean-subtract each trial by its own pre-spike baseline
            q_norm = np.zeros_like(q_trials)
            for ti in range(n_trials):
                bl = q_trials[ti, :20].mean()
                q_norm[ti] = q_trials[ti] - bl

            # spike region shade
            ax.axvspan(0, SPIKE_STEPS*DT_MS,
                       alpha=0.10, color=col_use, zorder=0)

            # individual trials (grey)
            for ti in range(n_trials):
                ax.plot(t_axis, q_norm[ti],
                        lw=0.6, color='#bbbbbb', alpha=0.70, zorder=1)

            # mean trace
            ax.plot(t_axis, q_norm.mean(axis=0),
                    lw=2.0, color=col_use, alpha=0.95, zorder=4,
                    label='Mean trace')

            # detected latency lines
            for lat in lats:
                ax.axvline(lat, color=col_use, lw=0.8,
                           ls='--', alpha=0.55, zorder=3)

            # zero / baseline
            ax.axhline(0, color='black', lw=1.0, ls='-', alpha=0.4)

            # stats box
            if len(lats) > 0:
                hit_pct = len(lats)/n_trials*100
                box_txt = (f'lat = {lats.mean():.1f}\u00b1{lats.std():.1f} ms\n'
                           f'\u03b1 = {alpha_from_lat(lats.mean()):.3f}\n'
                           f'hit = {hit_pct:.0f}%  ({len(lats)}/{n_trials})')
            else:
                box_txt = 'NO DETECTIONS'
            ec = FAULT_COL if is_fault else col_use
            ax.text(0.97, 0.97, box_txt,
                    transform=ax.transAxes, va='top', ha='right',
                    fontsize=7.5, color=ec, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.22',
                              fc='white', ec=ec, alpha=0.90))

            if is_fault:
                ax.text(0.50, 0.50, '⚠ RL FAULT\nSTICTION',
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=9, color=FAULT_COL, fontweight='bold',
                        alpha=0.40)

            ax.set_title(jn, fontsize=10.5,
                         color=FAULT_COL if is_fault else c,
                         fontweight='bold')
            ax.grid(True, alpha=0.18, axis='y')
            if row==2: ax.set_xlabel('Time from command (ms)', fontsize=8)
            if col==0: ax.set_ylabel(f'{gn}\n\u0394q (rad)', fontsize=9)

    legend_els = [
        Line2D([0],[0], color='#bbbbbb', lw=1.5, label='Individual trials'),
        Line2D([0],[0], color=GCOLS['Hip'], lw=2.0,
               label='Mean trace (Hip / Thigh / Knee in group colour)'),
        Line2D([0],[0], color='grey', lw=1.0, ls='--',
               label='Detected latency per trial'),
        mpatches.Patch(color='grey', alpha=0.15, label='Spike command window (0\u201340\,ms)'),
        Line2D([0],[0], color=FAULT_COL, lw=2.0, label='RL\_th (fault — stiction)'),
    ]
    fig.legend(handles=legend_els, loc='lower center', ncol=5,
               fontsize=8.5, bbox_to_anchor=(0.5,-0.01),
               framealpha=0.95, edgecolor='grey')
    plt.tight_layout(rect=[0,0.04,1,1])
    if save:
        out = f'{prefix}_fig1_traces.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'  Saved {out}')
    return fig


# ── Figure 2: Latency distributions per joint ─────────────────────────────────
def figure2_distributions(d, save=False, prefix='latency'):
    fig, axes = plt.subplots(3, 4, figsize=(22, 11))
    fig.suptitle(
        'Go1 Test 3 — Latency Distribution per Joint (20 Trials)\n'
        'Each circle = one trial latency. Box = IQR. Whiskers = 5th\u201395th percentile.\n'
        'Non-detected trials (RL\_th) shown as crosses at the time limit.',
        fontsize=10.5, fontweight='bold')

    N_TRIALS = 10
    SPIKE_STEPS = int(d['spike_steps'][0])
    DT_MS       = float(d['dt_ms'][0])
    TIME_LIMIT  = SPIKE_STEPS * DT_MS + 200*DT_MS   # max window

    for row, gn in enumerate(['Hip','Thigh','Knee']):
        for col, ji in enumerate(GROUPS[gn]):
            ax  = axes[row][col]
            jn  = JNAMES[ji]
            c   = gcol(ji)
            is_fault = (jn == FAULT_J)
            col_use  = FAULT_COL if is_fault else c

            lats     = d[f'{jn}_lat_all_ms']
            n_det    = len(lats)
            n_nodet  = N_TRIALS - n_det

            # box plot for detected
            if n_det >= 2:
                bp = ax.boxplot([lats],
                                positions=[0], widths=0.4,
                                patch_artist=True,
                                medianprops=dict(color='white', lw=2.0),
                                boxprops=dict(facecolor=col_use, alpha=0.35,
                                              edgecolor=col_use, lw=1.5),
                                whiskerprops=dict(color=col_use, lw=1.2),
                                capprops=dict(color=col_use, lw=1.5),
                                flierprops=dict(marker='', alpha=0),
                                showfliers=False)

            # strip plot — individual detections
            jitter = np.random.uniform(-0.12, 0.12, size=n_det)
            ax.scatter(jitter, lats,
                       s=40, color=col_use, alpha=0.80,
                       edgecolors='white', linewidths=0.8,
                       zorder=5)

            # non-detections as crosses at time limit
            if n_nodet > 0:
                jitter_nd = np.random.uniform(-0.12, 0.12, size=n_nodet)
                ax.scatter(jitter_nd,
                           np.full(n_nodet, TIME_LIMIT * 0.95),
                           s=60, marker='x', color='#cc0000',
                           linewidths=1.5, zorder=5,
                           label=f'{n_nodet} non-detected')
                ax.axhline(TIME_LIMIT*0.95, color='#cc0000',
                           lw=0.8, ls=':', alpha=0.5)
                ax.text(0.5, 0.88, f'{n_nodet} non-detected\n(stiction)',
                        transform=ax.transAxes, ha='center',
                        fontsize=8, color='#cc0000', fontweight='bold',
                        bbox=dict(boxstyle='round', fc='white',
                                  ec='#cc0000', alpha=0.85))

            # mean and alpha annotation
            if n_det > 0:
                mean_l = lats.mean()
                alph   = alpha_from_lat(mean_l)
                ax.axhline(mean_l, color=col_use,
                           lw=1.5, ls='--', alpha=0.7)
                ax.text(0.03, 0.97,
                        f'\u03bc = {mean_l:.1f} ms\n'
                        f'\u03c3 = {lats.std():.1f} ms\n'
                        f'\u03b1 = {alph:.3f}',
                        transform=ax.transAxes, va='top',
                        fontsize=8, color=col_use, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.22',
                                  fc='white', ec=col_use, alpha=0.90))

            ax.set_xlim(-0.5, 0.5)
            ax.set_xticks([])
            ax.set_title(jn, fontsize=10.5,
                         color=FAULT_COL if is_fault else c,
                         fontweight='bold')
            ax.grid(True, alpha=0.20, axis='y')
            if row==2: ax.set_xlabel('', fontsize=8)
            if col==0: ax.set_ylabel(f'{gn}\nLatency (ms)', fontsize=9)
            ax.set_facecolor(GBG[gn])

    legend_els = [
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor='#2166ac', ms=8,
               label='Detected trial latency'),
        mpatches.Patch(facecolor='#2166ac', alpha=0.35,
                       label='IQR box'),
        Line2D([0],[0], color='#2166ac', lw=1.5, ls='--',
               label='Mean latency'),
        Line2D([0],[0], marker='x', color='#cc0000',
               ms=8, lw=1.5, label='Non-detected trial (RL\_th stiction)'),
    ]
    fig.legend(handles=legend_els, loc='lower center', ncol=4,
               fontsize=9, bbox_to_anchor=(0.5,-0.01),
               framealpha=0.95, edgecolor='grey')
    plt.tight_layout(rect=[0,0.05,1,1])
    if save:
        out = f'{prefix}_fig2_distributions.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'  Saved {out}')
    return fig


# ── Figure 3: Summary bar chart ───────────────────────────────────────────────
def figure3_summary(d, save=False, prefix='latency'):
    JNAMES_PLOT = JNAMES
    x        = np.arange(12)
    bar_cols = [gcol(i) for i in range(12)]

    means  = []
    stds   = []
    alphas = []
    hits   = []

    for jn in JNAMES:
        lats = d[f'{jn}_lat_all_ms']
        if len(lats) > 0:
            means.append(lats.mean())
            stds.append(lats.std())
            alphas.append(alpha_from_lat(lats.mean()))
            hits.append(len(lats)/10*100)
        else:
            means.append(np.nan); stds.append(np.nan)
            alphas.append(np.nan); hits.append(0.0)

    means  = np.array(means)
    stds   = np.array(stds)
    alphas = np.array(alphas)

    fig, (ax_lat, ax_alp) = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle(
        'Go1 Test 3 — Latency and Derived \u03b1 per Joint\n'
        'Latency = time from command issue to first detectable encoder response (4\u03c3 threshold).\n'
        '\u03b1 = 0.02\,s / (0.02\,s + latency) — first-order lag filter coefficient at 50\,Hz policy rate.',
        fontsize=11, fontweight='bold')

    # group background
    for gn,(xlo,xhi) in [('Hip',(-0.6,3.6)),('Thigh',(3.6,7.6)),('Knee',(7.6,11.6))]:
        for ax in [ax_lat, ax_alp]:
            ax.axvspan(xlo, xhi, alpha=0.20, color=GBG[gn], zorder=0)
            ax.text((xlo+xhi)/2, 0, gn, ha='center', va='bottom',
                    fontsize=10, color=GCOLS[gn], fontweight='bold',
                    alpha=0.60, transform=ax.get_xaxis_transform())

    # latency bars
    bar_c = [FAULT_COL if JNAMES[i]==FAULT_J else bar_cols[i] for i in range(12)]
    bars = ax_lat.bar(x, means, color=bar_c, alpha=0.85,
                      edgecolor='white', lw=0.6)
    ax_lat.errorbar(x, means, yerr=stds, fmt='none',
                    ecolor='#333333', elinewidth=1.8, capsize=5)

    # healthy mean reference line
    healthy_idx = [i for i,jn in enumerate(JNAMES) if jn!=FAULT_J]
    healthy_mean = np.nanmean([means[i] for i in healthy_idx])
    ax_lat.axhline(healthy_mean, color='#333333', lw=1.5,
                   ls='--', alpha=0.7,
                   label=f'Healthy mean = {healthy_mean:.1f} ms')

    for i in range(12):
        if not np.isnan(means[i]):
            ax_lat.text(i, means[i]+stds[i]+0.4,
                        f'{means[i]:.1f}', ha='center', va='bottom',
                        fontsize=7.5, color=bar_c[i], fontweight='bold')

    # RL_th annotation
    rl_i = JNAMES.index('RL_th')
    ax_lat.annotate(f'RL\_th: {means[rl_i]:.0f}\u00b1{stds[rl_i]:.0f} ms\n40\% hit rate\n(stiction / gearbox fault)',
                    xy=(rl_i, means[rl_i]),
                    xytext=(rl_i+1.5, means[rl_i]+5),
                    arrowprops=dict(arrowstyle='->', color=FAULT_COL, lw=1.5),
                    fontsize=8.5, color=FAULT_COL, fontweight='bold',
                    bbox=dict(boxstyle='round', fc='lightyellow',
                              ec=FAULT_COL, alpha=0.90))

    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels(JNAMES, rotation=30, ha='right', fontsize=9.5)
    ax_lat.set_ylabel('Latency (ms)', fontsize=10)
    ax_lat.set_title('(A) Mean latency \u00b11\u03c3 per joint', fontsize=10.5, fontweight='bold')
    ax_lat.grid(True, alpha=0.25, axis='y')
    ax_lat.set_ylim(0, max(np.nanmax(means)+np.nanmax(stds), 45)+5)
    ax_lat.legend(fontsize=9, loc='upper left')

    # alpha bars
    alpha_c = [FAULT_COL if JNAMES[i]==FAULT_J else bar_cols[i] for i in range(12)]
    ax_alp.bar(x, alphas, color=alpha_c, alpha=0.85,
               edgecolor='white', lw=0.6)

    # healthy alpha reference
    healthy_alpha = np.nanmean([alphas[i] for i in healthy_idx])
    ax_alp.axhline(healthy_alpha, color='#333333', lw=1.5, ls='--', alpha=0.7,
                   label=f'Healthy mean \u03b1 = {healthy_alpha:.3f} (adopted)')
    ax_alp.axhline(1.0, color='black', lw=0.8, ls=':', alpha=0.4,
                   label='\u03b1 = 1.0 (zero lag)')

    for i in range(12):
        if not np.isnan(alphas[i]):
            ax_alp.text(i, alphas[i]+0.005,
                        f'{alphas[i]:.3f}', ha='center', va='bottom',
                        fontsize=7.5, color=alpha_c[i], fontweight='bold')

    ax_alp.annotate(f'RL\_th \u03b1 = {alphas[rl_i]:.3f}\n(high lag — not used)',
                    xy=(rl_i, alphas[rl_i]),
                    xytext=(rl_i+1.5, alphas[rl_i]-0.05),
                    arrowprops=dict(arrowstyle='->', color=FAULT_COL, lw=1.5),
                    fontsize=8.5, color=FAULT_COL, fontweight='bold',
                    bbox=dict(boxstyle='round', fc='lightyellow',
                              ec=FAULT_COL, alpha=0.90))

    ax_alp.set_xticks(x)
    ax_alp.set_xticklabels(JNAMES, rotation=30, ha='right', fontsize=9.5)
    ax_alp.set_ylabel('Lag filter coefficient \u03b1', fontsize=10)
    ax_alp.set_title('(B) Derived first-order lag coefficient \u03b1 = 0.02 / (0.02 + latency)',
                     fontsize=10.5, fontweight='bold')
    ax_alp.grid(True, alpha=0.25, axis='y')
    ax_alp.set_ylim(0, 1.05)
    ax_alp.legend(fontsize=9, loc='lower center', ncol=2)

    plt.tight_layout()
    if save:
        out = f'{prefix}_fig3_summary.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'  Saved {out}')
    return fig


# ── Console table ──────────────────────────────────────────────────────────────
def print_table(d):
    N_TRIALS = 10
    print()
    print('='*95)
    print('TEST 3 — CONTROL LATENCY RESULTS')
    print(f"Spike amp: hips={d['spike_amp'][0]:.2f}rad  "
          f"thighs={d['spike_amp'][4]:.2f}rad  knees={d['spike_amp'][8]:.2f}rad")
    print(f"Detection: 4\u03c3 first-crossing threshold. "
          f"\u03b1 = 0.02\u2009s / (0.02\u2009s + latency)")
    print('='*95)
    print(f"  {'Joint':10s}  {'Trials':>6}  {'Detect':>7}  {'Hit%':>5}  "
          f"{'Mean ms':>8}  {'Std ms':>7}  {'Med ms':>7}  {'P95 ms':>7}  "
          f"{'Alpha':>7}")
    print('  '+'─'*90)
    for ji, jn in enumerate(JNAMES):
        lats = d[f'{jn}_lat_all_ms']
        n_det = len(lats)
        hit = n_det/N_TRIALS*100
        if n_det>0:
            mean=lats.mean(); std=lats.std()
            med=np.median(lats); p95=np.percentile(lats,95)
            alph=alpha_from_lat(mean)
            note = '  \u2190 FAULT (stiction)' if jn==FAULT_J else ''
            print(f"  {jn:10s}  {N_TRIALS:>6d}  {n_det:>7d}  {hit:>5.0f}  "
                  f"{mean:>8.1f}  {std:>7.1f}  {med:>7.1f}  {p95:>7.1f}  "
                  f"{alph:>7.4f}{note}")
        else:
            print(f"  {jn:10s}  {N_TRIALS:>6d}  {n_det:>7d}  {hit:>5.0f}  "
                  f"{'nan':>8}  {'nan':>7}  {'nan':>7}  {'nan':>7}  {'nan':>7}")
    healthy = [jn for jn in JNAMES if jn!=FAULT_J]
    hmeans  = [d[f'{jn}_lat_all_ms'].mean() for jn in healthy]
    halpha  = alpha_from_lat(np.mean(hmeans))
    print()
    print(f"  Healthy mean latency: {np.mean(hmeans):.1f} ms  "
          f"range: {min(hmeans):.1f}\u2013{max(hmeans):.1f} ms")
    print(f"  Adopted \u03b1 = {halpha:.4f}  (from healthy mean {np.mean(hmeans):.1f} ms)")
    print(f"  RL_th: {d['RL_th_lat_all_ms'].mean():.1f} ms mean, "
          f"40% hit rate \u2192 \u03b1 = {alpha_from_lat(d['RL_th_lat_all_ms'].mean()):.4f} "
          f"(addressed by joint masking, not lag filter)")
    print('='*95)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('npz', type=str)
    p.add_argument('--save', action='store_true')
    p.add_argument('--outdir', type=str, default='.')
    args = p.parse_args()

    if args.save:
        matplotlib.use('Agg')
    else:
        try: matplotlib.use('TkAgg')
        except: matplotlib.use('Agg'); args.save=True

    d      = np.load(args.npz, allow_pickle=True)
    base   = os.path.basename(args.npz).replace('.npz','')
    os.makedirs(args.outdir, exist_ok=True)
    prefix = os.path.join(args.outdir, base)

    np.random.seed(42)
    print_table(d)
    print('\nGenerating figures...')
    figure1_traces(d,       save=args.save, prefix=prefix)
    figure2_distributions(d,save=args.save, prefix=prefix)
    figure3_summary(d,      save=args.save, prefix=prefix)

    if not args.save:
        plt.show()
    else:
        print(f'\nAll 3 figures saved to: {args.outdir}/')

if __name__=='__main__':
    main()