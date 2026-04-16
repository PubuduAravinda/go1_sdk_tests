#!/usr/bin/env python3
"""
Go1 Test 2 — Encoder Noise Calibration: Academic Plots
=======================================================
Four figures from a two-phase encoder .npz file:

  Fig 1 — Joint Position Time-Series: Commanded vs Actual (Ph0 + Ph1)
  Fig 2 — Joint Velocity Time-Series: Commanded vs Actual (Ph0 + Ph1)
  Fig 3 — Commanded vs Actual Position per Group (dot + errorbar)
  Fig 4 — Encoder Noise Sigma: Ph0 vs Ph1 (jpos and jvel)

Usage:
  python3 go1_test2_encoder_plots.py calib_encoder_*.npz
  python3 go1_test2_encoder_plots.py calib_encoder_*.npz --save
  python3 go1_test2_encoder_plots.py calib_encoder_*.npz --save --outdir ./plots
"""

import os, sys, argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Constants ─────────────────────────────────────────────────────────────────
JNAMES = ['FL_hip','FR_hip','RL_hip','RR_hip',
          'FL_th', 'FR_th', 'RL_th', 'RR_th',
          'FL_kn', 'FR_kn', 'RL_kn', 'RR_kn']

GROUPS   = {'Hip':   [0,1,2,3],
            'Thigh': [4,5,6,7],
            'Knee':  [8,9,10,11]}

GCOLS    = {'Hip':   '#2166ac',   # blue
            'Thigh': '#4dac26',   # green
            'Knee':  '#d6604d'}   # red

GBG      = {'Hip':   '#eef4fb',
            'Thigh': '#f0f9f0',
            'Knee':  '#fdf0ef'}

KP_TRAIN = [35]*4 + [65]*4 + [80]*4

PH0_COL  = '#888888'   # grey — Phase 0 hanging
PH0_BAND = '#cccccc'


def gcol(i):
    if i < 4: return GCOLS['Hip']
    if i < 8: return GCOLS['Thigh']
    return GCOLS['Knee']


def gname(i):
    if i < 4: return 'Hip'
    if i < 8: return 'Thigh'
    return 'Knee'


def clip_std(arr, n=3):
    """3-sigma clipped standard deviation."""
    m = np.abs(arr - arr.mean()) < n * arr.std()
    return float(arr[m].std())


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — jpos time-series: commanded vs actual, both phases
# ══════════════════════════════════════════════════════════════════════════════
def figure1_jpos(d, save=False, prefix='encoder'):
    """
    3×4 grid — one panel per joint.
    Y-axis: mean-subtracted position (rad) so both phases are centred
            at zero and the noise width is directly readable.
    Commanded = 0 after mean subtraction (shown as black dashed).
    Phase 0 (grey) and Phase 1 (group colour) plotted back-to-back
    with a thin separator line. No ramp/settle gap shown.
    ±1σ band per phase. Outliers (>3σ) shown as red dots.
    """
    ph0_jpos = d['ph0_jpos'];  ph1_jpos = d['ph1_jpos']
    ph0_time = d['ph0_time'];  ph1_time = d['ph1_time']

    t0_dur  = ph0_time[-1]
    GAP     = 1.5
    t1_plot = ph1_time + t0_dur + GAP

    sub0 = max(1, len(ph0_time)//3000)
    sub1 = max(1, len(ph1_time)//3000)

    fig, axes = plt.subplots(3, 4, figsize=(22, 12), sharey=False)
    fig.suptitle(
        'Go1 Test 2 — Joint Position Encoder Noise: Phase 0 (Hanging) and Phase 1 (Standing)\n'
        'Signal mean-subtracted per phase. Commanded target = 0 after subtraction (black dashed).\n'
        'Band = ±1\u03c3. Red markers = |signal| > 3\u03c3 (spike outliers).',
        fontsize=11, fontweight='bold')

    for row, gn in enumerate(['Hip','Thigh','Knee']):
        idxs = GROUPS[gn]
        gc   = GCOLS[gn]

        # shared y-range per row based on max sigma across group
        ymax_row = max(
            max(clip_std(ph0_jpos[:,i]) for i in idxs),
            max(clip_std(ph1_jpos[:,i]) for i in idxs)
        ) * 6

        for col, ji in enumerate(idxs):
            ax  = axes[row][col]
            c   = gcol(ji)

            s0  = ph0_jpos[:,ji] - ph0_jpos[:,ji].mean()
            s1  = ph1_jpos[:,ji] - ph1_jpos[:,ji].mean()
            std0 = clip_std(ph0_jpos[:,ji])
            std1 = clip_std(ph1_jpos[:,ji])

            # ±1σ bands
            ax.fill_betweenx([-std0, std0],
                             ph0_time[0], ph0_time[-1],
                             alpha=0.18, color=PH0_COL, zorder=0)
            ax.fill_betweenx([-std1, std1],
                             t1_plot[0], t1_plot[-1],
                             alpha=0.18, color=c, zorder=0)

            ax.axhline( std0, color=PH0_COL, lw=0.8, ls='--', alpha=0.7)
            ax.axhline(-std0, color=PH0_COL, lw=0.8, ls='--', alpha=0.7)
            ax.axhline( std1, color=c,        lw=0.8, ls='--', alpha=0.7)
            ax.axhline(-std1, color=c,        lw=0.8, ls='--', alpha=0.7)

            # commanded = 0 (full width)
            ax.axhline(0, color='black', lw=1.8, ls='--',
                       alpha=0.85, zorder=5)

            # signals
            ax.plot(ph0_time[::sub0], s0[::sub0],
                    lw=0.45, color=PH0_COL, alpha=0.90)
            ax.plot(t1_plot[::sub1], s1[::sub1],
                    lw=0.45, color=c, alpha=0.90)

            # spike outliers
            mask0 = np.abs(s0) > 3*std0
            mask1 = np.abs(s1) > 3*std1
            if mask0.sum():
                ax.scatter(ph0_time[mask0], s0[mask0],
                           s=12, color='red', zorder=8, alpha=0.85)
            if mask1.sum():
                ax.scatter(t1_plot[mask1], s1[mask1],
                           s=12, color='red', zorder=8, alpha=0.85)

            # phase separator
            ax.axvline(t0_dur + GAP/2,
                       color='#aaaaaa', lw=1.2, ls=':', alpha=0.7)

            # phase labels at top
            ax.text(ph0_time[-1]*0.5, ymax_row*0.92,
                    'Ph0', ha='center', fontsize=8,
                    color=PH0_COL, fontweight='bold')
            ax.text(t0_dur + GAP + ph1_time[-1]*0.5, ymax_row*0.92,
                    'Ph1', ha='center', fontsize=8,
                    color=c, fontweight='bold')

            # stats box
            ax.text(0.02, 0.97,
                    f'\u03c3\u2080 = {std0*1000:.4f} mrad\n'
                    f'\u03c3\u2081 = {std1*1000:.4f} mrad\n'
                    f'KP = {KP_TRAIN[ji]}',
                    transform=ax.transAxes, va='top', fontsize=7.5,
                    color=c, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.22',
                              fc='white', ec=c, alpha=0.90))

            ax.set_title(JNAMES[ji], fontsize=10.5,
                         color=c, fontweight='bold')
            ax.set_ylim(-ymax_row, ymax_row)
            ax.set_xlim(ph0_time[0]-0.5, t1_plot[-1]+0.5)
            ax.grid(True, alpha=0.18, axis='y')
            ax.tick_params(labelsize=8)

            if row == 2:
                ax.set_xlabel(
                    f'Time (s)   |   Ph0: 0\u2013{t0_dur:.0f}s'
                    f'  \u00b7  Ph1: {t0_dur+GAP:.0f}\u2013{t1_plot[-1]:.0f}s',
                    fontsize=8)
            if col == 0:
                ax.set_ylabel(f'{gn}\n\u0394q (rad)', fontsize=9)

    legend_els = [
        Line2D([0],[0], color='black', lw=2.0, ls='--',
               label='Commanded = 0 (mean-subtracted)'),
        Line2D([0],[0], color=PH0_COL, lw=1.5,
               label='Ph0 actual — hanging, soft KP'),
        Line2D([0],[0], color=GCOLS['Hip'], lw=1.5,
               label='Ph1 Hip actual — standing, KP=35'),
        Line2D([0],[0], color=GCOLS['Thigh'], lw=1.5,
               label='Ph1 Thigh actual — standing, KP=65'),
        Line2D([0],[0], color=GCOLS['Knee'], lw=1.5,
               label='Ph1 Knee actual — standing, KP=80'),
        mpatches.Patch(color='red', alpha=0.7, label='|signal| > 3\u03c3 spike'),
    ]
    fig.legend(handles=legend_els, loc='lower center', ncol=6,
               fontsize=8.5, bbox_to_anchor=(0.5,-0.01),
               framealpha=0.95, edgecolor='grey')
    plt.tight_layout(rect=[0,0.04,1,1])

    if save:
        out = f'{prefix}_fig1_jpos.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'  Saved {out}')
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — jvel time-series: commanded vs actual, both phases
# ══════════════════════════════════════════════════════════════════════════════
def figure2_jvel(d, save=False, prefix='encoder'):
    """
    3×4 grid — one panel per joint.
    Y-axis: joint velocity (rad/s). Commanded = 0 (static hold).
    All readings are sensor noise. ±1σ band shown per phase.
    """
    ph0_jvel = d['ph0_jvel'];  ph1_jvel = d['ph1_jvel']
    ph0_time = d['ph0_time'];  ph1_time = d['ph1_time']

    t0_dur  = ph0_time[-1]
    GAP     = 1.5
    t1_plot = ph1_time + t0_dur + GAP

    sub0 = max(1, len(ph0_time)//3000)
    sub1 = max(1, len(ph1_time)//3000)

    fig, axes = plt.subplots(3, 4, figsize=(22, 12), sharey='row')
    fig.suptitle(
        'Go1 Test 2 — Joint Velocity Encoder Noise: Phase 0 (Hanging) and Phase 1 (Standing)\n'
        'Commanded velocity = 0 rad/s (black dashed) — robot is stationary throughout both phases.\n'
        'All non-zero readings are pure sensor noise. Band = ±1\u03c3 per phase.',
        fontsize=11, fontweight='bold')

    for row, gn in enumerate(['Hip','Thigh','Knee']):
        idxs = GROUPS[gn]
        gc   = GCOLS[gn]

        ymax_row = max(
            max(np.percentile(np.abs(ph0_jvel[:,i]), 99.0) for i in idxs),
            max(np.percentile(np.abs(ph1_jvel[:,i]), 99.0) for i in idxs)
        ) * 1.5

        for col, ji in enumerate(idxs):
            ax  = axes[row][col]
            c   = gcol(ji)

            v0   = ph0_jvel[:,ji]
            v1   = ph1_jvel[:,ji]
            std0 = clip_std(v0)
            std1 = clip_std(v1)

            # ±1σ bands
            ax.fill_betweenx([-std0, std0],
                             ph0_time[0], ph0_time[-1],
                             alpha=0.18, color=PH0_COL, zorder=0)
            ax.fill_betweenx([-std1, std1],
                             t1_plot[0], t1_plot[-1],
                             alpha=0.18, color=c, zorder=0)

            ax.axhline( std0, color=PH0_COL, lw=0.8, ls='--', alpha=0.70)
            ax.axhline(-std0, color=PH0_COL, lw=0.8, ls='--', alpha=0.70)
            ax.axhline( std1, color=c,        lw=0.8, ls='--', alpha=0.70)
            ax.axhline(-std1, color=c,        lw=0.8, ls='--', alpha=0.70)

            # commanded = 0
            ax.axhline(0, color='black', lw=1.8, ls='--', alpha=0.85, zorder=5)

            # signals
            ax.plot(ph0_time[::sub0], v0[::sub0],
                    lw=0.45, color=PH0_COL, alpha=0.90)
            ax.plot(t1_plot[::sub1], v1[::sub1],
                    lw=0.45, color=c, alpha=0.90)

            # phase separator
            ax.axvline(t0_dur + GAP/2,
                       color='#aaaaaa', lw=1.2, ls=':', alpha=0.7)

            # phase labels
            ax.text(ph0_time[-1]*0.5, ymax_row*0.90,
                    'Ph0', ha='center', fontsize=8,
                    color=PH0_COL, fontweight='bold')
            ax.text(t0_dur + GAP + ph1_time[-1]*0.5, ymax_row*0.90,
                    'Ph1', ha='center', fontsize=8,
                    color=c, fontweight='bold')

            # stats
            ax.text(0.02, 0.97,
                    f'\u03c3\u2080 = {std0*1000:.3f} mrad/s\n'
                    f'\u03c3\u2081 = {std1*1000:.3f} mrad/s',
                    transform=ax.transAxes, va='top', fontsize=7.5,
                    color=c, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.22',
                              fc='white', ec=c, alpha=0.90))

            ax.set_title(JNAMES[ji], fontsize=10.5,
                         color=c, fontweight='bold')
            ax.set_ylim(-ymax_row, ymax_row)
            ax.set_xlim(ph0_time[0]-0.5, t1_plot[-1]+0.5)
            ax.grid(True, alpha=0.18, axis='y')
            ax.tick_params(labelsize=8)

            if row == 2:
                ax.set_xlabel(
                    f'Time (s)   |   Ph0: 0\u2013{t0_dur:.0f}s'
                    f'  \u00b7  Ph1: {t0_dur+GAP:.0f}\u2013{t1_plot[-1]:.0f}s',
                    fontsize=8)
            if col == 0:
                ax.set_ylabel(f'{gn}\ndq (rad/s)', fontsize=9)

    legend_els = [
        Line2D([0],[0], color='black', lw=2.0, ls='--',
               label='Commanded velocity = 0 rad/s (static hold)'),
        Line2D([0],[0], color=PH0_COL, lw=1.5,
               label='Ph0 actual — hanging'),
        Line2D([0],[0], color=GCOLS['Hip'], lw=1.5,
               label='Ph1 Hip — KP=35'),
        Line2D([0],[0], color=GCOLS['Thigh'], lw=1.5,
               label='Ph1 Thigh — KP=65'),
        Line2D([0],[0], color=GCOLS['Knee'], lw=1.5,
               label='Ph1 Knee — KP=80'),
    ]
    fig.legend(handles=legend_els, loc='lower center', ncol=5,
               fontsize=8.5, bbox_to_anchor=(0.5,-0.01),
               framealpha=0.95, edgecolor='grey')
    plt.tight_layout(rect=[0,0.04,1,1])

    if save:
        out = f'{prefix}_fig2_jvel.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'  Saved {out}')
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Commanded vs Actual Position per group
# ══════════════════════════════════════════════════════════════════════════════
def figure3_pos_actual(d, save=False, prefix='encoder'):
    """
    1×3 panels (Hip / Thigh / Knee).
    Y-axis: actual joint position (rad) — absolute physical value.
    X-axis: joint names (categorical).
    Per joint: commanded target (black dashed horizontal line)
               + Ph0 mean dot (grey) + Ph1 mean dot (colour).
    Error bars = ±1σ (scaled ×50 for visibility — noise is ~0.02 mrad).
    Offset labels in mrad next to each dot.
    """
    ph0_jpos = d['ph0_jpos']; ph1_jpos = d['ph1_jpos']
    dq_hw    = d['default_q_hw']

    cmd_vals = np.array([float(dq_hw[i]) for i in range(12)])
    ph0_mean = ph0_jpos.mean(axis=0)
    ph1_mean = ph1_jpos.mean(axis=0)
    ph0_std  = np.array([clip_std(ph0_jpos[:,i]) for i in range(12)])
    ph1_std  = np.array([clip_std(ph1_jpos[:,i]) for i in range(12)])

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(
        'Go1 Test 2 — Commanded vs Actual Joint Position per Group\n'
        'Black dashed = commanded target (DEFAULT_Q_HW, rad).  '
        'Grey circle = Phase 0 mean actual (hanging, soft KP).  '
        'Colour circle = Phase 1 mean actual (standing, training KP).\n'
        'Error bars = \u00b11\u03c3 encoder noise (\u00d750 for visibility, \u03c3 \u2248 0.02\u20130.09 mrad).  '
        'Labels = offset from commanded target (mrad).',
        fontsize=10.5, fontweight='bold')

    for col, (gn, idxs) in enumerate(GROUPS.items()):
        ax = axes[col]
        gc = GCOLS[gn]
        x  = np.arange(len(idxs))
        w  = 0.20
        ax.set_facecolor(GBG[gn])

        for xi, ji in enumerate(idxs):
            cmd = cmd_vals[ji]
            p0m = ph0_mean[ji]; s0 = ph0_std[ji]
            p1m = ph1_mean[ji]; s1 = ph1_std[ji]
            c   = gcol(ji)

            # commanded horizontal line
            ax.plot([xi-0.40, xi+0.40], [cmd, cmd],
                    color='black', lw=2.2, ls='--',
                    zorder=5, solid_capstyle='round')

            # Ph0 dot (grey open circle)
            ax.errorbar(xi-w, p0m, yerr=s0*50,
                        fmt='o', ms=11,
                        color='white', markeredgecolor=PH0_COL,
                        markeredgewidth=2.2,
                        ecolor=PH0_COL, elinewidth=2.0,
                        capsize=6, capthick=1.8, zorder=6)

            # Ph1 dot (filled colour)
            ax.errorbar(xi+w, p1m, yerr=s1*50,
                        fmt='o', ms=11,
                        color=c, markeredgecolor='white',
                        markeredgewidth=1.0,
                        ecolor=c, elinewidth=2.0,
                        capsize=6, capthick=1.8, zorder=7)

            # connector
            ax.plot([xi-w, xi+w], [p0m, p1m],
                    color=c, lw=0.9, ls='-', alpha=0.35, zorder=4)

            # offset labels
            off0 = (p0m - cmd)*1000
            off1 = (p1m - cmd)*1000
            all_y = [p0m, p1m, cmd]
            pad = (max(all_y)-min(all_y))*0.14 + 0.003

            ax.text(xi-w, min(all_y)-pad,
                    f'{off0:+.1f}\nmrad',
                    ha='center', va='top', fontsize=7.5,
                    color=PH0_COL, fontweight='bold')
            ax.text(xi+w, min(all_y)-pad,
                    f'{off1:+.1f}\nmrad',
                    ha='center', va='top', fontsize=7.5,
                    color=c, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([JNAMES[i] for i in idxs],
                           fontsize=10.5, fontweight='bold')
        ax.set_ylabel('Joint position (rad)', fontsize=10)
        ax.set_title(f'{gn} joints  |  KP\u2081 = {KP_TRAIN[idxs[0]]} Nm/rad',
                     fontsize=11, fontweight='bold', color=gc)
        ax.grid(True, alpha=0.28, axis='y')
        ax.set_xlim(-0.55, len(idxs)-0.45)

        all_v = np.concatenate([cmd_vals[idxs],
                                ph0_mean[idxs],
                                ph1_mean[idxs]])
        rng = max(all_v.max()-all_v.min(), 0.05)
        ax.set_ylim(all_v.min()-rng*0.50, all_v.max()+rng*0.20)

    legend_els = [
        Line2D([0],[0], color='black', lw=2.5, ls='--',
               label='Commanded target (DEFAULT_Q_HW)'),
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor='white', markeredgecolor=PH0_COL,
               markersize=11, markeredgewidth=2.2,
               label='Phase 0 actual mean \u00b11\u03c3  (hanging, soft KP)'),
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor=GCOLS['Hip'], markeredgecolor='white',
               markersize=11, markeredgewidth=1,
               label='Phase 1 actual mean \u00b11\u03c3  (standing, training KP)'),
    ]
    fig.legend(handles=legend_els, loc='lower center', ncol=3,
               fontsize=10, bbox_to_anchor=(0.5,-0.02),
               framealpha=0.95, edgecolor='grey')
    plt.tight_layout(rect=[0,0.07,1,1])

    if save:
        out = f'{prefix}_fig3_pos_actual.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'  Saved {out}')
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Sigma comparison Ph0 vs Ph1 (jpos and jvel)
# ══════════════════════════════════════════════════════════════════════════════
def figure4_sigma(d, save=False, prefix='encoder'):
    """
    1×2 panels: left = jpos sigma (mrad), right = jvel sigma (mrad/s).
    Per joint: grey bar = Ph0, colour bar = Ph1.
    Ratio Ph1/Ph0 shown above each pair.
    Group background shading (blue/green/red) matches joint colour coding.
    """
    ph0_jpos=d['ph0_jpos']; ph1_jpos=d['ph1_jpos']
    ph0_jvel=d['ph0_jvel']; ph1_jvel=d['ph1_jvel']

    ps0 = np.array([clip_std(ph0_jpos[:,i])*1000 for i in range(12)])
    ps1 = np.array([clip_std(ph1_jpos[:,i])*1000 for i in range(12)])
    vs0 = np.array([clip_std(ph0_jvel[:,i])*1000 for i in range(12)])
    vs1 = np.array([clip_std(ph1_jvel[:,i])*1000 for i in range(12)])

    x        = np.arange(12)
    w        = 0.35
    bar_cols = [gcol(i) for i in range(12)]

    fig, (ax_p, ax_v) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        'Go1 Test 2 — Encoder Noise \u03c3: Phase 0 (Hanging) vs Phase 1 (Standing)\n'
        'Grey bar = Phase 0 noise floor (hanging, soft KP).  '
        'Colour bar = Phase 1 deployment condition (standing, training KP).\n'
        'Ratio above each pair = \u03c3\u2081 / \u03c3\u2080. '
        'Ratio \u2248 1.0 confirms encoder noise is load-independent.',
        fontsize=10.5, fontweight='bold')

    for ax, d0, d1, unit, title in [
        (ax_p, ps0, ps1, 'mrad',
         'Joint Position Noise \u03c3\n'
         '(adopted as jpos\_noise in Isaac Lab obs\_noise\_std)'),
        (ax_v, vs0, vs1, 'mrad/s',
         'Joint Velocity Noise \u03c3\n'
         '(adopted as jvel\_noise in Isaac Lab obs\_noise\_std)'),
    ]:
        # group background shading
        for gn, (xlo, xhi) in [('Hip',(-0.6,3.6)),
                                ('Thigh',(3.6,7.6)),
                                ('Knee',(7.6,11.6))]:
            ax.axvspan(xlo, xhi, alpha=0.25, color=GBG[gn], zorder=0)
            ax.text((xlo+xhi)/2, 0, gn, ha='center', va='bottom',
                    fontsize=10, color=GCOLS[gn], fontweight='bold',
                    alpha=0.65, transform=ax.get_xaxis_transform())

        # bars
        ax.bar(x-w/2, d0, w,
               color='#d0d0d0', edgecolor='#777777',
               linewidth=1.0, alpha=0.92,
               label='Ph0  hanging  (noise floor)')
        ax.bar(x+w/2, d1, w,
               color=bar_cols, edgecolor='white',
               linewidth=0.6, alpha=0.92,
               label='Ph1  standing  (deployment)')

        ymax = max(d0.max(), d1.max())

        # value labels
        for xi in range(12):
            ax.text(xi-w/2, d0[xi]+ymax*0.012, f'{d0[xi]:.3f}',
                    ha='center', va='bottom', fontsize=6.5,
                    color='#444444', rotation=90, fontweight='bold')
            ax.text(xi+w/2, d1[xi]+ymax*0.012, f'{d1[xi]:.3f}',
                    ha='center', va='bottom', fontsize=6.5,
                    color=bar_cols[xi], rotation=90, fontweight='bold')

        # ratio
        for xi in range(12):
            ratio = d1[xi] / max(d0[xi], 0.0001)
            col_r = '#cc0000' if (ratio > 3 or ratio < 0.3) else '#333333'
            ax.text(xi, ymax*1.18, f'{ratio:.1f}\u00d7',
                    ha='center', va='bottom', fontsize=7.5,
                    color=col_r, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(JNAMES, rotation=30, ha='right', fontsize=9.5)
        ax.set_ylabel(f'Noise  \u03c3  ({unit})', fontsize=10)
        ax.set_title(title, fontsize=10.5, fontweight='bold')
        ax.grid(True, alpha=0.25, axis='y')
        ax.set_ylim(0, ymax*1.52)
        ax.set_xlim(-0.65, 11.65)
        ax.text(0.01, 0.99,
                'Commanded = 0  (static hold)\n'
                'All readings = sensor noise',
                transform=ax.transAxes, va='top', fontsize=8.5,
                color='#444444',
                bbox=dict(boxstyle='round,pad=0.3',
                          fc='white', ec='#bbbbbb', alpha=0.88))

    legend_els = [
        mpatches.Patch(facecolor='#d0d0d0', edgecolor='#777777',
                       linewidth=1.5,
                       label='Ph0  hanging  (soft KP_HANG)  — noise floor'),
        mpatches.Patch(facecolor='#2166ac', edgecolor='white',
                       label='Ph1  standing  (KP_TRAIN)  — deployment condition'),
    ]
    fig.legend(handles=legend_els, loc='lower center', ncol=2,
               fontsize=10, bbox_to_anchor=(0.5,-0.01),
               framealpha=0.95, edgecolor='grey')
    plt.tight_layout(rect=[0,0.06,1,1])

    if save:
        out = f'{prefix}_fig4_sigma.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'  Saved {out}')
    return fig


# ── Console table ─────────────────────────────────────────────────────────────
def print_table(d):
    ph0_jpos=d['ph0_jpos']; ph1_jpos=d['ph1_jpos']
    ph0_jvel=d['ph0_jvel']; ph1_jvel=d['ph1_jvel']
    ph0_time=d['ph0_time']; ph1_time=d['ph1_time']
    ph0_dt=d['ph0_dt'];     ph1_dt=d['ph1_dt']
    dq_hw=d['default_q_hw']

    print()
    print('='*108)
    print('TEST 2 — ENCODER NOISE RESULTS (3\u03c3-clipped sigma)')
    print('='*108)
    print(f'  Ph0 hanging : {ph0_jpos.shape[0]} samples  {ph0_time[-1]:.1f}s  '
          f'{ph0_jpos.shape[0]/ph0_time[-1]:.0f} Hz  '
          f'dt_mean={ph0_dt.mean()*1000:.2f}ms  dt_max={ph0_dt.max()*1000:.1f}ms')
    print(f'  Ph1 standing: {ph1_jpos.shape[0]} samples  {ph1_time[-1]:.1f}s  '
          f'{ph1_jpos.shape[0]/ph1_time[-1]:.0f} Hz  '
          f'dt_mean={ph1_dt.mean()*1000:.2f}ms  dt_max={ph1_dt.max()*1000:.1f}ms')
    print()
    print(f'  {"Joint":10s}  {"KP":>4}  {"CMD (rad)":>10}  '
          f'{"Ph0 mean":>10}  {"Ph0 \u0394":>10}  {"Ph0 \u03c3":>9}  '
          f'{"Ph1 mean":>10}  {"Ph1 \u0394":>10}  {"Ph1 \u03c3":>9}  '
          f'{"Ph0 v\u03c3":>9}  {"Ph1 v\u03c3":>9}')
    print('  '+'─'*107)
    for i,n in enumerate(JNAMES):
        cmd = float(dq_hw[i])
        p0m = ph0_jpos[:,i].mean(); p1m = ph1_jpos[:,i].mean()
        s0p = clip_std(ph0_jpos[:,i])*1000; s1p = clip_std(ph1_jpos[:,i])*1000
        s0v = clip_std(ph0_jvel[:,i])*1000; s1v = clip_std(ph1_jvel[:,i])*1000
        print(f'  {n:10s}  {KP_TRAIN[i]:>4}  {cmd:>10.4f}  '
              f'{p0m:>10.5f}  {(p0m-cmd)*1000:>+8.2f}mrad  {s0p:>6.4f}mrad  '
              f'{p1m:>10.5f}  {(p1m-cmd)*1000:>+8.2f}mrad  {s1p:>6.4f}mrad  '
              f'{s0v:>6.3f}m\u00b0/s  {s1v:>6.3f}m\u00b0/s')
    ps1 = np.array([clip_std(ph1_jpos[:,i]) for i in range(12)])
    vs1 = np.array([clip_std(ph1_jvel[:,i]) for i in range(12)])
    print()
    print('  Isaac Lab obs_noise_std — Phase 1 (3\u03c3-clipped):')
    print(f'  jpos_noise = {[round(float(v),6) for v in ps1]}  # rad')
    print(f'  jvel_noise = {[round(float(v),5) for v in vs1]}  # rad/s')
    print('='*108)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description='Go1 Test 2 — Encoder noise academic plots (final)')
    p.add_argument('npz',      type=str,
                   help='Path to calib_encoder_*.npz')
    p.add_argument('--save',   action='store_true',
                   help='Save PNG files instead of showing interactively')
    p.add_argument('--outdir', type=str, default='.',
                   help='Output directory for saved figures')
    args = p.parse_args()

    if args.save:
        matplotlib.use('Agg')
    else:
        try:    matplotlib.use('TkAgg')
        except: matplotlib.use('Agg'); args.save = True

    d      = np.load(args.npz, allow_pickle=True)
    base   = os.path.basename(args.npz).replace('.npz','')
    os.makedirs(args.outdir, exist_ok=True)
    prefix = os.path.join(args.outdir, base)

    print_table(d)
    print('\nGenerating figures...')
    figure1_jpos(d,      save=args.save, prefix=prefix)
    figure2_jvel(d,      save=args.save, prefix=prefix)
    figure3_pos_actual(d,save=args.save, prefix=prefix)
    figure4_sigma(d,     save=args.save, prefix=prefix)

    if not args.save:
        plt.show()
    else:
        print(f'\nAll 4 figures saved to: {args.outdir}/')

if __name__ == '__main__':
    main()