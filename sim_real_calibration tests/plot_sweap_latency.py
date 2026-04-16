#!/usr/bin/env python3
"""
Go1 Combined Latency Tests — Academic Plots
=============================================
Plots for Test 3 (spike latency) + Test 5 (frequency sweep).
Reads calib_freq_sweep_*.npz. Latency data embedded from known values.

Figures:
  Fig 1 — Bode magnitude plot: tracking ratio vs frequency, per group
  Fig 2 — Bandwidth + α summary bar chart per joint
  Fig 3 — Combined α: spike (fixed) vs sweep (group DR range)
  Fig 4 — ρ at 2 Hz (walking frequency) showing attenuation per joint

Usage:
  python3 go1_test_latency_sweep_plots.py calib_freq_sweep_*.npz
  python3 go1_test_latency_sweep_plots.py calib_freq_sweep_*.npz --save
"""
import os, sys, argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

JNAMES = ['FL_hip','FR_hip','RL_hip','RR_hip',
          'FL_th', 'FR_th', 'RL_th', 'RR_th',
          'FL_kn', 'FR_kn', 'RL_kn', 'RR_kn']
GROUPS   = {'Hip':[0,1,2,3],'Thigh':[4,5,6,7],'Knee':[8,9,10,11]}
GCOLS    = {'Hip':'#2166ac','Thigh':'#4dac26','Knee':'#d6604d'}
FAULT_J  = 'RL_th'
FAULT_C  = '#cc4400'
KP_TRAIN = [35]*4+[65]*4+[80]*4

# Spike latency results from Test 3 (embedded — no separate npz needed)
SPIKE_LAT = {  # mean latency ms per joint
    'FL_hip':17.1,'FR_hip':16.3,'RL_hip':15.6,'RR_hip':16.1,
    'FL_th':15.8, 'FR_th':15.4, 'RL_th':39.0, 'RR_th':15.0,
    'FL_kn':16.9, 'FR_kn':16.5, 'RL_kn':17.5, 'RR_kn':16.7}
SPIKE_STD = {
    'FL_hip':4.9,'FR_hip':1.7,'RL_hip':0.9,'RR_hip':1.3,
    'FL_th':0.6, 'FR_th':2.0, 'RL_th':0.9, 'RR_th':1.2,
    'FL_kn':0.9, 'FR_kn':1.2, 'RL_kn':3.7, 'RR_kn':2.0}
SPIKE_ALPHA = {jn: 0.02/(0.02+v/1000) for jn,v in SPIKE_LAT.items()}
HEALTHY_SPIKE_ALPHA = 0.552

DT = 0.02  # 50 Hz policy

def gcol(i):
    if i<4: return GCOLS['Hip']
    if i<8: return GCOLS['Thigh']
    return GCOLS['Knee']

def g(d, key):
    return float(d[key].flat[0])

def load(fname):
    return np.load(fname, allow_pickle=True)


# ── Figure 1: Bode magnitude — 3 panels (one per group) ──────────────────────
def figure1_bode(d, save=False, prefix='sweep'):
    freqs  = d['freqs_hz']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    fig.suptitle(
        'Go1 Test 5 — Actuator Frequency Response (Bode Magnitude)\n'
        'Tracking ratio $\\rho = \\sigma(q_{\\mathrm{actual}})/\\sigma(q_{\\mathrm{cmd}})$ '
        'versus command frequency at training KP, amplitude = 0.10 rad.\n'
        'Dashed horizontal line = $-3$\\,dB threshold ($0.707 \\times \\rho_{0.5\\,\\mathrm{Hz}}$). '
        'Vertical line = group mean bandwidth.',
        fontsize=10.5, fontweight='bold')

    for col, (gn, idxs) in enumerate(GROUPS.items()):
        ax = axes[col]
        gc = GCOLS[gn]
        bws = []

        for ji in idxs:
            jn = JNAMES[ji]
            is_fault = (jn == FAULT_J)
            c = FAULT_C if is_fault else gc
            ls = '--' if is_fault else '-'
            lw = 1.5 if is_fault else 2.0

            ratios = [g(d, f'{jn}_{f:.1f}Hz_ratio_mean') for f in freqs]
            stds   = [g(d, f'{jn}_{f:.1f}Hz_ratio_std')  for f in freqs]
            bw     = g(d, f'{jn}_bandwidth_hz')
            tau    = g(d, f'{jn}_tau_bw_ms')
            alpha  = g(d, f'{jn}_alpha_bw')

            if not is_fault:
                bws.append(bw)

            ax.plot(freqs, ratios, c=c, lw=lw, ls=ls,
                    marker='o', ms=5, label=f'{jn}' if not is_fault else f'{jn} ⚠')
            ax.fill_between(freqs,
                            np.array(ratios)-np.array(stds),
                            np.array(ratios)+np.array(stds),
                            alpha=0.12, color=c)

            # Mark bandwidth point
            if not is_fault and not np.isnan(bw):
                r_bw = ratios[0]*0.707
                ax.plot(bw, r_bw, marker='v', ms=9,
                        color=c, zorder=8, markeredgecolor='white', markeredgewidth=1)

        # Group mean BW line
        if bws:
            mean_bw = np.mean(bws)
            ax.axvline(mean_bw, color=gc, lw=1.5, ls=':', alpha=0.7,
                       label=f'Mean BW = {mean_bw:.1f} Hz')
            ax.text(mean_bw+0.15, 0.97, f'{mean_bw:.1f} Hz',
                    color=gc, fontsize=8.5, fontweight='bold', va='top',
                    transform=ax.get_xaxis_transform())

        # -3dB line (use first healthy joint's baseline)
        healthy_j = [JNAMES[i] for i in idxs if JNAMES[i] != FAULT_J][0]
        r_base    = g(d, f'{healthy_j}_0.5Hz_ratio_mean')
        ax.axhline(r_base*0.707, color='#555555', lw=1.2, ls='--', alpha=0.6,
                   label=f'$-3$dB = {r_base*0.707:.3f}')

        # Walking frequency reference
        ax.axvline(2.0, color='#888888', lw=1.0, ls=':', alpha=0.55)
        ax.text(2.05, 0.15, 'Walk\n2 Hz', color='#666666',
                fontsize=7.5, va='bottom',
                transform=ax.get_xaxis_transform())

        ax.set_xscale('log')
        ax.set_xlim(0.4, 12)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Command frequency (Hz)', fontsize=10)
        ax.set_ylabel('Tracking ratio $\\rho$', fontsize=10)
        ax.set_title(f'{gn} joints  |  $K_P = {KP_TRAIN[idxs[0]]}$\\,Nm/rad',
                     fontsize=11, fontweight='bold', color=gc)
        ax.grid(True, alpha=0.25, which='both')
        ax.set_xticks([0.5,1,2,3,4,5,6,8,10])
        ax.set_xticklabels([0.5,1,2,3,4,5,6,8,10], fontsize=8.5)
        ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    if save:
        out = f'{prefix}_fig1_bode.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'  Saved {out}')
    return fig


# ── Figure 2: Bandwidth + alpha bar chart ────────────────────────────────────
def figure2_bw_alpha(d, save=False, prefix='sweep'):
    x       = np.arange(12)
    bws     = [g(d, f'{jn}_bandwidth_hz')  for jn in JNAMES]
    taus    = [g(d, f'{jn}_tau_bw_ms')     for jn in JNAMES]
    alphas  = [g(d, f'{jn}_alpha_bw')      for jn in JNAMES]
    bar_c   = [FAULT_C if jn==FAULT_J else gcol(i) for i,jn in enumerate(JNAMES)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle(
        'Go1 Test 5 — Actuator Bandwidth and Derived Lag Coefficient per Joint\n'
        'Higher $K_P$ gives wider bandwidth: knees (KP=80) track faster than hips (KP=35).',
        fontsize=11, fontweight='bold')

    # BW bars
    ax1.bar(x, bws, color=bar_c, alpha=0.85, edgecolor='white', lw=0.6)
    for xi, (bw, c) in enumerate(zip(bws, bar_c)):
        ax1.text(xi, bw+0.05, f'{bw:.2f}', ha='center', va='bottom',
                 fontsize=7.5, color=c, fontweight='bold')
    ax1.axhline(2.0, color='#888888', lw=1.2, ls='--', alpha=0.6, label='Trotting frequency (2 Hz)')
    ax1.set_xticks(x); ax1.set_xticklabels(JNAMES, rotation=30, ha='right', fontsize=9)
    ax1.set_ylabel('Actuator bandwidth (Hz)', fontsize=10)
    ax1.set_title('(A) Actuator −3\\,dB bandwidth per joint', fontsize=10.5, fontweight='bold')
    ax1.grid(True, alpha=0.25, axis='y'); ax1.set_ylim(0, 7)
    ax1.legend(fontsize=9)
    for gn,(xlo,xhi) in [('Hip',(-0.6,3.6)),('Thigh',(3.6,7.6)),('Knee',(7.6,11.6))]:
        ax1.text((xlo+xhi)/2, 6.5, gn, ha='center', fontsize=10,
                 color=GCOLS[gn], fontweight='bold', alpha=0.55)

    # Alpha bars — sweep vs spike
    ax2.bar(x-0.2, alphas, 0.38, color=bar_c, alpha=0.85,
            edgecolor='white', lw=0.6, label='$\\alpha$ from freq sweep (BW-derived)')
    ax2.bar(x+0.2, [SPIKE_ALPHA[jn] for jn in JNAMES], 0.38,
            color=[c if jn!=FAULT_J else FAULT_C for jn,c in zip(JNAMES,bar_c)],
            alpha=0.40, edgecolor='white', lw=0.6, hatch='//',
            label='$\\alpha$ from spike test (comms delay only)')
    ax2.axhline(HEALTHY_SPIKE_ALPHA, color='black', lw=1.5, ls='--', alpha=0.7,
                label=f'Adopted spike $\\alpha$ = {HEALTHY_SPIKE_ALPHA} (uniform)')
    for xi, (a_sw, jn) in enumerate(zip(alphas, JNAMES)):
        ax2.text(xi-0.2, a_sw+0.005, f'{a_sw:.3f}', ha='center', va='bottom',
                 fontsize=6.5, color=bar_c[xi], fontweight='bold')

    ax2.set_xticks(x); ax2.set_xticklabels(JNAMES, rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel('Lag filter coefficient $\\alpha$', fontsize=10)
    ax2.set_title('(B) $\\alpha$ from frequency sweep (solid) vs spike test (hatched)\n'
                  'Spike $\\alpha$ captures comms delay only; sweep $\\alpha$ captures full actuator dynamics.',
                  fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.25, axis='y'); ax2.set_ylim(0, 0.75)
    ax2.legend(fontsize=8.5, loc='upper right')
    for gn,(xlo,xhi) in [('Hip',(-0.6,3.6)),('Thigh',(3.6,7.6)),('Knee',(7.6,11.6))]:
        ax2.text((xlo+xhi)/2, 0.68, gn, ha='center', fontsize=10,
                 color=GCOLS[gn], fontweight='bold', alpha=0.55)

    plt.tight_layout()
    if save:
        out = f'{prefix}_fig2_bw_alpha.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'  Saved {out}')
    return fig


# ── Figure 3: DR range visualisation ─────────────────────────────────────────
def figure3_dr_range(d, save=False, prefix='sweep'):
    """
    Shows the recommended α DR range per joint:
    Lower bound = sweep α (full actuator dynamics)
    Upper bound = spike α (comms delay only) = 0.552
    Centre = midpoint, used as nominal
    """
    alphas_sw  = np.array([g(d, f'{jn}_alpha_bw') for jn in JNAMES])
    alphas_sp  = np.array([SPIKE_ALPHA[jn] for jn in JNAMES])
    bar_c      = [FAULT_C if jn==FAULT_J else gcol(i) for i,jn in enumerate(JNAMES)]
    x          = np.arange(12)

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle(
        'Go1 Test 3+5 — Recommended Lag Filter $\\alpha$ Domain Randomisation Range\n'
        'Lower bound from frequency sweep (full actuator BW). '
        'Upper bound from spike test (comms delay). '
        'Policy trained to be robust across the full range.',
        fontsize=10.5, fontweight='bold')

    # Draw range bars
    for xi, (lo, hi, c, jn) in enumerate(zip(alphas_sw, alphas_sp, bar_c, JNAMES)):
        if jn == FAULT_J:
            ax.plot(xi, lo, marker='x', ms=14, color=FAULT_C, lw=3,
                    label='RL\_th (excluded — fault)')
            continue
        ax.plot([xi, xi], [lo, hi], color=c, lw=6, solid_capstyle='round', alpha=0.45)
        ax.plot(xi, lo,          marker='o', ms=10, color=c,
                markeredgecolor='white', markeredgewidth=1.5, zorder=5,
                label='Sweep $\\alpha$ (lower bound)' if xi==0 else '')
        ax.plot(xi, hi,          marker='s', ms=8,  color=c,
                markeredgecolor='white', markeredgewidth=1.0, alpha=0.7, zorder=5,
                label='Spike $\\alpha$ (upper bound)' if xi==0 else '')
        mid = (lo+hi)/2
        ax.text(xi, lo-0.015, f'{lo:.3f}', ha='center', va='top',
                fontsize=7, color=c, fontweight='bold')
        ax.text(xi, hi+0.008, f'{hi:.3f}', ha='center', va='bottom',
                fontsize=7, color=c, fontweight='bold')

    ax.axhline(HEALTHY_SPIKE_ALPHA, color='black', lw=1.5, ls='--', alpha=0.6,
               label=f'Nominal (spike mean) = {HEALTHY_SPIKE_ALPHA}')

    # Group DR boxes
    group_ranges = {
        'Hip':   (np.mean(alphas_sw[:4]),  HEALTHY_SPIKE_ALPHA),
        'Thigh': (np.mean([alphas_sw[i] for i in [4,5,7]]), HEALTHY_SPIKE_ALPHA),
        'Knee':  (np.mean(alphas_sw[8:]),  HEALTHY_SPIKE_ALPHA),
    }
    for gn, (lo, hi) in group_ranges.items():
        c = GCOLS[gn]
        ax.text(11.7, (lo+hi)/2, f'{gn}\n[{lo:.3f},\n {hi:.3f}]',
                ha='left', va='center', fontsize=7.5, color=c, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=c, alpha=0.85))

    ax.set_xticks(x); ax.set_xticklabels(JNAMES, rotation=30, ha='right', fontsize=9.5)
    ax.set_ylabel('Lag filter coefficient $\\alpha$', fontsize=10)
    ax.set_ylim(0.05, 0.70)
    ax.set_xlim(-0.6, 13.5)
    ax.grid(True, alpha=0.22, axis='y')
    ax.legend(fontsize=9, loc='upper left')

    plt.tight_layout()
    if save:
        out = f'{prefix}_fig3_dr_range.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'  Saved {out}')
    return fig


# ── Figure 4: ρ at walking frequency ─────────────────────────────────────────
def figure4_walking_attenuation(d, save=False, prefix='sweep'):
    """
    Shows tracking ratio at 2 Hz (trotting) and phase lag per joint.
    Motivates the action rate penalty weights and the per-group α choice.
    """
    r2    = [g(d, f'{jn}_2.0Hz_ratio_mean') for jn in JNAMES]
    r2std = [g(d, f'{jn}_2.0Hz_ratio_std')  for jn in JNAMES]
    ph2   = [g(d, f'{jn}_2.0Hz_phase_ms')   for jn in JNAMES]
    bar_c = [FAULT_C if jn==FAULT_J else gcol(i) for i,jn in enumerate(JNAMES)]
    x     = np.arange(12)
    w     = 0.38

    fig, (ax_r, ax_p) = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle(
        'Go1 Test 5 — Actuator Response at Trotting Frequency (2\\,Hz)\n'
        'At the nominal gait frequency, hip joints attenuate commanded motion '
        'by $\\approx$21\\% and introduce larger phase lags than thigh/knee joints.\n'
        'This motivates higher action-rate penalty weights for hip joints in the reward function.',
        fontsize=10.5, fontweight='bold')

    # Ratio bars
    ax_r.bar(x, r2, color=bar_c, alpha=0.85, edgecolor='white', lw=0.6)
    ax_r.errorbar(x, r2, yerr=r2std, fmt='none',
                  ecolor='#333333', elinewidth=1.8, capsize=5)
    ax_r.axhline(1.0, color='black', lw=0.8, ls=':', alpha=0.4, label='Perfect tracking')

    # Group mean lines
    for gn, idxs in GROUPS.items():
        healthy = [i for i in idxs if JNAMES[i]!=FAULT_J]
        gm = np.mean([r2[i] for i in healthy])
        ax_r.plot([healthy[0]-0.45, healthy[-1]+0.45], [gm,gm],
                  color=GCOLS[gn], lw=2.0, ls='--', alpha=0.7,
                  label=f'{gn} mean ρ = {gm:.3f}')

    for xi, (r, c, jn) in enumerate(zip(r2, bar_c, JNAMES)):
        if jn != FAULT_J:
            ax_r.text(xi, r+0.005, f'{r:.3f}', ha='center', va='bottom',
                      fontsize=7, color=c, fontweight='bold')
        else:
            ax_r.text(xi, r+0.005, '0.010\n⚠', ha='center', va='bottom',
                      fontsize=7, color=FAULT_C, fontweight='bold')

    ax_r.set_xticks(x); ax_r.set_xticklabels(JNAMES, rotation=30, ha='right', fontsize=9)
    ax_r.set_ylabel('Tracking ratio $\\rho$ at 2\\,Hz', fontsize=10)
    ax_r.set_title('(A) Tracking ratio at 2\\,Hz — attenuation of commanded motion '
                   '(1.0 = no attenuation)', fontsize=10, fontweight='bold')
    ax_r.grid(True, alpha=0.25, axis='y'); ax_r.set_ylim(0, 1.10)
    ax_r.legend(fontsize=8.5, loc='lower right')

    # Phase lag bars
    ph2_plot = [abs(p) for p in ph2]  # sign convention: positive = lag
    ax_p.bar(x, ph2_plot, color=bar_c, alpha=0.85, edgecolor='white', lw=0.6)
    for xi, (p, c, jn) in enumerate(zip(ph2_plot, bar_c, JNAMES)):
        if jn != FAULT_J:
            ax_p.text(xi, p+0.3, f'{p:.0f}', ha='center', va='bottom',
                      fontsize=7.5, color=c, fontweight='bold')
    ax_p.set_xticks(x); ax_p.set_xticklabels(JNAMES, rotation=30, ha='right', fontsize=9)
    ax_p.set_ylabel('Phase lag at 2\\,Hz (ms)', fontsize=10)
    ax_p.set_title('(B) Phase lag at 2\\,Hz — delay between command and response peaks',
                   fontsize=10, fontweight='bold')
    ax_p.grid(True, alpha=0.25, axis='y')

    plt.tight_layout()
    if save:
        out = f'{prefix}_fig4_walking.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f'  Saved {out}')
    return fig


# ── Console table ─────────────────────────────────────────────────────────────
def print_table(d):
    freqs = d['freqs_hz']
    print()
    print('='*100)
    print('TEST 5 — FREQUENCY SWEEP RESULTS')
    print('='*100)
    print(f"  {'Joint':10s}  {'KP':>4}  {'BW (Hz)':>8}  "
          f"{'τ (ms)':>8}  {'α_sweep':>9}  {'α_spike':>9}  "
          f"{'ρ@0.5Hz':>8}  {'ρ@2Hz':>8}  {'ρ@10Hz':>8}")
    print('  '+'─'*90)
    for i, jn in enumerate(JNAMES):
        bw  = g(d, f'{jn}_bandwidth_hz')
        tau = g(d, f'{jn}_tau_bw_ms')
        alp = g(d, f'{jn}_alpha_bw')
        r05 = g(d, f'{jn}_0.5Hz_ratio_mean')
        r2  = g(d, f'{jn}_2.0Hz_ratio_mean')
        r10 = g(d, f'{jn}_10.0Hz_ratio_mean')
        sp  = SPIKE_ALPHA[jn]
        note = '  ← FAULT' if jn==FAULT_J else ''
        print(f"  {jn:10s}  {KP_TRAIN[i]:>4}  {bw:>8.2f}  "
              f"{tau:>8.1f}  {alp:>9.4f}  {sp:>9.4f}  "
              f"{r05:>8.3f}  {r2:>8.3f}  {r10:>8.3f}{note}")

    print()
    print('  Group-level DR ranges for Isaac Lab lag filter:')
    for gn, jns in [('Hip',   ['FL_hip','FR_hip','RL_hip','RR_hip']),
                    ('Thigh', ['FL_th','FR_th','RR_th']),
                    ('Knee',  ['FL_kn','FR_kn','RL_kn','RR_kn'])]:
        alps = [g(d, f'{jn}_alpha_bw') for jn in jns]
        lo = np.mean(alps)
        print(f"    {gn:8s}: α ~ U({lo:.3f}, {HEALTHY_SPIKE_ALPHA:.3f})  "
              f"[sweep lower bound → spike upper bound]")
    print('='*100)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('npz',      type=str)
    p.add_argument('--save',   action='store_true')
    p.add_argument('--outdir', type=str, default='.')
    args = p.parse_args()

    if args.save:
        matplotlib.use('Agg')
    else:
        try:    matplotlib.use('TkAgg')
        except: matplotlib.use('Agg'); args.save=True

    d      = load(args.npz)
    base   = os.path.basename(args.npz).replace('.npz','')
    os.makedirs(args.outdir, exist_ok=True)
    prefix = os.path.join(args.outdir, base)

    print_table(d)
    print('\nGenerating figures...')
    figure1_bode(d,                 save=args.save, prefix=prefix)
    figure2_bw_alpha(d,             save=args.save, prefix=prefix)
    figure3_dr_range(d,             save=args.save, prefix=prefix)
    figure4_walking_attenuation(d,  save=args.save, prefix=prefix)

    if not args.save:
        plt.show()
    else:
        print(f'\nAll 4 figures saved to: {args.outdir}/')

if __name__=='__main__':
    main()