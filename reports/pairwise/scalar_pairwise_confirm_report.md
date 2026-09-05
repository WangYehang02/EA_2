# Scalar pairwise confirm report

**Status: CONFIRMED**

PAIRWISE CONFIRMED: scalar pairwise improved over both the fixed rescoring baseline and the residual-only control on the held-out confirm set (catalog-assisted S-phase candidate re-picking / reranking).

n_traces=43090 n_events=2669 n_ge2=27560 tau=0.50

## Metrics
- fixed  F1@0.5=0.837317 F1@0.1=0.503458 P95=2.4655
- resid  F1@0.5=0.845858 Δfixed=+0.008540
- scalar F1@0.5=0.853493 Δfixed=+0.016175 Δresid=+0.007635
- fixes=907 breaks=210 net=697 switch_prec=0.8120 Q2_rec=0.7732

## Bootstrap (event, 5000)
- vs fixed: {'mean': 0.016163862273741388, 'ci95': [0.01456127724079393, 0.01780921305127694]}
- vs resid: {'mean': 0.007632411520812627, 'ci95': [0.006464639853055701, 0.008829127388887319]}

## Effect replication
          split    fixed    resid   scalar  scalar-fixed  scalar-resid  ratio_scalar_fixed_vs_fulldev  ratio_scalar_fixed_vs_heldout  ratio_scalar_resid_vs_fulldev  ratio_scalar_resid_vs_heldout
  pilot-heldout 0.919600 0.935000 0.940300      0.020700      0.005256                       1.510949                       1.000000                       0.719980                       1.000000
phaseB-full-dev 0.866800 0.873200 0.880500      0.013700      0.007300                       1.000000                       0.661836                       1.000000                       1.388928
        confirm 0.837317 0.845858 0.853493      0.016175      0.007635                       1.180690                       0.781423                       1.045915                       1.452701

## Margin bins (post-hoc)
   margin_bin     n  n_events  fixed_f1@0.5  resid_f1@0.5  scalar_f1@0.5  delta_scalar_fixed  delta_scalar_resid  fixes  breaks  net_fixes  switch_precision  Q2_recovery_rate
 lowest_10pct  2756      1430      0.819303      0.863570       0.866836            0.047533            0.003266    152      21        131          0.878613          0.853933
      p10_p30  5512      1931      0.893324      0.912010       0.915639            0.022315            0.003628    156      33        123          0.825397          0.861878
      p30_p60  8268      2287      0.865989      0.878447       0.892477            0.026488            0.014030    274      55        219          0.832827          0.825301
highest_40pct 11024      2441      0.691491      0.695120       0.711811            0.020319            0.016691    325     101        224          0.762911          0.674274

Hard stop. No retrain. No confirm retune. No next stage.
