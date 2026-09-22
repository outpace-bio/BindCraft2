# ipSAE as an interface metric alongside ipTM

Date: 2026-09-22
Branch: `explore/ipsae-instead-of-iptm`
Status: design approved, not implemented

## Why

ipTM is insensitive on small and asymmetric interfaces — the regime most of this group's
campaigns live in (trimmed targets, membrane-protein loops, peptide epitopes). ipSAE
(Dunbrack 2025, biorxiv 2025.02.10.637595) is built to fix exactly that, and is already the
primary interface metric in the CLDN6 and GPC3 campaigns, computed post-hoc by the upstream
script. This design brings it inside BindCraft2 so it can gate stages, drive the multitarget
scheduler, weight mutations, and optionally drive the gradient.

Known limitation, from this group's own outcome data: on 78 Octet-tested GPC3 designs ipSAE
**enriches but does not separate** — the single highest-scoring design in the set was a
non-binder. ipSAE is a better filter than ipTM, not an oracle. `i_pTM` stays in the output.

## Definition — what "ipSAE" means here

The paper's headline number, reproduced exactly:

- **Variant `d0res`.** Upstream also emits `d0chn` and `d0dom`; neither is the headline score.
- **Reduced as `max` over both chain directions**, matching upstream's `Type == max` rows.
  This is what the GPC3 cut points (0.70 / 0.85) are calibrated against, so they transfer.
- **`pae_cutoff = 10`.**
- **`d0` from `calc_d0_array`, not `calc_d0`.** Upstream has two d0 functions that disagree:

  | | `ipsae.py:116` `calc_d0` | `ipsae.py:126` `calc_d0_array` |
  |---|---|---|
  | form | `1.24*(L-15)^(1/3) - 1.8 if L > 27 else 1.0` | `L = max(26, L)`, then formula |
  | floor | 1.0 | 1.0 |
  | used for | `d0chn`, `d0dom` | **`d0res` — i.e. ipSAE** |
  | at `L = 27` | **1.0** | **1.0389** |

  They agree everywhere except `L = 27`. `calc_d0_array` is correct for this score.

**ipSAE needs only the PAE matrix and chain assignment.** `dist_cutoff` feeds pDockQ and the
diagnostic residue counts (`ipsae.py:759`); it never enters ipSAE. No coordinates required.

## Architecture

### 1. `bindcraft/ipsae.py` — new module, pure JAX

```
d0(L)        = maximum(1.0, 1.24*(maximum(26, L) - 15)^(1/3) - 1.8)
ptm(x, d0)   = 1 / (1 + (x/d0)^2)
mask         = outer(from_mask, to_mask) & (pae < cutoff)
n0[i]        = stop_gradient(mask[i].sum())
byres[i]     = masked_mean(ptm(pae[i], d0(n0[i])), mask[i])          # 0 where n0[i] == 0
```

Public surface:

| function | reduction | consumer |
|---|---|---|
| `ipsae_by_residue(pae, from_mask, to_mask, cutoff)` | none | everything below |
| `ipsae(pae, binder_mask, target_mask, cutoff)` | `max` over both directions' byres | metric, filters, scheduler |
| `soft_ipsae(pae, binder_mask, target_mask, cutoff, temperature)` | `soft_maximum` over `concat` of both directions' byres | loss only |

`soft_maximum` already exists at `loss.py:227`.

### 2. Computed once, in the graph, beside ipTM

`af2.py:145 alphafold_prediction_metrics` already builds the full PAE matrix at line 158. Add:

```
metrics['ipsae']              # scalar, max over both directions, cutoff 10
metrics['ipsae_per_residue']  # binder->target byres array
```

`ipsae_per_residue` is deliberately **binder→target only**, not the max direction: it feeds
`sequence_optimization.py` mutation weighting, which needs one value per *binder* residue.
The target→binder array is indexed over target residues and cannot weight a binder mutation.

Requires a new `binder_residue_mask(chain_names, chain_lengths)` beside `interface_asym_ids`
(`af2.py:108`), because `interface_asym_ids` folds all binder chains into one assembly but
leaves target chains separate — it is not a binder/target binary on multi-chain targets.
Padded at `af2.py:316` with 0, as `interface_asym_id` is padded with `len(chain_names)`.

**Consequence: filters, scheduler, reporting and MPNN read `metrics['ipsae']` exactly as they
read `metrics['iptm']` today.** No new plumbing in any of them.

### 3. Loss — `weights_ipsae_loss`, default `0`

```python
@loss('ipsae_loss', target_weighting='binds_target')
def ipsae_loss(protein_states, predictions, prediction_state='complex',
               pae_cutoff=10.0, warmup_cutoff=30.0, temperature=0.1):
    # cutoff = warmup_cutoff - (warmup_cutoff - pae_cutoff) * sequence_hardness
    return 1 - soft_ipsae(..., cutoff, temperature)
```

**Why the loss deviates, and why nothing reported does.** At `cutoff = 10`, step 0 of a
hallucination trajectory has PAE ≈ 30 everywhere, no pair passes the mask, ipSAE is
identically 0 and the gradient is identically 0 — the loss cannot descend. ipTM has no cutoff
and always pulls. So inside the gradient only, the cutoff anneals 30 → 10 and `max` becomes
`soft_maximum`. Every value that is reported, filtered, ranked or scheduled on uses the
paper's fixed cutoff 10 and hard `max`.

The anneal rides `sequence_hardness` — the mean max-probability of the sequence, the existing
`loss.py:851` idiom — so it walks screen → refine → anneal → harden with no threading through
`run_gradient_design_stage`, and reaches exactly 10 where the sequence is discrete. The loss
recomputes from `metrics['pae']` rather than reading `metrics['ipsae']`, since it needs the
moving cutoff and the soft reduction.

### 4. Settings — purely additive, nothing existing changes meaning

| New setting | Mirrors | Default |
|---|---|---|
| `min_ipsae_{stage}`, `min_ipsae_final` | `min_iptm_*` | unset |
| `max_detarget_ipsae`, `max_detarget_ipsae_{stage}` | `max_detarget_iptm*` | unset |
| `weights_ipsae_loss` | `weights_iptm_loss` | `0` |
| `multitarget_swap_metric` (`'iptm'`\|`'ipsae'`) | — | `'iptm'` |

Touches `settings.py:36` `CAMPAIGN_SETTING_NAMES`, `:37` `FINAL_CONFIDENCE_FILTERS`
(`min_ipsae_final -> 'i_pSAE'`), `:410` the `{filter}_{stage}` product, `filters.py:812`
`design_stage_filters`, `preflight.py:167`.

**All defaults off.** ipSAE is computed and reported on every run; nothing gates or optimizes
on it until a weight or threshold is set. Every existing settings JSON and example keeps
behaving identically, so ipTM and ipSAE can be A/B'd from one binary. `max_detarget_ipsae`
is stricter than its ipTM twin for free: a `max` reduction demands *both* directions be low.

### 5. Reporting — column `i_pSAE`

`score.py:17` `PREDICTED_METRICS` · `campaign_output.py:89` `LEADING_CONFIDENCE_COLUMNS` ·
`rank.py:18` (`i_pSAE_detarget` joins lower-is-better) and `:29`/`:60` glossary ·
`protein.py:31` `QUALITY_METRIC_TYPES` · `MPNN_stage.py:98` bounds gain `'ipsae': (0.0, 1.0)`.

`sequence_optimization.py:145` gains `'interface_ipsae'` as a third `mutation_weighting`
value reading `ipsae_per_residue`, alongside the existing `'interface_iptm'`.
`RANKING_METRIC` stays `i_pDAE` — unchanged.

### 6. Validation

The repo has no `tests/` directory and no pytest config; both are created, with pytest added
as a dev dependency in `pyproject.toml`.

1. **Golden test against upstream.** Run
   `/home/bobbylangan/workdir/packages/IPSAE/ipsae.py` (v4) on a real AF2 PDB + PAE json from
   a past campaign and assert the JAX `ipsae` matches upstream's `ipSAE` value on the `Type ==
   max` row to ~1e-5. This is the test that makes the port trustworthy; nothing ships without it.
2. **`d0` boundary.** Assert `d0` at `n0res` 26 / 27 / 28 equals `calc_d0_array`, specifically
   1.0 / 1.0389 / 1.1157 — the case where the two upstream d0 functions disagree.
3. **Gradient liveness.** `soft_ipsae` is finite and strictly non-zero at PAE ≈ 30 with
   cutoff 30, and its gradient w.r.t. PAE is non-zero.
4. **Empty interface.** No pair under cutoff → exactly 0.0, no NaN, gradient finite.
5. **Direction.** On an asymmetric synthetic PAE, `ipsae` equals the larger of the two
   directional values and `ipsae_per_residue` has length equal to the binder.

## Out of scope

- Replacing or renaming any `iptm` setting, loss or column. `iptm_loss`, `min_iptm_*`,
  `max_detarget_iptm*` and `i_pTM` all stay and keep working.
- `pDockQ`, `pDockQ2`, `LIS` — upstream computes them; nothing here needs them.
- The `d0chn` and `d0dom` variants.
- Changing `RANKING_METRIC`.

## Risks

- **GPC3 cut points assume 5-seed means.** The table behind 0.70 / 0.85 averaged ipSAE over 5
  seeds; a BindCraft2 `i_pSAE` is one prediction. Thresholds need a spread check before reuse.
- **`max` can be carried by the target.** A large target chain confidently locating a small
  binder scores the pair well. Accepted deliberately, to match the paper; `i_pSAE` should be
  read next to `i_pTM` and the interface-residue counts, not alone.
- **The annealed loss is untested as an optimizer.** It may prove worse than `iptm_loss`. It
  defaults to weight 0 for that reason; the filter and scheduler work stands on its own.
