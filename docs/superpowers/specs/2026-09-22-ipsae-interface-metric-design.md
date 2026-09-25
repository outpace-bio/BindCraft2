# ipSAE as an interface metric alongside ipTM

Date: 2026-09-22
Branch: `explore/ipsae-instead-of-iptm`
Status: design approved, not implemented

## Why

ipTM is insensitive on small and asymmetric interfaces, which is the regime most of our
campaigns live in: trimmed targets, membrane-protein loops, peptide epitopes. ipSAE
(Dunbrack 2025, biorxiv 2025.02.10.637595) was built to fix that. We already use it as the
primary interface metric on CLDN6 and GPC3, computed after the fact by the upstream script.
This design moves it inside BindCraft2, where it can gate stages, drive the multitarget
scheduler, weight mutations, and optionally drive the gradient.

One limitation comes out of our own outcome data. Across 78 Octet-tested GPC3 designs, ipSAE
enriched for binders without separating them, and the highest-scoring design in the set did
not bind. It is a better filter than ipTM and nothing more, so `i_pTM` stays in the output.

## Definition: what "ipSAE" means here

We reproduce the paper's headline number exactly.

- Variant `d0res`. Upstream also emits `d0chn` and `d0dom`, and neither is the headline score.
- Reduced as `max` over both chain directions, matching upstream's `Type == max` rows. The
  GPC3 cut points (0.70 and 0.85) are calibrated against this, so they transfer.
- `pae_cutoff = 10`.
- `d0` from `calc_d0_array`, not `calc_d0`. Upstream carries two d0 functions that disagree:

  | | `ipsae.py:116` `calc_d0` | `ipsae.py:126` `calc_d0_array` |
  |---|---|---|
  | form | `1.24*(L-15)^(1/3) - 1.8 if L > 27 else 1.0` | `L = max(26, L)`, then formula |
  | floor | 1.0 | 1.0 |
  | used for | `d0chn`, `d0dom` | `d0res`, i.e. ipSAE |
  | at `L = 27` | 1.0 | 1.0389 |

  They agree everywhere but `L = 27`. `calc_d0_array` is the one that applies to this score.

ipSAE reads only the PAE matrix and the chain assignment. `dist_cutoff` feeds pDockQ and the
diagnostic residue counts (`ipsae.py:759`) and never enters ipSAE, so we need no coordinates.

## Architecture

### 1. `bindcraft/ipsae.py`, a new pure-JAX module

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

`af2.py:145 alphafold_prediction_metrics` already builds the full PAE matrix at line 158.
We add two entries there:

```
metrics['ipsae']              # scalar, max over both directions, cutoff 10
metrics['ipsae_per_residue']  # binder->target byres array
```

Both read the raw `pae` local at `af2.py:158`, not `metrics['pae']`. Line 159 stores
`(pae + pae.T) / 2`, and upstream ipSAE runs on the raw asymmetric AF2 matrix. On the golden
fixture the symmetrized matrix gives 0.033419 where upstream gives 0.047426, a 30% error, so
the distinction is not cosmetic.

`ipsae_per_residue` holds the binder->target direction rather than the max, because
`sequence_optimization.py` weights mutations and needs one value per binder residue. The
target->binder array indexes over target residues and cannot weight a binder mutation.

This needs a new `binder_residue_mask(chain_names, chain_lengths)` beside `interface_asym_ids`
(`af2.py:108`). `interface_asym_ids` folds every binder chain into one assembly but leaves the
target chains separate, so on a multi-chain target it does not give us a binder/target binary.
We pad it at `af2.py:316` with 0, where `interface_asym_id` pads with `len(chain_names)`.

Filters, scheduler, reporting and MPNN then read `metrics['ipsae']` the same way they read
`metrics['iptm']` today, so none of them needs new plumbing.

### 3. Loss: `weights_ipsae_loss`, default `0`

```python
@loss('ipsae_loss', target_weighting='binds_target')
def ipsae_loss(protein_states, predictions, prediction_state='complex',
               pae_cutoff=10.0, warmup_cutoff=30.0, temperature=0.1):
    # cutoff = warmup_cutoff - (warmup_cutoff - pae_cutoff) * sequence_hardness
    return 1 - soft_ipsae(..., cutoff, temperature)
```

The gradient needs three departures from the paper, and nothing we report shares them. At
`cutoff = 10`, step 0 of a hallucination trajectory sits at PAE around 30 everywhere. No pair
passes the mask, ipSAE is identically 0, the gradient is identically 0, and the loss has
nothing to descend. ipTM carries no cutoff and always pulls. So inside the gradient the cutoff
anneals from 30 down to 10 and `max` becomes `soft_maximum`. The loss also reads the
symmetrized `metrics['pae']` rather than the raw matrix, since the raw one is local to
`alphafold_prediction_metrics` and keeping a second N-by-N array on every prediction is not
worth it for a term that already departs twice. Anything we report, filter, rank or schedule
on still uses the raw PAE, the paper's fixed cutoff of 10, and a hard `max`.

The anneal rides `sequence_hardness`, the mean max-probability of the sequence that
`loss.py:851` already computes. That walks the cutoff through screen, refine, anneal and
harden without threading a step count through `run_gradient_design_stage`, and it reaches
exactly 10 once the sequence is discrete. The loss recomputes from `metrics['pae']` instead of
reading `metrics['ipsae']`, since it needs both the moving cutoff and the soft reduction.

### 4. Settings, all additive

| New setting | Mirrors | Default |
|---|---|---|
| `min_ipsae_{stage}`, `min_ipsae_final` | `min_iptm_*` | unset |
| `max_detarget_ipsae`, `max_detarget_ipsae_{stage}` | `max_detarget_iptm*` | unset |
| `weights_ipsae_loss` | `weights_iptm_loss` | `0` |
| `multitarget_swap_metric` (`'iptm'` or `'ipsae'`) | none | `'iptm'` |

These touch `settings.py:36` `CAMPAIGN_SETTING_NAMES`, `:37` `FINAL_CONFIDENCE_FILTERS`
(`min_ipsae_final -> 'i_pSAE'`), `:410` the `{filter}_{stage}` product, `filters.py:812`
`design_stage_filters`, and `preflight.py:167`.

Every default is off. We compute and report ipSAE on every run, and it gates nothing and
optimizes nothing until someone sets a weight or a threshold. Existing settings JSONs and
examples keep behaving as they do now, which lets us A/B ipTM against ipSAE from one binary.
`max_detarget_ipsae` comes out stricter than its ipTM twin, since a `max` reduction only
passes when both directions are low.

### 5. Reporting: column `i_pSAE`

Added to `score.py:17` `PREDICTED_METRICS`, `campaign_output.py:89`
`LEADING_CONFIDENCE_COLUMNS`, `rank.py:18` (`i_pSAE_detarget` joins lower-is-better) with
glossary entries at `:29` and `:60`, `protein.py:31` `QUALITY_METRIC_TYPES`, and
`MPNN_stage.py:98` bounds, which gain `'ipsae': (0.0, 1.0)`.

`sequence_optimization.py:145` gains `'interface_ipsae'` as a third `mutation_weighting`
value, reading `ipsae_per_residue` beside the existing `'interface_iptm'`. `RANKING_METRIC`
stays `i_pDAE`.

### 6. Validation

The repo has no `tests/` directory and no pytest config, so we create both and add pytest as a
dev dependency in `pyproject.toml`.

1. Golden test against upstream. BindCraft2 writes no PAE json, so the fixture is a committed
   synthetic two-chain model (12-residue binder, 30-residue target) whose interface patch
   drives `n0res` across 23 to 28 and so exercises the d0 clamp. Running
   `/home/bobbylangan/workdir/packages/IPSAE/ipsae.py` (v4) on it at cutoffs 10 and 15 gives
   binder->target 0.047426, target->binder 0.020000, and `max` 0.047426. The test asserts the
   JAX port hits all three to 1e-6. Nothing ships until this passes; it is what makes the port
   trustworthy.
2. `d0` boundary. Assert `d0` at `n0res` of 26, 27 and 28 equals `calc_d0_array`, giving
   1.0, 1.0389 and 1.1157. That is where the two upstream d0 functions disagree.
3. Gradient liveness. `soft_ipsae` stays finite and non-zero at PAE around 30 with a cutoff of
   30, and its gradient with respect to PAE is non-zero.
4. Empty interface. With no pair under the cutoff, the score is exactly 0.0, with no NaN and a
   finite gradient.
5. Direction. On an asymmetric synthetic PAE, `ipsae` equals the larger of the two directional
   values, and `ipsae_per_residue` has one entry per binder residue.

## Out of scope

- Replacing or renaming any `iptm` setting, loss or column. `iptm_loss`, `min_iptm_*`,
  `max_detarget_iptm*` and `i_pTM` all stay and keep working.
- `pDockQ`, `pDockQ2` and `LIS`. Upstream computes them and nothing here needs them.
- The `d0chn` and `d0dom` variants.
- Changing `RANKING_METRIC`.

## Risks

- The GPC3 cut points assume 5-seed means. The table behind 0.70 and 0.85 averaged ipSAE over
  5 seeds, while a BindCraft2 `i_pSAE` comes from one prediction. Check the seed-to-seed
  spread before reusing those thresholds.
- A `max` reduction can be carried by the target. A large target chain that confidently
  locates a small binder scores the pair well. We accept this to match the paper, so read
  `i_pSAE` alongside `i_pTM` and the interface-residue counts rather than by itself.
- We have not tested the annealed loss as an optimizer, and it may turn out worse than
  `iptm_loss`. That is why it defaults to weight 0. The filter and scheduler work does not
  depend on it.
