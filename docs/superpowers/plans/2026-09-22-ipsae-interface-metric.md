# ipSAE Interface Metric Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ipSAE (Dunbrack 2025) to BindCraft2 as a reported metric, a stage/final filter, a multitarget scheduler signal, a mutation weighting, and an optional gradient loss, without changing the meaning of anything ipTM does today.

**Architecture:** A new pure-JAX module `bindcraft/ipsae.py` computes the score from a PAE matrix and two chain masks. `af2.py` calls it once per prediction and stores `metrics['ipsae']` and `metrics['ipsae_per_residue']`, so every downstream consumer reads a scalar exactly as it reads `metrics['iptm']` today. A separate soft variant lives in the loss, where the PAE cutoff anneals and the max becomes a softmax.

**Tech Stack:** Python 3.12, JAX 0.11.x, pytest (added by Task 1), NumPy.

**Spec:** `docs/superpowers/specs/2026-09-22-ipsae-interface-metric-design.md`

## Global Constraints

- The reported score reproduces upstream `ipsae.py` v4 exactly: the `d0res` variant, `max` over both chain directions, `pae_cutoff = 10.0`.
- `d0` uses `calc_d0_array` semantics (`L = max(26, L)`, then `max(1.0, 1.24*(L-15)^(1/3) - 1.8)`), never the `calc_d0` scalar branch at `L > 27`. The two disagree at `L = 27`: 1.0389 against 1.0.
- `metrics['ipsae']` reads the raw asymmetric `pae` local at `af2.py:158`, never `metrics['pae']`, which `af2.py:159` symmetrizes. On the fixture, symmetrizing gives 0.033419 against upstream's 0.047426.
- Nothing existing is renamed or removed. `iptm_loss`, `min_iptm_*`, `max_detarget_iptm*`, `i_pTM` and `RANKING_METRIC = 'i_pDAE'` all stay and keep their meaning.
- Every new setting defaults to off: `weights_ipsae_loss = 0`, every `min_ipsae_*` and `max_detarget_ipsae*` unset, `multitarget_swap_metric = 'iptm'`.
- Upstream reference script, used only by the golden test: `/home/bobbylangan/workdir/packages/IPSAE/ipsae.py`.
- Branch: `explore/ipsae-instead-of-iptm`, off `dev`. Do not merge; the branch is reviewed as a whole.
- **Interpreter: `/home/bobbylangan/.conda/envs/bindcraft2/bin/python`.** This box has 37 conda envs and the default `python` has no jax. Every command below uses `$PY`, so export it once per shell:

  ```bash
  export PY=/home/bobbylangan/.conda/envs/bindcraft2/bin/python
  ```

  That env holds jax 0.11.2 and an editable `bindcraft` pointing at this repo.

---

## File Structure

| Path | Responsibility | Task |
|---|---|---|
| `bindcraft/ipsae.py` | Create. The score itself: `d0`, `ipsae_by_residue`, `ipsae`, `soft_ipsae`. No I/O, no settings, no knowledge of BindCraft types. | 1 |
| `tests/test_ipsae.py` | Create. Golden check against upstream, d0 boundary, gradient liveness, empty interface, direction. | 1 |
| `tests/fixtures/ipsae_model.pdb` | Create. Two-chain synthetic model for the golden test. | 1 |
| `tests/fixtures/ipsae_pae.json` | Create. Matching asymmetric PAE matrix. | 1 |
| `tests/fixtures/make_ipsae_fixture.py` | Create. Regenerates both fixtures deterministically. | 1 |
| `pyproject.toml` | Modify. Add a `dev` extra carrying pytest. | 1 |
| `bindcraft/af2.py` | Modify. `binder_residue_mask`, plus the two new metrics. | 2 |
| `bindcraft/filters.py` | Modify. `ipsae_metric`, and `min_ipsae_*` / `max_detarget_ipsae_*` stage filters. | 3 |
| `bindcraft/settings.py` | Modify. Register the new setting names. | 3 |
| `bindcraft/preflight.py` | Modify. Let an ipSAE ceiling satisfy the detarget check. | 3 |
| `bindcraft/loss.py` | Modify. `ipsae_loss`. | 4 |
| `bindcraft/target_schedule.py` | Modify. `multitarget_swap_metric`. | 5 |
| `bindcraft/score.py`, `campaign_output.py`, `rank.py`, `protein.py`, `MPNN_stage.py`, `sequence_optimization.py` | Modify. Reporting columns and `interface_ipsae` weighting. | 6 |
| `docs/reference.md`, `docs/outputs.md` | Modify. Document the setting and the column. | 7 |

---

### Task 1: The ipSAE module, verified against upstream

**Files:**
- Create: `bindcraft/ipsae.py`
- Create: `tests/__init__.py` (empty), `tests/test_ipsae.py`
- Create: `tests/fixtures/make_ipsae_fixture.py`, `tests/fixtures/ipsae_model.pdb`, `tests/fixtures/ipsae_pae.json`
- Modify: `pyproject.toml` (add `[project.optional-dependencies] dev`)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `d0(length: Array) -> Array`
  - `ipsae_by_residue(pae: Array, from_mask: Array, to_mask: Array, pae_cutoff: float = 10.0) -> Array` — returns one value per row of `pae`, zero for rows outside `from_mask`.
  - `ipsae(pae: Array, binder_mask: Array, target_mask: Array, pae_cutoff: float = 10.0) -> Array` — scalar, `max` over both directions.
  - `soft_ipsae(pae: Array, binder_mask: Array, target_mask: Array, pae_cutoff: float = 10.0, temperature: float = 0.1) -> Array` — scalar, `soft_maximum` over both directions concatenated.

- [ ] **Step 1: Add the pytest dev extra**

In `pyproject.toml`, inside the existing `[project.optional-dependencies]` block, add this line directly after the `oneapi = [...]` line:

```toml
dev = ["pytest>=8"]
```

Declaring it installs nothing, so install it into the env now:

```bash
export PY=/home/bobbylangan/.conda/envs/bindcraft2/bin/python
$PY -m pip install 'pytest>=8'
$PY -m pytest --version
```

Expected: a version at or above 8.

- [ ] **Step 2: Write the fixture generator**

Create `tests/fixtures/make_ipsae_fixture.py`. BindCraft2 writes no PAE json, so the golden test needs a synthetic model. The interface patch drives `n0res` from 23 to 28, which straddles the `d0` clamp at 26, and the two directions are deliberately asymmetric so the test catches any accidental symmetrization.

```python
"""Regenerate the ipSAE golden-test fixtures.

Run from the repository root:

    $PY tests/fixtures/make_ipsae_fixture.py

Then re-derive the expected values with upstream ipsae.py v4:

    $PY /home/bobbylangan/workdir/packages/IPSAE/ipsae.py \
        tests/fixtures/ipsae_pae.json tests/fixtures/ipsae_model.pdb 10 15

and copy the ipSAE column of the two `asym` rows and the one `max` row into
UPSTREAM_* in tests/test_ipsae.py.
"""
import json
import os

import numpy as np

BINDER_RESIDUES = 12
TARGET_RESIDUES = 30
RESIDUES = BINDER_RESIDUES + TARGET_RESIDUES
FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))


def build_pae() -> np.ndarray:
    pae = np.full((RESIDUES, RESIDUES), 25.0)
    pae[:BINDER_RESIDUES, :BINDER_RESIDUES] = 2.0
    pae[BINDER_RESIDUES:, BINDER_RESIDUES:] = 2.5
    for residue in range(3, 9):
        contacts = 23 + (residue - 3)
        pae[residue, BINDER_RESIDUES:BINDER_RESIDUES + contacts] = 5.0
        pae[BINDER_RESIDUES:BINDER_RESIDUES + contacts, residue] = 7.0
    np.fill_diagonal(pae, 0.0)
    return pae


def build_pdb(plddt: np.ndarray) -> str:
    lines, serial = [], 0
    for chain, count, offset in (('A', BINDER_RESIDUES, 0), ('B', TARGET_RESIDUES, BINDER_RESIDUES)):
        for residue in range(count):
            for atom, shift in (('CA', 0.0), ('CB', 1.5)):
                serial += 1
                x = residue * 3.8 + shift + (0.0 if chain == 'A' else 12.0)
                lines.append(f'ATOM  {serial:>5}  {atom:<3} ALA {chain}{residue + 1:>4}    '
                             f'{x:>8.3f}{0.0:>8.3f}{0.0:>8.3f}  1.00{plddt[offset + residue]:>6.2f}           C')
    return '\n'.join(lines + ['END']) + '\n'


def main() -> None:
    pae = build_pae()
    plddt = np.linspace(70.0, 95.0, RESIDUES)
    with open(os.path.join(FIXTURE_DIR, 'ipsae_pae.json'), 'w') as handle:
        json.dump({'pae': pae.tolist(), 'plddt': plddt.tolist(), 'ptm': 0.5, 'iptm': 0.4}, handle)
    with open(os.path.join(FIXTURE_DIR, 'ipsae_model.pdb'), 'w') as handle:
        handle.write(build_pdb(plddt))


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Generate the fixtures and confirm upstream's numbers**

```bash
cd /home/bobbylangan/workdir/packages/BindCraft2
$PY tests/fixtures/make_ipsae_fixture.py
$PY /home/bobbylangan/workdir/packages/IPSAE/ipsae.py \
    "$PWD/tests/fixtures/ipsae_pae.json" "$PWD/tests/fixtures/ipsae_model.pdb" 10 15
cat tests/fixtures/ipsae_model_10_15.txt
```

Expected: three rows. `A B ... asym` with ipSAE `0.047426`, `B A ... asym` with `0.020000`, `A B ... max` with `0.047426`. The `n0res` column reads 28 on the A->B rows and 6 on the B->A row.

Delete the three files upstream writes beside the fixture, they are not committed:

```bash
rm tests/fixtures/ipsae_model_10_15.txt tests/fixtures/ipsae_model_10_15_byres.txt tests/fixtures/ipsae_model_10_15.pml
```

- [ ] **Step 4: Write the failing tests**

Create `tests/__init__.py` as an empty file, then create `tests/test_ipsae.py`:

```python
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bindcraft.ipsae import d0, ipsae, ipsae_by_residue, soft_ipsae

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')
BINDER_RESIDUES = 12
# From upstream ipsae.py v4 on the committed fixture at cutoffs 10 and 15.
# Regenerate with tests/fixtures/make_ipsae_fixture.py; see its docstring.
UPSTREAM_BINDER_TO_TARGET = 0.047426
UPSTREAM_TARGET_TO_BINDER = 0.020000
UPSTREAM_MAX = 0.047426


@pytest.fixture
def fixture_pae():
    with open(os.path.join(FIXTURE_DIR, 'ipsae_pae.json')) as handle:
        return jnp.asarray(json.load(handle)['pae'])


@pytest.fixture
def chain_masks(fixture_pae):
    residues = fixture_pae.shape[0]
    binder = jnp.arange(residues) < BINDER_RESIDUES
    return binder.astype(jnp.float32), (~binder).astype(jnp.float32)


def test_matches_upstream_on_both_directions(fixture_pae, chain_masks):
    binder, target = chain_masks
    forward = float(ipsae_by_residue(fixture_pae, binder, target).max())
    reverse = float(ipsae_by_residue(fixture_pae, target, binder).max())
    assert forward == pytest.approx(UPSTREAM_BINDER_TO_TARGET, abs=1e-6)
    assert reverse == pytest.approx(UPSTREAM_TARGET_TO_BINDER, abs=1e-6)


def test_matches_upstream_max(fixture_pae, chain_masks):
    binder, target = chain_masks
    assert float(ipsae(fixture_pae, binder, target)) == pytest.approx(UPSTREAM_MAX, abs=1e-6)


def test_symmetrized_pae_would_disagree(fixture_pae, chain_masks):
    """Guards the af2.py:158 vs :159 distinction: ipSAE must read the raw matrix."""
    binder, target = chain_masks
    symmetrized = (fixture_pae + fixture_pae.T) / 2
    assert float(ipsae(symmetrized, binder, target)) != pytest.approx(UPSTREAM_MAX, abs=1e-3)


def test_d0_matches_calc_d0_array_across_the_clamp():
    assert [round(float(d0(jnp.asarray(float(n)))), 4) for n in (25, 26, 27, 28)] == [1.0, 1.0, 1.0389, 1.1157]


def test_empty_interface_is_zero_with_finite_gradient(chain_masks):
    binder, target = chain_masks
    pae = jnp.full((BINDER_RESIDUES + 30, BINDER_RESIDUES + 30), 30.0)
    assert float(ipsae(pae, binder, target)) == 0.0
    gradient = jax.grad(lambda matrix: soft_ipsae(matrix, binder, target))(pae)
    assert bool(jnp.isfinite(gradient).all())


def test_soft_ipsae_has_live_gradient_at_a_wide_cutoff(chain_masks):
    binder, target = chain_masks
    pae = jnp.full((BINDER_RESIDUES + 30, BINDER_RESIDUES + 30), 30.0)
    value = soft_ipsae(pae, binder, target, pae_cutoff=31.0)
    gradient = jax.grad(lambda matrix: soft_ipsae(matrix, binder, target, pae_cutoff=31.0))(pae)
    assert float(value) > 0.0
    assert float(jnp.abs(gradient).sum()) > 0.0


def test_max_takes_the_reverse_direction_when_it_leads():
    """A target that locates the binder confidently while the binder does not reciprocate."""
    residues = 40
    binder = (jnp.arange(residues) < 10).astype(jnp.float32)
    target = (jnp.arange(residues) >= 10).astype(jnp.float32)
    pae = np.full((residues, residues), 30.0)
    pae[10:40, 0:10] = 1.0
    pae = jnp.asarray(pae)
    forward = float(ipsae_by_residue(pae, binder, target).max())
    reverse = float(ipsae_by_residue(pae, target, binder).max())
    assert reverse > forward
    assert float(ipsae(pae, binder, target)) == pytest.approx(reverse, abs=1e-6)


def test_by_residue_is_zero_outside_the_from_mask(fixture_pae, chain_masks):
    binder, target = chain_masks
    values = ipsae_by_residue(fixture_pae, binder, target)
    assert values.shape == (fixture_pae.shape[0],)
    assert float(jnp.abs(values[BINDER_RESIDUES:]).max()) == 0.0
```

- [ ] **Step 5: Run the tests to verify they fail**

Run: `cd /home/bobbylangan/workdir/packages/BindCraft2 && $PY -m pytest tests/test_ipsae.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'bindcraft.ipsae'`.

- [ ] **Step 6: Write the module**

Create `bindcraft/ipsae.py`:

```python
"""ipSAE, the interface score of Dunbrack 2025 (biorxiv 2025.02.10.637595).

Reproduces the `d0res` variant that upstream ipsae.py reports as its headline `ipSAE`
column, reduced as the max over both chain directions. The score reads only a PAE matrix
and two chain masks; no coordinates are involved, because upstream's `dist_cutoff` feeds
pDockQ and the diagnostic residue counts rather than ipSAE itself.

Pass the RAW predicted aligned error. Upstream runs on AlphaFold's asymmetric matrix, so a
symmetrized one gives a different number.
"""
import jax
import jax.numpy as jnp
from jax import Array

DEFAULT_PAE_CUTOFF = 10.0
#upstream calc_d0_array clamps the length rather than branching on it, which is what the
#d0res variant uses; the scalar calc_d0 branches at L > 27 and is for d0chn and d0dom only
D0_MINIMUM_LENGTH = 26.0
D0_MINIMUM = 1.0


def d0(length: Array) -> Array:
    return jnp.maximum(D0_MINIMUM, 1.24 * jnp.cbrt(jnp.maximum(D0_MINIMUM_LENGTH, length) - 15.0) - 1.8)


def ipsae_by_residue(pae: Array, from_mask: Array, to_mask: Array, pae_cutoff: float=DEFAULT_PAE_CUTOFF) -> Array:
    """One ipSAE value per residue of the aligned chain, zero outside from_mask."""
    pair_mask = (from_mask[:, None] * to_mask[None, :]) * (pae < pae_cutoff)
    scored_residues = jax.lax.stop_gradient(pair_mask.sum(-1))
    aligned_error = 1.0 / (1.0 + jnp.square(pae / d0(scored_residues)[:, None]))
    return jnp.where(scored_residues > 0, (aligned_error * pair_mask).sum(-1) / jnp.maximum(scored_residues, 1.0), 0.0)


def _both_directions(pae: Array, binder_mask: Array, target_mask: Array, pae_cutoff: float) -> Array:
    return jnp.concatenate([ipsae_by_residue(pae, binder_mask, target_mask, pae_cutoff),
                            ipsae_by_residue(pae, target_mask, binder_mask, pae_cutoff)])


def ipsae(pae: Array, binder_mask: Array, target_mask: Array, pae_cutoff: float=DEFAULT_PAE_CUTOFF) -> Array:
    """The reported score: upstream's `max` row for the binder/target chain pair."""
    return _both_directions(pae, binder_mask, target_mask, pae_cutoff).max()


def soft_ipsae(pae: Array, binder_mask: Array, target_mask: Array, pae_cutoff: float=DEFAULT_PAE_CUTOFF, temperature: float=0.1) -> Array:
    """The gradient variant, with the max softened so more than one residue carries signal."""
    from bindcraft.loss import soft_maximum
    return soft_maximum(_both_directions(pae, binder_mask, target_mask, pae_cutoff), temperature)
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `cd /home/bobbylangan/workdir/packages/BindCraft2 && $PY -m pytest tests/test_ipsae.py -v`
Expected: 8 passed.

If `test_d0_matches_calc_d0_array_across_the_clamp` fails at index 2 with 1.0 instead of 1.0389, the clamp was written as a branch on `L > 27`. Re-read the Global Constraints.

- [ ] **Step 8: Commit**

```bash
cd /home/bobbylangan/workdir/packages/BindCraft2
git add bindcraft/ipsae.py tests/ pyproject.toml
git commit -m "Add ipSAE, checked against the upstream Dunbrack script

The d0res variant reduced as max over both chain directions, which is
what upstream reports as its ipSAE column. d0 follows calc_d0_array
(clamp at 26), not the calc_d0 scalar branch at L > 27; they disagree at
L = 27 and only the array form applies to d0res.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Compute ipSAE on every prediction

**Files:**
- Modify: `bindcraft/af2.py:108-113` (add `binder_residue_mask` after `interface_asym_ids`)
- Modify: `bindcraft/af2.py:145-166` (`alphafold_prediction_metrics`)
- Modify: `bindcraft/af2.py:277-283`, `:304`, `:316`, `:345` (thread the mask through)
- Test: `tests/test_ipsae_metrics.py`

**Interfaces:**
- Consumes: `bindcraft.ipsae.ipsae`, `bindcraft.ipsae.ipsae_by_residue` from Task 1.
- Produces:
  - `binder_residue_mask(chain_names: tuple[str, ...], chain_lengths: tuple[int, ...]) -> Array` — float32, 1.0 on binder-chain residues.
  - `metrics['ipsae']` — scalar Array, max over both directions, cutoff 10.
  - `metrics['ipsae_per_residue']` — Array over all residues, binder->target direction, zero on target and padding rows.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ipsae_metrics.py`:

```python
import jax.numpy as jnp

from bindcraft.af2 import binder_residue_mask


def test_binder_mask_marks_only_binder_chains():
    mask = binder_residue_mask(('binder', 'target'), (3, 4))
    assert mask.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_binder_mask_handles_multi_chain_binders_and_multi_chain_targets():
    mask = binder_residue_mask(('binder_a', 'binder_b', 'target', 'target_2'), (2, 2, 3, 1))
    assert mask.tolist() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_binder_mask_is_all_binder_for_a_binder_alone_state():
    assert binder_residue_mask(('binder',), (5,)).tolist() == [1.0] * 5


def test_binder_mask_follows_chain_name_order_not_sorted_order():
    """concatenate_chain_arrays walks chain_names in the order given, so the mask must too."""
    mask = binder_residue_mask(('target', 'binder'), (2, 3))
    assert mask.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /home/bobbylangan/workdir/packages/BindCraft2 && $PY -m pytest tests/test_ipsae_metrics.py -v`
Expected: `ImportError: cannot import name 'binder_residue_mask' from 'bindcraft.af2'`.

- [ ] **Step 3: Add the mask builder**

In `bindcraft/af2.py`, directly after the `interface_asym_ids` function that ends at line 113, add:

```python
def binder_residue_mask(chain_names: tuple[str, ...], chain_lengths: tuple[int, ...]) -> Array:
    """1.0 on residues of a designed chain. interface_asym_ids cannot stand in for this: it folds
    every binder chain into one assembly but leaves the target chains apart, so on a multi-chain
    target it is not a binder/target binary."""
    return jnp.concatenate([jnp.full((length,), float(is_binder_chain(name)), dtype=jnp.float32) for name, length in zip(chain_names, chain_lengths)])
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /home/bobbylangan/workdir/packages/BindCraft2 && $PY -m pytest tests/test_ipsae_metrics.py -v`
Expected: 4 passed.

- [ ] **Step 5: Thread the mask into the metrics function**

In `bindcraft/af2.py`, add the import. Change line 17's import block by adding `ipsae` and `ipsae_by_residue`:

```python
from bindcraft.ipsae import DEFAULT_PAE_CUTOFF, ipsae, ipsae_by_residue
```

Change the signature at line 145 from:

```python
def alphafold_prediction_metrics(alphafold_outputs: dict, seq_mask: Array, interface_asym_id: Array) -> dict[str, Array]:
```

to:

```python
def alphafold_prediction_metrics(alphafold_outputs: dict, seq_mask: Array, interface_asym_id: Array, binder_mask: Array) -> dict[str, Array]:
```

Then inside the `if 'predicted_aligned_error' in alphafold_outputs:` block, directly after the existing line 161 (`metrics['iptm'], metrics['iptm_per_residue'] = ...`), add:

```python
        #ipSAE reads the RAW asymmetric matrix, not metrics['pae'], which line above symmetrizes;
        #upstream runs on AlphaFold's asymmetric PAE and the two differ by ~30% on a real interface
        real_binder = binder_mask * seq_mask
        real_target = (1.0 - binder_mask) * seq_mask
        metrics['ipsae'] = ipsae(pae, real_binder, real_target, DEFAULT_PAE_CUTOFF)
        metrics['ipsae_per_residue'] = ipsae_by_residue(pae, real_binder, real_target, DEFAULT_PAE_CUTOFF)
```

- [ ] **Step 6: Update the three call sites**

At `af2.py:277`, add `binder_mask: Array` to the `predict_complex_arrays` parameter list, directly after `interface_asym_id: Array`:

```python
            def predict_complex_arrays(model_parameters: Array, key: Array, sequence: Array, atoms: Array, atom_mask: Array, residue_index: Array, asym_id: Array, entity_id: Array, interface_asym_id: Array, binder_mask: Array, seq_mask: Array, flags: Array, dropout: Array, softmax_weight: Array, one_hot_weight: Array, temperature: Array, logit_scale: Array):
```

At `af2.py:283`, pass it through:

```python
                return predicted_atom_positions, predicted_atom_mask, alphafold_prediction_metrics(alphafold_outputs, seq_mask, interface_asym_id, binder_mask)
```

At `af2.py:304`, directly after the `interface_asym_id = interface_asym_ids(...)` line, add:

```python
        binder_mask = binder_residue_mask(chain_names, chain_lengths)
```

At `af2.py:316`, directly after the `interface_asym_id = jnp.pad(...)` line inside `if padding_length:`, add:

```python
            binder_mask = jnp.pad(binder_mask, [0, padding_length])
```

At `af2.py:318`, add `binder_mask` to the call, directly after `interface_asym_id`:

```python
        positions, mask, metrics = self._compiled_complex_prediction(model, padded_residue_count)(self.model_parameters[model], self.key, sequence, atoms, atom_mask, residue_index, asym_id, entity_id, interface_asym_id, binder_mask, seq_mask, flags, jnp.asarray(self.dropout), jnp.asarray(softmax_weight), jnp.asarray(one_hot_weight), jnp.asarray(temperature), jnp.asarray(logit_scale))
```

At `af2.py:345`:

```python
                metrics = alphafold_prediction_metrics(alphafold_outputs, seq_mask, interface_asym_ids(chain_names, chain_lengths), binder_residue_mask(chain_names, chain_lengths))
```

- [ ] **Step 7: Add the padding and ensembling tests**

Append to `tests/test_ipsae_metrics.py`:

```python
import json
import os

from bindcraft.ipsae import ipsae

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')


def test_padding_residues_score_as_neither_chain():
    """A padded prediction must give the same ipSAE as the unpadded one."""
    with open(os.path.join(FIXTURE_DIR, 'ipsae_pae.json')) as handle:
        pae = jnp.asarray(json.load(handle)['pae'])
    residues = pae.shape[0]
    binder = (jnp.arange(residues) < 12).astype(jnp.float32)
    unpadded = float(ipsae(pae, binder, 1.0 - binder))

    padding = 8
    padded_pae = jnp.pad(pae, [[0, padding], [0, padding]])
    padded_binder = jnp.pad(binder, [0, padding])
    seq_mask = jnp.pad(jnp.ones(residues, dtype=jnp.float32), [0, padding])
    padded = float(ipsae(padded_pae, padded_binder * seq_mask, (1.0 - padded_binder) * seq_mask))
    assert padded == unpadded
```

That test matters because padded PAE entries are zero, which is below any cutoff. Without multiplying both masks by `seq_mask`, every padded column would join the target and inflate the score.

- [ ] **Step 8: Run the tests**

Run: `cd /home/bobbylangan/workdir/packages/BindCraft2 && $PY -m pytest tests/ -v`
Expected: 13 passed.

- [ ] **Step 9: Confirm the ensemble path accepts the new metric**

Run:

```bash
cd /home/bobbylangan/workdir/packages/BindCraft2
$PY -c "
from bindcraft.MPNN_stage import REACHABLE_CONFIDENCE_BOUNDS
print('ipsae' in REACHABLE_CONFIDENCE_BOUNDS)
"
```

Expected: `False`. Task 6 adds it. `MPNN_stage.py:90-95` averages every metric it finds, so `ipsae` and `ipsae_per_residue` already ensemble correctly; only `best_reachable_ensemble` needs the bound.

- [ ] **Step 10: Commit**

```bash
cd /home/bobbylangan/workdir/packages/BindCraft2
git add bindcraft/af2.py tests/test_ipsae_metrics.py
git commit -m "Compute ipSAE on every AlphaFold prediction

Reads the raw asymmetric PAE rather than metrics['pae'], which is
symmetrized a line earlier; upstream runs on the asymmetric matrix and
the two differ by about 30% on a real interface.

Both chain masks are multiplied by seq_mask, because padded PAE entries
are zero and would otherwise pass any cutoff and join the target.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Filters, settings and preflight

**Files:**
- Modify: `bindcraft/filters.py:90-93` (add `ipsae_metric`), `:812-825` (`design_stage_filters`)
- Modify: `bindcraft/settings.py:36` (`CAMPAIGN_SETTING_NAMES`), `:37` (`FINAL_CONFIDENCE_FILTERS`), `:410` (setting-name product)
- Modify: `bindcraft/preflight.py:167`
- Test: `tests/test_ipsae_filters.py`

**Interfaces:**
- Consumes: `metrics['ipsae']` from Task 2.
- Produces: `ipsae_metric(protein_states, predictions, prediction_state='complex') -> float`, registered under filter names `i_pSAE` and `i_pSAE_detarget`. Stage filters keyed `i_pSAE` (or `i_pSAE.<state>` on multi-state campaigns).

- [ ] **Step 1: Write the failing test**

Create `tests/test_ipsae_filters.py`:

```python
from bindcraft.settings import CAMPAIGN_SETTING_NAMES, FINAL_CONFIDENCE_FILTERS, known_campaign_settings


def test_final_ipsae_filter_is_registered():
    assert FINAL_CONFIDENCE_FILTERS['min_ipsae_final'] == 'i_pSAE'


def test_detarget_ipsae_ceiling_is_a_campaign_setting():
    assert 'max_detarget_ipsae' in CAMPAIGN_SETTING_NAMES
    assert 'multitarget_swap_metric' in CAMPAIGN_SETTING_NAMES


def test_per_stage_ipsae_settings_are_known():
    names = known_campaign_settings()
    for stage in ('screen', 'refine', 'anneal', 'harden', 'mutate', 'final'):
        assert f'min_ipsae_{stage}' in names
        assert f'max_detarget_ipsae_{stage}' in names


def test_existing_iptm_settings_still_known():
    names = known_campaign_settings()
    assert 'min_iptm_final' in names
    assert 'min_iptm_anneal' in names
    assert 'max_detarget_iptm' in CAMPAIGN_SETTING_NAMES
    assert FINAL_CONFIDENCE_FILTERS['min_iptm_final'] == 'i_pTM'
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /home/bobbylangan/workdir/packages/BindCraft2 && $PY -m pytest tests/test_ipsae_filters.py -v`
Expected: `KeyError: 'min_ipsae_final'`.

- [ ] **Step 3: Register the metric**

In `bindcraft/filters.py`, directly after the `iptm_metric` function that ends at line 93, add:

```python
@filter_metric('i_pSAE')
@filter_metric('i_pSAE_detarget')
def ipsae_metric(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex') -> float:
    return float(predictions[resolve_prediction_state(predictions, prediction_state)].metrics['ipsae'])
```

- [ ] **Step 4: Register the settings**

In `bindcraft/settings.py:36`, inside the `CAMPAIGN_SETTING_NAMES` frozenset, add `'max_detarget_ipsae'` directly after `'max_detarget_iptm'`, and add `'multitarget_swap_metric'` directly after `'multitarget_swap_patience'`.

In `bindcraft/settings.py:37`, change:

```python
FINAL_CONFIDENCE_FILTERS = {'min_monomer_plddt_final': 'Unbound_Binder_pLDDT', 'min_ptm_final': 'pTM', 'min_iptm_final': 'i_pTM', 'max_ipae_final': 'i_pAE'}
```

to:

```python
FINAL_CONFIDENCE_FILTERS = {'min_monomer_plddt_final': 'Unbound_Binder_pLDDT', 'min_ptm_final': 'pTM', 'min_iptm_final': 'i_pTM', 'min_ipsae_final': 'i_pSAE', 'max_ipae_final': 'i_pAE'}
```

In `bindcraft/settings.py:410`, change the filter tuple in the `{f'{filter}_{stage}' for filter in (...)}` comprehension from:

```python
for filter in ('min_plddt', 'min_iptm', 'max_detarget_iptm')
```

to:

```python
for filter in ('min_plddt', 'min_iptm', 'min_ipsae', 'max_detarget_iptm', 'max_detarget_ipsae')
```

- [ ] **Step 5: Add the stage filter**

In `bindcraft/filters.py`, change the `design_stage_filters` signature at line 812 from:

```python
def design_stage_filters(design_settings: 'BinderDesignSettings', protein_states: ProteinStates, stage: str, iptm: bool=True, plddt: bool=True, campaign_filters: dict[str, DesignFilter] | None=None) -> dict[str, DesignFilter]:
```

to:

```python
def design_stage_filters(design_settings: 'BinderDesignSettings', protein_states: ProteinStates, stage: str, iptm: bool=True, plddt: bool=True, campaign_filters: dict[str, DesignFilter] | None=None, ipsae: bool=True) -> dict[str, DesignFilter]:
```

Then inside the `for state_name in complex_states:` loop, directly after the existing `if iptm and state_name != BINDER_ALONE:` block (which ends with the `stage_filters[state_filter_name('i_pTM', ...)] = DesignFilter(...)` line), add a parallel block:

```python
        if ipsae and state_name != BINDER_ALONE:
            is_detarget = state_name in detarget_states
            threshold = settings.get(f'max_detarget_ipsae_{stage}' if is_detarget else f'min_ipsae_{stage}')
            entry = settings.get('losses', {}).get('ipsae_loss', {})
            bound_metric, required_states = bind_state_metric(ipsae_metric, {**entry, 'prediction_state': state_name})
            stage_filters[state_filter_name('i_pSAE', state_name, complex_states)] = DesignFilter(bound_metric, None if threshold is None else float(threshold), not is_detarget, required_states, threshold is not None)
```

Note the local name `ipsae` now shadows nothing, because `ipsae_metric` is the imported symbol used here, not the module.

- [ ] **Step 6: Let an ipSAE ceiling satisfy the detarget preflight check**

In `bindcraft/preflight.py:167`, change:

```python
    if settings.get('max_detarget_interface_residues_final') is not None or any(settings.get(f'max_detarget_iptm_{stage}') is not None for stage in DESIGN_STAGE_NAMES):
```

to:

```python
    if settings.get('max_detarget_interface_residues_final') is not None or any(settings.get(f'max_detarget_{metric}_{stage}') is not None for metric in ('iptm', 'ipsae') for stage in DESIGN_STAGE_NAMES):
```

Without this, a campaign that sets only `max_detarget_ipsae_harden` would still be warned that its off-target is avoided by weight alone.

- [ ] **Step 7: Add the behaviour test**

Append to `tests/test_ipsae_filters.py`:

```python
from bindcraft.preflight import undecided_avoidance


class _Settings:
    def __init__(self, settings, prepared_states=()):
        self.settings = settings
        self.prepared_states = prepared_states


def test_ipsae_ceiling_counts_as_deciding_an_off_target():
    decided = _Settings({'max_detarget_ipsae_harden': 0.3})
    assert undecided_avoidance(decided) == ''
```

If `undecided_avoidance` needs more of `BinderDesignSettings` than this stub provides, build the stub out until it runs; do not change the production signature to suit the test.

- [ ] **Step 8: Run the tests**

Run: `cd /home/bobbylangan/workdir/packages/BindCraft2 && $PY -m pytest tests/ -v`
Expected: all pass.

- [ ] **Step 9: Verify no existing campaign config broke**

```bash
cd /home/bobbylangan/workdir/packages/BindCraft2
for f in settings/core/*.json settings/modality/*.json examples/*.json; do
  $PY -c "
import json, sys
from bindcraft.settings import known_campaign_settings
names = known_campaign_settings()
unknown = [k for k in json.load(open('$f')) if k not in names]
print('$f', 'UNKNOWN:', unknown) if unknown else None
"
done
```

Expected: no output. Every existing key still validates.

- [ ] **Step 10: Commit**

```bash
cd /home/bobbylangan/workdir/packages/BindCraft2
git add bindcraft/filters.py bindcraft/settings.py bindcraft/preflight.py tests/test_ipsae_filters.py
git commit -m "Gate design stages and finals on ipSAE

min_ipsae_{stage}, min_ipsae_final and max_detarget_ipsae{,_stage},
mirroring their ipTM twins and unset by default. An ipSAE ceiling now
also satisfies the preflight check that an off-target is decided by
something other than loss weight.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: The gradient loss

**Files:**
- Modify: `bindcraft/loss.py` (register `ipsae_loss` directly after `iptm_loss` at line 377-379)
- Test: `tests/test_ipsae_loss.py`

**Interfaces:**
- Consumes: `bindcraft.ipsae.soft_ipsae` from Task 1, `soft_maximum` at `loss.py:227`, `chain_residue_slices` and `amino_acid_probabilities` already in `loss.py`.
- Produces: `ipsae_loss` in `REGISTERED_LOSSES`, weighted by `weights_ipsae_loss`, default 0.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ipsae_loss.py`:

```python
import jax.numpy as jnp

from bindcraft.ipsae import soft_ipsae
from bindcraft.loss import REGISTERED_LOSSES, LOSS_TARGET_WEIGHTING, annealed_pae_cutoff, build_losses


def test_ipsae_loss_is_registered_and_targets_the_bound_state():
    assert 'ipsae_loss' in REGISTERED_LOSSES
    assert LOSS_TARGET_WEIGHTING['ipsae_loss'] == 'binds_target'


def test_registering_the_loss_makes_its_weight_a_known_setting():
    """known_campaign_settings builds weights_* from REGISTERED_LOSSES, so this is only
    true once the decorator above has run. Task 3 deliberately does not assert it."""
    from bindcraft.settings import known_campaign_settings
    assert 'weights_ipsae_loss' in known_campaign_settings()


def test_ipsae_loss_is_off_by_default():
    assert 'ipsae_loss' not in build_losses({})


def test_ipsae_loss_switches_on_with_a_weight():
    assert 'ipsae_loss' in build_losses({'weights_ipsae_loss': 0.05})


def test_cutoff_anneals_from_warmup_down_to_the_paper_value():
    soft_sequence = jnp.asarray(1.0 / 20.0)
    hard_sequence = jnp.asarray(1.0)
    assert float(annealed_pae_cutoff(soft_sequence, 10.0, 30.0)) > 28.0
    assert float(annealed_pae_cutoff(hard_sequence, 10.0, 30.0)) == 10.0


def test_a_wide_cutoff_scores_above_the_paper_cutoff_on_a_dispersed_interface():
    """The reason the anneal exists: at cutoff 10 a step-0 trajectory scores exactly zero."""
    residues = 40
    binder = (jnp.arange(residues) < 10).astype(jnp.float32)
    target = 1.0 - binder
    pae = jnp.full((residues, residues), 22.0)
    assert float(soft_ipsae(pae, binder, target, pae_cutoff=10.0)) == 0.0
    assert float(soft_ipsae(pae, binder, target, pae_cutoff=30.0)) > 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /home/bobbylangan/workdir/packages/BindCraft2 && $PY -m pytest tests/test_ipsae_loss.py -v`
Expected: `ImportError: cannot import name 'annealed_pae_cutoff' from 'bindcraft.loss'`.

- [ ] **Step 3: Write the loss**

In `bindcraft/loss.py`, add `is_binder_chain` to the existing `bindcraft.protein` import at line 10. It is not there today, and the new loss needs it. Change:

```python
from bindcraft.protein import AMINO_ACIDS, ATOM_INDEX, BINDER_ALONE, Protein, ProteinStates, ResidueFlags, StructurePredictions, alignment_matrix_product, has_residue_flag, kabsch, real_residue_count, real_residue_mask, real_residue_weights, redesignable_residue_mask
```

to:

```python
from bindcraft.protein import AMINO_ACIDS, ATOM_INDEX, BINDER_ALONE, Protein, ProteinStates, ResidueFlags, StructurePredictions, alignment_matrix_product, has_residue_flag, is_binder_chain, kabsch, real_residue_count, real_residue_mask, real_residue_weights, redesignable_residue_mask
```

Then add the module import beside the other `bindcraft` imports:

```python
from bindcraft.ipsae import soft_ipsae
```

`chain_residue_slices`, `amino_acid_probabilities` and `_masked_mean` are already defined in `loss.py` (at lines 128, 124 and 120 respectively); `AMINO_ACIDS` and `real_residue_weights` are already imported.

Then directly after `iptm_loss` (which ends at line 379), add:

```python
def annealed_pae_cutoff(sequence_hardness: Array, pae_cutoff: float, warmup_cutoff: float) -> Array:
    """Walk the PAE cutoff down to the paper's value as the sequence hardens.

    At the paper's cutoff of 10 a step-0 trajectory sits near PAE 30, no pair passes the mask,
    ipSAE is identically zero and so is its gradient. sequence_hardness is the mean maximum
    amino-acid probability, so it runs from 1/20 on a uniform sequence to 1 on a discrete one,
    which tracks screen through harden without threading a step count into the loss."""
    return warmup_cutoff - (warmup_cutoff - pae_cutoff) * jnp.clip((sequence_hardness - 1.0 / len(AMINO_ACIDS)) / (1.0 - 1.0 / len(AMINO_ACIDS)), 0.0, 1.0)


@loss('ipsae_loss', target_weighting='binds_target')
def ipsae_loss(protein_states: ProteinStates, predictions: StructurePredictions, prediction_state: str='complex', binder: str='binder', pae_cutoff: float=10.0, warmup_cutoff: float=30.0, temperature: float=0.1) -> Array:
    prediction_state = resolve_prediction_state(predictions, prediction_state)
    protein_complex = protein_states[prediction_state]
    binder_chains = [name for name in sorted(protein_complex) if is_binder_chain(name)]
    if not binder_chains or len(binder_chains) == len(protein_complex):
        return jnp.asarray(0.0)
    chain_slices = chain_residue_slices(protein_complex)
    residue_count = sum(len(protein_complex[name]) for name in sorted(protein_complex))
    binder_mask = jnp.zeros(residue_count, dtype=jnp.float32)
    for name in binder_chains:
        binder_mask = binder_mask.at[chain_slices[name]].set(1.0)
    real_residues = jnp.concatenate([real_residue_weights(protein_complex[name].flags) for name in sorted(protein_complex)])
    sequence_hardness = _masked_mean(jnp.concatenate([amino_acid_probabilities(protein_complex[name].sequence).max(-1) for name in binder_chains]),
                                     jnp.concatenate([real_residue_weights(protein_complex[name].flags) for name in binder_chains]))
    cutoff = annealed_pae_cutoff(sequence_hardness, pae_cutoff, warmup_cutoff)
    #the loss reads the symmetrized metrics['pae']; the raw matrix is local to
    #alphafold_prediction_metrics, and this term already departs from the paper twice
    return 1 - soft_ipsae(predictions[prediction_state].metrics['pae'], binder_mask * real_residues, (1.0 - binder_mask) * real_residues, cutoff, temperature)
```

`soft_ipsae` takes `pae_cutoff` as a traced Array here rather than a Python float. That is fine: it only ever feeds a comparison and no shape depends on it.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd /home/bobbylangan/workdir/packages/BindCraft2 && $PY -m pytest tests/test_ipsae_loss.py -v`
Expected: 6 passed.

If `bindcraft.ipsae` importing `soft_maximum` from `bindcraft.loss` while `bindcraft.loss` imports `soft_ipsae` from `bindcraft.ipsae` raises a circular import, note that `soft_ipsae` does its import inside the function body for exactly that reason. Leave it there.

- [ ] **Step 5: Check the loss is differentiable end to end**

Append to `tests/test_ipsae_loss.py`:

```python
import jax


def test_the_loss_has_a_live_gradient_on_a_dispersed_interface():
    residues = 40
    binder = (jnp.arange(residues) < 10).astype(jnp.float32)
    target = 1.0 - binder
    pae = jnp.full((residues, residues), 22.0)
    gradient = jax.grad(lambda matrix: 1 - soft_ipsae(matrix, binder, target, pae_cutoff=30.0))(pae)
    assert bool(jnp.isfinite(gradient).all())
    assert float(jnp.abs(gradient).sum()) > 0.0
```

Run: `cd /home/bobbylangan/workdir/packages/BindCraft2 && $PY -m pytest tests/ -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
cd /home/bobbylangan/workdir/packages/BindCraft2
git add bindcraft/loss.py tests/test_ipsae_loss.py
git commit -m "Add an optional ipSAE design loss, weight 0 by default

The gradient variant softens the max and anneals the PAE cutoff from 30
to the paper's 10 as the sequence hardens. At a fixed cutoff of 10 a
step-0 trajectory scores exactly zero and has no gradient to descend, so
the term would never fire. Nothing reported, filtered or ranked shares
these departures.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: The multitarget scheduler

**Files:**
- Modify: `bindcraft/target_schedule.py:18` (`build_target_schedule`), `:128-145` (`MultitargetSchedule.__init__`), `:177-178` (`record_stage_peak`), `:219` (`select_protein_states`)
- Test: `tests/test_ipsae_schedule.py`

**Interfaces:**
- Consumes: `metrics['ipsae']` from Task 2, `multitarget_swap_metric` from Task 3.
- Produces: `MultitargetSchedule.confidence_metric: str`, defaulting to `'iptm'`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ipsae_schedule.py`:

```python
from bindcraft.target_schedule import MultitargetSchedule


def _schedule(metric):
    return MultitargetSchedule(targets={}, target_objectives={}, iterations=10, confidence_metric=metric)


def test_schedule_defaults_to_iptm():
    assert MultitargetSchedule(targets={}, target_objectives={}, iterations=10).confidence_metric == 'iptm'


def test_schedule_accepts_ipsae():
    assert _schedule('ipsae').confidence_metric == 'ipsae'
```

`MultitargetSchedule.__init__` (`target_schedule.py:128`) takes `targets`, `target_objectives` and `iterations` positionally and defaults everything else, so those three are the whole construction.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /home/bobbylangan/workdir/packages/BindCraft2 && $PY -m pytest tests/test_ipsae_schedule.py -v`
Expected: `TypeError: __init__() got an unexpected keyword argument 'confidence_metric'`.

- [ ] **Step 3: Add the field**

In `bindcraft/target_schedule.py:128`, add `confidence_metric: str='iptm'` to the end of the `__init__` parameter list, and in the body directly after `self.iptm_threshold = iptm_threshold` (line 133) add:

```python
        self.confidence_metric = confidence_metric
```

- [ ] **Step 4: Read the metric by name**

At `target_schedule.py:177-178`, change:

```python
        peak, interface_confidence = (self.stage_peak_predictions.get(target_name), predictions[target_name].metrics['iptm'])
        if peak is None or (interface_confidence < peak.metrics['iptm'] if objective == 'detarget' else interface_confidence > peak.metrics['iptm']):
```

to:

```python
        peak, interface_confidence = (self.stage_peak_predictions.get(target_name), predictions[target_name].metrics[self.confidence_metric])
        if peak is None or (interface_confidence < peak.metrics[self.confidence_metric] if objective == 'detarget' else interface_confidence > peak.metrics[self.confidence_metric]):
```

At `target_schedule.py:219`, change:

```python
            interface_confidence = float(predictions[target_name].metrics['iptm'])
```

to:

```python
            interface_confidence = float(predictions[target_name].metrics[self.confidence_metric])
```

- [ ] **Step 5: Wire the setting**

At `target_schedule.py:18`, inside the `MultitargetSchedule(...)` construction, add directly after `iptm_threshold=settings.get('multitarget_swap_threshold', 0.5)`:

```python
confidence_metric=settings.get('multitarget_swap_metric', 'iptm'),
```

- [ ] **Step 6: Run the tests**

Run: `cd /home/bobbylangan/workdir/packages/BindCraft2 && $PY -m pytest tests/ -v`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
cd /home/bobbylangan/workdir/packages/BindCraft2
git add bindcraft/target_schedule.py tests/test_ipsae_schedule.py
git commit -m "Let the multitarget scheduler swap on ipSAE

multitarget_swap_metric picks which confidence drives target swaps,
detarget exits and the stage peak. Defaults to iptm, so an existing
multitarget campaign schedules exactly as it does now.

Note the swap and detarget thresholds are calibrated against ipTM; a
campaign switching to ipsae must recalibrate multitarget_swap_threshold
and max_detarget_ipsae, since ipSAE runs far lower.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Reporting and mutation weighting

**Files:**
- Modify: `bindcraft/score.py:17`, `bindcraft/campaign_output.py:89`, `bindcraft/rank.py:18` and `:29` and `:60`, `bindcraft/protein.py:31`, `bindcraft/MPNN_stage.py:98`
- Modify: `bindcraft/sequence_optimization.py:142-152`, `:169`, `:212`
- Test: `tests/test_ipsae_reporting.py`

**Interfaces:**
- Consumes: `metrics['ipsae']` and `metrics['ipsae_per_residue']` from Task 2, `i_pSAE` filter name from Task 3.
- Produces: the `i_pSAE` output column and `mutation_weighting='interface_ipsae'`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ipsae_reporting.py`:

```python
from bindcraft.campaign_output import LEADING_CONFIDENCE_COLUMNS
from bindcraft.MPNN_stage import REACHABLE_CONFIDENCE_BOUNDS
from bindcraft.protein import QUALITY_METRIC_TYPES
from bindcraft.rank import LOWER_IS_BETTER_METRICS, MODALITY_METRICS
from bindcraft.score import PREDICTED_METRICS


def test_ipsae_is_a_reported_metric():
    assert 'i_pSAE' in PREDICTED_METRICS
    assert 'i_pSAE' in LEADING_CONFIDENCE_COLUMNS


def test_ipsae_keeps_its_ipTM_neighbour():
    assert 'i_pTM' in PREDICTED_METRICS
    assert 'i_pTM' in LEADING_CONFIDENCE_COLUMNS


def test_detarget_ipsae_is_lower_is_better():
    assert 'i_pSAE_detarget' in LOWER_IS_BETTER_METRICS
    assert 'i_pSAE' not in LOWER_IS_BETTER_METRICS


def test_ipsae_is_documented_in_the_rank_glossary():
    assert 'i_pSAE' in MODALITY_METRICS['every campaign']
    assert 'i_pSAE_detarget' in MODALITY_METRICS['multi-target, cross-reactivity and detargeting']


def test_ipsae_has_a_quality_metric_type_and_an_ensemble_bound():
    assert QUALITY_METRIC_TYPES[('i_pSAE', 'global')] == 'ipTM'
    assert REACHABLE_CONFIDENCE_BOUNDS['ipsae'] == (0.0, 1.0)
```

`QUALITY_METRIC_TYPES` maps to a validator name, and ipSAE shares ipTM's `[0, 1]` shape, so `'ipTM'` is the right validator to reuse. Confirm by reading how the value is consumed before assuming.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /home/bobbylangan/workdir/packages/BindCraft2 && $PY -m pytest tests/test_ipsae_reporting.py -v`
Expected: `AssertionError` on the first test.

- [ ] **Step 3: Add the column**

`bindcraft/score.py:17`:

```python
PREDICTED_METRICS = ('pLDDT', 'pTM', 'i_pTM', 'i_pSAE', 'i_pAE', RANKING_METRIC)
```

`bindcraft/campaign_output.py:89`:

```python
LEADING_CONFIDENCE_COLUMNS = (RANKING_METRIC, 'i_pTM', 'i_pSAE', 'pLDDT', 'pTM', 'i_pAE', 'Unbound_Binder_pLDDT', 'Target_pLDDT')
```

`bindcraft/rank.py:18`, add `'i_pSAE_detarget'` to the frozenset directly after `'i_pTM_detarget'`.

`bindcraft/rank.py:29`, add directly after the `'i_pTM'` entry:

```python
                    'i_pSAE': 'interface pSAE, the Dunbrack interface score (stricter than i_pTM on small epitopes)',
```

`bindcraft/rank.py:60`, add directly after the `'i_pTM_detarget'` entry:

```python
                                                    'i_pSAE_detarget': 'interface pSAE against a state the binder should miss',
```

`bindcraft/protein.py:31`, add `('i_pSAE', 'global'): 'ipTM'` to `QUALITY_METRIC_TYPES`.

`bindcraft/MPNN_stage.py:98`:

```python
REACHABLE_CONFIDENCE_BOUNDS = {'plddt': (0.0, 1.0), 'ptm': (0.0, 1.0), 'iptm': (0.0, 1.0), 'ipsae': (0.0, 1.0), 'pae': (0.0, 0.0)}
```

- [ ] **Step 4: Add the mutation weighting**

In `bindcraft/sequence_optimization.py`, change `interface_confidence_weights` at line 142 to take the metric name:

```python
def interface_confidence_weights(prediction: StructurePrediction, cutoff: float=INTERFACE_CUTOFF, metric: str='iptm_per_residue') -> dict[str, Array] | None:
```

and change line 145 and line 148 to use it:

```python
    if metric not in prediction.metrics or not binder_chains or len(binder_chains) == len(protein_complex):
        return None
    interface_mask = pseudo_beta_interface_mask(protein_complex, cutoff)
    interface_confidence = jnp.where(interface_mask, 1.0 - prediction.metrics[metric], 0.0)
```

Add a module-level map directly above that function:

```python
INTERFACE_WEIGHTING_METRICS = {'interface_iptm': 'iptm_per_residue', 'interface_ipsae': 'ipsae_per_residue'}
```

Change line 169 from:

```python
        if self.mutation_weighting != 'interface_iptm' or not self.chain_interface_weights:
```

to:

```python
        if self.mutation_weighting not in INTERFACE_WEIGHTING_METRICS or not self.chain_interface_weights:
```

Change line 212 from:

```python
        if self.mutation_weighting == 'interface_iptm':
```

to:

```python
        if self.mutation_weighting in INTERFACE_WEIGHTING_METRICS:
```

and the `interface_confidence_weights(prediction)` call two lines below it to:

```python
                for name, weights in (interface_confidence_weights(prediction, metric=INTERFACE_WEIGHTING_METRICS[self.mutation_weighting]) or {}).items():
```

- [ ] **Step 5: Test the weighting**

Append to `tests/test_ipsae_reporting.py`:

```python
from bindcraft.sequence_optimization import INTERFACE_WEIGHTING_METRICS


def test_both_interface_weightings_are_available():
    assert INTERFACE_WEIGHTING_METRICS['interface_iptm'] == 'iptm_per_residue'
    assert INTERFACE_WEIGHTING_METRICS['interface_ipsae'] == 'ipsae_per_residue'
```

- [ ] **Step 6: Run the full suite**

Run: `cd /home/bobbylangan/workdir/packages/BindCraft2 && $PY -m pytest tests/ -v`
Expected: all pass.

- [ ] **Step 7: Confirm the CLI still starts**

```bash
cd /home/bobbylangan/workdir/packages/BindCraft2
$PY -c "import bindcraft.cli, bindcraft.rank, bindcraft.score, bindcraft.campaign; print('imports clean')"
$PY -m bindcraft.rank --help 2>&1 | grep -i "i_pSAE"
```

Expected: `imports clean`, then the two glossary lines mentioning `i_pSAE`. `rank.py:319` builds the glossary into the argparse epilog and only fills it in when `-h` or `--help` is present, so `--help` is the flag that prints it.

- [ ] **Step 8: Commit**

```bash
cd /home/bobbylangan/workdir/packages/BindCraft2
git add bindcraft/score.py bindcraft/campaign_output.py bindcraft/rank.py bindcraft/protein.py bindcraft/MPNN_stage.py bindcraft/sequence_optimization.py tests/test_ipsae_reporting.py
git commit -m "Report i_pSAE and allow mutation weighting on it

i_pSAE joins the scored columns next to i_pTM rather than replacing it,
because on our own GPC3 outcome data ipSAE enriches for binders without
separating them and the top scorer did not bind. Ranking still defaults
to i_pDAE.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Documentation

**Files:**
- Modify: `docs/reference.md`, `docs/outputs.md`

- [ ] **Step 1: Find where ipTM is documented**

```bash
cd /home/bobbylangan/workdir/packages/BindCraft2
grep -n "min_iptm\|i_pTM\|max_detarget_iptm" docs/reference.md docs/outputs.md
```

- [ ] **Step 2: Document the settings in `docs/reference.md`**

Beside each `min_iptm_*` / `max_detarget_iptm` entry the grep found, add the ipSAE twin using the same table or list format the surrounding lines use. The text for each:

- `min_ipsae_{stage}` / `min_ipsae_final`: "minimum ipSAE (Dunbrack 2025) for the stage. Unset by default. ipSAE is far stricter than ipTM on small interfaces, so a threshold copied across from `min_iptm_*` will reject almost everything."
- `max_detarget_ipsae` / `max_detarget_ipsae_{stage}`: "ipSAE ceiling an off-target state must stay under. Unset by default."
- `weights_ipsae_loss`: "weight on the ipSAE design loss. 0 by default. The loss anneals its PAE cutoff from 30 to 10 as the sequence hardens, because at a fixed cutoff of 10 it has no gradient at the start of a trajectory."
- `multitarget_swap_metric`: "`iptm` (default) or `ipsae`, which confidence drives multitarget swaps and detarget exits. Switching this invalidates `multitarget_swap_threshold`, which is calibrated against ipTM."

- [ ] **Step 3: Document the column in `docs/outputs.md`**

Beside the `i_pTM` entry, add:

"`i_pSAE` interface pSAE (Dunbrack 2025, biorxiv 2025.02.10.637595), reproducing the paper's `d0res` variant at a PAE cutoff of 10, taken as the maximum over both chain directions. Built to stay sensitive on small and asymmetric interfaces where ipTM saturates. Read it beside `i_pTM`: on our GPC3 set it enriched for binders without separating them, and the highest-scoring design did not bind."

- [ ] **Step 4: Verify and commit**

```bash
cd /home/bobbylangan/workdir/packages/BindCraft2
grep -c "i_pSAE\|ipsae" docs/reference.md docs/outputs.md
$PY -m pytest tests/ -q
git add docs/
git commit -m "Document the ipSAE settings and the i_pSAE column

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Verification before calling the branch done

```bash
cd /home/bobbylangan/workdir/packages/BindCraft2
$PY -m pytest tests/ -v                  # every test green
git log --oneline dev..HEAD                 # 7 implementation commits on top of the 2 spec commits
git diff dev --stat
grep -rn "min_iptm\|iptm_loss\|i_pTM" bindcraft/ --include='*.py' | grep -v 'af/alphafold' | wc -l
```

The last count must be no lower than it was on `dev`. Nothing ipTM-related was removed, and a drop means something was renamed that the spec put out of scope.

Run one short real campaign against an example before merging, and confirm `i_pSAE` appears in the output CSV with values in `[0, 1]`:

```bash
cd /home/bobbylangan/workdir/packages/BindCraft2
# use whatever the README gives as the smallest example; pdl1_denovo with a low trajectory count
grep -n "i_pSAE" <project_folder>/*.csv | head
```
