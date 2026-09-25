"""Regenerate the multi-chain ipSAE fixtures: two binder copies against one target.

This is the `copies > 1` geometry, where pooling the binder chains into one mask stops
agreeing with upstream. Upstream ipsae.py emits one row per ordered chain pair, so the
reported score is the max over (binder chain, target chain) pairs and the binder/binder
pair is not part of it.

The PAE values are snapped to the decoded centres of the 64-bin PAE head that
tests/test_ipsae_metrics.py builds, so upstream and the port read the identical matrix.

Run from the repository root:

    $PY tests/fixtures/make_ipsae_multichain_fixture.py

Then re-derive the expected values with upstream ipsae.py v4:

    $PY /home/bobbylangan/workdir/packages/IPSAE/ipsae.py \
        tests/fixtures/ipsae_multichain_pae.json tests/fixtures/ipsae_multichain_model.pdb 10 15

and copy the ipSAE column of the three `max` rows into UPSTREAM_* in
tests/test_ipsae_metrics.py.
"""
import json
import os

import numpy as np

BINDER_A_RESIDUES = 20
BINDER_B_RESIDUES = 20
TARGET_RESIDUES = 30
RESIDUES = BINDER_A_RESIDUES + BINDER_B_RESIDUES + TARGET_RESIDUES
CHAINS = (('A', BINDER_A_RESIDUES), ('B', BINDER_B_RESIDUES), ('C', TARGET_RESIDUES))
A = slice(0, BINDER_A_RESIDUES)
B = slice(BINDER_A_RESIDUES, BINDER_A_RESIDUES + BINDER_B_RESIDUES)
C = slice(BINDER_A_RESIDUES + BINDER_B_RESIDUES, RESIDUES)

PAE_BINS = 64
PAE_MAX = 32.0
FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))


def pae_bin_centres() -> np.ndarray:
    """The decoded centres of tests/test_ipsae_metrics.py's one-hot PAE head."""
    breaks = np.linspace(0.0, PAE_MAX, PAE_BINS)[:-1]
    width = breaks[1] - breaks[0]
    return np.append(breaks + width / 2, breaks[-1] + 1.5 * width)


def snap(value: float) -> float:
    centres = pae_bin_centres()
    return float(centres[int(np.argmin(np.abs(centres - value)))])


def build_pae() -> np.ndarray:
    """Every interface is asymmetric, and the target->binder direction is the one that leads.

    That direction is where pooling the two copies into one mask breaks: a target residue's
    n0res counts its good pairs against BOTH copies at once, so n0res goes 20 -> 40 and d0
    goes 1.0000 -> 1.8258, which lifts every pTM term. The target sees copy A a little more
    confidently than copy B, so the max over pairs is a specific pair rather than a tie.

    The two copies also pack against each other more confidently than either binds the
    target. That is deliberate: the reported score is the max over BINDER/TARGET pairs, so
    the binder/binder pair being the strongest interface in the complex must not raise it."""
    pae = np.full((RESIDUES, RESIDUES), snap(25.0))
    pae[A, A] = snap(2.0)
    pae[B, B] = snap(2.0)
    pae[C, C] = snap(2.5)
    #target -> copies: every target residue locates both copies, copy A the better of the two
    pae[C, A] = snap(5.0)
    pae[C, B] = snap(7.0)
    #copies -> target: a weaker reciprocal signal, so the max comes from the direction above
    pae[A.start + 2:A.start + 8, C.start:C.start + 24] = snap(7.0)
    pae[B.start + 3:B.start + 6, C.start:C.start + 10] = snap(9.0)
    #the two copies against each other: the most confident interface in the complex
    pae[B, A] = snap(4.0)
    pae[A, B] = snap(3.0)
    np.fill_diagonal(pae, snap(0.0))
    return pae


def build_pdb(plddt: np.ndarray) -> str:
    lines, serial, offset = [], 0, 0
    for chain_index, (chain, count) in enumerate(CHAINS):
        for residue in range(count):
            for atom, shift in (('CA', 0.0), ('CB', 1.5)):
                serial += 1
                x = residue * 3.8 + shift + chain_index * 12.0
                lines.append(f'ATOM  {serial:>5}  {atom:<3} ALA {chain}{residue + 1:>4}    '
                             f'{x:>8.3f}{chain_index * 6.0:>8.3f}{0.0:>8.3f}  1.00{plddt[offset + residue]:>6.2f}           C')
        offset += count
    return '\n'.join(lines + ['END']) + '\n'


def main() -> None:
    pae = build_pae()
    plddt = np.linspace(70.0, 95.0, RESIDUES)
    with open(os.path.join(FIXTURE_DIR, 'ipsae_multichain_pae.json'), 'w') as handle:
        json.dump({'pae': pae.tolist(), 'plddt': plddt.tolist(), 'ptm': 0.5, 'iptm': 0.4}, handle)
    with open(os.path.join(FIXTURE_DIR, 'ipsae_multichain_model.pdb'), 'w') as handle:
        handle.write(build_pdb(plddt))


if __name__ == '__main__':
    main()
