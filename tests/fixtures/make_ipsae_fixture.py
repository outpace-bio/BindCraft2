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
