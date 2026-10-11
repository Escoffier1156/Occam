"""One SMILES parser for every module, so dialects cannot diverge silently.

External datasets write SMILES in dialects RDKit reads differently from what
the author meant. Two were found in GramNegAccum and both are silent: the
molecule parses, no error is raised, and only a descriptor comes out wrong.

  1. `[N+]` for a protonated amine. SMILES says a bracket atom has exactly the
     hydrogens it writes, so RDKit reads a nitrogen cation with none. A
     one-bond nitrogen cation with no hydrogen is not a real species; the
     writer expected the hydrogens to be inferred. Clearing the no-implicit
     flag where the valence falls short of what the element and its charge
     allow puts them back, which works regardless of spelling.

  2. `[N+H2]` with the charge ahead of the hydrogen count. RDKit requires the
     opposite order and returns None, so this one at least fails loudly.

Restricted to the SMILES organic subset on purpose. A bare `[Li]`, `[Na]` or
`[Fe]` is a legitimate metal atom or ion, and the valence rule would turn it
into a hydride: on B3DB it rewrote `[Li]` to `[LiH]`, a different compound.
Outside the organic subset the writer was not relying on a default valence.

Repairs are counted, never silent. `parse_report` returns what was touched so
a caller can print it or refuse to continue.

Effect on the datasets in use, measured over every row:

  CHAMP (1,132)        unchanged
  IMI-COMBINE (22,052) unchanged
  B3DB (7,807)         5 molecules repaired, all under-specified organic atoms
                       (`[CH]`, `[NH]`, `[O+]`, `[N]`, `[S]`)
  GramNegAccum         66 molecules repaired; primary amines in table 4 go
                       from 7/68 to 67/68 and unique structures from 217 to 197
"""

import re

from rdkit import Chem

# A bracket atom with the charge written before the hydrogen count. RDKit
# wants [NH2+] and GramNegAccum's tables also write [N+H2] and [N+@@H].
_BRACKET_CHARGE_FIRST = re.compile(r"\[([A-Z][a-z]?)(@{0,2})([+-]\d*)(@{0,2})(H\d*)\]")

# B, C, N, O, P, S and the halogens: the SMILES organic subset, the elements
# whose default valence the writer of a short bracket atom was relying on.
ORGANIC_SUBSET = frozenset((5, 6, 7, 8, 15, 16, 9, 17, 35, 53))

_PERIODIC_TABLE = Chem.GetPeriodicTable()


def reorder_brackets(smiles):
    """Moves a bracket atom's charge after its hydrogen count."""
    return _BRACKET_CHARGE_FIRST.sub(
        lambda m: f"[{m.group(1)}{m.group(2)}{m.group(4)}{m.group(5)}{m.group(3)}]", smiles
    )


def _recount_hydrogens(mol):
    """Lets RDKit infer hydrogens on organic atoms whose valence falls short.

    Returns the number of atoms whose hydrogen count was handed back to RDKit.
    """
    repaired = 0
    for atom in mol.GetAtoms():
        if not atom.GetNoImplicit() or atom.GetAtomicNum() not in ORGANIC_SUBSET:
            continue
        used = int(sum(b.GetBondTypeAsDouble() for b in atom.GetBonds())) + atom.GetNumExplicitHs()
        allowed = _PERIODIC_TABLE.GetDefaultValence(atom.GetAtomicNum()) + atom.GetFormalCharge()
        if allowed > 0 and used < allowed:
            atom.SetNoImplicit(False)
            repaired += 1
    return repaired


def parse_report(smiles):
    """Parses one SMILES; returns (mol, repaired_atoms, reordered).

    `mol` is None when neither the written form nor the reordered one parses.
    `repaired_atoms` counts atoms whose hydrogens were recounted, and
    `reordered` says whether the bracket rewrite was needed. A molecule that
    parses as written and needs no repair comes back exactly as
    Chem.MolFromSmiles would return it.
    """
    if not smiles:
        return None, 0, False
    for candidate in (smiles, reorder_brackets(smiles)):
        mol = Chem.MolFromSmiles(candidate, sanitize=False)
        if mol is None:
            continue
        repaired = _recount_hydrogens(mol)
        try:
            Chem.SanitizeMol(mol)
        except (Chem.AtomValenceException, Chem.KekulizeException, ValueError, RuntimeError):
            # Recounting produced something unsanitisable, so take the plain
            # reading of this spelling rather than a repaired broken molecule.
            mol = Chem.MolFromSmiles(candidate)
            if mol is None:
                continue
            return mol, 0, candidate != smiles
        # [LOCK: ast name="smiles_matches_rdkit_default"]
        # UNLOCK_REASON (2026-10-08): AssignStereochemistry added alongside
        # RemoveHs. Both lines restore a step that Chem.MolFromSmiles performs
        # for itself and the sanitize=False path above skips. Neither is
        # optional and neither belongs to the repair.
        #
        # RemoveHs: without it the graph keeps [H] atoms and every descriptor
        # shifts on any input that writes one. This looked like the valence
        # repair changing CHAMP, and CHAMP has no atom the repair touches.
        #
        # AssignStereochemistry: without it ring double-bond stereo is never
        # perceived. MolToSmiles writes the bond directions back either way, so
        # the canonical SMILES is identical and a SMILES-level comparison sees
        # nothing -- but InChI reads the flags, and py/candidates.py breaks ties
        # on the InChIKey. One IMI molecule with a ring `/C=C\` picked a
        # different stereoisomer as its representative because of this.
        mol = Chem.RemoveHs(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        # [/LOCK]
        return mol, repaired, candidate != smiles
    return None, 0, False


def parse(smiles):
    """Parses one SMILES, discarding the repair report. None if unreadable."""
    return parse_report(smiles)[0]


def radical_atoms(mol):
    """Atoms left with unpaired electrons after parsing.

    A short valence this module could not account for. Some are real radicals;
    a lot of them at once means the input uses a dialect not handled here, and
    that failure shows up in the descriptors rather than in a parse error --
    `[N+]` read as a zero-hydrogen cation simply misses the primary-amine
    pattern. Callers compare this against a share of the dataset.
    """
    return sum(1 for a in mol.GetAtoms() if a.GetNumRadicalElectrons() > 0)
