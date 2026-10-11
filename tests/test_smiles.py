"""The shared SMILES parser: what it repairs, and what it must leave alone.

Two of these tests exist because the repair was nearly shipped without them.
The parser was verified on GramNegAccum, where it is right, and both defects
only showed up on datasets it was not being tested against.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))

from rdkit import Chem, RDLogger  # noqa: E402

import smiles as smiles_mod  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# A primary amine written three ways. The first is what RDKit wants; the other
# two are GramNegAccum's spellings and must come out as the same molecule.
PRIMARY_AMINE_SMARTS = Chem.MolFromSmarts("[$([N;H2;X3][CX4]),$([N;H3;X4+][CX4])]")


def skeleton(mol):
    return Chem.MolToInchiKey(mol).split("-")[0]


class RepairsTheDialects(unittest.TestCase):
    def test_bracket_nitrogen_without_hydrogens_becomes_an_amine(self):
        """`[N+]` means a protonated amine, not a zero-hydrogen cation."""
        written = "[N+]C1COC2C(OS(=O)(=O)c3ccc(-c4ccccc4)cc3)COC12"
        as_rdkit_reads_it = Chem.MolFromSmiles(written)
        self.assertIsNotNone(as_rdkit_reads_it)
        self.assertFalse(
            as_rdkit_reads_it.HasSubstructMatch(PRIMARY_AMINE_SMARTS),
            "if RDKit already found the amine this test no longer proves anything",
        )
        repaired, atoms, reordered = smiles_mod.parse_report(written)
        self.assertTrue(repaired.HasSubstructMatch(PRIMARY_AMINE_SMARTS))
        self.assertEqual(atoms, 1)
        self.assertFalse(reordered)

    def test_charge_before_hydrogen_count_parses(self):
        """`[N+H2]` is the other spelling; RDKit returns None for it.

        Reordering makes it readable as `[NH2+]CC`, which is still one hydrogen
        short of a real species -- a nitrogen cation with one bond takes three
        -- so the valence repair then completes it to ethylammonium. Both
        corrections are needed to land on the molecule that was meant.
        """
        written = "[N+H2]CC"
        self.assertIsNone(Chem.MolFromSmiles(written))
        mol, atoms, reordered = smiles_mod.parse_report(written)
        self.assertIsNotNone(mol)
        self.assertTrue(reordered)
        self.assertEqual(atoms, 1)
        self.assertEqual(Chem.MolToSmiles(mol), "CC[NH3+]")
        self.assertEqual(Chem.MolToSmiles(smiles_mod.parse("[NH2+]CC")), "CC[NH3+]")

    def test_the_two_spellings_give_one_molecule(self):
        """The reason the compound count fell from 217 to 197."""
        stereo = "[NH3+][C@H]1CO[C@H]2[C@@H]1OC[C@H]2OS(=O)(=O)c1ccc(-c2ccccc2)cc1"
        flat = "[N+]C1COC2C(OS(=O)(=O)c3ccc(-c4ccccc4)cc3)COC12"
        self.assertEqual(skeleton(smiles_mod.parse(stereo)), skeleton(smiles_mod.parse(flat)))

    def test_radical_atoms_counts_what_was_not_repaired(self):
        self.assertEqual(smiles_mod.radical_atoms(smiles_mod.parse("CCO")), 0)
        self.assertEqual(smiles_mod.radical_atoms(smiles_mod.parse("[N+]C1CCCCC1")), 0)
        # Deliberately not repaired, so the short valence stays visible rather
        # than being papered over: the count is a report, not a promise of zero.
        self.assertEqual(smiles_mod.radical_atoms(smiles_mod.parse("[Li]")), 1)


class LeavesTheRestAlone(unittest.TestCase):
    def test_ring_double_bond_stereo_survives(self):
        """Taboo 38: the sanitize=False path never perceives it.

        MolToSmiles writes the bond directions back regardless, so comparing
        canonical SMILES shows nothing wrong. InChI reads the stereo flags, and
        py/candidates.py breaks ties on the InChIKey, so the difference decides
        which stereoisomer represents a skeleton.
        """
        written = r"N[C@@H]1CCN2c3c(F)cc4c(=O)c(C(=O)O)cn(C5CC5)c4c3/C=C\CC[C@H]12"
        plain = Chem.MolFromSmiles(written)
        parsed = smiles_mod.parse(written)
        self.assertEqual(Chem.MolToSmiles(plain), Chem.MolToSmiles(parsed))
        self.assertEqual(Chem.MolToInchiKey(plain), Chem.MolToInchiKey(parsed))

    def test_explicit_hydrogens_are_stripped_as_rdkit_would(self):
        """Taboo 29: the sanitize=False path skips MolFromSmiles' own RemoveHs.

        Without it the graph keeps [H] atoms and every descriptor shifts. This
        molecule is from CHAMP, which has no atom the valence repair touches,
        so any difference here is the missing RemoveHs and nothing else.
        """
        written = "CC(C)C(C=C1)=CC2=C1[C@]3(C)[C@](CC2)([H])[C@@](CN=[N+]=[N-])(C)CCC3"
        mol, atoms, _ = smiles_mod.parse_report(written)
        self.assertEqual(atoms, 0, "no atom in this molecule should be repaired")
        self.assertEqual(
            Chem.MolToSmiles(mol), Chem.MolToSmiles(Chem.MolFromSmiles(written))
        )

    def test_metals_are_not_turned_into_hydrides(self):
        """Taboo 30: a bare [Li] is lithium, not lithium hydride."""
        for written in ("[Li]", "[Na]", "[Fe]", "[Fe+2]"):
            with self.subTest(written=written):
                mol, atoms, _ = smiles_mod.parse_report(written)
                self.assertEqual(atoms, 0)
                self.assertEqual(
                    Chem.MolToSmiles(mol), Chem.MolToSmiles(Chem.MolFromSmiles(written))
                )

    def test_ordinary_molecules_are_untouched(self):
        for written in ("CCO", "c1ccccc1", "CC(=O)Nc1ccc(O)cc1",
                        "C[C@H](N)C(=O)O", "O=S(=O)([O-])[O-]"):
            with self.subTest(written=written):
                mol, atoms, reordered = smiles_mod.parse_report(written)
                self.assertEqual((atoms, reordered), (0, False))
                self.assertEqual(
                    Chem.MolToSmiles(mol), Chem.MolToSmiles(Chem.MolFromSmiles(written))
                )

    def test_unreadable_input_returns_none(self):
        for written in ("", None, "not a molecule", "C1CC"):
            with self.subTest(written=written):
                self.assertIsNone(smiles_mod.parse(written))


class AgreesWithTheDatasetsAlreadyMeasured(unittest.TestCase):
    """The no-op claim behind keeping the published CHAMP and IMI numbers."""

    def _is_noop_over(self, path, column, limit=None):
        import pyarrow.parquet as pq

        if not path.exists():
            self.skipTest(f"{path} not built")
        changed = seen = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=[column]):
            for s in batch.column(0).to_pylist():
                if not s:
                    continue
                plain = Chem.MolFromSmiles(s)
                repaired = smiles_mod.parse(s)
                if plain is None or repaired is None:
                    continue
                seen += 1
                if (Chem.MolToSmiles(plain) != Chem.MolToSmiles(repaired)
                        or Chem.MolToInchiKey(plain) != Chem.MolToInchiKey(repaired)):
                    changed += 1
                if limit and seen >= limit:
                    return seen, changed
        return seen, changed

    def test_champ_is_unchanged(self):
        seen, changed = self._is_noop_over(
            ROOT / "data" / "work" / "champ.parquet", "smiles"
        )
        self.assertEqual(changed, 0, f"{changed} of {seen} CHAMP molecules changed")

    def test_imi_is_unchanged(self):
        seen, changed = self._is_noop_over(
            ROOT / "data" / "work" / "imi_clean.parquet", "smiles"
        )
        self.assertEqual(changed, 0, f"{changed} of {seen} IMI molecules changed")


if __name__ == "__main__":
    unittest.main()
