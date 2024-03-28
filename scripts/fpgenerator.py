import functools
import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import rdkit.Chem.FilterCatalog as FilterCatalog
import scipy.sparse as sparse
from bidict import bidict

class FragmentFingerprint:
    def __init__(self, substructure_list):
        # super(FragmentFingerprint, self).__init__()
        self._substructure_list = substructure_list
        self._substructure_obj_list = []

        self._filter = FilterCatalog.FilterCatalog()

        for i, substructure in enumerate(self._substructure_list):
            # Validating Smarts
            smarts_obj = Chem.MolFromSmarts(substructure)
            if smarts_obj is None:
                raise ValueError(f"Invalid SMARTS pattern #{i} : {substructure}")
            self._substructure_obj_list.append(smarts_obj)

            # Adding pattern to the filter catalogue
            pattern = FilterCatalog.SmartsMatcher(f"Pattern {i}", substructure, 1)
            self._filter.AddEntry(FilterCatalog.FilterCatalogEntry(str(i), pattern))

    @functools.cached_property
    def n_bits(self):
        return len(self._substructure_list)

    def _gen_features(self, mol_obj):
        return [int(match.GetDescription()) for match in self._filter.GetMatches(mol_obj)]

    def _transform(self, mol_fp_list) -> sparse.csr_matrix:
        data = []
        rows = []
        cols = []
        n_col = 0
        for i, mol_fp in enumerate(mol_fp_list):
            data.extend([1] * len(mol_fp))
            rows.extend(mol_fp)
            cols.extend([i] * len(mol_fp))
            n_col += 1
        return sparse.csr_matrix((data, (cols, rows)), shape=(n_col, self.n_bits))

    def fit(self, mol_obj_list) -> None:
        pass

    def fit_transform(self, mol_obj_list) -> sparse.csr_matrix:
        return self.transform(mol_obj_list)

    def transform(self, mol_obj_list) -> sparse.csr_matrix:
        mol_feature_iterator = (self._gen_features(mol_obj) for mol_obj in mol_obj_list)
        return self._transform(mol_feature_iterator)
    
    def transform_smiles(self, smiles_list):
        mol_feature_iterator = (self._gen_features(Chem.MolFromSmiles(smiles)) for smiles in smiles_list)
        return self._transform(mol_feature_iterator)
