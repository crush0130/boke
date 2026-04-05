import os
import torch
import pickle
from pathlib import Path
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from foldingdiff.datasets import CathCanonicalAnglesDataset

LOCAL_DATA_DIR = Path(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
)

CATH_DIR = LOCAL_DATA_DIR / "cath"
SS8_TO_SS3 = {
    "H": "H", "B": "E", "E": "E", "G": "H", "I": "H", "T": "C", "S": "C", " ": "C"
}
SS3_TO_IDX = {"H": 0, "E": 1, "C": 2}


def extract_and_cache_dssp_labels(dataset_dir, output_file="cached_dssp_labels.pkl"):
    dataset = CathCanonicalAnglesDataset(pdbs="cath", use_cache=True, zero_center=False)
    parser = PDBParser(QUIET=True)

    all_dssp = {}
    for struct in dataset.structures:
        fname = struct["fname"]
        try:
            structure = parser.get_structure("X", fname)
            model = structure[0]
            dssp = DSSP(model, fname, dssp="mkdssp", file_type="PDB")
            ss8 = [dssp[key][2] for key in dssp.keys()]
            ss3 = [SS8_TO_SS3.get(c, "C") for c in ss8]
            indices = [SS3_TO_IDX[c] for c in ss3]
            all_dssp[os.path.basename(fname)] = torch.tensor(indices, dtype=torch.long)
        except Exception as e:
            L = struct["angles"].shape[0]
            all_dssp[os.path.basename(fname)] = torch.full((L,), SS3_TO_IDX["C"], dtype=torch.long)

    with open(output_file, "wb") as f:
        pickle.dump(all_dssp, f)



if __name__ == "__main__":
    # 替换为你的 PDB 文件夹路径
    extract_and_cache_dssp_labels("cath")
