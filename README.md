## Installation

To install, clone this using `git clone`. This software is written in Python, notably using PyTorch, PyTorch Lightning, and the HuggingFace transformers library. The required conda environment is defined within the `environment.yml` file. To set this up, make sure you have conda (or [mamba](https://mamba.readthedocs.io/en/latest/index.html)) installed, clone this repository, and run:

```bash
conda env create -f environment.yml
conda activate foldingdiff
pip install -e ./  # make sure ./ is the dir including setup.py
```

### Downloading data

We require some data files not packaged on Git due to their large size. These are not required for sampling (as long as you are not using the `--testcomparison` option, see below); this is required for training your own model. We provide a script in the `data` dir to download requisite CATH data.

```bash
# Download the CATH dataset
cd data  # Ensure that you are in the data subdirectory within the codebase
chmod +x download_cath.sh
./download_cath.sh
```

If the download link in the `.sh` file is not working, the tarball is also mirrored at the following [Dropbox link](https://www.dropbox.com/s/ka5m5lx58477qu6/cath-dataset-nonredundant-S40.pdb.tgz?dl=0).

## Training models

To train your own model on the CATH dataset, use the script at `bin/train.py` in combination with one of the
json config files under `config_jsons` (or write your own). An example usage of this is as follows:

```bash
python train.py cath_full_angles_cosine.json --dryrun
```

By default, the training script will calculate the KL divergence at each timestep before starting training, which can be quite computationally expensive with more timesteps. To skip this, append the `--dryrun` flag. The output of the model will be in the `results` folder with the following major files present:

```
results/
    - config.json           # Contains the config file for the huggingface BERT model itself
    - logs/                 # Contains the logs from training
    - models/               # Contains model checkpoints. By default we store the best 5 models by validation loss and the best 5 by training loss
    - training_args.json    # Full set of arguments, can be used to reproduce run
```

## Sampling protein backbones

To sample protein backbones, use the script `bin/sample.py`. Example commands to do this using the pretrained weights described above are as follows.

```bash
# To sample 10 backbones per length ranging from [50, 128) with a batch size of 512 - reproduces results in our manuscript
python sample.py -l 50 128 -n 10 -b 512 --device cuda:0
```

This will run the trained model hosted at [wukevin/foldingdiff_cath](https://huggingface.co/wukevin/foldingdiff_cath) and generate sequences of varying lengths. If you wish to load the test dataset and include test chains in the generated plots, use the option `--testcomparison`; note that this requires downloading the CATH dataset, see above. Running `sample.py` will create the following directory structure in the diretory where it is run:

```
some_dir/
    - plots/            # Contains plots comparing the distribution of training/generated angles
    - sampled_angles/   # Contains .csv.gz files with the sampled angles
    - sampled_pdb/      # Contains .pdb files from converting the sampled angles to cartesian coordinates
    - model_snapshot/   # Contains a copy of the model used to produce results
```

### Maximum training similarity TM scores

After generating sequences, we can calculate TM-scores to evaluate the simliarity of the generated sequences and the original sequences. This is done using the script under `bin/tmscore_training.py` and requires data to have been downloaded prior (see above).


## Evaluating designability of generated backbones

One way to evaluate the quality of generated backbones is via their "designability". This refers to whether or not we can design an amino acid chain that will fold into the designed backbone. To evaluate this, we use an inverse folding model to generate amino acid sequences that are predicted to fold into our generated backbone, and check whether those generated sequences actually fold into a structure comparable to our backbone.

### Inverse folding

Inverse folding is the task of predicting a sequence of amino acids that will produce a given protein backbone structure. In our analysis, we used the ProteinMPNN model to generate eight distinct amino acid sequences for each structure produced by FoldingDiff.

#### ProteinMPNN

To set up [ProteinMPNN](https://www.science.org/doi/10.1126/science.add2187), see the authors guide on their [GitHub](https://github.com/dauparas/ProteinMPNN).

After this, we follow a similar procedure as for ESM-IF1 (above) where we `cd` into the directory containing the `sampled_pdb` folder and run:

```bash
python ~/bin/pdb_to_residue_proteinmpnn.py sampled_pdb
```

This will create a new directory called `proteinmpnn_residues` containing 8 amino acid chains per sampled PDB structure.

### Structural prediction

#### OmegaFold

We use [OmegaFold](https://github.com/HeliXonProtein/OmegaFold) to fold the amino acid sequences produced by ProteinMPNN. This is due to OmegaFold's relatively fast runtime compared to AlphaFold2, and due to the fact that OmegaFold is natively designed to be run without MSA information - making it more suitable for our protein design task.

After creating and activating a separate conda environment and following the authors' instructions for installing OmegaFold, we use the following script to split our input amino acid fasta files across GPUs for inference, and subsequently calculate the self-consistency TM (scTM) scores.

```bash
# Fold each fasta, spreading the work over GPUs 0 and 1, outputs to omegafold_predictions folder
python ~/bin/omegafold_across_gpus.py proteinmpnn_residues/*.fasta -g 0 1
```




### Binding Pocket Detection and Selection
Potential ligand-binding pockets were identified using **Fpocket**.

Pockets with **druggability score > 0.5** and **volume > 200 Å³** were selected
A total of **62 candidate pockets** were obtained
These pockets were further **clustered into three groups**
Representative pockets were selected for subsequent molecular docking analysis


### Ligand Selection and Molecular Docking
Candidate small molecules were retrieved from the **ChEMBL database** based on:

Spatial proximity to the predicted pockets
Physicochemical properties of the binding pockets

#### Protein and Ligand Preparation
Prior to docking:

Water molecules were removed
Polar hydrogen atoms were added
**Gasteiger charges** were assigned
Both receptor and ligand structures were converted to **PDBQT format**

#### Docking Procedure
Molecular docking was performed using **AutoDock Vina** to evaluate binding affinity and generate protein–ligand complex structures.


### Protein–Ligand Interaction Analysis
The resulting complexes were analyzed using **PLIP (Protein–Ligand Interaction Profiler)**.

PLIP automatically identifies key non-covalent interactions, including:

Hydrogen bonds
Hydrophobic interactions
π–π stacking
Cation–π interactions
Salt bridges

It also generates both **3D binding mode visualizations** and **2D interaction diagrams**, enabling detailed analysis of key binding residues and interaction patterns.



### ADMET Evaluation
To assess the drug-likeness and safety of selected ligands:

**ADMET properties** (absorption, distribution, metabolism, excretion, and toxicity) were predicted
Promising candidates were further evaluated based on their pharmacokinetic profiles and potential toxicity risks


## Summary
This pipeline integrates protein design, structural validation, pocket detection, molecular docking, interaction analysis, and ADMET prediction to identify promising protein–ligand systems for downstream applications.
