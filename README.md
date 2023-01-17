# HyFactor

HyFactor is an open-source architecture for structure generation using graph-based approaches.
It is based on the new type of molecular graph - Hydrogen-count Labelled Graph (HLG).
This graph, similar to the InChI linear notation, considers the number 
of hydrogens attached to the heavy atoms instead of the bond types.
Additionally, with HyFactor we add ReFactor architecture, which is based molecular graph-based 
architecture with defactorization procedure adopted from the reported DEFactor architecture.

This repository includes official implementations of HyFactor, ReFactor and 
functions for translation molecular graph to HLG and back.

__For more details please refer to the [paper](https://pubs.acs.org/doi/10.1021/acs.jcim.2c00744)__

If you are using this repository in your paper, please cite us as:

```
HyFactor: A Novel Open-Source, Graph-Based Architecture for Chemical Structure Generation
Tagir Akhmetshin, Arkadii Lin, Daniyar Mazitov, Yuliana Zabolotna, Evgenii Ziaikin, Timur Madzhidov, and Alexandre Varnek
Journal of Chemical Information and Modeling 2022 62 (15), 3524-3534
DOI: 10.1021/acs.jcim.2c00744
```

## Data
All materials used in the publication are availible 
on [Figshare project page](https://figshare.com/projects/HyFactor_Hydrogen-count_labelled_graph-based_defactorization_Autoencoder/127103)

### Data sets
The standardized data sets and training/validation splits:
1. [ZINC 250K standardized data set](https://figshare.com/articles/dataset/ZINC_250K_data_sets/17122427) 
2. [ChEMBL v.27 standardized data set](https://figshare.com/articles/dataset/ChEMBL_data_sets/17121986)
3. The MOSES data set was used as it is

The original data sets were taken from:
1. [Original ZINC 250K data set](https://github.com/mkusner/grammarVAE/tree/master/data) 
2. [ChEMBL page](https://www.ebi.ac.uk/chembl/)
3.  [MOSES benchmarking GitHub repository](https://github.com/molecularsets/moses)

### Models weights
The weights of Autoencoders from the experiments are available 
on [Figshare](https://figshare.com/articles/software/HyFactor_and_ReFactor_models_weights/17122622)

## Installation
### Installation with conda (preffered)
First, download the repository on your machine. Then, create conda enviroment with the folowing code:
    
    conda env create -f enviroment.yml

The enviroment file is made for use on GPU with CUDA version = 11.3. 
If you have different versions of drivers or want to use a CPU version,
please modify this file before the installation. For additional support, please,
visit the tensorflow documentation page. 

When your enviroment is ready, activate it and execute command to install the architecture:

    python3 setup.py install

### Installation with pip
In this case you should create enviroment folder anywhere you prefer, install here the enviroment and activate it:

    mkdir hyfactor_env
    python3 -m venv hyfactor_env/
    source hyfactor_env/bin/activate

Then, similarly as with conda, you just run the folowing code:

    python3 setup.py install

## Usage

### Before start
This tool works in two modes: command-line and as usual python package. 
In both ways you should specify config file which will be used for every task.
The examples of config file you can find in the folder `examples/configs`.

### Command-line interface
Once you specified your config file, execute the AutoEncoder with folowing command:

    hyfactor -cfg YOUR_CONFIG_FILE.yaml

### Python interface
Here you can simply import the HYFactor package in folowing way:

    from HYFactor import task_preparer
    import yaml
    
    with open('YOUR_CONFIG_FILE.yaml', 'r') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    task_preparer(config)

### HLG to molecular graph conversion

The code for conversion of the HLG to molecular graph is implemented in 
function `HYFactor.nn_utils.reconstruction_hyfactor`. 

Here is a custom example of HLG conversion:

    from HYFactor.nn_utils import recompile_rules
    from HYFactor.mol_utils import reconstruct_molecule_hyfactor

    atoms = [...]  # list of atoms indices according to tuple atom_types
    hydrogens = [...]  # list of hydrogens attached to heavy atoms from 0 to 4
    adjs = [[...]]  # list of lists or binary matrix of connectivity between atoms
    
    recompiled_rules = recompile_rules(atom_types)
    molecule = reconstruct_molecule_hyfactor(atoms, hydrogens, adjs, atom_types, recompiled_rules)

## Contributing

We welcome contributions, in the form of issues or pull requests.

If you have a question or want to report a bug, please submit an issue.

To contribute with code to the project, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the remote branch: `git push`
5. Create the pull request.


## Copyright
* Tagir Akhmetshin tagirshin@gmail.com
* Arkadii Lin arkadiyl18@gmail.com
* Timur Madzhidov tmadzhidov@gmail.com
* Alexandre Varnek varnek@unistra.fr
