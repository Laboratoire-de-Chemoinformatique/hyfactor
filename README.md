# HyFactor

Graph-based architectures are becoming increasingly popular as a tool for structure generation. 
Here, we introduce a novel open-source architecture HyFactor which is inspired by previously 
reported DEFactor architecture and based on hydrogen labeled graphs. 
Since the original DEFactor code was not available, 
its updated implementation (ReFactor) was prepared 
in this work for benchmarking purposes.

__For more details please refer to the [paper](https://chemrxiv.org/engage/chemrxiv/article-details/61aa38576d4e8f3bdba8aead)__

If you are using this repository in your paper, please cite us as:

```
Akhmetshin T, Lin A, Mazitov D, Ziaikin E, Madzhidov T, Varnek A (2021) 
HyFactor: Hydrogen-count labelled graph-based defactorization Autoencoder. 
ChemRxiv. doi: 10.26434/chemrxiv-2021-18x0d
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
