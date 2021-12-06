Configuration file 
===============

Configuration file consists of three parts: `Parameters`, 
`Model_files` and `Data_files`. In each section you can 
specify parameters or paths to files.

## Parameters
* `max_bond_order` - Maximal bond order. Used only in ReFactor;
* `max_atoms` - Maximal number of atoms in a molecule.
  Should be +1 more, than in the given datasets;
* `num_convs` - Number of convolution layers;
* `atomic_vector_length` - Dimension of atomic representation vector;
* `molecule_vector_length` - Dimension of molecular representation vector;
* `batch` - Batch size;
* `n_epoch` - Number of epochs;
* `learning_rate` - Learning rate;
* `task` - name of possible tasks:
  * `train` - training of chosen model; 
  * `evaluate` - evaluation of chosen model with NN stats (loss, rec_rate);
  * `encode` - encoding of a _SDF_ file given in `input_data_file` 
    to the molecular latent vector;
  * `decode` - decoding of a _SVM_ file given in `input_data_file`
    to the graphs;
  * `reconstruct` - get reconstructed structures from AutoEncoder.
* `GPUs` - which GPU or GPUs on which model will work. 
  If more than 1 will be present, the architecture will use 
  a distributed training. To specify more than 1 number of GPU 
  use list of numbers, for example `[0, 1]`. 
  For details see the page tensorflow.org/tutorials/distribute/keras;
* `model` - Which model to use. 
  `refactor` will give ReFactor AE, 
  `hyfactor` will give HYFactor;
* `scheduler` -  Which scheduler to use. 
  `exp` is exponential scheduler, `cos` is cosinus scheduler. If not specified, 
  no scheduler will be used;
* `cos_min_learning_rate` - Parameter for `cos` scheduler. 
  If not specified, standard one (`1e-6`) would be used;
* `exp_coef` - Parameter for `exp` scheduler. 
  If not specified, standard one (`-0.05`) would be used;
* `exp_coef` - Parameter for `exp` scheduler. 
  If not specified, standard one (`4`) would be used;
* `chunks` - Specifies if tasks should be done in chunks.
  Used in encode and decode task. 
  
## Model_files
* `input_model_file` - Path to model's weights. 
  If not specified, model without weights would be build.
  Arbitary in all tasks, except training;
* `output_model_file` - Path to save model's weights. 
  Arbitrary in training task, otherwise ignored;
* `prepared_atoms_types` - Path to prepared pickle of tuple of atom types, optional. 
  If file doesn't exist, tool automatically will create 
  a new one and will save to the specified path;
* `log` - Way to log file, optional;
* `plot_file` - Path and file name to save picture of AutoEncoder graph, optional;
* `summary_file` - Path and file name to save AutoEncoder's summary, optional;
## Data_files
* `input_data_file` - SDF file used to train the model or to be encoded.
  Also, it might be an SVM file containing latent vectors that have to be decoded;
* `validation_file` - Path to validation file. Arbitary in training mode, otherwise ignored;
* `test_file` - Path to validation file. Optional in training mode, otherwise ignored;
* `preprocessed_tmp_file` - Path to preprocessed file;