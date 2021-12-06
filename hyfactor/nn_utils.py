# -*- coding: utf-8 -*-
#
#  Copyright Laboratoire de Chemoinformatique
#  Copyright Laboratory of chemoinformatics and molecular modeling
#  This file is part of hyfactor.
#
#  hyfactor is free software; you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program; if not, see <https://www.gnu.org/licenses/>.

from collections import defaultdict
from contextlib import redirect_stdout
from functools import partial
from multiprocessing import Pool

import tensorflow as tf
import tensorflow.keras.backend as K
from CGRtools.containers import MoleculeContainer
from CGRtools.files import SDFWrite
from adabelief_tf import AdaBeliefOptimizer
from numpy import pi
from tensorflow.keras import callbacks as cb
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tqdm import tqdm

from .layers import refactor_encoder, refactor_decoder, hyfactor_encoder, hyfactor_decoder
from .metrics import log_like_atoms, log_like_bonds, log_like_adj, log_like_hydrogens, accuracy_atoms, \
    accuracy_hydrogens, accuracy_bonds, accuracy_adj, reconstruction_rate_refactor, reconstruction_rate_hyfactor
from .mol_utils import matrix_chunking, load_latent_vectors, create_chem_graph, reconstruct_molecule_hyfactor, \
    write_lv
from .preprocessing import prepare_data


class StopCallback(cb.Callback):
    def __init__(self, metric, stop_value, **kwargs):
        self.metric = metric
        self.stop_value = stop_value

    def on_epoch_end(self, epoch, logs=None):
        if logs[self.metric] >= self.stop_value:
            print(f'Epoch {epoch}: Asked metric value is reached')
            self.model.stop_training = True


def scheduler_cos(min_lr, max_lr, n_epoch):
    def lr(epoch):
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + tf.math.cos(epoch / n_epoch * pi))

    return lr


def scheduler_exp(epoch, lr):
    if epoch < 1:
        return lr
    else:
        return lr * 0.98


def choose_scheduler(config):
    if config['scheduler'] == 'exp':
        scheduler = cb.LearningRateScheduler(scheduler_exp)
    elif config['scheduler'] == 'cos':
        scheduler = cb.LearningRateScheduler(scheduler_cos(1e-6, config['learning_rate'], config['n_epoch']))
    else:
        raise ValueError("I don't know which scheduler you specified")

    return scheduler


def config_names_converter(config: dict) -> dict:
    short_names = {
        'atomic_representation_vector_length': 'avl',
        'molecule_representation_vector_length': 'mvl',
        'input_data_file': 'idf',
        'output_data_file': 'odf',
        'input_model_file': 'imf',
        'output_model_file': 'omf',
        'validation_file': 'val',
        'test_file': 'test',
        'prepared_atoms_types': 'atom_types',
        'preprocessed_tmp_file': 'tmp',
        'plot_file': 'plot',
        'summary_file': 'sum'
    }

    new_config = {}
    for level in config.values():
        for name, value in level.items():
            if name in short_names.keys():
                new_config[short_names[name]] = value
            else:
                new_config[name] = value

    for k in ['vae', 'vectorized_atoms', 'tl_features', 'plot', 'sum', 'chunks',
              'odf', 'imf', 'val', 'test', 'tmp', 'prop_vec']:
        try:
            _ = new_config[k]
        except (ValueError, KeyError):
            if k == 'hyf_rec_processes':
                new_config[k] = 1
            else:
                new_config[k] = None

    return new_config


def check_configs(config: dict):
    if not config['task']:
        raise ValueError("The task was not provided")

    if not config['idf']:
        raise ValueError("The input file was not provided")

    params = not config['avl'] or not config['mvl'] or not config[
        'max_bond_order'] or not config['max_atoms'] or not config['batch'] or not config['n_epoch'] or not config[
        'learning_rate']
    if params:
        raise ValueError("Some of model params are missed")

    if config['task'] == 'train':
        if not config['val']:
            raise ValueError("The validation set was not provided for the task 1")
        if not config['omf']:
            raise ValueError("The output model file was not provided for the task 1")

    elif config['task'] == 'evaluate':
        if not config['imf']:
            raise ValueError("The model file was not provided for the task 2")

    elif config['task'] == 'reconstruct':
        if not config['odf']:
            raise ValueError("The output file was not provided for the task 5")
        if not config['imf']:
            raise ValueError("The model file was not provided for the task 5")


def model_visualisation(config, model):
    if config['sum']:
        sum_file = config['sum']
        with open(config['sum'], 'w') as f:
            with redirect_stdout(f):
                model.summary()
        print(f'Summary file saved at {sum_file}')
    else:
        print('Continue without saving summary')

    if config['plot']:
        plot_file = config['plot']
        plot_model(model, to_file=plot_file, show_shapes=True, show_layer_names=True, expand_nested=True)
        print(f'Plot file saved at {plot_file}')
    else:
        print('Continue without saving plot')


def reconstruction_refactor(model, inputs, out_file, config, atom_types, mol_i, i_chunk=None):
    atoms_seq, bonds_matrices = model.predict(inputs, batch_size=config['batch'], verbose=1)
    del inputs

    atoms_seq, bonds_matrices = K.argmax(atoms_seq), K.argmax(bonds_matrices)
    atoms_seq = atoms_seq.numpy().tolist()
    bonds_matrices = bonds_matrices.numpy().tolist()
    if i_chunk:
        print(f'Started convertation of molecules for chunk {i_chunk}')
    else:
        print('Started convertation of molecules')

    empty_mols = []
    for atoms, bonds in tqdm(zip(atoms_seq, bonds_matrices)):
        molecule = create_chem_graph(atoms, bonds, atom_types)
        if list(molecule.atoms()):
            molecule.meta.update({'index': mol_i})
            out_file.write(molecule)
        else:
            empty_mols.append(mol_i)
        mol_i += 1

    if empty_mols:
        print(f'Reconstructed {len(empty_mols)} empty molecules')

    del atoms_seq, bonds_matrices

    return mol_i


def recompile_rules(atoms_types):
    tmp_mol = MoleculeContainer()
    for atom, charge in atoms_types:
        tmp_mol.add_atom(atom=atom, charge=charge)

    all_recompiled = {}
    for n, atom in tmp_mol.atoms():
        recompiled_rules = defaultdict(set)
        all_rules = atom._compiled_valence_rules
        for atom_rule, neighbors_rules in all_rules.items():
            for _, neighbors_count, hydrogens in neighbors_rules:
                enviroment = defaultdict(int)
                if neighbors_count:
                    for (_, neighbor_type), neigh_count in neighbors_count.items():
                        enviroment[neighbor_type] += neigh_count
                enviroment = tuple(sorted(enviroment.items()))
                recompiled_rules[(enviroment, hydrogens)].add(atom_rule)
        all_recompiled[atom.atomic_number] = recompiled_rules
    return all_recompiled


def reconstruction_hyfactor(model, inputs, out_file, config, atom_types, mol_i, i_chunk=None):
    atoms, hydrogens, adj_matrices = model.predict(inputs, batch_size=config['batch'], verbose=1)
    del inputs

    atoms, hydrogens, adj_matrices = K.argmax(atoms), K.argmax(hydrogens), tf.cast(adj_matrices > 0.5, tf.int32)

    if i_chunk:
        print(f'Started convertation of molecules for chunk {i_chunk}')
    else:
        print('Started convertation of molecules')

    pool_chunks = atoms.shape[0] // 100000 + int(bool(atoms.shape[0] % 100000))
    atoms = atoms.numpy().tolist()
    hydrogens = hydrogens.numpy().tolist()
    adj_matrices = adj_matrices.numpy().tolist()

    empty_mols = []
    recompiled_rules = recompile_rules(atom_types)
    for pool_chunk in tqdm(range(pool_chunks)):
        atoms_pool = atoms[pool_chunk * 100000:(pool_chunk + 1) * 100000]
        hydrogens_pool = hydrogens[pool_chunk * 100000:(pool_chunk + 1) * 100000]
        adj_matrices_pool = adj_matrices[pool_chunk * 100000:(pool_chunk + 1) * 100000]
        with Pool(config['hyf_rec_processes']) as p:
            result = p.starmap(partial(reconstruct_molecule_hyfactor,
                                       recompiled_rules=recompiled_rules, atom_types=atom_types),
                               zip(atoms_pool, hydrogens_pool, adj_matrices_pool))

        for molecule in result:
            if list(molecule.atoms()):
                molecule.meta.update({'index': mol_i})
                out_file.write(molecule)
            else:
                empty_mols.append(mol_i)
            mol_i += 1

    if empty_mols:
        print(f'Reconstructed {len(empty_mols)} empty molecules')

    del atoms, hydrogens, adj_matrices


def generate_mols(model, inputs, config, atom_types):
    mol_index = 0
    with SDFWrite(config['odf']) as out:
        if config['chunks']:
            print('Started reconstruction of molecules with chunking')

            if config['task'] == 'reconstruct':
                chunk_size = matrix_chunking(inputs[0], config['chunks'], config['batch'])
            else:
                chunk_size = matrix_chunking(inputs, config['chunks'], config['batch'])

            print(chunk_size)
            for i in range(config['chunks']):
                if config['task'] == 'reconstruct':
                    input_chunk = [matrix[i * chunk_size:(i + 1) * chunk_size] for matrix in inputs]
                else:
                    input_chunk = inputs[i * chunk_size:(i + 1) * chunk_size]
                if config['model'] == 'hyfactor':
                    reconstruction_hyfactor(model, input_chunk, out, config, atom_types, mol_index, i)
                else:
                    mol_index = reconstruction_refactor(model, input_chunk, out, config, atom_types, mol_index, i)

                K.clear_session()
                if config['task'] == 'decode':
                    model = load_decoder(config, load_ae(config))
                elif config['task'] == 'reconstruct':
                    model = load_ae(config)

        else:
            print('Started reconstruction of molecules without chunking')
            if config['model'] == 'hyfactor':
                reconstruction_hyfactor(model, inputs, out, config, atom_types, mol_index)
            else:
                reconstruction_refactor(model, inputs, out, config, atom_types, mol_index)


def build_model(config):
    atoms_input = Input(shape=(config['max_atoms'],), batch_size=config['batch'], name='Main_Atoms_Input')
    inputs = [atoms_input]
    if config['model'] == 'hyfactor':
        hydro_input = Input(shape=(config['max_atoms'],), batch_size=config['batch'], name='Main_Hydro_Input')
        inputs.append(hydro_input)
    bonds_input = Input(shape=(config['max_atoms'], config['max_atoms']), batch_size=config['batch'],
                        name='Main_Bonds_Input')
    inputs.append(bonds_input)

    mask = tf.cast(atoms_input != 0, atoms_input.dtype)

    if config['model'] == 'refactor':
        x = refactor_encoder(config['max_atoms'], config['batch'], config['max_atom_num'],
                             config['avl'], config['mvl'], config['distributed'])([atoms_input, bonds_input, mask])
        atoms_pred, bonds_pred = refactor_decoder(config['max_atoms'], config['batch'], config['max_atom_num'],
                                                  config['avl'], config['mvl'], config['distributed'])(x)

    elif config['model'] == 'hyfactor':
        x = hyfactor_encoder(config['max_atoms'], config['batch'], config['max_atom_num'],
                             config['avl'], config['mvl'], config['distributed'])([atoms_input, hydro_input,
                                                                                   bonds_input, mask])
        atoms_pred, hydro_pred, bonds_pred = hyfactor_decoder(config['max_atoms'], config['batch'],
                                                              config['max_atom_num'],
                                                              config['avl'], config['mvl'], config['distributed'])(x)

    outputs = [atoms_pred]
    if config['model'] == 'hyfactor':
        outputs.append(hydro_pred)
    outputs.append(bonds_pred)

    model = Model(inputs=inputs, outputs=outputs)

    model_visualisation(config, model)

    mask = tf.cast(mask, tf.float32)

    atoms_pred = tf.cast(atoms_pred, tf.float32)
    bonds_pred = tf.cast(bonds_pred, tf.float32)
    length = tf.cast(tf.math.count_nonzero(mask, axis=1), tf.float32)

    loss_atoms = log_like_atoms(atoms_input, atoms_pred, length, config)
    loss = loss_atoms
    model.add_metric(loss_atoms, name='atoms_loss', aggregation='mean')
    model.add_metric(accuracy_atoms(atoms_input, atoms_pred, mask, length),
                     name='atoms_error', aggregation='mean')

    if config['model'] == 'refactor':
        loss_bonds = log_like_bonds(bonds_input, bonds_pred, mask, length, config)
        loss += loss_bonds
        model.add_metric(loss_bonds, name='bonds_loss', aggregation='mean')
        model.add_metric(accuracy_bonds(bonds_input, bonds_pred, mask, length),
                         name='bonds_error', aggregation='mean')
        model.add_metric(reconstruction_rate_refactor(atoms_input, bonds_input, atoms_pred, bonds_pred, mask, length),
                         name='reconstruction_rate', aggregation='mean')

    elif config['model'] == 'hyfactor':
        loss += log_like_hydrogens(hydro_input, hydro_pred, mask, length)
        loss += log_like_adj(bonds_input, bonds_pred, mask, length)
        model.add_metric(reconstruction_rate_hyfactor(atoms_input, hydro_input, bonds_input,
                                                      atoms_pred, hydro_pred, bonds_pred,
                                                      mask, length), name='reconstruction_rate', aggregation='mean')

        model.add_metric(accuracy_hydrogens(hydro_input, hydro_pred, mask, length),
                         name='hydrogens_error', aggregation='mean')
        model.add_metric(accuracy_adj(bonds_input, bonds_pred, mask, length),
                         name='bonds_error', aggregation='mean')

    model.add_loss(K.mean(loss))
    optimizer = AdaBeliefOptimizer(learning_rate=config['learning_rate'], epsilon=1e-14,
                                   print_change_log=False)
    _ = optimizer.lr
    model.compile(optimizer=optimizer)

    return model


def choose_model(config):
    model = build_model(config)
    if config['imf']:
        model.load_weights(filepath=config['imf']).expect_partial()
        K.set_value(model.optimizer.lr, tf.constant(config['learning_rate']))
        print('Loaded model with weights')
    else:
        print('Loaded model without weights')
    return model


def load_ae(config):
    physical_devices = tf.config.list_physical_devices('GPU')

    if config['distributed']:
        for i in config['GPUs']:
            tf.config.experimental.set_memory_growth(physical_devices[i], enable=True)
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = choose_model(config)

    else:
        tf.config.set_visible_devices(physical_devices[config['GPUs']], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[config['GPUs']], enable=True)
        model = choose_model(config)

    return model


def load_encoder(config, autoencoder):
    if config['model'] == 'refactor':
        atoms_input = Input(shape=(config['max_atoms'],), batch_size=config['batch'], name='enc_Atoms_Input')
        bonds_input = Input(shape=(config['max_atoms'], config['max_atoms']), batch_size=config['batch'],
                            name='enc_Bonds_Input')
        mask = tf.cast(atoms_input != 0, tf.float32)
        latent_vector = autoencoder.layers[4]([atoms_input, bonds_input, mask])
        encoder = Model(inputs=[atoms_input, bonds_input], outputs=latent_vector)
        encoder.summary()
    elif config['model'] == 'hyfactor':
        atoms_input = Input(shape=(config['max_atoms'],), batch_size=config['batch'], name='enc_Atoms_Input')
        hydro_input = Input(shape=(config['max_atoms'],), batch_size=config['batch'], name='enc_Hydro_Input')
        bonds_input = Input(shape=(config['max_atoms'], config['max_atoms']), batch_size=config['batch'],
                            name='enc_Bonds_Input')
        mask = tf.cast(atoms_input != 0, tf.float32)
        latent_vector = autoencoder.layers[5]([atoms_input, hydro_input, bonds_input, mask])
        encoder = Model(inputs=[atoms_input, hydro_input, bonds_input], outputs=latent_vector)
        encoder.summary()
    else:
        raise ValueError('Bad model specified. Use only "refactor" or "hyfactor"')

    return encoder


def load_decoder(config, autoencoder):
    if config['model'] == 'refactor':
        decoder = autoencoder.layers[5]
    elif config['model'] == 'hyfactor':
        decoder = autoencoder.layers[6]
    else:
        raise ValueError('Bad model specified. Use only "refactor" or "hyfactor"')
    return decoder


def train(config, model, atom_types):
    train_data = prepare_data(config, 'train', atom_types)
    validation_data = prepare_data(config, 'val', atom_types)
    callbacks = [cb.ModelCheckpoint(config['omf'], save_weights_only=True, save_best_only=True,
                                    monitor='val_reconstruction_rate', mode='max')]

    if config['scheduler']:
        scheduler = choose_scheduler(config)
        callbacks.append(scheduler)

    if config['log']:
        callbacks.append(cb.CSVLogger(config['log'], append=False))

    model.fit(train_data, batch_size=config['batch'], epochs=config['n_epoch'],
              validation_data=(validation_data, None), use_multiprocessing=True, callbacks=callbacks)

    if config['test']:
        test_data = prepare_data(config, 'test', atom_types)
        model.evaluate(x=test_data, batch_size=config['batch'], use_multiprocessing=True)


def evaluate(config, model, atom_types):
    input_data = prepare_data(config, 'encode', atom_types)
    model.evaluate(x=input_data, batch_size=config['batch'], use_multiprocessing=True)


def encode(config, model, atom_types):
    encoder = load_encoder(config, model)

    data = prepare_data(config, 'encode', atom_types)
    line_index = 0
    with open(config['odf'], "w") as out:
        if config['chunks']:
            chunk_size = matrix_chunking(data[0], config['chunks'], config['batch'])
            for n in range(config['chunks']):
                input_chunk = [matrices[n * chunk_size:(n + 1) * chunk_size] for matrices in data]

                latent_vectors = encoder.predict(input_chunk, batch_size=config['batch'],
                                                 use_multiprocessing=True, verbose=1)
                del input_chunk
                line_index = write_lv(line_index, latent_vectors, out)
                del latent_vectors
        else:
            latent_vectors = encoder.predict(data, batch_size=config['batch'], use_multiprocessing=True, verbose=1)
            del data
            write_lv(line_index, latent_vectors, out)
            K.clear_session()


def decode(config, model, atom_types):
    decoder = load_decoder(config, model)
    decoder.summary()
    latent_vectors = load_latent_vectors(config['idf'])
    generate_mols(decoder, latent_vectors, config, atom_types)
