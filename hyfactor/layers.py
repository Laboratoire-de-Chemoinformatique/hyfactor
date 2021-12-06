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

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, GRU, RepeatVector, Embedding, Bidirectional, Dense, \
    BatchNormalization, Activation, Lambda, Layer, LayerNormalization
from tensorflow.keras.layers.experimental import SyncBatchNormalization
from tensorflow.keras.models import Model


def multi_term_calc():
    def _multi_term_calc(adjacency_matrix):
        # Prepare bond type-specific adjacency slices
        adjacency_matrix = tf.cast(adjacency_matrix, tf.float32)
        ones = tf.ones_like(adjacency_matrix, dtype=adjacency_matrix.dtype)
        twos = tf.multiply(ones, 2)
        threes = tf.multiply(ones, 3)

        E = tf.concat([tf.expand_dims(tf.cast(tf.math.equal(adjacency_matrix, ones), tf.float32), 1),
                       tf.expand_dims(tf.cast(tf.math.equal(adjacency_matrix, twos), tf.float32), 1),
                       tf.expand_dims(tf.cast(tf.math.equal(adjacency_matrix, threes), tf.float32), 1)], axis=1)

        diagonal = tf.reduce_sum(tf.cast(tf.math.not_equal(adjacency_matrix, 0), tf.float32), axis=2)
        diagonal = tf.math.divide_no_nan(1.0, tf.math.sqrt(diagonal))

        D = tf.linalg.diag(diagonal)
        D = tf.expand_dims(D, 1)
        D = tf.repeat(D, repeats=3, axis=1)

        # Compute D_E_D term
        E_term = tf.matmul(D, tf.matmul(E, D))

        return E_term

    return Lambda(_multi_term_calc)


def term_calc():
    def _terms_calc(adjacency_matrix):
        adjacency_matrix = tf.cast(adjacency_matrix, tf.float32)
        diagonal = tf.reduce_sum(adjacency_matrix, axis=-1)
        diagonal = tf.math.divide_no_nan(1.0, tf.math.sqrt(diagonal))
        degree = tf.linalg.diag(diagonal)

        return tf.linalg.matmul(degree, tf.linalg.matmul(adjacency_matrix, degree))

    return Lambda(_terms_calc)


def multi_bond_gcn(max_atoms: int, batch_size: int, atom_vector_dim: int, distributed: bool):
    """
    Implementation of multi-channel convolution layer described in DEFactor (arXiv:1811.09766)
    :param batch_size: number of elements in batch
    :param distributed: If distributed training will be used, bool
    :param max_atoms: Maximal number of atoms in molecule, int
    :param atom_vector_dim: Dimension of atom vector, int
    :return: Updated atom vectors
    """
    atoms_vectors = Input(shape=(max_atoms, atom_vector_dim), batch_size=batch_size, name='Input_GCN_Atoms')
    e_term = Input(shape=(None, max_atoms, max_atoms), batch_size=batch_size, name='Input_GCN_term')

    x_1 = Dense(units=atom_vector_dim, kernel_initializer='he_normal')(atoms_vectors)
    x_2 = Dense(units=atom_vector_dim, kernel_initializer='he_normal')(atoms_vectors)
    x_3 = Dense(units=atom_vector_dim, kernel_initializer='he_normal')(atoms_vectors)
    x_self = Dense(units=atom_vector_dim, kernel_initializer='he_normal')(atoms_vectors)

    x_1 = LayerNormalization(axis=-2)(x_1)
    x_2 = LayerNormalization(axis=-2)(x_2)
    x_3 = LayerNormalization(axis=-2)(x_3)
    x_self = LayerNormalization(axis=-2)(x_self)

    x = K.concatenate([K.expand_dims(x_1, axis=1), K.expand_dims(x_2, axis=1), K.expand_dims(x_3, axis=1)], axis=1)

    e_term_val = tf.cast(e_term, x.dtype)
    x = tf.linalg.matmul(e_term_val, x)
    x = tf.reduce_sum(x, axis=1)
    x = tf.add(x, x_self)
    x = K.relu(x)

    return Model(inputs=[atoms_vectors, e_term], outputs=x)


def gcn(max_atoms: int, batch: int, atom_vector_dim: int, distributed: bool):
    """
    Implementation of one-channel convolution layer described in DEFactor (arXiv:1811.09766)
    :param distributed: If distributed training will be used, bool
    :param max_atoms: number of maximum number of atoms in molecule, int
    :param batch: number of instances in batch
    :param atom_vector_dim: number of  atoms vector length, int
    :return:
    """
    atoms_vectors = Input(shape=(max_atoms, atom_vector_dim), batch_size=batch, name='Kipf_Atoms')
    e_term = Input(shape=(max_atoms, max_atoms), batch_size=batch, name='Kipf_E_term')

    x_self = Dense(units=atom_vector_dim, kernel_initializer='he_normal')(atoms_vectors)
    x_neighbors = Dense(units=atom_vector_dim, kernel_initializer='he_normal')(atoms_vectors)

    x_self = LayerNormalization(axis=-2)(x_self)
    x_neighbors = LayerNormalization(axis=-2)(x_neighbors)

    e_term_val = tf.cast(e_term, x_neighbors.dtype)
    x_neighbors = tf.linalg.matmul(e_term_val, x_neighbors)

    x = tf.add(x_neighbors, x_self)
    x = K.relu(x)

    return Model(inputs=[atoms_vectors, e_term], outputs=x)


class MultiDefactorization(Layer):
    def __init__(self, *args, **kwargs):
        super(MultiDefactorization, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.no_bond_weights = self.add_weight(shape=(input_shape[-1],), initializer='he_normal', trainable=True,
                                               name='no_bond_weights')
        self.no_bond_bias = self.add_weight(shape=(input_shape[-1],), initializer='he_normal', trainable=True,
                                            name='no_bond_bias')
        self.single_bond_weights = self.add_weight(shape=(input_shape[-1],), initializer='he_normal',
                                                   trainable=True, name='single_bond_weights')
        self.single_bond_bias = self.add_weight(shape=(input_shape[-1],), initializer='he_normal', trainable=True,
                                                name='single_bond_bias')
        self.double_bond_weights = self.add_weight(shape=(input_shape[-1],), initializer='he_normal',
                                                   trainable=True, name='double_bond_weights')
        self.double_bond_bias = self.add_weight(shape=(input_shape[-1],), initializer='he_normal', trainable=True,
                                                name='double_bond_bias')
        self.triple_bond_weights = self.add_weight(shape=(input_shape[-1],), initializer='he_normal',
                                                   trainable=True, name='triple_bond_weights')
        self.triple_bond_bias = self.add_weight(shape=(input_shape[-1],), initializer='he_normal', trainable=True,
                                                name='triple_bond_bias')

    def call(self, inputs, training=None, mask=None):
        x_1 = K.bias_add(tf.multiply(inputs, self.no_bond_weights), self.no_bond_bias)
        x_2 = K.bias_add(tf.multiply(inputs, self.single_bond_weights), self.single_bond_bias)
        x_3 = K.bias_add(tf.multiply(inputs, self.double_bond_weights), self.double_bond_bias)
        x_4 = K.bias_add(tf.multiply(inputs, self.triple_bond_weights), self.triple_bond_bias)

        x_1 = tf.linalg.matmul(x_1, inputs, transpose_b=True)
        x_1 = Activation('sigmoid', dtype='float32', name='single_bonds_predictions')(x_1)

        x_2 = tf.linalg.matmul(x_2, inputs, transpose_b=True)
        x_2 = Activation('sigmoid', dtype='float32', name='double_bonds_predictions')(x_2)

        x_3 = tf.linalg.matmul(x_3, inputs, transpose_b=True)
        x_3 = Activation('sigmoid', dtype='float32', name='triple_bonds_predictions')(x_3)

        x_4 = tf.linalg.matmul(x_4, inputs, transpose_b=True)
        x_4 = Activation('sigmoid', dtype='float32', name='self_bonds_predictions')(x_4)

        return K.concatenate([K.expand_dims(x_1), K.expand_dims(x_2), K.expand_dims(x_3), K.expand_dims(x_4)])


class Defactorization(Layer):
    def __init__(self, *args, **kwargs):
        super(Defactorization, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.def_weights = self.add_weight(shape=(input_shape[-1],), initializer='he_normal', trainable=True,
                                           name='def_weights')
        self.def_bias = self.add_weight(shape=(input_shape[-2], input_shape[-2]), initializer='he_normal',
                                        trainable=True, name='bias')

    def call(self, inputs, **kwargs):
        """
        Implementation of one-channel de-factorization layer described in DEFactor (arXiv:1811.09766)
        Formula: A = V^T * W * V + B, where V - is atoms vectors input, W - diagonal matrix of trainable weights,
        B - trainable bias with shape (number of atoms, number of atoms).
        :param inputs: atoms vectors with the shape (batch, number of atoms, length of atom vector)
        :param kwargs: done to maintain compatibility with Keras
        :return: the values to convert into adjacency matrix with shape (number of atoms, number of atoms)
        """

        updated_vec = tf.multiply(inputs, self.def_weights)
        adj = tf.linalg.matmul(updated_vec, inputs, transpose_b=True)
        adj = K.bias_add(adj, self.def_bias)
        adj = Activation('sigmoid', dtype='float32', name='bonds_predictions')(adj)

        return adj


def refactor_encoder(max_atoms, batch_size, max_atomic_num, atom_vector_dim, mol_vector_dim, distributed, num_convs=5):
    atoms = Input(shape=(max_atoms,), batch_size=batch_size, name='Encoder_Atoms')
    adjacency_matrix = Input(shape=(max_atoms, max_atoms), batch_size=batch_size, name='Encoder_Adj')
    mask = Input(shape=(max_atoms,), batch_size=batch_size, name='Encoder_Mask')

    e_term = multi_term_calc()(adjacency_matrix)
    expanded_mask = tf.expand_dims(mask, 2)

    masked_atoms = atoms * mask
    emb = Embedding(max_atomic_num + 2, atom_vector_dim,
                    mask_zero=True, name='embedding_atoms')(masked_atoms)

    expanded_mask = tf.cast(expanded_mask, emb.dtype)
    x = tf.multiply(emb, expanded_mask)

    # Convolution
    for _ in range(num_convs):
        x = multi_bond_gcn(max_atoms, batch_size, mol_vector_dim, distributed)([x, e_term])
        x = tf.multiply(x, expanded_mask)

    # Aggregation, l_v - molecular latent vector
    l_v = Bidirectional(GRU(atom_vector_dim))(x)
    l_v = Dense(units=mol_vector_dim, kernel_initializer='he_normal', name='aggregator_dense')(l_v)
    if distributed:
        l_v = SyncBatchNormalization()(l_v)
    else:
        l_v = BatchNormalization()(l_v)
    l_v = K.relu(l_v)

    return Model(inputs=[atoms, adjacency_matrix, mask], outputs=l_v)


def refactor_decoder(max_atoms, batch_size, max_atom_num, atom_vector_dim, mol_vector_dim, distributed):
    mol_vectors = Input(shape=(mol_vector_dim,), batch_size=batch_size, name='Decode_Vec')

    molecular_vectors = RepeatVector(max_atoms)(mol_vectors)
    atoms_vectors = GRU(mol_vector_dim, return_sequences=True)(molecular_vectors)
    atoms_vectors = K.concatenate([molecular_vectors, atoms_vectors], axis=-1)
    atoms_vectors = Dense(atom_vector_dim, kernel_initializer='he_normal')(atoms_vectors)
    atoms_vectors = K.relu(atoms_vectors)
    atoms_vectors = GRU(atom_vector_dim, return_sequences=True)(atoms_vectors)

    atom_type = Dense(max_atom_num + 1, kernel_initializer='he_normal')(atoms_vectors)
    atom_type = K.relu(atom_type)

    atom_type = Dense(max_atom_num + 1, kernel_initializer='he_normal')(atom_type)
    atom_type = Activation('softmax', dtype='float32', name='atoms_predictions')(atom_type)

    adj_vectors = Dense(int(atom_vector_dim / 8), kernel_initializer='he_normal')(atoms_vectors)
    adj_matrix = MultiDefactorization(int(atom_vector_dim / 8))(adj_vectors)

    return Model(inputs=mol_vectors, outputs=[atom_type, adj_matrix])


def hyfactor_encoder(max_atoms, batch_size, max_atomic_num, atom_vector_dim, mol_vector_dim, distributed, num_convs=5):
    atoms = Input(shape=(max_atoms,), batch_size=batch_size, name='Encode_Atoms')
    hydrogens = Input(shape=(max_atoms,), batch_size=batch_size, name='Encode_Hydrogens')
    adj_matrix = Input(shape=(max_atoms, max_atoms), batch_size=batch_size, name='Encode_Adj')
    mask = Input(shape=(max_atoms,), batch_size=batch_size, name='Encode_Mask')

    e_term = term_calc()(adj_matrix)

    expanded_mask = tf.expand_dims(mask, 2)
    emb = Embedding(max_atomic_num + 2, 64, mask_zero=True, name='atom_embedding')(atoms)
    expanded_mask = tf.cast(expanded_mask, emb.dtype)
    emb = tf.multiply(emb, expanded_mask)

    hydrogens_emb = Embedding(4, 4, name='hydrogens_embedding')(hydrogens)
    hydrogens_emb = tf.multiply(hydrogens_emb, expanded_mask)

    atom_vec = K.concatenate([emb, hydrogens_emb], axis=-1)
    atom_vec = Dense(units=atom_vector_dim, kernel_initializer='he_normal')(atom_vec)
    x = K.relu(atom_vec)

    # Convolution
    for _ in range(num_convs):
        x = gcn(max_atoms, batch_size, atom_vector_dim, distributed)([x, e_term])
        x = tf.multiply(x, expanded_mask)

    # Aggregation
    # l_v - molecular latent vector
    l_v = Bidirectional(GRU(atom_vector_dim))(x)
    l_v = Dense(units=mol_vector_dim, kernel_initializer='he_normal')(l_v)
    if distributed:
        l_v = SyncBatchNormalization()(l_v)
    else:
        l_v = BatchNormalization()(l_v)
    l_v = K.relu(l_v)

    return Model(inputs=[atoms, hydrogens, adj_matrix, mask], outputs=l_v)


def hyfactor_decoder(max_atoms, batch_size, max_atom_num, atom_vector_dim, mol_vector_dim, distributed):
    mol_vectors = Input(shape=(mol_vector_dim,), batch_size=batch_size, name='Decode_Vec')

    molecular_vectors = RepeatVector(max_atoms)(mol_vectors)
    atoms_vectors = GRU(mol_vector_dim, return_sequences=True)(molecular_vectors)
    atoms_vectors = K.concatenate([molecular_vectors, atoms_vectors], axis=-1)
    atoms_vectors = Dense(atom_vector_dim, kernel_initializer='he_normal')(atoms_vectors)
    atoms_vectors = K.relu(atoms_vectors)
    atoms_vectors = GRU(atom_vector_dim, return_sequences=True)(atoms_vectors)

    # Prediction of atom type
    atom_type_vec = Dense(int(atom_vector_dim / 8), kernel_initializer='he_normal')(atoms_vectors)
    atom_type_vec = K.relu(atom_type_vec)
    atom_type = Dense(max_atom_num + 1, kernel_initializer='he_normal')(atom_type_vec)
    atom_type = Activation('softmax', dtype='float32', name='atoms_predictions')(atom_type)

    # Prediction of adjacency matrix
    adj_vectors = Dense(int(atom_vector_dim / 8), kernel_initializer='he_normal')(atoms_vectors)
    adj_vectors = K.relu(adj_vectors)
    adj_matrix = Defactorization(atom_vector_dim)(adj_vectors)

    # Prediction of hydrogens count
    hydrogens_count = Dense(int(atom_vector_dim / 8), kernel_initializer='he_normal')(atoms_vectors)
    hydrogens_count = K.relu(hydrogens_count)
    hydrogens_count = Dense(4, kernel_initializer='he_normal')(hydrogens_count)
    hydrogens_count = Activation('softmax', dtype='float32', name='hydrogens_predictions')(hydrogens_count)

    return Model(inputs=mol_vectors, outputs=[atom_type, hydrogens_count, adj_matrix])
