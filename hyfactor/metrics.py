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


def adj_masking(true, mask):
    batch, max_atoms, _ = K.int_shape(true)
    expanded_mask = tf.expand_dims(mask, 2)
    adj_mask = tf.matmul(expanded_mask, expanded_mask, transpose_b=True)
    adj_mask -= tf.linalg.diag(tf.ones([batch, max_atoms], true.dtype))
    adj_mask = tf.cast(adj_mask > 0, true.dtype)
    return adj_mask


def log_like_atoms(a_true, a_pred, length, config):
    a_true = tf.subtract(a_true, 1)
    one_hot = tf.one_hot(tf.cast(a_true, tf.int64), depth=config['max_atom_num'] + 1)

    loss = tf.math.log(tf.add(a_pred, K.epsilon()))
    loss = tf.multiply(one_hot, loss)
    loss = tf.reduce_sum(loss, axis=(-1, -2))
    loss = tf.divide(loss, length)

    return tf.multiply(loss, tf.constant(-1.0))


def log_like_bonds(b_true, b_pred, mask, length, config):
    normalization = tf.cast(tf.multiply(length + 1, length + 1), b_pred.dtype)
    bonds_mask = adj_masking(b_true, mask)
    bonds_true = b_true + bonds_mask - 1

    bonds_one_hot = tf.one_hot(tf.cast(bonds_true, tf.int64), depth=config['max_bond_order'] + 1)
    zero_one_hot = tf.subtract(tf.ones_like(bonds_one_hot), bonds_one_hot)

    zero_mask = tf.cast(tf.tile(tf.expand_dims(bonds_mask, axis=-1), [1, 1, 1, 4]), zero_one_hot.dtype)
    zero_one_hot = tf.multiply(zero_one_hot, zero_mask)

    log_1 = tf.math.log(tf.add(b_pred, K.epsilon()))
    log_2 = tf.math.log(tf.add(tf.subtract(1.0, b_pred), K.epsilon()))

    term_1 = tf.multiply(bonds_one_hot, log_1)
    term_2 = tf.multiply(zero_one_hot, log_2)

    loss_bond = tf.add(term_1, term_2)
    loss_bond = tf.reduce_sum(loss_bond, axis=3)
    loss_bond = tf.multiply(loss_bond, tf.cast(bonds_mask, loss_bond.dtype))

    loss_bond = tf.reduce_sum(loss_bond, axis=(1, 2))
    loss_bond = tf.divide(loss_bond, normalization)

    return tf.multiply(loss_bond, tf.constant(-1.0))


def log_like_adj(adj_true, adj_pred, mask, length):
    normalization = tf.cast(tf.multiply(length, length), adj_true.dtype)
    adj_mask = adj_masking(adj_true, mask)

    real_bonds = tf.cast(adj_true == 1, adj_pred.dtype)
    zero_bonds = tf.cast(adj_true == 0, adj_pred.dtype)

    log_1 = tf.math.log(tf.add(adj_pred, K.epsilon()))
    log_2 = tf.math.log(tf.add(tf.subtract(1.0, adj_pred), K.epsilon()))

    term_1 = tf.multiply(real_bonds, log_1)
    term_2 = tf.multiply(zero_bonds, log_2)

    loss_bond = tf.add(term_1, term_2)
    loss_bond = tf.multiply(loss_bond, adj_mask)
    loss_bond = tf.reduce_sum(loss_bond, axis=(1, 2))
    loss_bond = tf.divide(loss_bond, normalization)

    return tf.multiply(loss_bond, tf.constant(-1.0))


def log_like_hydrogens(h_true, h_pred, mask, length):
    one_hot = tf.one_hot(tf.cast(h_true, tf.int64), depth=4)

    loss = tf.math.log(tf.add(h_pred, K.epsilon()))
    loss = tf.multiply(one_hot, loss)
    loss = tf.reduce_sum(loss, axis=-1)
    loss = tf.multiply(mask, loss)
    loss = tf.reduce_sum(loss, axis=-1)
    loss = tf.divide(loss, length)

    return tf.multiply(loss, tf.constant(-1.0))


def accuracy_atoms(atoms_true, atoms_pred, mask, length):
    atoms_chosen = tf.cast(K.argmax(atoms_pred), atoms_true.dtype) + 1
    atoms_chosen = tf.multiply(atoms_chosen, mask)
    atoms_true = tf.multiply(atoms_true, mask)

    score = tf.cast(tf.math.not_equal(atoms_true, atoms_chosen), atoms_true.dtype)
    score = tf.reduce_sum(score, axis=-1)
    score = tf.math.multiply(tf.math.divide(score, length), tf.constant(100, score.dtype))

    return score


def accuracy_hydrogens(atoms_true, atoms_pred, mask, length):
    atoms_chosen = tf.cast(K.argmax(atoms_pred), atoms_true.dtype)
    atoms_chosen = tf.multiply(atoms_chosen, mask)

    score = tf.cast(tf.math.not_equal(atoms_true, atoms_chosen), atoms_true.dtype)
    score = tf.reduce_sum(score, axis=-1)
    score = tf.math.multiply(tf.math.divide(score, length), tf.constant(100, score.dtype))

    return score


def accuracy_bonds(bonds_true, bonds_pred, mask, length):
    bonds_chosen = tf.cast(K.argmax(bonds_pred), bonds_true.dtype)
    bonds_mask = adj_masking(bonds_true, mask)
    bonds_chosen = tf.multiply(bonds_chosen, bonds_mask)
    bonds_true = tf.multiply(bonds_true, bonds_mask)

    score = tf.cast(tf.math.not_equal(bonds_true, bonds_chosen), bonds_true.dtype)
    score = tf.reduce_sum(score, axis=(-1, -2))
    normalization = tf.cast(tf.multiply(length, length), score.dtype)
    score = tf.math.divide(score, normalization)
    score = tf.math.multiply(score, tf.constant(100, score.dtype))

    return score


def accuracy_adj(adj_true, adj_pred, mask, length):
    connect_chosen = tf.cast(adj_pred > 0.5, adj_true.dtype)
    bonds_mask = adj_masking(adj_true, mask)
    bonds_chosen = tf.multiply(connect_chosen, bonds_mask)

    score = tf.cast(tf.math.not_equal(adj_true, bonds_chosen), adj_true.dtype)
    score = tf.reduce_sum(score, axis=(-1, -2))
    normalization = tf.cast(tf.multiply(length, length - 1), score.dtype)
    score = tf.math.divide(score, normalization)
    score = tf.math.multiply(score, tf.constant(100, score.dtype))

    return score


def reconstruction_rate_refactor(atoms_true, bonds_true, atoms_pred, bonds_pred, mask, length):

    atoms_errors = accuracy_atoms(atoms_true, atoms_pred, mask, length)
    bonds_errors = accuracy_bonds(bonds_true, bonds_pred, mask, length)
    error_score = tf.add(atoms_errors, bonds_errors)

    reconstruction_score = tf.math.multiply(tf.cast(tf.math.equal(error_score, 0), atoms_true.dtype),
                                            tf.constant(100, atoms_true.dtype))

    return reconstruction_score


def reconstruction_rate_hyfactor(atoms_true, hydrogens_true, adj_true,
                                 atoms_pred, hydrogens_pred, adj_pred,
                                 mask, length):
    atoms_errors = accuracy_atoms(atoms_true, atoms_pred, mask, length)
    hydrogens_errors = accuracy_hydrogens(hydrogens_true, hydrogens_pred, mask, length)
    bonds_errors = accuracy_adj(adj_true, adj_pred, mask, length)

    error_score = atoms_errors + bonds_errors + hydrogens_errors
    reconstruction_score = tf.math.multiply(tf.cast(tf.math.equal(error_score, 0), atoms_true.dtype),
                                            tf.constant(100, atoms_true.dtype))

    return reconstruction_score
