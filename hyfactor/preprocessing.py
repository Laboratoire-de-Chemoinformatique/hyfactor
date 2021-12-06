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

from pathlib import Path

import numpy as np
from CGRtools.containers import MoleculeContainer
from CGRtools.files import SDFRead
from tqdm import tqdm


def _generate_atom_types(input_file: str):
    atom_types = set()
    with SDFRead(input_file) as inp:
        for molecule in inp:
            for n, atom in molecule.atoms():
                atom_types.add((atom.atomic_number, atom.charge,))

    return tuple(atom_types)


def extract_atom_types(input_file: str, output_file=None):
    if output_file:
        import pickle
        if Path(output_file).is_file():
            print('Using the created tuple of atom types')
            with open(output_file, 'rb') as inp:
                atom_types = pickle.load(inp)
        else:
            print('Creating a new tuple of atom types')
            atom_types = _generate_atom_types(input_file)
            with open(output_file, 'wb') as out:
                pickle.dump(atom_types, out)

    else:
        print('Creating a new tuple of atom types without saving')
        atom_types = _generate_atom_types(input_file)

    return atom_types


def atom_to_vector(atom, normalization=False):
    vector = np.zeros(14, dtype=np.int8)
    vector[0] += atom.atomic_number
    vector[1] += max(atom.isotopes_distribution, key=atom.isotopes_distribution.get) - atom.atomic_number
    vector[2] += atom.implicit_hydrogens
    e = atom.atomic_number + atom.charge
    i = 3
    orbitals = [2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6]
    while e > 0:
        e_level = orbitals.pop(0)
        if e > e_level:
            vector[i] += e_level
        else:
            vector[i] += e
        i += 1
        e -= e_level

    if normalization:
        vector /= np.array([54, 77, 4, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6], dtype=np.int32)

    return vector


def graph_to_atoms_vectors(molecule: MoleculeContainer, config_set: dict):
    atoms_vectors = np.zeros((config_set['max_atoms'], 14), dtype=np.int8)
    for n, atom in molecule.atoms():
        atoms_vectors[n - 1] = atom_to_vector(atom)
    return atoms_vectors


def graph_to_atoms_numbers(molecule: MoleculeContainer, atom_types: tuple, max_atoms: int):
    atoms = np.zeros((max_atoms,), dtype=np.int8)
    for n, atom in molecule.atoms():
        atoms[n - 1] = atom_types.index(tuple([atom.atomic_number, atom.charge])) + 1
    atoms[n] = len(atom_types) + 1
    return atoms


def create_atoms_mask(molecule: MoleculeContainer, max_atoms: int):
    atoms = np.zeros((max_atoms,), dtype=np.int8)
    for n, atom in molecule.atoms():
        atoms[n - 1] = 1
    return atoms


def graph_to_adj_matrix(molecule: MoleculeContainer, max_atoms: int):
    adj_matrix = np.zeros((max_atoms, max_atoms), dtype=np.int8)

    for a, n, order in molecule.bonds():
        adj_matrix[a - 1][n - 1] = adj_matrix[n - 1][a - 1] = 1

    return adj_matrix


def graph_to_bond_matrix(molecule: MoleculeContainer, max_atoms: int):
    bond_matrix = np.zeros((max_atoms, max_atoms), dtype=np.int8)

    for a, n, order in molecule.bonds():
        bond_matrix[a - 1][n - 1] = bond_matrix[n - 1][a - 1] = int(order)
        if int(order) == 4:
            raise ValueError('Found a structure with aromatic bond')

    return bond_matrix


def generate_h_true(molecule: MoleculeContainer, max_atoms: int):
    hydrogens = np.zeros((max_atoms,), dtype=np.int8)

    for n, atom in molecule.atoms():
        hydrogens[n - 1] = int(atom.implicit_hydrogens)

    return hydrogens


def generate_h_one_hot(molecule: MoleculeContainer, max_atoms: int):
    hydrogens = np.zeros((max_atoms, 4), dtype=np.int8)

    for n, atom in molecule.atoms():
        hydrogens[n - 1][int(atom.implicit_hydrogens)] = 1

    return hydrogens


def check_input_file(config, file_type):
    if file_type == 'train' or file_type == 'encode':
        return config['idf']
    elif file_type == 'val':
        return config['val']
    else:
        return config['test']


def preprocess_refactor(atom_types, dataset, num_samples, max_atoms):
    atom_matrices = np.zeros([num_samples, max_atoms], dtype=np.int8)
    bonds_matrices = np.zeros([num_samples, max_atoms, max_atoms], dtype=np.int8)
    for n, molecule in tqdm(enumerate(dataset), total=num_samples):

        if len(molecule) >= max_atoms:
            raise ValueError('Found molecule with size bigger than defined')

        if n == num_samples:
            break

        atom_matrices[n, :] = graph_to_atoms_numbers(molecule, atom_types, max_atoms)
        bonds_matrices[n, :, :] = graph_to_bond_matrix(molecule, max_atoms)

    data = [atom_matrices, bonds_matrices]
    return data


def preprocess_hyfactor(atom_types, dataset, num_samples, max_atoms):
    atom_matrices = np.zeros([num_samples, max_atoms], dtype=np.int8)
    hydrogens_matrices = np.zeros([num_samples, max_atoms], dtype=np.int8)
    adj_matrices = np.zeros([num_samples, max_atoms, max_atoms], dtype=np.int8)
    for n, molecule in tqdm(enumerate(dataset), total=num_samples):

        if len(molecule) >= max_atoms:
            raise ValueError('Found molecule with size bigger than defined')

        if n == num_samples:
            break

        atom_matrices[n, :] = graph_to_atoms_numbers(molecule, atom_types, max_atoms)
        hydrogens_matrices[n, :, ] = generate_h_true(molecule, max_atoms)
        adj_matrices[n, :, :] = graph_to_adj_matrix(molecule, max_atoms)

    data = [atom_matrices, hydrogens_matrices, adj_matrices]

    return data


def preprocess_mol_graphs(config: dict, file_type: str, atom_types: tuple):
    input_file = check_input_file(config, file_type)
    with SDFRead(input_file, indexable=True) as dataset:
        dataset.reset_index()
        print(f'Size of input dataset: {len(dataset)}')
        if file_type == 'encode':
            num_samples = (len(dataset) // config['batch'] + 1) * config['batch']
            print(f'Number of zero structures that will remain: {num_samples - len(dataset)}')
        else:
            num_samples = (len(dataset) // config['batch']) * config['batch']
            print(f'Number of molecules that will remain: {num_samples}')

        if config['model'] == 'refactor':
            data = preprocess_refactor(atom_types, dataset, num_samples, config['max_atoms'])
        elif config['model'] == 'hyfactor':
            data = preprocess_hyfactor(atom_types, dataset, num_samples, config['max_atoms'])

        return data


def prepare_data(config: dict, file_type: str, atom_types: tuple):
    if config['tmp']:
        file = config['tmp'] + '_' + file_type + '.npz'
        if Path(file).is_file():
            print(f'Using the saved file {file}')
            data = np.load(file)
            data = [data[x] for x in data]
        else:
            print(f'Started generation of {file_type} dataset')
            data = preprocess_mol_graphs(config, file_type, atom_types)
            np.savez(file, *data)

    else:
        print(f'Started generation of {file_type} dataset without saving')
        data = preprocess_mol_graphs(config, file_type, atom_types)

    return data
