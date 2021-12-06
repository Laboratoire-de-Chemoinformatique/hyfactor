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

from collections import defaultdict, Counter
from itertools import product
from random import choice, shuffle
from typing import Generator, Tuple, DefaultDict, Union, Any, NoReturn, TextIO

from CGRtools.containers import MoleculeContainer
from numpy import zeros, ndarray, count_nonzero
from tqdm import tqdm


def iterate_svm(svm_file: str) -> Generator[Tuple[Union[str, float], DefaultDict[int, float]], None, None]:
    with open(svm_file) as svm_inp:
        for i, line in enumerate(svm_inp):
            column_value = line.split()
            if column_value[0].isdigit():
                vector_index = float(column_value[0])
            else:
                vector_index = column_value[0]
            vector = defaultdict(float)
            for val in column_value[1:]:
                tmp = val.split(':')
                vector[int(tmp[0])] = float(tmp[1])
            yield vector_index, vector


def count_svm_lines(input_file: str) -> int:
    with open(input_file, 'r') as inp:
        n_lines = sum(1 for _ in inp)
    return n_lines


def count_svm_columns(input_file: str) -> int:
    with open(input_file, 'r') as inp:
        line = inp.readline()
        nums = line.split(' ')[1:]
        last_node, _ = nums[-1].split(':')

    return int(last_node)


def write_svm(index: Any, vector: ndarray, file: TextIO) -> NoReturn:
    num_columns = vector.shape[0] - 1
    file.write(str(index))
    for j in range(num_columns):
        if vector[j]:
            file.write(f' {j + 1}:{vector[j]}')
    file.write(f' {num_columns + 1}:{vector[num_columns]}\n')


def write_lv(index, vectors, file):
    num_vectors = vectors.shape[0]
    for i in tqdm(range(num_vectors)):
        if count_nonzero(vectors[i]):
            write_svm(index, vectors[i], file)
            index += 1
    return index


def load_latent_vectors(input_file):
    print('Calculating latent vectors file size')
    n_lines = count_svm_lines(input_file)
    last_i = count_svm_columns(input_file)

    vectors = zeros((n_lines, int(last_i),))
    print(f'Reading latent vectors from svm file with size {n_lines}x{last_i}')
    for n, (_, raw_vector) in tqdm(enumerate(iterate_svm(input_file)), total=n_lines):
        for ind, val in raw_vector.items():
            vectors[n, ind - 1] = val
    return vectors


def matrix_chunking(input_data, num_of_chunks, batch):
    num_input_batches = input_data.shape[0] / batch
    chunk_size = int(num_input_batches // num_of_chunks * batch)

    all_data = chunk_size * num_of_chunks
    if all_data != input_data.shape[0]:
        dropped = input_data.shape[0] - all_data
        print(f'{dropped} inputs will be dropped')

    return chunk_size


def create_chem_graph(atoms_sequence, connectivity_matrix, atom_types) -> MoleculeContainer:
    """
    Create molecular graph or basis for HLG
    :param atoms_sequence: sequence of atoms
    :param atom_types: atoms types
    :param connectivity_matrix: adjacency matrix
    :return: molecular graph
    """
    molecule = MoleculeContainer()

    for n, atom in enumerate(atoms_sequence):
        if atom == len(atom_types):
            break
        atomic_symbol, charge = atom_types[int(atom)]
        molecule.add_atom(atom=atomic_symbol, charge=charge)

    for i in range(len(molecule)):
        for j in range(i + 1, len(molecule)):
            if connectivity_matrix[i][j]:
                molecule.add_bond(i + 1, j + 1, int(connectivity_matrix[i][j]))

    return molecule


def generate_unsaturated_adjacency(desaturation, neighbors):
    unsaturated_adj = defaultdict(set)

    for n in list(desaturation.keys()):
        if desaturation[n]:
            for neighbour in neighbors[n]:
                if desaturation[neighbour]:
                    unsaturated_adj[n].add(neighbour)

    return unsaturated_adj


def find_unsaturated_atoms(molecule, h_sequence, all_rules):
    adj = {atom: set(neighbors) for atom, neighbors in molecule._bonds.items()}

    desaturation = defaultdict(list)
    for n, atom in molecule.atoms():
        h_count = h_sequence[n - 1]
        atom_charge = atom.charge
        atom_radical = atom.is_radical
        num_neighbours = len(adj[n])

        counts = Counter([molecule.atom(neighbor).atomic_number for neighbor in adj[n]])

        neighbors_for_rule = []
        neighbors_count = 0
        for atom_number, count in counts.items():
            neighbors_for_rule.append((atom_number, count,))
            neighbors_count += count
        neighbors_for_rule = tuple(sorted(neighbors_for_rule))

        atom_valence_rules = all_rules[atom.atomic_number]
        possible_valences = []
        if atom.atomic_symbol == 'S' and h_count == 0 and neighbors_count == 2:
            possible_valences == [(0, False, 2)]
        elif atom.atomic_symbol == 'P' and h_count == 0 and neighbors_count == 3:
            possible_valences == [(0, False, 3)]
        else:
            if (neighbors_for_rule, h_count) in atom_valence_rules:
                possible_valences = atom_valence_rules[(neighbors_for_rule, h_count)]
            elif ((), h_count) in atom_valence_rules:
                possible_valences = atom_valence_rules[((), h_count)]
            else:
                continue

        for charge, radical, valence in possible_valences:
            if (charge, radical) == (atom_charge, atom_radical):
                desaturation_value = valence - num_neighbours
                if desaturation_value > 0:
                    desaturation[n].append(desaturation_value)

    return desaturation, adj


def generate_options(raw_options):
    options = []
    keys = raw_options.keys()
    for element in product(*raw_options.values()):
        options.append(dict(zip(keys, element)))
    return options


def update_bond(molecule: MoleculeContainer, update: int, atom: int, neighbor: int) -> MoleculeContainer:
    """
    Updating bonds in molecule.
    :param molecule: input molecule
    :param update: number of bonds that we want to add
    :param atom: central atom
    :param neighbor: neighbor of central atom
    :return: updated molecule
    """
    old_bond = int(molecule.bond(atom, neighbor))
    molecule.delete_bond(atom, neighbor)
    new_bond = int(old_bond + update)
    molecule.add_bond(atom, neighbor, new_bond)
    return molecule


def update_unsaturated_neighbors(num_atom, unsaturated_neighbors_copy):
    for atom, neighbors in unsaturated_neighbors_copy.items():
        if num_atom in neighbors:
            neighbors.remove(num_atom)
    return unsaturated_neighbors_copy


def update_atoms_desaturation(atoms_desaturation, unsat_neighbors, atom, neighbor, choices):
    atom_desaturation = atoms_desaturation[atom]
    neighbor_desaturation = atoms_desaturation[neighbor]
    if atom_desaturation > neighbor_desaturation:
        update = neighbor_desaturation
        unsat_neighbors[atom].remove(neighbor)
        unsat_neighbors = update_unsaturated_neighbors(neighbor,
                                                       unsat_neighbors)
    elif atom_desaturation == neighbor_desaturation:
        update = atom_desaturation
        unsat_neighbors = update_unsaturated_neighbors(atom, unsat_neighbors)
        unsat_neighbors = update_unsaturated_neighbors(neighbor,
                                                       unsat_neighbors)
    elif atom_desaturation < neighbor_desaturation:
        update = atom_desaturation
        unsat_neighbors[neighbor].remove(atom)
        unsat_neighbors = update_unsaturated_neighbors(atom, unsat_neighbors)
    choices.append((atom, neighbor, update + 1))

    atoms_desaturation[atom] -= update
    atoms_desaturation[neighbor] -= update
    return atoms_desaturation, unsat_neighbors


def set_multiple_bonds(molecule, desaturation_options, unsaturated_neighbors):
    if len(molecule) > 40:
        num_trials = 10000
    else:
        num_trials = 200

    molecule_copy = molecule.copy()

    for trial in range(num_trials):
        choices = []
        option_id = choice([i for i in range(len(desaturation_options))])

        atoms_desaturation = defaultdict(int)
        for atom, desaturation in desaturation_options[option_id].items():
            atoms_desaturation[atom] = desaturation

        unsat_neighbors = defaultdict(list)
        for atom, neighbors in unsaturated_neighbors.items():
            unsat_neighbors[atom] = list(neighbors)

        rec_error = False
        atoms = list(atoms_desaturation.keys())
        shuffle(atoms)
        for atom in atoms:
            while atoms_desaturation[atom]:
                if not unsat_neighbors[atom]:
                    rec_error = True
                    break
                else:
                    neighbor = choice(unsat_neighbors[atom])
                    if atoms_desaturation[neighbor] == 0:
                        rec_error = True
                        break
                    else:
                        atoms_desaturation, unsat_neighbors = update_atoms_desaturation(atoms_desaturation,
                                                                                        unsat_neighbors,
                                                                                        atom, neighbor,
                                                                                        choices)

            if rec_error:
                break
        if not rec_error:
            break

    for atom, neighbor, bond in choices:
        molecule_copy.delete_bond(atom, neighbor)
        molecule_copy.add_bond(atom, neighbor, bond)

    return molecule_copy, rec_error, trial


def reconstruct_molecule_hyfactor(atoms, hydrogens, adjs, atom_types, recompiled_rules):
    saturated_molecule = create_chem_graph(atoms, adjs, atom_types)
    desaturation, neighbors = find_unsaturated_atoms(saturated_molecule, hydrogens, recompiled_rules)
    desaturation_options = generate_options(desaturation)
    unsaturated_adjacency = generate_unsaturated_adjacency(desaturation, neighbors)
    new_molecule, rec_error, trials = set_multiple_bonds(saturated_molecule, desaturation_options,
                                                         unsaturated_adjacency)
    new_molecule.meta.update({'Rec_error': rec_error, 'Num_of_trials': trials})
    return new_molecule
