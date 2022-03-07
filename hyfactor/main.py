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

from os import environ

environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # to silent debugger of tf

import argparse
import yaml
from .preprocessing import prepare_data, extract_atom_types
from .nn_utils import config_names_converter, check_configs, load_ae, train, evaluate, encode, decode, generate_mols
from tensorflow.keras import mixed_precision


def task_preparer(config):
    config = config_names_converter(config)
    check_configs(config)

    if config['use_mixed_precision']:
        print('Mixed precision applied')
        mixed_precision.set_global_policy('mixed_float16')

    atom_types = extract_atom_types(config['idf'], config['atom_types'])
    config['max_atom_num'] = len(atom_types)
    config['distributed'] = isinstance(config['GPUs'], list)

    autoencoder = load_ae(config)

    if config['task'] == 'train':
        train(config, autoencoder, atom_types)

    elif config['task'] == 'evaluate':
        evaluate(config, autoencoder, atom_types)

    elif config['task'] == 'encode':
        encode(config, autoencoder, atom_types)

    elif config['task'] == 'decode':
        decode(config, autoencoder, atom_types)

    elif config['task'] == 'reconstruct':
        input_data = prepare_data(config, 'encode', atom_types)
        generate_mols(autoencoder, input_data, config, atom_types)

    else:
        raise ValueError('I do not know which task you have specified')


def main():
    """
    Entry point for "hyfactor" command
    """
    print('Hello there...')
    parser = argparse.ArgumentParser("hyfactor")

    parser.add_argument('-cfg', '--config', type=str, help='Way to configuration file')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    task_preparer(config)


if __name__ == '__main__':
    main()
