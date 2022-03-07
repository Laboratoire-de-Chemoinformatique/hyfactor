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

from setuptools import find_packages, setup

setup(
    name="hyfactor",
    version="0.2.0",
    packages=['hyfactor'],
    description="Hydrogen-count labeled graph-based AutoEncoder",
    author="Laboratoire de Chemoinformatique, Laboratory of chemoinformatics and molecular modeling",
    author_email="tagirshin@gmail.com",
    license="LGPLv3",
    python_requires='<3.10',
    install_requires=[],
    entry_points={
        "console_scripts": ["hyfactor = hyfactor.main:main"]
    },
)
