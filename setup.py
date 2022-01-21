#!/usr/bin/env python

"""The setup script."""

import re
from pathlib import Path
from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

python_requires = '>=3.8'
requirements = [
    'pyparsing>=3.0.6',
    'Click>=8.0.3',
    'ase', 'matplotlib',
    'simplejson',
    'spglib', 'plotly', 'flask', 'pymatgen',
    'phonopy',
]

setup_requirements = []

extras_require = {'docs': ['sphinx', 'sphinx-autoapi',
                           'sphinxcontrib-programoutput']}

txt = Path('asr/__init__.py').read_text()
version = re.search("__version__ = '(.*)'", txt).group(1)

package_data = {'asr': ['database/templates/*.html',
                        'setup/substitution.dat',
                        'setup/testsystems.dat']}


setup(
    author="Morten Niklas Gjerding",
    author_email='mortengjerding@gmail.com',
    python_requires=python_requires,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
    ],
    description="ASE recipes for calculating material properties",
    package_data=package_data,
    entry_points={
        'console_scripts': [
            'asr=asr.core.cli:cli',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='asr',
    name='asr',
    packages=find_packages(include=['asr', 'asr.*']),
    setup_requires=setup_requirements,
    url='https://gitlab.com/asr-dev/asr',
    version=version,
    zip_safe=False,
)
