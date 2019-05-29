import re
from pathlib import Path
from setuptools import setup, find_packages

txt = Path('asr/__init__.py').read_text()
version = re.search("__version__ = '(.*)'", txt).group(1)

long_description = Path('README.md').read_text()

setup(
    name='asr',
    version=version,
    description='Atomic Simulation Recipes',
    long_description=long_description,
    author='M. N. Gjerding',
    author_email='mogje@fysik.dtu.dk',
    url='https://gitlab.com/mortengjerding/asr',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['Click', 'pytest', 'matplotlib',
                      'spglib', 'plotly'],
    entry_points='''
        [console_scripts]
        asr=asr.utils.cli:cli
    ''',
    classifiers=[
        'Development Status :: 0 - Beta', 'Environment :: Console',
        'Intended Audience :: Developers', 'License :: OSI Approved :: '
        'GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ])
