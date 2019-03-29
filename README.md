Atomic Simulation Recipes
=========================
Recipes for Atomic Scale Materials Research.

Collection of python recipes that just work(!) for common (and not so common)
tasks perfomed in atomic scale materials research. These tasks include
relaxation of structures, calculating ground states, calculating band
structures, calculating dielectric functions and so on.

Requirements
------------

* ASE (Atomic Simulation Environment)
* GPAW
* click
* spglib

Additionally, but not a requirement, it can be nice to have
* myqueue

Installation
------------

```console
$ cd ~ && git clone https://gitlab.com/mortengjerding/asr.git pythonpackages/asr
$ echo  'export PYTHONPATH=~/pythonpackages:$PYTHONPATH' >> ~/.bashrc 
```

How to use
----------
Lets calculate the properties of Silicon. To do this, we start by creating
a new folder and put a 'start.json' file into the directory containing
the atomic structure of Silicon. Then we relax the structure.
```console
$ mkdir ~/silicon && cd ~/silicon
$ ase build -x diamond Si start.json
$ python3 -m asr.relax
```

This generates a new folder `~/silicon/nm/` containing a new `start.json`
file that contains the final relaxed structure. Going into this directory we
get some quick information about this structure by running the `asr.quickinfo`
recipe, collect the data to a database using `asr.collect` recipe and view it
in a browser with the `asr.browser` recipe. This is done below

```console
$ cd ~/silicon/nm
$ python3 -m asr.quickinfo
$ python3 -m asr.collect .
$ python3 -m asr.browser
```

Change default settings in scripts
----------------------------------
All material folders can contain a `params.json`-file. This file can
changed to overwrite default settings in scripts. For example:

```javascript
{
    "asr.gs": {"gpw": "otherfile.gpw",
               "ecut": 800},
    "asr.relax": {"states": ["nm", ]}
}
```

In this way all default parameters exposed through the CLI of a recipe
can be corrected.

See help for a recipe
---------------------
We assume that you have cloned the project into `~/asr/` and have added
this folder to your `PYTHONPATH`. To see the command line interface (CLI)
help of the relax recipe we simply do

```console
$ python3 -m asr.gs --help
Usage: gs.py [OPTIONS]

  Calculate ground state density

Options:
  -a, --atomfile TEXT     Atomic structure  [default: start.json]
  --gpwfilename TEXT      filename.gpw  [default: gs.gpw]
  --ecut FLOAT            Plane-wave cutoff  [default: 800.0]
  -k, --kptdensity FLOAT  K-point density  [default: 6.0]
  --xc TEXT               XC-functional  [default: PBE]
  --help                  Show this message and exit.
```

Submit a recipe to a computer-cluster
-------------------------------------
It is also recommended to use these recipes together with the `myqueue`
job managing package. We assume that you have installed the `myqueue`-package
and are familiar with its usage. If you are not, then take a look at its excellent
documentation. To submit a job that relaxes a structure simply do

```console
$ mq submit asr.relax@24:10h
```

Developing
==========
To see the current status of all recipes write
```console
$ python3 -m asr
```

Skeleton of recipes
-------------------
A recipe contains some specific functionality implemented in separate functions:

```python
from asr.utils import update_defaults
import click

@click.command()
@update_defaults('asr.scriptname')  # Name in params.json
@click.option('-a1', '--arg1', default=1.0, help='Help for arg1')
def main(arg1):
    """Main functionality"""
    pass

def collect_data(kvp, data, atoms, key_descriptions):
    """Collect data to ASE database"""
    pass

def webpanel(row, key_descriptions):
    """Construct web panel for ASE database"""
    pass

dependencies = ['asr.otherscript']
resources = '8:10h'

if __name__ == '__main__':
    main()

```

In all recipes the `main()` function implements the main functionality of
the recipe. The `collect_data()` tells another recipe (`asr.collect`) how
pick up data and put it into a database.


Types of recipes
----------------
The recipes are divided into two groups:

- Property recipes: Recipes that calculate a property for a given materials.
  These scripts should only assume the existence of files in the same folder.
  For example: The ground state recipe gs.py should only require an existence
  of a starting atomic structure, in our case this is called `start.json`

- Structure recipes: These are recipes that produce a new atomic structure.
  When these scripts are run they produce a new folder containing a `start.json`
  such that all property-recipes can be evaluated for the new structure in
  the new folder. For example: The relax recipe which relaxes the atomic
  structure produces new folders "nm/" "fm/" and "afm/" if these structures
  are close to the lowest energy structure. Each of these folders contain
  a new `start.json` from which the property recipes can be evaluated.

- Post-processing recipes: Recipes that do no actual calculations and only
  serves to collect and present data.