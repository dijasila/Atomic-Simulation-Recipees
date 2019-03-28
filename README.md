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

Additionally, but not a requirement, it can be nice to have
* myqueue

Installation
------------

```console
$ cd ~ && git clone https://gitlab.com/mortengjerding/asr.git pythonmodules/asr
$ echo  'export PYTHONPATH=~/pythonmodules:$PYTHONPATH' >> ~/.bashrc 
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

Skeleton of recipes
-------------------
A recipe contains some specific functionality

```python
import click

@click.command()
@click.option('-a1', '--arg1', default=1.0, help='Help for arg1')
def main(arg1):
    pass

def collect_data(kvp, data, atoms, key_descriptions):
    pass

def webpanel(row, key_descriptions):
    pass


if __name__ == '__main__':
    main()

```

Developing
----------
To see the current status of all recipes write
```console
$ python3 -m asr
```

Types of recipes
----------------
The recipes are divided into two groups:

- Property recipes: Recipes that calculate a property for a given materials.
  These scripts should only assume the existence of files in the same folder.
  For example: The ground state recipe gs.py should only require an existence
  of a starting atomic structure, in our case this is called `start.traj`

- Structure recipes: These are recipes that produce a new atomic structure.
  When these scripts are run they produce a new folder containing a `start.traj`
  such that all property-recipes can be evaluated for the new structure in
  the new folder. For example: The relax recipe which relaxes the atomic
  structure produces new folders "nm/" "fm/" and "afm/" if these structures
  are close to the lowest energy structure. Each of these folders contain
  a new `start.traj` from which the property recipes can be evaluated.

To start a calculation
----------------------
- Make a new folder. Name doesn't matter. We call such a folder a
  "material folder".
- Make a start.traj file containing the starting atomic structure.
- In this folder you can evaluate all property-recipes and
  structure-recipes. Be aware structure-recipes produce new folders.

See help for a recipe
---------------------
We assume that you have cloned the project into `~/asr/` and have added
this folder to your `PYTHONPATH`. To see the command line interface (CLI)
help of the relax recipe we simply do

```console
$ python3 -m asr.gs -h
usage: gs.py [-h] [-a ATOMS] [-g GPW] [-e ECUT] [-k KPTDENSITY] [--xc XC]

optional arguments:
  -h, --help            show this help message and exit
  -a ATOMS, --atoms ATOMS
                        Atomic structure (default: start.traj)
  -g GPW, --gpw GPW     Name of ground state file (default: gs.gpw)
  -e ECUT, --ecut ECUT  Plane-wave cutoff (default: 800)
  -k KPTDENSITY, --kptdensity KPTDENSITY
                        K-point density (default: 6.0)
  --xc XC               XC-functional (default: PBE)
```

Locally run a recipe in a materials folder
------------------------------------------

Simply do
```console
$ python3 -m asr.relax
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

