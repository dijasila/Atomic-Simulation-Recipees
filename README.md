# Atomic Simulation Recipes:
**Recipes for Atomic Scale Materials Research**

Collection of a wide range of python recipes (scripts) for common (and not so
common) tasks perfomed in atomic scale materials research that (are supposed to)
just work! These tasks include relaxation of structures, calculating ground
states, calculating band structures, calculating dielectric functions and so on.

## How to use
To use the recipes you need ASE and GPA. First, clone the code into `~/asr`.
Now, if you want to relax an atomic structure then save your structure to
`start.traj` and simply do
```console
$ python -m asr.relax
```
in the same folder.

## Grouping of recipes
These recipes follow the template of the `myrecipes` python package for python
recipes. It is also recommended to use these recipes together with the `myqueue`
job managing package.

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

## To start a calculation 
- Make a new folder. Name doesn't matter. We call such a folder a
  "material folder".
- Make a start.traj file containing the starting atomic structure.
- In this folder you can evaluate all property-recipes and
  structure-recipes. Be aware structure-recipes produce new folders.

## See help for a recipe
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

## Locally run a recipe in a materials folder

Simply do
```console
$ python3 -m asr.relax
```

## Submit a recipe to a computer-cluster
We assumed that you have installed the `myqueue`-package and are familiar
with its usage. If you are not, then take a look at its excellent
documentation. To submit a job that relaxes a structure simply do

```console
$ mq submit asr.relax@24:10h
```

## Change default settings in scripts
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

