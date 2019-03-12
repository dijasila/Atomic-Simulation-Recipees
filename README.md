Recipes for Materials Research
==============================

Collection of recipes for materials research. These recipes follow
the template of the `myrecipes` python package for python recipes.
It is also recommended to use these recipes together with the
`myqueue` job managing package.

To use the recipes you need GPAW and ASE. The recipes are divided into two groups:

- Property recipes: Recipes that calculate a property for a given materials.
  These scripts should only assume the existence of files in the same folder.
  For example: The ground state recipe gs.py should only require an existence
  of a starting atomic structure, in our case this is called start.traj

- Structure recipes: These are recipes that produce a new atomic structure.
  When these scripts are run they produce a new folder containing a start.traj
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

Change default settings in scripts
-------------------------------------
All material folders can contain a `params.json`-file. This file can
changed to overwrite default settings in scripts. For example:

```javascript
{
    "gs.py": {"gpw": "otherfile.gpw"},
    "relax.py": {"states": "nm",
                 "ecut": 800}
}
```

See the command line interface for a recipe
-------------------------------------------
We assume that you have cloned the project into `~/rmr/` and have added
this folder to your `PYTHONPATH`. To see the command line interface (CLI)
of the relax recipe we simply do

```console
[mogje@dirac MoS2-MoS2]$ python3 -m rmr.relax -h
usage: relax.py [-h] [-U] [--states STATES [STATES ...]]

Relax atomic structure

optional arguments:
  -h, --help            show this help message and exit
  -U, --plusu           Do +U calculation
  --states STATES [STATES ...]
                        list of nm, fm, afm
```