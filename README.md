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
* pytest

Additionally, but not a requirement, it can be nice to have
* myqueue

Installation
------------

```console
$ cd ~ && git clone https://gitlab.com/mortengjerding/asr.git
$ echo  'export PYTHONPATH=~/asr:$PYTHONPATH' >> ~/.bashrc
$ python3 -m pip install -e ~/asr
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
recipe which creates a `quickinfo.json` file that contains some simple
information about the atomic structure. Then we collect the data to a database
`database.db` using `asr.collect` recipe and view it
in a browser with the `browser` subcommand. This is done below

```console
$ cd ~/silicon/nm
$ python3 -m asr.quickinfo
$ python3 -m asr.collect
$ python3 -m asr browser
```

Notice the space in the last command between `asr` and `browser`.
`browser` is a subcommand of `asr` and not(!) a recipe. To see the available
subcommands of ASR simply do
`
python3 -m asr
`

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
For example, to see the command line interface (CLI) help of the relax recipe
simply do

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
In the following you will find the necessary information needed to implement new
recipes into the ASR framework. The first section gives an ultra short
description of how to implement new recipes, and the following sections go
into more detail.

Guide to making new recipes for geniuses
----------------------------------------

- Start by copying the template [template_recipe.py](asr/utils/something.py) 
  into your asr/asr/ directory. The filename of this file is important since
  this is the name that is used when executing the script. We assume that you
  script is called `something.py`.
- Implement your main functionality into the `main` function. This is the 
  function that is called when executing the script directly. Please save your
  results into a .json file if possible. In the template we save the results to
  `something.json`.
- Implement the `collect_data` function which ASR uses to collect the data (in
  this case it collects the data in `something.json`). It is important that this
  function returns a dict of key-value pairs `kvp` alongside their
  `key-descriptions` and any data you want to save in the collected database.
- Now implement the `web_panel` function which tells ASR how to present the data
  on the website. This function returns a `panel` and a `things` list. The panel
  is simply a tuple of the title that goes into the panel title and a list of
  columns and their contents. This should be clear from studying the example.
- Finally, implement the additional metadata keys `group` (see below for 
  possible groups), `creates` which tells ASR what files are created and
  `dependencies` which should be a list of ASR recipes (e. g. ['asr.gs']).


When you have implemented your first draft of your recipe make sure to test it.
See the section below for more information.


Testing
-------
When you make a new recipe it will be automatically added to a test that runs a
full workflow for Silicon, Iron, 2D h-BN, 2D-VS2. However, if you want more
extended testing of your recipe you will have to implement them manually. The
tests can be found in `asr/asr/tests/` where you will find folders containing
the specific materials. To run all tests execute
```
python3 -m asr test
```
ASR uses the pytest module for its tests.


Special recipe metadata keywords
--------------------------------
A recipe contains some specific functionality implemented in separate functions:
[template_recipe.py](asr/utils/something.py). Below you will find a description
of each special keyword in the recipe.

- `main()` Implement the main functionality of the script. This is where the heavy
  duty stuff goes.
- `collect_data()` tells another recipe (`asr.collect`) how pick up data and put
  it into a database.
- `webpanel()` tells ASR how to present the data on a webpage.
- `group` See "Types of recipes" section below.
- `creates` is a list of filenames created by `main()`. The files in this list 
  should be the files that contain the essential data that would be needed
  later.
- `resources` is a `myqueue` specific keyword which is a string in the specific
  format `ncores:timelimit` e. g. `1:10m`. These are the resources that myqueue
  uses when submitting the jobs to your cluster. This can also be a `callable`
  in the future but this functionality is not currently well tested.
- `diskspace` is a `myqueue` specific keyword which is a number in arbitrary 
  units that can be
  parsed by myqueue to make sure that not too many diskspace intensive jobs are
  running simultaneously.
- `restart` is a `myqueue` specific keyword which is an integer that tells
  myqueue whether it makes sense to restart the job if it timeout or had a
  memory error and how many times it makes sense to try. If it doesn't make
  sense then set this number to 0.

Types of recipes
----------------
The recipes are divided into the following groups:

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