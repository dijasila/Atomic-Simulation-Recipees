Atomic Simulation Recipes
=========================
Recipes for Atomic Scale Materials Research.

Collection of python recipes for common (and not so common)
tasks perfomed in atomic scale materials research. These tasks include
relaxation of structures, calculating ground states, calculating band
structures, calculating dielectric functions and so on.

Installation
------------
To install ASR first clone the code and pip-install the code
```console
$ cd ~ && git clone https://gitlab.com/mortengjerding/asr.git
$ python3 -m pip install -e ~/asr
```

XXX You need a brand new ase version for this code to work!

We do relaxations with the D3 van-der-Waals contribution. To install the van 
der Waals functional DFTD3 do
```console
$ cd
$ mkdir functional
$ cd functional
$ mkdir PBED3
$ wget http://chemie.uni-bonn.de/pctc/mulliken-center/software/dft-d3/dftd3.tgz
$ tar -zxf dftd3.tgz
$ make
$ echo 'export ASE_DFTD3_COMMAND=$HOME/functional/PBED3/dftd3' >> ~/.bashrc
$ source ~/.bashrc
```

To make Bader analysis we use another program. Download the executable for Bader 
analysis and put in path (this is for Linux, find the appropriate executable):
```console
$ cd ~ && mkdir baderext && cd baderext
$ wget http://theory.cm.utexas.edu/henkelman/code/bader/download/bader_lnx_64.tar.gz
$ tar -zxf bader_lnx_64.tar.gz
$ echo  'export PATH=~/baderext:$PATH' >> ~/.bashrc
```

Additionally, you might also want
* myqueue

if you want to run jobs on a computer-cluster.

Requirements
------------
When you have done this installation you with have the following pythhon
packages
* ASE (Atomic Simulation Environment)
* GPAW
* click
* spglib
* pytest
* plotly

Packages you need to compile yourself:
* bader (see instructions above)
* DFTD3 functional (see instructions aobove)

Additionally, but not a requirement, it can be nice to have
* myqueue


How to use
----------
Lets calculate the properties of Silicon. To do this, we start by creating
a new folder and put a 'structure.json' file into the directory containing
the atomic structure of Silicon. Then we relax the structure.
```console
$ mkdir ~/silicon && cd ~/silicon
$ ase build -x diamond Si structure.json
$ python3 -m asr.relax
```

This generates a new folder `~/silicon/nm/` containing a new `structure.json`
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
`browser` is a subcommand of `asr` and not a recipe. To see the available
subcommands of ASR simply do
```console
$ python3 -m asr
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
For example, to see the command line interface (CLI) help of the relax recipe
simply do

```console
$ python3 -m asr.gs --help
Usage: gs.py [OPTIONS]

  Calculate ground state density

Options:
  -a, --atomfile TEXT     Atomic structure  [default: structure.json]
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

Make a screening study
----------------------
A screening study what we call a simultaneous automatic study of many materials. ASR
has a set of tools to make such studies easy to handle. Suppose we have an ASE
database that contain many atomic structures. In this case we take OQMD12 database
that contain all unary and binary compounds on the convex hull.

The first thing we do is to get the database:
```console
$ mkdir ~/oqmd12 && cd ~/oqmd12
$ wget https://cmr.fysik.dtu.dk/_downloads/oqmd12.db
```
We then use the `unpackdatabase` function of ASR to unpack the database into a
directory tree
```console
$ python3 -m asr.setup.unpackdatabase oqmd12.db -s u=False --run
```
(we have made the selection `u=False` since we are not interested in the DFT+U values).
This function produces a new folder `~oqmd12/tree/` where you can find the tree. To see the contents of the tree
it is recommended to use the linux command `tree`
```console
$ tree tree/
```
You will see that the unpacking of the database has produced many `unrelaxed.json`
files that contain the unrelaxed atomic structures. Because we dont know the
magnetic structure of the materials we also want to sample different magnetic structures.
This can be done with the `magnetize` function of asr
```console
$ python3 -m asr run asr.setup.magnetize */*/*/*/
```
We use the `run` function because that gives us the option to deal with many folders
at once. You are now ready to run a
workflow on the entire tree. A simple workflow would be to relax all structures:
```console
$ cat workflow.py
from myqueue.task import task


def create_tasks():
    tasks = [task('asr.relax@8:1d'),
             task('asr.gs@8:1h', deps='asr.relax')]
    return tasks
```
(copy this and save it to `workflow.py`). We now ask `myqueue` what jobs 
it wants to run.
```console
$ mq workflow -z workflow.py tree/*/*/*/*/
```
To submit the jobs simply remove the `-z`, and run the command again.

For more complex workflows the `mq workflow` function would have to be run 
periodically to check for new jobs. In this case it is smart to set up a crontab
to do the work for you. To do this write
```console
$ crontab -e
```
choose your editor and put the line 
`*/5 * * * * . ~/.bashrc; cd ~/oqmd12; mq kick; mq workflow -z workflow.py tree/*/*/*/*/`
into the file. This will restart any timeout jobs and run the workflow command 
to see if any new tasks should be spawned with a 5 minute interval. 


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
Tests can be run using
```
python3 -m asr test
```
When you make a new recipe ASR will automatically generate a test thats tests
its dependencies and itself to make sure that all dependencies have been
included. These automatically generated tests are generated from
[test_template.py](asr/tests/template.py).

ASR uses the `pytest` module for its tests. To see what tests will run use
```
python3 -m asr test --collect-only
```
To execute a single test use 
```
python3 -m asr test -k my_test.py
```
If you want more extended testing of your recipe you will have to implement them
manually. Your test should be placed in the `asr/asr/tests/`-folder where other
tests are located as well. where you will find folders containing
the specific materials.


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

- Property recipes: Recipes that calculate a property for a given atomic structure.
  The scripts should use the file in the current folder called `structure.json`.
  These scripts should only assume the existence of files in the same folder.
  Example recipes: `asr.gs`, `asr.bandstructure`, `asr.structureinfo`.

- Structure recipes: These are recipes that can produce a new atomic structure in
  this folder.
  Example: `asr.relax` takes the atomic structure in `unrelaxed.json`
  in the current folder and produces a relaxed structure in `structure.json` 
  that the property recipes can use.

- Setup recipes: These recipes are located in the asr.setup folder and the 
  purpose of these recipes is to set up new atomic structures in new folders.
  Example: `asr.setup.magnetize`, `asr.push`, `asr.setup.unpackdatabase` all
  takes some sort of input and produces folders with new atomic structures that 
  can be relaxed.


