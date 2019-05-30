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

ASE has to be installed manually since we need the newest version:
```console
$ python3 -m pip install git+https://gitlab.com/ase/ase.git
```
Install a database of reference energies to calculate HOF and convex hull. Here 
we use a database of one- and component-structures from OQMD
```
$ cd ~ && wget https://cmr.fysik.dtu.dk/_downloads/oqmd12.db
$ echo 'export ASR_REFERENCES=~/oqmd12.db' >> ~/.bashrc
```

We do relaxations with the D3 van-der-Waals contribution. To install the van 
der Waals functional DFTD3 do
```console
$ mkdir ~/DFTD3 && cd ~/DFTD3
$ wget http://chemie.uni-bonn.de/pctc/mulliken-center/software/dft-d3/dftd3.tgz
$ tar -zxf dftd3.tgz
$ make
$ echo 'export ASE_DFTD3_COMMAND=$HOME/DFTD3/dftd3' >> ~/.bashrc
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

How to use
----------
ASR comes with some built in functions. To see these simply write
```console
$ asr
Usage: asr [OPTIONS] COMMAND [ARGS]...

Options:
  -h, --help  Show this message and exit.

Commands:
  help      See help for recipe.
  list      Show a list of all recipes.
  run       Run recipe or ASE command.
  status    Show the status of the current folder for all ASR recipes.
  test      Run test of recipes.
  workflow  Helper function to make workflows for MyQueue.
```

Let's put these functions into use by calculating some properties of 
Silicon. To get an overview of the possible recipes we can use the `list`
command to list the known recipes:
```console
$ asr list
Recipe                Description
------                -----------
collect               Collect data in ase database.
structureinfo         Get quick information about structure based...
browser               Open results in web browser.
relax                 Relax atomic positions and unit cell.
setup.decorate        Decorate structure with different atoms.
setup.params          Make a new params file.
setup.unpackdatabase  Set up folders with atomic structures based...
setup.magnetize       Set up magnetic moments of atomic structure.
gs                    Calculate ground state density.
dos                   Calculate DOS.
polarizability        Calculate linear response polarizability or...
convex_hull
phonons               Calculate Phonons.
anisotropy
bader                 Calculate bader charges.
borncharges           Calculate Born charges.
bandstructure         Calculate electronic band structure.
push                  Push structure along some phonon mode and...
```

Let's say we want to relax a structure. We can search for `relax` and only get a
subset of this list:
```console
$ asr list relax
Recipe                Description
------                -----------
relax                 Relax atomic positions and unit cell.
setup.unpackdatabase  Set up folders with atomic structures based...
setup.magnetize       Set up magnetic moments of atomic structure.
push                  Push structure along some phonon mode and...
```

from which is clear that we will probably want to use the `relax` recipe. To see
more details about this recipe we can use the `help` function:
```console
$ asr help relax
Usage: asr run relax [OPTIONS]

  Relax atomic positions and unit cell.

  By default, this recipe takes the atomic structure in 'unrelaxed.json' and
  relaxes the structure including the DFTD3 van der Waals correction. The
  relaxed structure is saved to `structure.json` which can be processed by
  other recipes.

Options:
  --skip-deps / --run-deps  Skip execution of dependencies?  [default: False]
  --ecut INTEGER            Energy cutoff in electronic structure calculation
                            [default: 800]
  --kptdensity FLOAT        Kpoint density  [default: 6.0]
  -U, --plusu               Do +U calculation  [default: False]
  --xc TEXT                 XC-functional  [default: PBE]
  --d3 / --nod3             Relax with vdW D3  [default: True]
  --width FLOAT             Fermi-Dirac smearing temperature  [default: 0.05]
  --help                    Show this message and exit.
```

So to relax a structure, we start by creating
a new folder and put an `unrelaxed.json` file into the directory containing
the atomic structure of Silicon.
```console
$ mkdir ~/silicon && cd ~/silicon
$ ase build -x diamond Si unrelaxed.json
```
We can relax the structure by using the `asr run` command.
```console
$ asr run relax
```

To see what happened we can use the `status` command
```console
$ asr status
asr.relax           Done -> ['results_relax.json']
asr.gs              Todo
asr.dos             Todo
...
```
which shows that we have run the relax recipe and that the results have been 
stored to the `results_relax.json` file. In the process of looking for
interesting recipes we also found the `structureinfo` recipe which computes
some information about the atomic structure of the materials. Let's run that:
```console
$ asr run structureinfo
```


ASR lets us save all data to a database by running the `collect` recipe. The 
database is saved to a file `database.db`. This database is an ASE database and
can be browsed using the `ase db` module:
```console
$ asr run collect
$ ase db database.db
id|age|user |formula|calculator| energy| fmax|pbc|volume|charge|  mass| smax
 1| 7s|mogje|Si2    |dftd3     |-10.738|0.000|TTT|41.204| 0.000|56.170|0.001
Rows: 1
```
We can also browse this database by using the `browser` recipe which starts a
local server and lets you browse the database interactively:
```console
$ asr run browser
```

The ASR run command
-------------------
As you have just seen, the `run` command is used to execute run the recipes of ASR.
In most cases the run command is identical to executing the recipes as modules, ie.,
`asr run relax` is equivalent to `python -m asr.relax`. However, another usecase 
encountered frequently enough is to want to run a recipe in multiple directories.

The asr run command enables this with the following syntax:
```console
$ asr run relax in folder1/ folder1/
```
which makes it easy to run commands in multiple folders. If you want to provide
arguments for the recipe (the relax recipe in this example) you can use
```console
$ asr run relax --ecut 100 in folder1/ folder1/
```
The last option that the run commands provides is to execute other python modules
like `ase`. For example, suppose you have a lot of folders with a `structure.traj`
that you want to convert to `structure.json`. This can be done with the ase command
`python -m ase convert structure.traj structure.json`. `run` can run this script in
many folders for you with
```console
$ asr run "ase convert structure.traj structure.json" in materials/*/
```


The setup recipes
-----------------
ASR also includes some special `setup` recipes. These recipes are meant to give
the user some easy tools to setup atomic structures. Here we provice some examples
of their usage.

The `setup.magnetize` recipe is useful if you don't know the magnetic configuration
of the material you are currently investigation. It sets up non-magnetic (nm), magnetic (fm)
and anti-ferro magnetic (afm, only for exactly two magnetic atoms in the unit cell) 
configurations of the inital magnetic moments of the structure in new subfolders `nm/` `fm/`
and `afm`, respectively. For another example of using the magnetize recipe see the 
"Advanced Example: Make a screening study" section. For more information see `asr help setup.magnetize`

The `setup.decorate` recipe is useful if you want to create new atomic that are similar
to an existing atomic structure. The decorate recipe contains a table describing the
likelyhood of two atoms to be substituted. By default the decorate recipe creates a
new ASE database with the decorated atomic structure (including itself). For more information see `asr help setup.decorate`.

The `setup.unpackdatabase` is useful if you have a database of materials that you wish
to conduct some calculations on. By default, running `asr run setup.unpackdatabase` creates a new
folder `tree/` in the current directory with all mateirals distributed according to the 
following folder structure `tree/{stoi}/{spg}/{formula:metal}-{stoi}-{spg}-{wyck}-{uid}` 
where `stoi` is the stoichiometry, `spg` is the space group number, `wyck` are the alphabetically
sorted unique Wyckoff positions of the materials, `formula:metal` is the chemical formula 
sorted after metal atoms first and `uid` is a unique identifier to avoid collisions between
materials that would otherwise end up in the same folder. For another example of using the 
unpackdatabase recipe see the "Advanced Example: Make a screening study" section. For more
information see `asr help setup.unpackdatabase`.


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

Submit a recipe to a computer-cluster
-------------------------------------
It is also recommended to use these recipes together with the `myqueue`
job managing package. We assume that you have installed the `myqueue`-package
and are familiar with its usage. If you are not, then take a look at its excellent
documentation. To submit a job that relaxes a structure simply do

```console
$ mq submit asr.relax@24:10h
```

Advanced Example: Make a screening study
----------------------------------------
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
$ asr run setup.unpackdatabase oqmd12.db -s u=False --run
```
(we have made the selection `u=False` since we are not interested in the DFT+U values).
This function produces a new folder `~oqmd12/tree/` where you can find the tree. 
To see the contents of the tree it is recommended to use the linux command `tree`
```console
$ tree tree/
```
You will see that the unpacking of the database has produced many `unrelaxed.json`
files that contain the unrelaxed atomic structures. Because we don't know the
magnetic structure of the materials we also want to sample different magnetic structures.
This can be done with the `magnetize` function of asr
```console
$ asr run setup.magnetize in */*/*/*/
```
We use the `run` function because that gives us the option to deal with many folders
at once. You are now ready to run a
workflow on the entire tree. ASR has a `workflow` function that let's us build
workflows based on the recipes in ASR. This function is meant as an help to
start on new recipes. Familiarize yourself with the function by
running `asr workflow -h`. The help shows that it is possible to create a
workflow by
```console
$ asr workflow -t asr.relax,asr.bandstructure,asr.convex_hull --doforstable asr.bandstructure > workflow.py
$ cat worflow.py
from myqueue.task import task


def is_stable():
    # Example of function that looks at the heat of formation
    # and returns True if the material is stable
    from asr.utils import read_json
    from pathlib import Path
    fname = 'results_convex_hull.json'
    if not Path(fname).is_file():
        return False

    data = read_json(fname)
    if data['hform'] < 0.05:
        return True
    return False


def create_tasks():
    tasks = []
    tasks += [task('asr.relax@8:xeon8:10h')]
    tasks += [task('asr.structureinfo@1:10m')]
    tasks += [task('asr.gs@8:10h', deps='asr.structureinfo')]
    tasks += [task('asr.gaps@1:10m', deps='asr.structureinfo,asr.gs')]
    tasks += [task('asr.convex_hull@1:10m', deps='asr.structureinfo,asr.gs')]
    if is_stable():
        tasks += [task('asr.bandstructure@1:10m', deps='asr.structureinfo,asr.gaps,asr.gs')]

    return tasks
```

This workflow relaxes the structures and if the `convex_hull` recipe calculates
low heat of formation the workflow will make sure that the bandstructure is
calculated. We now ask `myqueue` what jobs it wants to run.
```console
$ mq workflow -z workflow.py tree/*/*/*/*/
```
To submit the jobs simply remove the `-z`, and run the command again.

For complex workflows, like the one above where we have to check the stability of
materials which has to wait until the `convex_hull` recipe has finished, the 
`mq workflow` function should have to be run periodically to check for new tasks.
In this case it is smart to set up a crontab to do the work for you. 
To do this write
```console
$ crontab -e
```
choose your editor and put the line 
`*/5 * * * * . ~/.bashrc; cd ~/oqmd12; mq kick; mq workflow workflow.py tree/*/*/*/*/`
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
asr test
```
When you make a new recipe ASR will automatically generate a test thats tests
its dependencies and itself to make sure that all dependencies have been
included. These automatically generated tests are generated from
[test_template.py](asr/tests/template.py).

ASR uses the `pytest` module for its tests. To see what tests will run use
```
asr test --collect-only
```
To execute a single test use 
```
asr test -k my_test.py
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


