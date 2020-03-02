Getting started
===============


ASR comes with some built in functions. To see these simply write

.. command-output:: asr

Let's put these functions into use by calculating some properties of 
Silicon. To get an overview of the possible recipes we can use the `list`
command to list the known recipes

.. command-output:: asr list

Let's say we want to relax a structure. We can search for `relax` and only get a
subset of this list

.. command-output:: asr list relax

from which is clear that we will probably want to use the `relax` recipe. To see
more details about this recipe we can use the `help` function:

.. command-output:: asr run "relax -h"

So to relax a structure, we start by creating
a new folder and put an `unrelaxed.json` file into the directory containing
the atomic structure of Silicon.::

  $ mkdir ~/silicon && cd ~/silicon
  $ ase build -x diamond Si unrelaxed.json

We can relax the structure by using the `asr run` command.::

  $ asr run relax

To see what happened we can use the `status` command::

  $ asr status
  asr.relax           Done -> ['results_relax.json']
  asr.gs              Todo
  asr.dos             Todo
  ...

which shows that we have run the relax recipe and that the results have been 
stored to the `results_relax.json` file. In the process of looking for
interesting recipes we also found the `structureinfo` recipe which computes
some information about the atomic structure of the materials. Let's run that::

  $ asr run structureinfo


ASR lets us save all data to a database by running the `collect` recipe. The 
database is saved to a file `database.db`. This database is an ASE database and
can be browsed using the `ase db` module::

  $ asr run collect
  $ ase db database.db
  id|age|user |formula|calculator| energy| fmax|pbc|volume|charge|  mass| smax
   1| 7s|mogje|Si2    |dftd3     |-10.738|0.000|TTT|41.204| 0.000|56.170|0.001
  Rows: 1

We can also browse this database by using the `browser` recipe which starts a
local server and lets you browse the database interactively::

  $ asr run browser

The ASR run command
-------------------
As you have just seen, the `run` command is used to execute run the recipes of ASR.
In most cases the run command is identical to executing the recipes as modules, ie.,
`asr run relax` is equivalent to `python -m asr.relax`. However, another usecase 
encountered frequently enough is to want to run a recipe in multiple directories.

The asr run command enables this with the following syntax::

  $ asr run relax folder1/ folder1/

which makes it easy to run commands in multiple folders. If you want to provide
arguments for the recipe (the relax recipe in this example) you can use::

  $ asr run "relax --ecut 100" folder1/ folder1/

The last option that the run commands provides is to execute other python modules
like `ase`. For example, suppose you have a lot of folders with a `structure.traj`
that you want to convert to `structure.json`. This can be done with the ase command
`ase convert structure.traj structure.json`. `run` can run this script in
many folders for you with::

  $ asr run --shell "ase convert structure.traj structure.json" materials/*/

where the `command` `asr run command` is used to tell ASR that the command you
wish to run is not a recipe.


The setup recipes
-----------------
ASR also includes some special `setup` recipes. These recipes are meant to give
the user some easy tools to setup atomic structures. Here we provide some explanations
of their usage.

* The `setup.magnetize` recipe is useful if you don't know the magnetic configuration
  of the material you are currently investigation. It sets up non-magnetic (nm), magnetic (fm)
  and anti-ferro magnetic (afm, only for exactly two magnetic atoms in the unit cell) 
  configurations of the inital magnetic moments of the structure in new subfolders `nm/` `fm/`
  and `afm`, respectively. For another example of using the magnetize recipe see the 
  "Advanced Example: Make a screening study" section. For more information see 
  `asr help setup.magnetize`
* The `setup.decorate` recipe is useful if you want to create new atomic that are similar
  to an existing atomic structure. The decorate recipe contains a table describing the
  likelyhood of two atoms to be substituted. By default the decorate recipe creates a
  new ASE database with the decorated atomic structure (including itself). For more 
  information see `asr help setup.decorate`.
* The `setup.unpackdatabase` recipe is useful if you have a database of materials that you wish
  to conduct some calculations on. By default, running `asr run setup.unpackdatabase` creates a new
  folder `tree/` in the current directory with all mateirals distributed according to the 
  following folder structure `tree/{stoi}/{spg}/{formula:metal}-{stoi}-{spg}-{wyck}-{uid}` 
  where `stoi` is the stoichiometry, `spg` is the space group number, `wyck` are the alphabetically
  sorted unique Wyckoff positions of the materials, `formula:metal` is the chemical formula 
  sorted after metal atoms first and `uid` is a unique identifier to avoid collisions between
  materials that would otherwise end up in the same folder. For another example of using the 
  unpackdatabase recipe see the "Advanced Example: Make a screening study" section. For more
  information see `asr run "setup.unpackdatabase -h"`.
* The `setup.params` recipe is useful as it makes a `params.json` file containing the default
  parameters of all recipes. This makes it possible to modify the input parameters used by each
  recipe. See the "Change default settings in scripts" section for more information on 
  how this works.
* The `setup.scanparams` recipe is useful if you want to conduct a convergence study
  of a given recipe. As argument it takes a number of different values for the input arguments
  to a recipe and generates a series of folders that contain a `params.json` file with a specific
  combination of those parameters. When you are done with you calculations you can collect
  the data in the folders and plot them in the browser.

Change default settings in scripts
----------------------------------
All material folders can contain a `params.json`-file. This file can
changed to overwrite default settings in scripts. For example:

.. code-block:: json

   {
   "asr.gs": {"gpw": "otherfile.gpw",
              "ecut": 800},
   "asr.relax": {"d3": true}
   }


In this way all default parameters exposed through the CLI of a recipe
can be corrected.

Submit a recipe to a computer-cluster
-------------------------------------
It is also recommended to use these recipes together with the `myqueue`
job managing package. We assume that you have installed the `myqueue`-package
and are familiar with its usage. If you are not, then take a look at its excellent
documentation. To submit a job that relaxes a structure simply do::

  $ mq submit asr.relax@24:10h
