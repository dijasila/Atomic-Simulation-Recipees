Getting started: The ASR command-line interface
===============================================

ASR comes with a simple command-line interface which can be invoked using

.. doctest::
   :hide:

   >>> import asr
   >>> from asr.core.cli import cli
   >>> cli(args=[], prog_name="asr", standalone_mode=False)
   Usage: asr [OPTIONS] COMMAND [ARGS]...
   <BLANKLINE>
   Options:
     -h, --help  Show this message and exit.
   <BLANKLINE>
   Commands:
     list  List and search for recipes.
     run   Run recipe, python function or shell command in multiple folders.
   ...

.. code-block:: console

   $ asr
   Usage: asr [OPTIONS] COMMAND [ARGS]...

   Options:
     -h, --help  Show this message and exit.

   Commands:
     list    List and search for recipes.
     run     Run recipe, python function or shell command in multiple folders.

From this output it is clear that the ``asr`` command has two
sub-commands: ``list`` and ``run``. The ``list`` subcommand can be
used to show a list of all known recipes. To show the help for the ``list``
sub-command do

.. doctest::
   :hide:

   >>> from asr.core.cli import cli
   >>> cli(args=['list', '-h'], prog_name="asr", standalone_mode=False)
   Usage: asr list [OPTIONS] [SEARCH]
   <BLANKLINE>
     List and search for recipes.
   <BLANKLINE>
     If SEARCH is specified: list only recipes containing SEARCH in their
     description.
   <BLANKLINE>
   Options:
     -h, --help  Show this message and exit.
   ...

.. code-block:: console

   $ asr list -h
   Usage: asr list [OPTIONS] [SEARCH]

     List and search for recipes.

     If SEARCH is specified: list only recipes containing SEARCH in their
     description.

   Options:
     -h, --help  Show this message and exit.

So we can see a list of all recipes using

.. doctest:: console
   :hide:

   >>> from asr.core.cli import cli
   >>> cli(args=['list'], prog_name="asr", standalone_mode=False)
   Name ... Description ...
   ...
   relax ... Relax atomic positions and unit cell...
   ...


.. code-block:: console

   $ asr list
   Name                           Description
   ----                           -----------
   ...
   relax                          Relax atomic positions and unit cell.
   ...


To run a recipe we use the ``run`` sub-command

.. doctest::
   :hide:

   >>> from asr.core.cli import cli
   >>> cli(args=['run', '-h'], prog_name="asr", standalone_mode=False)
   Usage: asr run [OPTIONS] COMMAND [FOLDERS]...
   <BLANKLINE>
     Run recipe, python function or shell command in multiple folders.
   ...

For example to run the above ``relax`` recipe we would do

.. code-block:: console

   $ asr run relax

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
