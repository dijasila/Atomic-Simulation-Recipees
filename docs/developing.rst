Developing new recipes
======================
In the following you will find the necessary information needed to implement new
recipes into the ASR framework. The first section gives an ultra short
description of how to implement new recipes, and the following sections go
into more detail.

Guide to making new recipes
---------------------------

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
- Now implement the `webpanel` function which tells ASR how to present the data
  on the website. This function returns a `panel` and a `things` list. The panel
  is simply a tuple of the title that goes into the panel title and a list of
  columns and their contents. This should be clear from studying the example.
- Finally, implement the additional metadata keys `group` (see below for 
  possible groups), `creates` which tells ASR what files are created and
  `dependencies` which should be a list of ASR recipes (e. g. ['asr.gs']).

When you have implemented your first draft of your recipe make sure to test it.
See the section below for more information.

The ASRCommand decorator
------------------------
As you will have seen in the template recipe [template_recipe.py](asr/utils/something.py)
all main functions in ASR are decorated using a special `command` decorator that ASR 
provides. In practice, the `command` decorator basically instantiates a new class
`ASRCommand()` which essentially is a `click.Command` with extra attributes.

When wrapping a command in this decorator the function will automatically:

* Make sure that the returned results of the function
  are safely stored to a file named `results_RECIPENAME.json`, where `RECIPENAME` is the
  filename of the recipe like `relax`, `gs` or `borncharges`. This means that you should
  make sure to return your results using the `return results` statement in python.
* The results file with additionally also contain the version number of ASR, ASE and GPAW.
  This is nice since they will be stored together with the actual results.
* Update the command defaults based on the `params.json` file in the current directory.
* Execute any missing dependencies if they haven't been run yet.
  Additionally, the `ASRCommand` will also add a `skip-deps` flag argument if you for some
  reason don't care about running the dependencies.
* Store to the the results file in the data key of ASR under the key
  `data['results_RECIPENAME']`.

The `command` decorator also support the `known_exceptions` keyword.
This keyword is a dictionary of errors (in python lingo: an exception) and some
multiplication factors that the parameters should be updated with if the programs
fails with that excact error. For example, take a look at how this is used in the
`relax` recipe below:
.. code-block:: python

   known_exceptions = {KohnShamConvergenceError: {'kptdensity': 1.5,
		'width': 0.5}}
   @command('asr.relax',
		known_exceptions=known_exceptions)

In other words, if the relax recipe encounters a `KohnShamConvergenceError` it execute
the recipe again with a 50% larger kpoint density and half the Fermi temperature smearing.

For some `recipes` it is necessary to use the exact some parameters that some other recipes
used. This is specifically true for the ground state recipe which need to use the same 
kpoint density and Fermi Temperature as the relax recipe. The command decorator supports 
the `overwrite_params` keyword which lets you load in other default parameters. In practice,
the ground state recipe reads parameters from the `gs_params.json` file (if it exists) which
is produced by the relax recipe. Below you can see how this works for the ground state recipe:

.. code-block:: python

   # Get some parameters from structure.json
   defaults = {}
   if Path('gs_params.json').exists():
       from asr.utils import read_json
       dct = read_json('gs_params.json')
       if 'ecut' in dct.get('mode', {}):
           defaults['ecut'] = dct['mode']['ecut']

       if 'density' in dct.get('kpts', {}):
           defaults['kptdensity'] = dct['kpts']['density']

   @command('asr.gs', defaults)



Setting a different calculator
==============================
The default DFT calculator of ASR is `GPAW`, however, at the moment some recipes
support using the EMT calculator of ASE, specifically the `relax` and the `gs` recipes.
This is mostly meant for testing, however in the future, we might want to support other
calculators. To change change the calculator simply add the keyword `_calculator: EMT`
in you `params.json` file:
.. code-block:: json

   {
   "_calculator": "EMT"
   }

We use the `_calculator` keyword as opposed to `calculator` because this functionality 
is not meant to be used for anything else than testing at the moment.

	
Testing
-------
Tests can be run using::

  $ asr test

When you make a new recipe ASR will automatically generate a test thats tests
its dependencies and itself to make sure that all dependencies have been
included. These automatically generated tests are generated from
[test_template.py](asr/tests/template.py).

To execute a single test use::

  $ asr test -k my_test.py

If you want more extended testing of your recipe you will have to implement them
manually. Your test should be placed in the `asr/asr/tests/`-folder and prefixed
with `test_` which is how ASR locates the tests.


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
	     


