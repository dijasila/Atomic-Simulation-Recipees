.. _Getting started:

Getting started
===============

The atomic simulation recipes (ASR) is a Python package that
implements functionality for storing results calculated by
"instructions". Instructions are simply python functions decorated
with the :func:`asr.instruction` decorator. These instructions are
grouped into "recipes" which are python modules with one or more
instructions and a single "main" instruction he main instruction is
the main entrypoint for the user.

To see how this works in practice let's look at an example of 


.. contents:: Contents
   :local:


Command-line interface
----------------------

The primary user interface of ASR is the command-line interface. Here 

ASR comes with a simple command-line interface which can be invoked using

.. doctest::
   :hide:

   >>> import asr
   >>> from asr.core.cli import cli
   >>> cli(args=[], prog_name="asr", standalone_mode=False)
   Usage: asr [OPTIONS] COMMAND [ARGS]...
   <BLANKLINE>
   Options:
     --version   Show the version and exit.
     -h, --help  Show this message and exit.
   <BLANKLINE>
   Commands:
     cache     Inspect results.
     database  ASR material project database.
     init      Initialize ASR Repository.
     list      List and search for recipes.
     params    Compile a params.json file with all options and defaults.
     results   Show results from records.
     run       Run recipe or python function in multiple folders.
   ...

.. code-block:: console

   $ asr
   Usage: asr [OPTIONS] COMMAND [ARGS]...

   Options:
     -h, --help  Show this message and exit.

   Commands:
     cache     Inspect results.
     database  ASR material project database.
     list      List and search for recipes.
     params    Compile a params.json file with all options and defaults.
     results   Show results from records.
     run       Run recipe or python function in multiple folders.

From this output it is clear that the ``asr`` command has multiple
sub-commands, but let's highlight a couple: ``list`` and ``run``. The
``list`` subcommand can be used to show a list of all known
recipes. To show the help for the ``list`` sub-command do

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
   Name ... Description...
   ...
   asr.relax ... Relax atomic positions and unit cell...
   ...


.. code-block:: console

   $ asr list
   Name                           Description
   ----                           -----------
   ...
   asr.relax                      Relax atomic positions and unit cell.
   ...


To run a recipe we use the ``run`` sub-command. For example to run the
above ``relax`` recipe we would do

.. doctest::
   :hide:

   >>> from asr.core.cli import cli
   >>> cli(args=['run', '-h'], prog_name="asr", standalone_mode=False)
   Usage: asr run [OPTIONS] COMMAND [FOLDERS]...
   <BLANKLINE>
     Run recipe or python function in multiple folders.
   ...

.. code-block:: console

   $ asr run asr.relax
