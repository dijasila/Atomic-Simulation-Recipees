.. _Getting started:

Getting started
===============

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


To run a recipe we use the ``run`` sub-command. For example to run the
above ``relax`` recipe we would do

.. doctest::
   :hide:

   >>> from asr.core.cli import cli
   >>> cli(args=['run', '-h'], prog_name="asr", standalone_mode=False)
   Usage: asr run [OPTIONS] COMMAND [FOLDERS]...
   <BLANKLINE>
     Run recipe, python function or shell command in multiple folders.
   ...

.. code-block:: console

   $ asr run relax
