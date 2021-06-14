.. _Getting started:

=================
 Getting started
=================

The atomic simulation recipes (ASR) is a Python package that assists
computational scientists with tools for storing calculation results
and related contextual data. To see how this works in practice, we
will in the following be implementing functionality for calculating
the most stable crystal structure of common metals.

Before we begin we have to initialize a data repository. This is where
ASR stores all its data. In practice, it is nothing more than a ".asr"
directory, which can be initialized with


.. code-block:: console

   $ asr init .
   $ ls -a
   .
   ..
   .asr

When running, ASR will search for the data repository in the current
folder or (stepwise) in any of the parent folders and use the first
one found for storage.

Instructions
============

The most basic element of ASR is an Instruction. Instructions are
simply python functions decorated with the :func:`asr.instruction`
decorator.

To see how this works in practice let's look at an example: 

..
   $ cp $ASRHOME/docs/src/tutorials/getting-started.py $ASRLIB/tutorial.py

.. literalinclude:: getting-started.py
   :pyobject: energy

In this example we have made an instruction for calculating the total
energy of a bulk metal in a given crystal structure using the
effective medium theory (EMT) calculator. The :func:`asr.argument`
helps ASR to construct a command-line interface to the
instruction. Here we have used it to tell ASR that the two arguments
of our instruction is to be interpreted as arguments on the command
line (:func:`asr.option` serves the sames purpose but for command line
options in stead).

The instruction is then easily run through the command-line interface

.. code-block:: console

   $ asr run "asr.tutorial:energy Ag fcc"
   In folder: . (1/1)
   Running asr.tutorial:energy(element='Ag', crystal_structure='fcc')


The cache
=========

Whenever an instruction has been run ASR generates a
:py:class:`asr.Record` containing contextual information about the
run, such as the name of the instruction, the parameters and the
result. The Record is stored in the "Cache" which is a kind of
database for storing calculations. The record can be viewed on the
command-line using

.. code-block:: console

   $ asr cache ls
                  name                       parameters    result
   asr.tutorial:energy element=Ag,crystal_structure=fcc -0.000367

An important feature of ASR is that of "caching" results. If we run
the instruction again with the same input parameters ASR will skip the
actual evaluation of the instruction and simply reuse the old
result.

.. code-block:: console

   $ asr run "asr.tutorial:energy Ag fcc"
   In folder: . (1/1)
   asr.tutorial:energy: Found cached record.uid=e84186a08eaf4523bb44d804071aed6c

This is useful in workflows where it is beneficial to not
redo expensive calculation steps when it has already been performed
once.

Continuing the task of calculating the lowest energy crystal structure
we will implement an additional instruction that loops over crystal
structures and finds the most stable one


.. literalinclude:: getting-started.py
   :pyobject: main


We say that the python modules which contains our two
instructions is a "recipe", ie. a collection of collection of
instructions that achieve our goal. The "main"-instruction is the
primary user interface to our recipe and as such it is not necesarry
to state so explicitly when running the main-instruction.

.. code-block:: console

   $ asr run "asr.tutorial Ag"
   In folder: . (1/1)
   Running asr.tutorial:main(element='Ag')
   Running asr.tutorial:energy(element='Ag', crystal_structure='sc')
   asr.tutorial:energy: Found cached record.uid=343d7f48aad3434493b5bc7e6cbdf94c
   Running asr.tutorial:energy(element='Ag', crystal_structure='bcc')
   Running asr.tutorial:energy(element='Ag', crystal_structure='diamond')


We can now check the result using the command-line tool

.. code-block:: console

   $ asr cache ls name=asr.tutorial:main
                name parameters result
   asr.tutorial:main element=Ag    fcc

Notice here we applied the "name=asr.tutorial" selection to select
only the record of relevance. As we can see the EMT calculator
correctly predicts FCC as the most stable crystal structure for
silver.

Let's continue and calculate the most stable crystal structures for
various other metals

.. code-block:: console

   $ asr run "asr.tutorial Al"
   In folder: . (1/1)
   Running asr.tutorial:main(element='Al')
   Running asr.tutorial:energy(element='Al', crystal_structure='sc')
   Running asr.tutorial:energy(element='Al', crystal_structure='fcc')
   Running asr.tutorial:energy(element='Al', crystal_structure='bcc')
   Running asr.tutorial:energy(element='Al', crystal_structure='diamond')
   $ asr run "asr.tutorial Ni"
   In folder: . (1/1)
   Running asr.tutorial:main(element='Ni')
   Running asr.tutorial:energy(element='Ni', crystal_structure='sc')
   Running asr.tutorial:energy(element='Ni', crystal_structure='fcc')
   Running asr.tutorial:energy(element='Ni', crystal_structure='bcc')
   Running asr.tutorial:energy(element='Ni', crystal_structure='diamond')
   $ asr run "asr.tutorial Cu"
   In folder: . (1/1)
   Running asr.tutorial:main(element='Cu')
   Running asr.tutorial:energy(element='Cu', crystal_structure='sc')
   Running asr.tutorial:energy(element='Cu', crystal_structure='fcc')
   Running asr.tutorial:energy(element='Cu', crystal_structure='bcc')
   Running asr.tutorial:energy(element='Cu', crystal_structure='diamond')
   $ asr run "asr.tutorial Pd"
   In folder: . (1/1)
   Running asr.tutorial:main(element='Pd')
   Running asr.tutorial:energy(element='Pd', crystal_structure='sc')
   Running asr.tutorial:energy(element='Pd', crystal_structure='fcc')
   Running asr.tutorial:energy(element='Pd', crystal_structure='bcc')
   Running asr.tutorial:energy(element='Pd', crystal_structure='diamond')
   $ asr run "asr.tutorial Pt"
   In folder: . (1/1)
   Running asr.tutorial:main(element='Pt')
   Running asr.tutorial:energy(element='Pt', crystal_structure='sc')
   Running asr.tutorial:energy(element='Pt', crystal_structure='fcc')
   Running asr.tutorial:energy(element='Pt', crystal_structure='bcc')
   Running asr.tutorial:energy(element='Pt', crystal_structure='diamond')
   $ asr run "asr.tutorial Au"
   In folder: . (1/1)
   Running asr.tutorial:main(element='Au')
   Running asr.tutorial:energy(element='Au', crystal_structure='sc')
   Running asr.tutorial:energy(element='Au', crystal_structure='fcc')
   Running asr.tutorial:energy(element='Au', crystal_structure='bcc')
   Running asr.tutorial:energy(element='Au', crystal_structure='diamond')

We can now take a look at the results with

.. code-block:: console

   $ asr cache ls name=asr.tutorial:main
                name parameters result
   asr.tutorial:main element=Ag    fcc
   asr.tutorial:main element=Al    fcc
   asr.tutorial:main element=Ni    fcc
   asr.tutorial:main element=Cu    fcc
   asr.tutorial:main element=Pd    fcc
   asr.tutorial:main element=Pt    fcc
   asr.tutorial:main element=Au    fcc

From which we can see that the EMT calculator predicts the FCC crystal
structure to be the most stable crystal structure for all tested
metals which is true in reality as well.

=====================================
Getting started - part 2 - migrations
=====================================

..
   $ cp $ASRHOME/docs/src/tutorials/getting-started-ver2.py $ASRLIB/tutorial.py

It often happens that you want/have to make changes to an existing instruction.
For example, you want to add an additional argument, change the return type of
the result, change the implementation which requires thinking about what should
happen to existing Records in the cache. This is what "migrations" are for.

ASR implements a revisioning system for Records for handling this
problem which revolves around defining functions for updating existing
records to be compatible with the newest implementation of the
instructions.

In the following we will continue with the example of calculating the
most stable crystal structure. It would be interesting to compare some
of ASE's other calculators. However, at the moment, the `energy`
instruction is hardcoded to use the EMT calculator and we will have to
update the instruction to supply the calculator as an input argument

.. literalinclude:: getting-started-ver2.py
   :pyobject: energy

To update the existing records to be consistent with the new
implementation we make a migration that adds the `calculator`
parameter to the existing records and sets it to `emt`

.. literalinclude:: getting-started-ver2.py
   :pyobject: add_missing_calculator_parameter

As is appararent, a migration is nothing more than a regular python
function decorated with the :py:func:`asr.migration` decorator. The
decorator takes a `selector` argument which is used to select records
to be migrated. The selector will be applied to all existing records
and should return a boolean. The function will then be applied for all
record that fulfill the selector criterion.

The migrations are applied with `asr cache migrate`

.. code-block:: console

   $ asr cache migrate
   There are 35 unapplied migrations, 0 erroneous migrations and 0 records are up to date.
   
   Run
       $ asr cache migrate --apply
   to apply migrations.
