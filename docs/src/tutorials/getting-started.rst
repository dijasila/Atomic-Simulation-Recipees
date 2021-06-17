.. _Getting started:

=================
 Getting started
=================

The atomic simulation recipes (ASR) is a Python package that assists
computational scientists with tools for storing calculation results
and related contextual data. To see how this works in practice, we
will in the following be implementing functionality for calculating
the most stable crystal structure of common metals.

Before we begin we have initialize a data repository. This is where
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
