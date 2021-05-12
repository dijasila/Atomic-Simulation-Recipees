.. _Getting started:

=================
 Getting started
=================

The atomic simulation recipes (ASR) is a Python package that assists
computational scientists by implementing tools for storing calculation
results and related contextual data. To see how this works in
practice, we will in the following be implementing functionality for
calculating most stable crystal structure of common metals.


Before we begin we have initialize a data repository. This is where
ASR stores all its data. In practice it is nothing more than a ".asr"
directory, which can be initialized with

.. code-block::

   $ asr init .
   $ ls -a
   .
   ..
   .asr


Instructions
============

The most basic element of ASR is an Instruction. Instructions are
simply python functions decorated with the :func:`asr.instruction`
decorator.

To see how this works in practice let's look at an example of 

.. literalinclude:: getting-started.py

In this example we have made an instruction for calculating the total
energy of some input atomic structure using the effective medium
theory (EMT) calculator. The :func:`asr.atomsopt` option decorator
informs ASR that the instruction takes an atomic structure as an input
named "atoms" and sets up a command line interface for this argument
(this decorator is a special case of the more general
:func:`asr.option` decorator). We will get back to this concept later
in the tutorial.

To run the instruction we require an atomic structure which can be
generated with ASE's command-line interface

.. code-block::

   $ ase build Ag structure.json -x fcc

The instruction is then easily run through the command-line interface

.. code-block::

   $ asr run "asr.tutorial:energy --atoms structure.json"
   In folder: . (1/1)
   Running asr.tutorial:energy(atoms=Atoms(symbols='Ag', pbc=True, cell=[[0.0, 2.045, 2.045], [2.045, 0.0, 2.045], [2.045, 2.045, 0.0]]))


The cache
=========

Whenever an instruction has been run, the result will be stored in the
"Cache" which is a kind of database for storing calculations. The
result can be viewed on the command-line using

.. code-block:: console

   $ asr cache ls
                  name                                  parameters                result
   asr.tutorial:energy {'atoms': Atoms(symbols='Ag', pbc=True, ... 0.0015843277481568663

When calling an instruction the decorator
takes care of storing the input parameters as well as the other
contextual data (see :py:class:`asr.Record` for information on exactly
what contextual data is stored).

These instructions are grouped into "recipes" which are
python modules with one or more instructions and a single "main"
instruction he main instruction is the main entrypoint for the user.
