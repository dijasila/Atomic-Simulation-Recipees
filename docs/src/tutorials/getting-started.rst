.. _Getting started:

Getting started
===============

The atomic simulation recipes (ASR) is a Python package that assists
the computational scientist by implementing tools for storing results
and related contextual data. As an example to see how this works in
practice, we will in the following be implementing functionality for
calculating most stable crystal structure of common metals.

Instructions
------------

The most basic element of ASR is an Instruction. Instructions are
simply python functions decorated with the :func:`asr.instruction`
decorator. When calling the decoratored instruction the decorator
takes care of storing the input parameters as well as the other
contextual data (see :py:class:`asr.Record` for information on exactly
what contextual data is stored).

These instructions are grouped into "recipes" which are
python modules with one or more instructions and a single "main"
instruction he main instruction is the main entrypoint for the user.

To see how this works in practice let's look at an example of 

.. literalinclude:: example_metal_crystals.py

In this example we have made an instruction for calculating the total
energy of some input atomic structure using the effective medium
theory (EMT) calculator. The :func:`asr.atomsopt` option decorator informs
ASR that the instruction takes an atomic structure as an input named
"atoms" and sets up a command line interface for this argument (this
decorator is a special case of the :func:`asr.option` decorator). We
will get back to this concept later in the tutorial.

To run the instruction we require an atomic structure which can be
generated with ASE's command-line interface


.. code-block::

   $ asr init .

.. code-block::

   $ ase build Ag structure.json -x fcc

The instruction is then easily run through the command-line interface

.. code-block::

   $ asr run "asr.tutorial:energy --atoms structure.json"
   In folder: . (1/1)
   Running asr.tutorial:energy(atoms=Atoms(symbols='Ag', pbc=True, cell=[[0.0, 2.045, 2.045], [2.045, 0.0, 2.045], [2.045, 2.045, 0.0]]))
   


The cache
---------
