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

An important feature of ASR is that of "caching". If we run
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

The migrations can be applied through the CLI

.. code-block:: console

   $ asr cache migrate
   There are 35 unapplied migrations, 0 erroneous migrations and 0 records are up to date.
   
   Run
       $ asr cache migrate --apply
   to apply migrations.

To apply the migrations we do

..  code-block:: console

   $ asr cache migrate --apply
   There are 35 unapplied migrations, 0 erroneous migrations and 0 records are up to date.
   
   record.uid=3b0d67ba5c054355b8d064bf839e9c3d
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=671c1fbaff1a4d598697c66105bf9754
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=d4f16b0378af43e7853d86df69b3416d
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=f0698d8879514ed6940e2a6c6c1fa881
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=35e7aaebc43b4033b60db13aee408da4
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=949c4a5c984348df88e0e8f33e82f8b1
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=c48a680aaa694afc9a37caf62bcfeb06
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=2c85e8243e9f49a28cc1d22a566e8b7f
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=db7ed56b532a4291b1060fb46d5c289a
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=573a25d64fc64758ae37b251daf7fdb5
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=9418a28e713a4bfca0c2814908c6ca0b
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=d6247127069f4656beec0342c62360ae
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=9573dc6793794cc48f0a7019ebc750b0
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=0f0937c6d5984bbf8d34cb8d080d77e7
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=9766172df9734d27b4cab16723125efb
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=ae579266475741aaaf8427c6abe43e86
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=ea70e777ef0a4f26beb0b422ba28190e
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=c18e2aaf08894c0a82762785291d9984
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=ba56801d4d4c44408de193fd2a26ef55
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=b3a3b948b36c41bcb0229c5e9efc47ac
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=16fe7608a15f410e8daad1b7700425bc
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=a28af577beea40cd896a9b73fb945476
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=133d144d0d5c47e98c03d887e18a394e
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=c7f362e4877d4b9396b3a05fc7e68a42
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=b4c81fa4af684f31896091c6977aee09
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=97859da642d940aebe72308ab70faa99
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=bd9510157c314d7f9a36e577ea32bcf6
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=404a012bdfe746c891276799fb7a48e9
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=5e3094cc679848bfadbb4dae7249f12a
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=15b7b216045145f59c7b94419e06fb0a
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=0bb767138f564125a96b6b1beecd04af
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=d60cbaa29a324c89a3e595a5dd8eb96e
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=909e738761ea47ec91ab45a981ed202e
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=a81807077bd8403cb43b895ab4168039
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=8392bc08787142849c53af49b9722f25
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)

The output informs us of the changes made to the existing records. We
can now run our updated instructions employing other calculators.

.. code-block:: console

   $ asr run "asr.tutorial Ag --calculator lj"
   In folder: . (1/1)
   Running asr.tutorial:main(element='Ag', calculator='lj')
   Running asr.tutorial:energy(element='Ag', crystal_structure='sc', calculator='lj')
   Running asr.tutorial:energy(element='Ag', crystal_structure='fcc', calculator='lj')
   Running asr.tutorial:energy(element='Ag', crystal_structure='bcc', calculator='lj')
   Running asr.tutorial:energy(element='Ag', crystal_structure='diamond', calculator='lj')
   $ asr run "asr.tutorial Al --calculator lj"
   In folder: . (1/1)
   Running asr.tutorial:main(element='Al', calculator='lj')
   Running asr.tutorial:energy(element='Al', crystal_structure='sc', calculator='lj')
   Running asr.tutorial:energy(element='Al', crystal_structure='fcc', calculator='lj')
   Running asr.tutorial:energy(element='Al', crystal_structure='bcc', calculator='lj')
   Running asr.tutorial:energy(element='Al', crystal_structure='diamond', calculator='lj')
   $ asr run "asr.tutorial Ni --calculator lj"
   In folder: . (1/1)
   Running asr.tutorial:main(element='Ni', calculator='lj')
   Running asr.tutorial:energy(element='Ni', crystal_structure='sc', calculator='lj')
   Running asr.tutorial:energy(element='Ni', crystal_structure='fcc', calculator='lj')
   Running asr.tutorial:energy(element='Ni', crystal_structure='bcc', calculator='lj')
   Running asr.tutorial:energy(element='Ni', crystal_structure='diamond', calculator='lj')
   $ asr run "asr.tutorial Cu --calculator lj"
   In folder: . (1/1)
   Running asr.tutorial:main(element='Cu', calculator='lj')
   Running asr.tutorial:energy(element='Cu', crystal_structure='sc', calculator='lj')
   Running asr.tutorial:energy(element='Cu', crystal_structure='fcc', calculator='lj')
   Running asr.tutorial:energy(element='Cu', crystal_structure='bcc', calculator='lj')
   Running asr.tutorial:energy(element='Cu', crystal_structure='diamond', calculator='lj')
   $ asr run "asr.tutorial Pd --calculator lj"
   In folder: . (1/1)
   Running asr.tutorial:main(element='Pd', calculator='lj')
   Running asr.tutorial:energy(element='Pd', crystal_structure='sc', calculator='lj')
   Running asr.tutorial:energy(element='Pd', crystal_structure='fcc', calculator='lj')
   Running asr.tutorial:energy(element='Pd', crystal_structure='bcc', calculator='lj')
   Running asr.tutorial:energy(element='Pd', crystal_structure='diamond', calculator='lj')
   $ asr run "asr.tutorial Pt --calculator lj"
   In folder: . (1/1)
   Running asr.tutorial:main(element='Pt', calculator='lj')
   Running asr.tutorial:energy(element='Pt', crystal_structure='sc', calculator='lj')
   Running asr.tutorial:energy(element='Pt', crystal_structure='fcc', calculator='lj')
   Running asr.tutorial:energy(element='Pt', crystal_structure='bcc', calculator='lj')
   Running asr.tutorial:energy(element='Pt', crystal_structure='diamond', calculator='lj')
   $ asr run "asr.tutorial Au --calculator lj"
   In folder: . (1/1)
   Running asr.tutorial:main(element='Au', calculator='lj')
   Running asr.tutorial:energy(element='Au', crystal_structure='sc', calculator='lj')
   Running asr.tutorial:energy(element='Au', crystal_structure='fcc', calculator='lj')
   Running asr.tutorial:energy(element='Au', crystal_structure='bcc', calculator='lj')
   Running asr.tutorial:energy(element='Au', crystal_structure='diamond', calculator='lj')

Let's check the results

.. code-block:: console

   $ asr cache ls name=asr.tutorial:main
                name                parameters result
   asr.tutorial:main element=Ag,calculator=emt    fcc
   asr.tutorial:main element=Al,calculator=emt    fcc
   asr.tutorial:main element=Ni,calculator=emt    fcc
   asr.tutorial:main element=Cu,calculator=emt    fcc
   asr.tutorial:main element=Pd,calculator=emt    fcc
   asr.tutorial:main element=Pt,calculator=emt    fcc
   asr.tutorial:main element=Au,calculator=emt    fcc
   asr.tutorial:main  element=Ag,calculator=lj    fcc
   asr.tutorial:main  element=Al,calculator=lj    fcc
   asr.tutorial:main  element=Ni,calculator=lj    fcc
   asr.tutorial:main  element=Cu,calculator=lj    fcc
   asr.tutorial:main  element=Pd,calculator=lj    fcc
   asr.tutorial:main  element=Pt,calculator=lj    fcc
   asr.tutorial:main  element=Au,calculator=lj    fcc

We can confirm that the Lennard-Jones calculator also predict fcc as
the most stable crystal structure.
