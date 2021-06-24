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
   :caption: asr/tutorial.py

In this example, we have created a file `tutorial.py` in the `asr/`
package-folder containing an instruction for calculating the total energy of a
bulk metal in a given crystal structure using the effective medium theory (EMT)
calculator. The :func:`asr.argument` helps ASR to construct a command-line
interface to the instruction. Here we have used it to tell ASR that the two
arguments of our instruction is to be interpreted as arguments on the command
line (:func:`asr.option` serves the sames purpose but for command-line options
in stead).

The instruction can be run through the `asr run` command-line interface

.. code-block:: console

   $ asr run "asr.tutorial:energy Ag fcc"
   In folder: . (1/1)
   Running asr.tutorial:energy(element='Ag', crystal_structure='fcc')

As shown in this example, the instruction is assigned a name which is
constructed from filename of the containing file, the python package containing
that file and the function name, in this case amounting to
"asr.tutorial:energy". The arguments for the instruction is given along with
the instruction name enclosed in quotes.

The cache
=========

Whenever an instruction has been run ASR generates an
:py:class:`asr.Record` containing contextual information about the
run, such as the name of the instruction, the parameters and the
result. The Record is stored in the "Cache" which is a kind of
database for storing calculations. The record can be viewed on the
command-line using

.. code-block:: console

   $ asr cache ls
                  name                       parameters    result
   asr.tutorial:energy element=Ag,crystal_structure=fcc -0.000367

The full contents of the record can be shown with

.. code-block:: console

   $ asr cache detail name=asr.tutorial:energy
   dependencies=None
   history=None
   metadata=
    created=2021-06-23 12:13:01.703021
    directory=.
    modified=2021-06-23 12:13:01.703021
   resources=
    execution_duration=0.32183194160461426
    execution_end=1624443181.702959
    execution_start=1624443181.381127
    ncores=1
   result=-0.000367
   run_specification=
    codes=
     code=
      git_hash=None
      package=ase
      version=3.22.0b1
     code=
      git_hash=None
      package=asr
      version=0.4.1
    name=asr.tutorial:energy
    parameters=element=Ag,crystal_structure=fcc
    uid=d16222c829d4489ababa5a6b8d1ba111
    version=0
   tags=None

As is evident, there is a lot of information stored in a
:py:class:`asr.Record`, however, right now we want to highlight a
couple of these:

 - The :attr:`asr.Record.run_specification.uid` property is a random
   unique identifier given to all records. This can be used to
   uniquely select records.  The Record provides a shortcut to the UID
   through :attr:`asr.Record.uid`.
 - The `run_specification.name` property stores the name of the instruction.
   The Record provides a shortcut to the name through :attr:`Record.name`.
 - The `run_specification.parameters` stores the parameters of the
   given run. The Record object provides a shortcut the parameters
   through :attr:`Record.parameters`.
 - The :attr:`result` property stores the result of the instruction.

An important feature of ASR is that of "caching". If we run
the instruction again with the same input parameters ASR will skip the
actual evaluation of the instruction and simply reuse the old
result.

.. code-block:: console

   $ asr run "asr.tutorial:energy Ag fcc"
   In folder: . (1/1)
   asr.tutorial:energy: Found cached record.uid=d16222c829d4489ababa5a6b8d1ba111

This is useful in workflows where it is beneficial to not
redo expensive calculation steps when it has already been performed
once.

Continuing the task of calculating the lowest energy crystal structure
we will implement an additional instruction that loops over crystal
structures and finds the most stable one


.. literalinclude:: getting-started.py
   :pyobject: main
   :caption: asr/tutorial.py

We say that the python module which contains our two
instructions is a "recipe", ie. a collection of collection of
instructions that achieve our goal. The "main"-instruction is the
primary user interface to our recipe and as such it is not necessary
to state so explicitly when running the main-instruction.

.. code-block:: console

   $ asr run "asr.tutorial Ag"
   In folder: . (1/1)
   Running asr.tutorial:main(element='Ag')
   Running asr.tutorial:energy(element='Ag', crystal_structure='sc')
   asr.tutorial:energy: Found cached record.uid=d16222c829d4489ababa5a6b8d1ba111
   Running asr.tutorial:energy(element='Ag', crystal_structure='bcc')
   Running asr.tutorial:energy(element='Ag', crystal_structure='diamond')


We can now check the result using the command-line tool

.. code-block:: console

   $ asr cache ls name=asr.tutorial:main
                name parameters result
   asr.tutorial:main element=Ag    fcc

Notice here we applied the "name=asr.tutorial:main" selection to
select only the record of relevance. As we can see the EMT calculator
correctly predicts FCC as the most stable crystal structure for
silver.

Let's also look at the detailed contents of this record

.. code-block:: console

   $ asr cache detail name=asr.tutorial:main
   dependencies=
    dependency=uid=266acddf1a684156a2153e339efdee44 revision=None
    dependency=uid=d16222c829d4489ababa5a6b8d1ba111 revision=None
    dependency=uid=e76a9d0c290e4dc0a0eb1d809f063640 revision=None
    dependency=uid=9f979d5697a442a6ae52522466f23bf6 revision=None
   history=None
   metadata=
    created=2021-06-23 12:13:05.658912
    directory=.
    modified=2021-06-23 12:13:05.658912
   resources=
    execution_duration=1.2135305404663086
    execution_end=1624443185.6588616
    execution_start=1624443184.445331
    ncores=1
   result=fcc
   run_specification=
    codes=
     code=
      git_hash=None
      package=ase
      version=3.22.0b1
     code=
      git_hash=None
      package=asr
      version=0.4.1
    name=asr.tutorial:main
    parameters=element=Ag
    uid=6e77df7431cd46eda37c8f953965bd63
    version=0
   tags=None

Here we want to highlight :attr:`asr.Record.dependencies` which stores any
dependencies of the selected record on any other records, ie., whether data
from any other records has been used in the construction of the selected
record. A dependency stores the UID and something called the `revision` (we
will get back to this concept later) which can be used to locate the
dependencies.

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
metals which is also true in reality.

=====================================
Getting started - part 2 - migrations
=====================================

..
   $ cp $ASRHOME/docs/src/tutorials/getting-started-ver2.py $ASRLIB/tutorial.py

It often happens that you want/have to make changes to an existing instruction.
For example, you want to add an additional argument, change the return type of
the result or change the implementation which requires thinking about what
should happen to existing Records in the cache. This is what "migrations" are
for.

ASR implements a revisioning system for Records for handling this
problem which revolves around defining functions for updating existing
records to be compatible with the newest implementation of the
instructions.

In the following, we will continue with the example of calculating the
most stable crystal structure. It would be interesting to compare some
of ASE's other calculators. However, at the moment, the `energy`
instruction is hardcoded to use the EMT calculator and we will have to
update the instruction to supply the calculator as an input argument

.. literalinclude:: getting-started-ver2.py
   :pyobject: energy
   :caption: asr/tutorial.py

To update the existing records to be consistent with the new
implementation we make a migration that adds the `calculator`
parameter to the existing records and sets it to `emt`

.. literalinclude:: getting-started-ver2.py
   :pyobject: add_missing_calculator_parameter
   :caption: asr/tutorial.py

As is appararent, a migration is nothing more than a regular python
function decorated with the :py:func:`asr.migration` decorator. The
decorator takes a `selector` argument which is used to select records
to be migrated. The selector will be applied to all existing records
and should return a boolean. The function will then be applied for all
records that fulfill the selector criterion.

The migrations can be applied through the CLI

.. code-block:: console

   $ asr cache migrate
   There are 35 unapplied migrations, 0 erroneous migrations and 0 records are up to date.
   
   Run
       $ asr cache migrate --apply
   to apply migrations.

To apply the migrations we do

.. code-block:: console

   $ asr cache migrate --apply
   There are 35 unapplied migrations, 0 erroneous migrations and 0 records are up to date.
   
   record.uid=d16222c829d4489ababa5a6b8d1ba111
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=266acddf1a684156a2153e339efdee44
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=e76a9d0c290e4dc0a0eb1d809f063640
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=9f979d5697a442a6ae52522466f23bf6
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=6e77df7431cd46eda37c8f953965bd63
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=d5b7bace79594946bec68b0e14f12e02
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=f4f0d34aada8429fb4eba435718768cf
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=0a9e6cd5e8ca474e87d9dca67d372e94
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=2c88de4283a34dd6b936d3e4c34182fc
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=67cb99228b01427780e985873923ae58
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=3baeef6c0bf14225be887bba3988ea93
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=e5b93aef45514a4580c3475e3cea3dc3
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=9830e18d6cab4e598eb460bacade26fe
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=5502be76918e40b8bfe6f1892c941129
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=0f702ca4850d497d9bd42bf8f79a9a68
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=1fcf1f3aa801414cacbe242791f919ae
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=91bc236c859d413c9ae40e12313b7715
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=d55daa8aca2b4c148d087f15fcf045e4
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=a44235ade2fb4fd7ad1d9081b912e32c
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=c98b028150724591a9dd6b3f16c7fc8c
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=66894b00e64a4261ae596a5ec562e4c5
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=707f39c9d4704d489a7ff76ae9dc19c2
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=deb875583bd048518c781c4b065d187c
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=30574476b6554549a07aff540bb85e53
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=79e9721a7d084c28a11544adcef49b10
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=ad55e71ef7aa4e83ad0b40ae1eff2bb6
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=abdb6321d4714cd4be374db62fb74ea2
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=ae79577f3d504fb6b66ffa0987868dd1
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=aabd8e0cd57c4df9b6a4aa290e5ccf8d
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=206e9dae7b634e1783384c39ca00c338
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=652cbb82405e4a4fa341cb4121483f8c
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=5a6e68e4376f456da3898c28f23e79a8
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=9fab0bddf4ca411abfd472f99a0af80d
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=8806bbcb96fa4f98804b4175666c5d2a
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)
   record.uid=8017e43bc37d4301a04d4ad0c61e17a3
   Revision #0 Fix old records that are missing calculator='emt'. (New attribute=.run_specification.parameters.calculator value=emt)

The output informs us of the changes made to the existing records, in
particular that the parameters have been updated with a new calculator
attribute named EMT. ASR does this by comparing the input and output record of
the migration and translates this into to a list of actual changes. This makes
it possible to revert these changes in case of errors. Let's look at the
details of one of the updated records

.. code-block:: console

   $ asr cache detail name=asr.tutorial:energy element=Ag



We can now
run our updated instructions employing other calculators. Here we will
investigate the results when using the Lennard-Jones calculator identified as
"lj" according to ASE.


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
