.. _Getting started:

=================
 Getting started
=================

.. contents::
   :local:

Part 1 - Implementing recipes and instructions
==============================================

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
------------

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
---------

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
    created=2021-11-04 09:42:09.921302
    directory=.
    modified=2021-11-04 09:42:09.921302
   resources=
    execution_duration=0.14464545249938965
    execution_end=1636015329.921233
    execution_start=1636015329.7765875
    ncores=1
   result=-0.000367
   run_specification=
    codes=
     code=
      git_hash=None
      package=ase
      version=3.23.0b1
     code=
      git_hash=None
      package=asr
      version=0.4.1
    name=asr.tutorial:energy
    parameters=element=Ag,crystal_structure=fcc
    uid=5799aa96b84e4207aadab4603d146a33
    version=0
   tags=None

As is evident, there is a lot of information stored in a
:py:class:`asr.Record`, however, right now we want to highlight a
couple of these:

 - The :attr:`asr.Record.run_specification.uid` property is a random
   unique identifier given to all records. This can be used to
   uniquely select records.  The Record provides a shortcut to the UID
   through :attr:`asr.Record.uid`.
 - The :attr:`run_specification.name` property stores the name of the instruction.
   The Record provides a shortcut to the name through :attr:`Record.name`.
 - The :attr:`run_specification.parameters` stores the parameters of the
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
   asr.tutorial:energy: Found cached record.uid=5799aa96b84e4207aadab4603d146a33

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
   Running asr.tutorial(element='Ag')
   Running asr.tutorial:energy(element='Ag', crystal_structure='sc')
   asr.tutorial:energy: Found cached record.uid=5799aa96b84e4207aadab4603d146a33
   Running asr.tutorial:energy(element='Ag', crystal_structure='bcc')
   Running asr.tutorial:energy(element='Ag', crystal_structure='diamond')


We can now check the result using the command-line tool

.. code-block:: console

   $ asr cache ls name=asr.tutorial
           name parameters result
   asr.tutorial element=Ag    fcc

Notice here we applied the "name=asr.tutorial" selection to
select only the record of relevance. As we can see the EMT calculator
correctly predicts FCC as the most stable crystal structure for
silver.

Let's also look at the detailed contents of this record

.. code-block:: console

   $ asr cache detail name=asr.tutorial
   dependencies=
    dependency=uid=da2adfba4643469994320d176470fcda revision=None
    dependency=uid=5799aa96b84e4207aadab4603d146a33 revision=None
    dependency=uid=aa16b032ef8248f98a8fc991ef9d6417 revision=None
    dependency=uid=2fcd3c99e1b840ec8faa680141d8f5fb revision=None
   history=None
   metadata=
    created=2021-11-04 09:42:11.662077
    directory=.
    modified=2021-11-04 09:42:11.662077
   resources=
    execution_duration=0.6687977313995361
    execution_end=1636015331.6620367
    execution_start=1636015330.993239
    ncores=1
   result=fcc
   run_specification=
    codes=
     code=
      git_hash=None
      package=ase
      version=3.23.0b1
     code=
      git_hash=None
      package=asr
      version=0.4.1
    name=asr.tutorial
    parameters=element=Ag
    uid=2deb079207f540568495389abce99ef7
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
   Running asr.tutorial(element='Al')
   Running asr.tutorial:energy(element='Al', crystal_structure='sc')
   Running asr.tutorial:energy(element='Al', crystal_structure='fcc')
   Running asr.tutorial:energy(element='Al', crystal_structure='bcc')
   Running asr.tutorial:energy(element='Al', crystal_structure='diamond')
   $ asr run "asr.tutorial Ni"
   In folder: . (1/1)
   Running asr.tutorial(element='Ni')
   Running asr.tutorial:energy(element='Ni', crystal_structure='sc')
   Running asr.tutorial:energy(element='Ni', crystal_structure='fcc')
   Running asr.tutorial:energy(element='Ni', crystal_structure='bcc')
   Running asr.tutorial:energy(element='Ni', crystal_structure='diamond')
   $ asr run "asr.tutorial Cu"
   In folder: . (1/1)
   Running asr.tutorial(element='Cu')
   Running asr.tutorial:energy(element='Cu', crystal_structure='sc')
   Running asr.tutorial:energy(element='Cu', crystal_structure='fcc')
   Running asr.tutorial:energy(element='Cu', crystal_structure='bcc')
   Running asr.tutorial:energy(element='Cu', crystal_structure='diamond')
   $ asr run "asr.tutorial Pd"
   In folder: . (1/1)
   Running asr.tutorial(element='Pd')
   Running asr.tutorial:energy(element='Pd', crystal_structure='sc')
   Running asr.tutorial:energy(element='Pd', crystal_structure='fcc')
   Running asr.tutorial:energy(element='Pd', crystal_structure='bcc')
   Running asr.tutorial:energy(element='Pd', crystal_structure='diamond')
   $ asr run "asr.tutorial Pt"
   In folder: . (1/1)
   Running asr.tutorial(element='Pt')
   Running asr.tutorial:energy(element='Pt', crystal_structure='sc')
   Running asr.tutorial:energy(element='Pt', crystal_structure='fcc')
   Running asr.tutorial:energy(element='Pt', crystal_structure='bcc')
   Running asr.tutorial:energy(element='Pt', crystal_structure='diamond')
   $ asr run "asr.tutorial Au"
   In folder: . (1/1)
   Running asr.tutorial(element='Au')
   Running asr.tutorial:energy(element='Au', crystal_structure='sc')
   Running asr.tutorial:energy(element='Au', crystal_structure='fcc')
   Running asr.tutorial:energy(element='Au', crystal_structure='bcc')
   Running asr.tutorial:energy(element='Au', crystal_structure='diamond')

We can now take a look at the results with

.. code-block:: console

   $ asr cache ls name=asr.tutorial
           name parameters result
   asr.tutorial element=Ag    fcc
   asr.tutorial element=Al    fcc
   asr.tutorial element=Ni    fcc
   asr.tutorial element=Cu    fcc
   asr.tutorial element=Pd    fcc
   asr.tutorial element=Pt    fcc
   asr.tutorial element=Au    fcc

From which we can see that the EMT calculator predicts the FCC crystal
structure to be the most stable crystal structure for all tested
metals which is also true in reality.

Part 2 - Migrations
===================

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
   
   record.uid=5799aa96b84e4207aadab4603d146a33
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=8fc8e150ff7b4f51847d15750810baf8
   
   record.uid=da2adfba4643469994320d176470fcda
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=fc59143ee7884dd099cadaec8b88fdfa
   
   record.uid=aa16b032ef8248f98a8fc991ef9d6417
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=b7168dd364244954b59b5d250dd6665c
   
   record.uid=2fcd3c99e1b840ec8faa680141d8f5fb
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=7f10ce813c7342d8be1c358f911b684f
   
   record.uid=2deb079207f540568495389abce99ef7
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=f3c0c34eb183432988727ff6b60e750c
   
   record.uid=da75f7e682d643689c025b2b851a0522
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=d72f0763a312410ab03f6c31c5547cc8
   
   record.uid=ca150639cba64d87ac621e6ea06d8fa9
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=94aca2e9ef834750808e07123b5f930f
   
   record.uid=c625d4bd900a48a980e5e2f29fae394d
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=d3b44194035341639eddf494ae055d3d
   
   record.uid=82fdbaa9ddb84b3298924d4dfb191310
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=253a6e6a7bb5446fb9e0a486504d225b
   
   record.uid=729212b49ccd4ffa83c7e020eb1fc5d2
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=d4dc806eec094e56a3d442173d8d4638
   
   record.uid=a3c2116190724144b4dacd3fdcef4f96
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=a4378db78c954229b5629bf4caeddb4a
   
   record.uid=5a8b309d73474787897ccb42f214a960
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=4137dd78691a4334834f3c2c91cac48d
   
   record.uid=3ef7ef9436fb4dbca78eedb718fc5600
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=e1d3a99955d24378a34680b228369405
   
   record.uid=49c483ac906b4e4fb970420755ab55b3
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=ac6ff82daff24653bd6180fcda551e59
   
   record.uid=2ab99e2bf19443a896740f8c39d37944
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=f603f23214e24f5f9b2c3a89e67e979c
   
   record.uid=86d9181e6aad495bbf71be12a30b4fe3
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=5b519e851d8343169b6e87075d7d1f5a
   
   record.uid=7a1058b3e7f6432a8d70f6702ec60d3b
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=cb0794bfe00f44faace00ab8497f9411
   
   record.uid=07ebdc1e1ec54d0c8ba203d0b843c58f
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=2781de71364f4def83a8e0e270aee643
   
   record.uid=a5c2f3f073f3459194002d3f2d63cc09
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=eee9ee183c944cc59e4ac83faef57b0f
   
   record.uid=b4e7563be0da449caa14e8dfa711a750
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=91143ebba191473bb5c2928fa9a5258e
   
   record.uid=626b2a3cbbb245d4924e7052f17c65bd
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=925d0d2f412c40c28b85df7a9d7c61c1
   
   record.uid=68e1a2a7e9b24cfda3d621077435ad83
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=e1f794200d194cc8ae2edf98adb7932c
   
   record.uid=3aed6f9558c74a3ba33310ac461e00a4
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=b6682fa73d2d4f56a61974e0e8a13db3
   
   record.uid=24768298982a4816b22eed2bf223467c
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=1cbac782729942c497e176792cb910d4
   
   record.uid=5f5b362f91224ca284d2ea678fdc5f5f
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=e38058daa4684d31a070b1cbc5514d49
   
   record.uid=8794960772c740f79d8e47c5ec7deb23
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=a70fa60e1de04d888565ce514eb524b6
   
   record.uid=ea984cc41767477e9edf0afe9d36208a
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=e016c39e073649a5b5f75976e4bd7513
   
   record.uid=e71b4961666644f1a9f1eba3e9faff58
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=07da74d798fa40b0a3e1536817bbadc1
   
   record.uid=ee4b0a910b5a4fdd850aac55b8546506
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=233dd820a9154ab194123a6a4090edd5
   
   record.uid=fa0199aaf2a049fab4516c8f33ad338f
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=0614c83c1d2a4c7bb63e6315b8d5e3a7
   
   record.uid=661fc2e4f13a4933856bf8c99a9777f6
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=ebb35b70532f4ababb3bbd8410a9db93
   
   record.uid=0dbe5fc9fe204d149cdfcefe75cc1b94
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=29346196be8f478591262bf5d51ed689
   
   record.uid=18b3641843a44a5290bf4cd2b9e20dbc
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=9f709130db3a43c591579f4a488259c5
   
   record.uid=e30516903b904befa3c2018a8f5f95db
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=d661d20891f24e89b822589ea0a129fa
   
   record.uid=cfbeef9ba8e2449c8cd0b7d0689f152a
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=7369dfb47bd445e4b0a06d7f6a7fc681
   

The output informs us of the changes made to the existing records, in
particular that the parameters have been updated with a new calculator
attribute named EMT. ASR figures this out by comparing the input and
output record of the migration and translates this into to a list of
actual changes. This makes it possible to revert these changes in case
of errors. Let's look at the details of one of the updated records

.. code-block:: console

   $ asr cache detail name=asr.tutorial:energy parameters.element=Ag parameters.crystal_structure=fcc
   dependencies=None
   history=
    latest_revision=8fc8e150ff7b4f51847d15750810baf8
    revision=
     description=Fix old records that are missing calculator='emt'.
     migration_uid=None
     modification=New attribute=.run_specification.parameters.calculator value=emt
     uid=8fc8e150ff7b4f51847d15750810baf8
   metadata=
    created=2021-11-04 09:42:09.921302
    directory=.
    modified=2021-11-04 09:42:09.921302
   resources=
    execution_duration=0.14464545249938965
    execution_end=1636015329.921233
    execution_start=1636015329.7765875
    ncores=1
   result=-0.000367
   run_specification=
    codes=
     code=
      git_hash=None
      package=ase
      version=3.23.0b1
     code=
      git_hash=None
      package=asr
      version=0.4.1
    name=asr.tutorial:energy
    parameters=element=Ag,crystal_structure=fcc,calculator=emt
    uid=5799aa96b84e4207aadab4603d146a33
    version=0
   tags=None

The changes to the record has been stored in the
:attr:`asr.record.history` attribute as a series of revisions (in this
case we only have one revision). Each revision is assigned a random
UID which can be used to identify the record revision. It is the
latest revision ID that is used when identifying dependencies.

We can now run our updated instructions employing other
calculators. Here we will investigate the results when using the
Lennard-Jones calculator identified as "lj" according to ASE.


.. code-block:: console

   $ asr run "asr.tutorial Ag --calculator lj"
   In folder: . (1/1)
   Running asr.tutorial(element='Ag', calculator='lj')
   Running asr.tutorial:energy(element='Ag', crystal_structure='sc', calculator='lj')
   Running asr.tutorial:energy(element='Ag', crystal_structure='fcc', calculator='lj')
   Running asr.tutorial:energy(element='Ag', crystal_structure='bcc', calculator='lj')
   Running asr.tutorial:energy(element='Ag', crystal_structure='diamond', calculator='lj')
   $ asr run "asr.tutorial Al --calculator lj"
   In folder: . (1/1)
   Running asr.tutorial(element='Al', calculator='lj')
   Running asr.tutorial:energy(element='Al', crystal_structure='sc', calculator='lj')
   Running asr.tutorial:energy(element='Al', crystal_structure='fcc', calculator='lj')
   Running asr.tutorial:energy(element='Al', crystal_structure='bcc', calculator='lj')
   Running asr.tutorial:energy(element='Al', crystal_structure='diamond', calculator='lj')
   $ asr run "asr.tutorial Ni --calculator lj"
   In folder: . (1/1)
   Running asr.tutorial(element='Ni', calculator='lj')
   Running asr.tutorial:energy(element='Ni', crystal_structure='sc', calculator='lj')
   Running asr.tutorial:energy(element='Ni', crystal_structure='fcc', calculator='lj')
   Running asr.tutorial:energy(element='Ni', crystal_structure='bcc', calculator='lj')
   Running asr.tutorial:energy(element='Ni', crystal_structure='diamond', calculator='lj')
   $ asr run "asr.tutorial Cu --calculator lj"
   In folder: . (1/1)
   Running asr.tutorial(element='Cu', calculator='lj')
   Running asr.tutorial:energy(element='Cu', crystal_structure='sc', calculator='lj')
   Running asr.tutorial:energy(element='Cu', crystal_structure='fcc', calculator='lj')
   Running asr.tutorial:energy(element='Cu', crystal_structure='bcc', calculator='lj')
   Running asr.tutorial:energy(element='Cu', crystal_structure='diamond', calculator='lj')
   $ asr run "asr.tutorial Pd --calculator lj"
   In folder: . (1/1)
   Running asr.tutorial(element='Pd', calculator='lj')
   Running asr.tutorial:energy(element='Pd', crystal_structure='sc', calculator='lj')
   Running asr.tutorial:energy(element='Pd', crystal_structure='fcc', calculator='lj')
   Running asr.tutorial:energy(element='Pd', crystal_structure='bcc', calculator='lj')
   Running asr.tutorial:energy(element='Pd', crystal_structure='diamond', calculator='lj')
   $ asr run "asr.tutorial Pt --calculator lj"
   In folder: . (1/1)
   Running asr.tutorial(element='Pt', calculator='lj')
   Running asr.tutorial:energy(element='Pt', crystal_structure='sc', calculator='lj')
   Running asr.tutorial:energy(element='Pt', crystal_structure='fcc', calculator='lj')
   Running asr.tutorial:energy(element='Pt', crystal_structure='bcc', calculator='lj')
   Running asr.tutorial:energy(element='Pt', crystal_structure='diamond', calculator='lj')
   $ asr run "asr.tutorial Au --calculator lj"
   In folder: . (1/1)
   Running asr.tutorial(element='Au', calculator='lj')
   Running asr.tutorial:energy(element='Au', crystal_structure='sc', calculator='lj')
   Running asr.tutorial:energy(element='Au', crystal_structure='fcc', calculator='lj')
   Running asr.tutorial:energy(element='Au', crystal_structure='bcc', calculator='lj')
   Running asr.tutorial:energy(element='Au', crystal_structure='diamond', calculator='lj')

Let's check the results

.. code-block:: console

   $ asr cache ls name=asr.tutorial
           name                parameters result
   asr.tutorial element=Ag,calculator=emt    fcc
   asr.tutorial element=Al,calculator=emt    fcc
   asr.tutorial element=Ni,calculator=emt    fcc
   asr.tutorial element=Cu,calculator=emt    fcc
   asr.tutorial element=Pd,calculator=emt    fcc
   asr.tutorial element=Pt,calculator=emt    fcc
   asr.tutorial element=Au,calculator=emt    fcc
   asr.tutorial  element=Ag,calculator=lj    fcc
   asr.tutorial  element=Al,calculator=lj    fcc
   asr.tutorial  element=Ni,calculator=lj    fcc
   asr.tutorial  element=Cu,calculator=lj    fcc
   asr.tutorial  element=Pd,calculator=lj    fcc
   asr.tutorial  element=Pt,calculator=lj    fcc
   asr.tutorial  element=Au,calculator=lj    fcc

In conclusion, the Lennard-Jones calculator also predict fcc as the
most stable crystal structure.
