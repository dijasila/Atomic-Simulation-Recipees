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
    created=2021-06-24 18:53:20.353366
    directory=.
    modified=2021-06-24 18:53:20.353366
   resources=
    execution_duration=0.22190260887145996
    execution_end=1624553600.3533049
    execution_start=1624553600.1314023
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
    uid=cb55e8269ef14f11943d7273399f9d4f
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
   asr.tutorial:energy: Found cached record.uid=cb55e8269ef14f11943d7273399f9d4f

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
   asr.tutorial:energy: Found cached record.uid=cb55e8269ef14f11943d7273399f9d4f
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
    dependency=uid=4813f96fdd0447888f96996514e549cb revision=None
    dependency=uid=cb55e8269ef14f11943d7273399f9d4f revision=None
    dependency=uid=736496a649574aa5b51f47261773dcb6 revision=None
    dependency=uid=1ce77b0d881f4fc5a24791f526a70402 revision=None
   history=None
   metadata=
    created=2021-06-24 18:53:23.115231
    directory=.
    modified=2021-06-24 18:53:23.115231
   resources=
    execution_duration=1.1910970211029053
    execution_end=1624553603.1151726
    execution_start=1624553601.9240756
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
    uid=e04f84d8f9af4294bcd354c2acf2a23e
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
   
   record.uid=cb55e8269ef14f11943d7273399f9d4f
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=768c45ede5594333b2d7f8ee76297a31
   
   record.uid=4813f96fdd0447888f96996514e549cb
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=7c920581f365450488910c0f557bc10a
   
   record.uid=736496a649574aa5b51f47261773dcb6
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=5f9652c2319246f98307abbce089b241
   
   record.uid=1ce77b0d881f4fc5a24791f526a70402
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=474eef393aeb448980ffb37ba872d766
   
   record.uid=e04f84d8f9af4294bcd354c2acf2a23e
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=d0cf68a6fc434d30926deed578795cf4
   
   record.uid=3edae22c00b54e3b81f72dfd8f73472f
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=ef9b17889a564a52bd23d6c88b8da1c8
   
   record.uid=5cc5d6f64a0a47fd8915610aaf4e4053
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=56b0883f578f4e9db534491ca262f65b
   
   record.uid=5aaded28bd9e44a1a69ea5f1b07fc90a
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=3405a6725f414a989780e924b5767a3b
   
   record.uid=e4107d7beaf2492ea95384252113be9d
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=b07405e4804a4f0f92601a725e45d0dd
   
   record.uid=3eb6b63e64c74598b8f1634c1acb9472
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=41a58e8f5fc94d5b829e5631cb066f09
   
   record.uid=b02b8bd937bf43ceae63c7d799ab3103
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=4278435cc6ef4c789b4ed4d8fbc3344f
   
   record.uid=9192fd59187c4f06aeb18c4cb9259063
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=93cc43009c8e4ff2954c3cdab47772c3
   
   record.uid=7049fd27fc9d487e86205de52c393e6b
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=696b486931db449a95978dbf7858a942
   
   record.uid=021c5cd741da4f79af1e77b13cd48fb9
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=29e79f37a0674a5abfdecf51e2a39f4b
   
   record.uid=75fe827361c549d18cbf8ef58b6aa5bb
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=39db96406fe24a8f9766e5a8435b9135
   
   record.uid=8379adc2f6624abc89c972637a1be2ea
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=385c1d739df744118f00f3919cd46063
   
   record.uid=ac461e27094549f9bf55e72cd6fae8c0
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=46fd22c29dda484493a8dd0f82949763
   
   record.uid=f3bd89cde5d748168316e786a6bd41e1
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=30fab30f473c4260b9a1c648e76642ab
   
   record.uid=ab5cc9ef9ea8476a89ed9b8b9755224a
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=8e91e092b49d4ee8a274b57f05b1c7e6
   
   record.uid=42fa760fc0d142c5a4c6229ef686dd3d
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=61482d03a755460cbe287be2b7ab6714
   
   record.uid=1498354017164ced90332e937faea740
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=2810b87a3b9f48ce98f688fd03b17649
   
   record.uid=dfde85c49eaa41a1b2f6a4f72cf151b1
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=90f1bcde4b134435bd9d422f885666a8
   
   record.uid=f271c4fd876f48588af523a4f2ba26a2
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=19fbb37cc5734d5783952dc1e00c5ef4
   
   record.uid=b42ccfbe8b2540778575e405c9b2b01a
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=0f8b8d53e8c74bb69910aa364df188e5
   
   record.uid=45c3fcb292e846118da176f2bded709a
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=637c80190ba949239584d5d4d26e26dc
   
   record.uid=cb3a3335d2464c0d913807a9b8d65454
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=65c158935b0f45e1ab38a1bb0da92ff6
   
   record.uid=6b514260e5c546d89e94e1b68e879b8d
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=38734b89c7e44a3bab165e4661f09162
   
   record.uid=d5d66200434d4123891e51cd09c5eeec
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=525f5485db9b4d6cbf6b28761687efdb
   
   record.uid=c291974b137941f8875f0c40cf1b1658
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=b385c5237f3e419884f6c32bdc7e51a8
   
   record.uid=664e7979812c4c618df4138261dc0ce4
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=dc6ba34054fd408a97c5b17de01c1f7f
   
   record.uid=0c2fe18091c94bbc80bd59e481821a31
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=1477d4dd810d4c0181bf071b78810756
   
   record.uid=0bd12e8242fd4f52a7e5f94cae6ef9e3
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=3559b7f4dc2348ddad8810e298bb1c84
   
   record.uid=42dd6573b35f496d938e2b341d2ddb2d
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=59e0816a29e44da0b7ab0dbdb4c77ef8
   
   record.uid=8585e7bbbde44c6cb2049cd7d690cae3
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=8528f37dd45e48e9803f76d66358c55a
   
   record.uid=08ca3f9a4b81440a89c21e777982cdb9
   Revision #0 description=Fix old records that are missing calculator='emt'.
   migration_uid=None
   modification=New attribute=.run_specification.parameters.calculator value=emt
   uid=1d00b5ed62ef4ecc915ad68419cfefd2
   

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
    latest_revision=768c45ede5594333b2d7f8ee76297a31
    revision=
     description=Fix old records that are missing calculator='emt'.
     migration_uid=None
     modification=New attribute=.run_specification.parameters.calculator value=emt
     uid=768c45ede5594333b2d7f8ee76297a31
   metadata=
    created=2021-06-24 18:53:20.353366
    directory=.
    modified=2021-06-24 18:53:20.353366
   resources=
    execution_duration=0.22190260887145996
    execution_end=1624553600.3533049
    execution_start=1624553600.1314023
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
    parameters=element=Ag,crystal_structure=fcc,calculator=emt
    uid=cb55e8269ef14f11943d7273399f9d4f
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

In conclusion, the Lennard-Jones calculator also predict fcc as the
most stable crystal structure.
