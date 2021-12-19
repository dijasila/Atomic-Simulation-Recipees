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
    created=2021-12-16 21:17:30.774443
    directory=.
    modified=2021-12-16 21:17:30.774443
   resources=
    execution_duration=0.5718867778778076
    execution_end=1639685850.774387
    execution_start=1639685850.2025
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
    uid=aee3d863d8234913b6f1b4aa53aee324
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
   asr.tutorial:energy: Found cached record.uid=aee3d863d8234913b6f1b4aa53aee324

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
   asr.tutorial:energy: Found cached record.uid=aee3d863d8234913b6f1b4aa53aee324
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
    dependency=uid=41d1addb90f64c9980f0d0212003a90a revision=None
    dependency=uid=aee3d863d8234913b6f1b4aa53aee324 revision=None
    dependency=uid=4b9f72b4c5264decaa626bafe7c162a2 revision=None
    dependency=uid=f853317e67694da697fd3d11610f6354 revision=None
   history=None
   metadata=
    created=2021-12-16 21:17:34.642078
    directory=.
    modified=2021-12-16 21:17:34.642078
   resources=
    execution_duration=2.222498893737793
    execution_end=1639685854.642043
    execution_start=1639685852.4195442
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
    name=asr.tutorial:main
    parameters=element=Ag
    uid=870759861aa34db18598db7c6dc7a9e4
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

Part 2 - Migrations and mutations
=================================

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
implementation we make a mutation that adds the `calculator`
parameter to the existing records and sets it to `emt`

.. literalinclude:: getting-started-ver2.py
   :pyobject: add_missing_calculator_parameter
   :caption: asr/tutorial.py

As is appararent, a mutation is nothing more than a regular python
function decorated with the :py:func:`asr.mutation` decorator. The
decorator takes a `selector` argument which is used to select records
to be migrated. The selector will be applied to all existing records
and should return a boolean. The function will then be applied for all
records that fulfill the selector criterion.

The mutations can be applied through the CLI

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
   
   record.uid=aee3d863d8234913b6f1b4aa53aee324
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=c3356e5aee5946cab47024a64757969e
   
   record.uid=41d1addb90f64c9980f0d0212003a90a
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=c04ea3c8651247f9ac251122a29a4e70
   
   record.uid=4b9f72b4c5264decaa626bafe7c162a2
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=c3f213d54d9147aab515d279c60a42cc
   
   record.uid=f853317e67694da697fd3d11610f6354
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=0179a0f174ae4a049c948b6fcf9c764e
   
   record.uid=870759861aa34db18598db7c6dc7a9e4
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=8cac007e53df45e09e3e0369e3f481e4
   
   record.uid=bbaba2bd17474a29952986c214e4dbb5
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=6ae7c38213314b5593d8a2d000587a63
   
   record.uid=c04e7cabbc744374bbb7038a6caf5d3e
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=0861bbb600ff48208c6bda998d100d21
   
   record.uid=0c0a237f3e65484fa7fbefa5e23c8ef7
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=31044787a0cf462ab3c0f949aed96e25
   
   record.uid=3d41be859fab4a26857f65a9e96debe0
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=1598240c2fe944dda47a4903fb27392c
   
   record.uid=086563a7a55d49939dab2eb9f78fb887
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=497da4e0ce8f40ec9eaaa343c80d16b8
   
   record.uid=57fe85c80dbd47fa9fe428b1249aaafb
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=60e46cd15e43411bb53b7afbf185ec57
   
   record.uid=01840acf1387412f9be923e6c5c32ded
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=50fff3e3e4a8494086a170c14c6638c7
   
   record.uid=04ec872f905542c583a55fe9f1596b66
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=6d6e4d904ec14928891d459b4b51cbba
   
   record.uid=86a0abab07694ec29aa3105b657d4243
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=993e8bb7c0a44d2bbf51957c174bd578
   
   record.uid=b4e4a5951a984b90aeced1b1ad5525b6
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=b336411b3d34421b87f94ab2682cdb07
   
   record.uid=82d25401af124703b66e818360ba79aa
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=495b5571ea204124a2c0e9bd78f9eb50
   
   record.uid=50fbadc8c6a0416fbaaad79a213d8cc1
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=db795e27fa5b49e1bbf1551a6aa97ac6
   
   record.uid=be782f46b9354562b91683b07a6ea76a
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=b8af4123403d478ea6c35cc57189fdfd
   
   record.uid=0f88976a51e146d0927cbb8d385e774a
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=ce7e3b3e8f2140838453c0d015ecc497
   
   record.uid=67aa9298f0fa4cb28ae59575553ba81f
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=f61ea5b779834bb68c5eedd35dba3d47
   
   record.uid=deb0725d8bb74ebbacad962fa8d7a78c
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=4ced27638b2f455990bbbd86ea1eace3
   
   record.uid=01cdb7dc5ce14c6bb2a12cf4bb3b8dac
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=ab0364f6e9c94da4b303208aaec0c474
   
   record.uid=f573c6905e244bbc9c10c77055aa4268
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=b7b01e2378b6405db9fe56781a1219a6
   
   record.uid=51c37ede9ffb458c82d4cd02bbb7d02f
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=6d650434c7454702b99ab972c97c8f63
   
   record.uid=26d1393694bf407dbc013d1b44bb55f9
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=d7b10b0f53614ac490373fc107b27875
   
   record.uid=d3c30b8ddc894927b248536f4bca0d1d
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=462f89f6d25a48b99e22f87c4f3752f3
   
   record.uid=e3bc52fec04a4b09908cb025b77c1bd9
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=1d569d08793049bd838d70ef1af37092
   
   record.uid=f218f986b5ce4208ad5a5600ae2e49cd
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=af8cfe4c520247e6b7074d8241919123
   
   record.uid=87e653f0a2034ea0a6f7affb0a04781b
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=709b90040a37450cb64ce0d5f30b031c
   
   record.uid=fd52cb80907e4315acfc6ecf6208c43b
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=63066dbcc0b74b9d9cb0d584fe161820
   
   record.uid=175438c794324c70851f5b70b5f3932d
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=daabe888f26c45929515fd2f84c3015e
   
   record.uid=2cd44b66beed4e2e858bc84867f50038
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=b556fbf7697f4157940c9299e8e1e75e
   
   record.uid=f226125b99eb4aeebf1a038bd4b8673b
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=85da87912a3c4518bc674146e514cb2b
   
   record.uid=dbf420f6f1bb4bdeb06d78e1b4771bcc
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=47da0cd3986643ef98554168d59f09a2
   
   record.uid=12138065a3fd48ea9a5e68088145d11c
   Revision #0 changes=New attribute=.run_specification.parameters.calculator value=emt
   description=Fix old records that are missing calculator='emt'.
   mutation_uid=None
   uid=77c6d8b392c7417699ecbd09bcbcab2a
   

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
    latest_revision=c3356e5aee5946cab47024a64757969e
    revision=
     changes=New attribute=.run_specification.parameters.calculator value=emt
     description=Fix old records that are missing calculator='emt'.
     mutation_uid=None
     uid=c3356e5aee5946cab47024a64757969e
   metadata=
    created=2021-12-16 21:17:30.774443
    directory=.
    modified=2021-12-16 21:17:30.774443
   resources=
    execution_duration=0.5718867778778076
    execution_end=1639685850.774387
    execution_start=1639685850.2025
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
    uid=aee3d863d8234913b6f1b4aa53aee324
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
