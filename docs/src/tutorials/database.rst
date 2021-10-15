.. _Database tutorial:

======================
Creating ASR databases
======================

.. contents::
   :local:


In this tutorial we will walk through how to collect results together in an ASR
database and how to use the ASR web application to browse the database in a
web-browser. Furthermore, we will see how to customize the layout of web
application using database project configuration files.

First we need some records in a cache in order to make a database, so let's
first make those.

.. code-block:: console

   $ asr init .
   $ ase build Ag -x fcc
   $ ase build Au -x fcc
   $ mkdir Ag
   $ mv Ag.json Ag/structure.json
   $ mkdir Au
   $ mv Au.json Au/structure.json
   $ asr run "structureinfo --atoms Ag/structure.json"
   $ asr run "structureinfo --atoms Au/structure.json"
   $ asr cache ls
                name                                  parameters                                      result
   asr.structureinfo atoms=Atoms(symbols='Ag', pbc=True, cell... Result(formula=Ag,stoichiometry=A,has_in...
   asr.structureinfo atoms=Atoms(symbols='Au', pbc=True, cell... Result(formula=Au,stoichiometry=A,has_in...


The easiest way to create an ASR database is to use the CLI command database
fromtree. This command reads structures from set a directories and creates
one row per structures. To associate the structures with a particular record
the fromtree command compares with the `atoms` parameter of each record. Go ahead
and create a database:

.. code-block:: console

   $ asr database fromtree Ag/ Au/
   Collecting folder Ag/ (1/2)
   Collecting folder Au/ (2/2)
   Row #
   0

This creates a new file `database.db` which is actually an ASE database. The database can be browsed
using the CLI provided by ASE:

.. code-block:: console

   $ ase db database.db
   id|age|user |formula|natoms|pbc|volume|charge|   mass
    1| 0s|mogje|Ag     |     1|TTT|17.104| 0.000|107.868
    2| 0s|mogje|Au     |     1|TTT|16.979| 0.000|196.967
   Rows: 2
   Keys: crystal_type, folder, has_inversion_symmetry, pointgroup, spacegroup, spgnum, uid

From the last line of the output we can see that the data contained in the
`structureinfo` record has made its way into the database, which now also
contains information about the point group, space group etc.

To view all this information interactively in the ASR web application it is most convenient
again to use the CLI:

.. code-block:: console

   $ asr database app database.db

Now open your favourite browser and head to `http://localhost:5000/` to browse
the web application.

Project configuration file
--------------------------

It is possible to customize the looks of the database app by creating a project
configuration see :py:func:`asr.database.DatabaseProject`. Such a configuration
can be specified in a python file that contains at least the required arguments:

.. literalinclude:: project.py
   :caption: Example of project configuration file

We can then run the CLI again with the project configration as input:

.. code-block:: console

   $ asr database app project.py


Application scripting interface
-------------------------------

Full control of the web application is provided through the application
scripting interface. Through the scripting interface we can create database
projects and run the web application as shown below


.. literalinclude:: scripting-interface.py
   :caption: Example of using the scripting interface to the database web application.


Which would spin up the server like before.