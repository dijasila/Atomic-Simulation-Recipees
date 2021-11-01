Making a convergence study
==========================

Problem
-------

For example, a convergence study of Silicon:

.. code-block:: console

   $ mkdir Si
   $ cd Si
   $ ase build Si structure.json -x diamond

   $ asr run "gs@calculate --calculator {'name':'gpaw','mode':{'name':'pw','ecut':300},'kpts':{'density':2},'txt':None}"
   $ asr run gs

   $ ls
   gs.gpw                                results-asr.magstate.json
   results-asr.gs@calculate.json         results-asr.structureinfo.json
   results-asr.gs.json                   structure.json
   results-asr.magnetic_anisotropy.json

Then presumably you would run

.. code-block:: console

   $ asr run "gs@calculate --calculator {'name':'gpaw','mode':{'name':'pw','ecut':400},'kpts':{'density':2},'txt':None}"
   $ asr run gs

   $ ls
   gs.gpw                                results-asr.magstate.json
   results-asr.gs@calculate.json         results-asr.structureinfo.json
   results-asr.gs.json                   structure.json
   results-asr.magnetic_anisotropy.json

but that doesn't work since files are simply overwritten. What you in
reality have to do right now is to make separate folders for each
parameter set.

.. code-block:: console

   $ mkdir converge_si
   $ cd converge_si

   $ mkdir ecut300eV
   $ mkdir ecut400eV

   $ ase build Si structure.json -x diamond
   $ cp structure.json ecut300eV
   $ cp structure.json ecut400eV

   $ asr run "asr.gs@calculate --calculator {'name':'gpaw','mode':{'name':'pw','ecut':300},'kpts':{'density':2},'txt':None}" ecut300eV/
   $ asr run "asr.gs@calculate --calculator {'name':'gpaw','mode':{'name':'pw','ecut':400},'kpts':{'density':2},'txt':None}" ecut400eV/
   $ asr run gs ecut*eV/


But even that doesn't work which is seen when a database is collected.

.. code-block:: console

   $ asr run "database.fromtree ecut*eV/"
   asr.database.fromtree.MissingUIDS: Duplicate uids in database.
   $ asr run "database.app database.db"


The materials in each folder are identical and therefore they have the
same UID's, which gives an error.
