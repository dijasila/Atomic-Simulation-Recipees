What can ASR do right now?
==========================

To explain what ASR can do it is easier to just show it. Calculate the ground state of silicon

.. code-block:: console

   $ ase build Si structure.json -x diamond
   $ asr run gs
   Running asr.gs@calculate(...)
   Running asr.magstate()
   Running asr.magnetic_anisotropy()
   Running asr.structureinfo()
   Running asr.gs()
   $ ls
   gs.gpw                         results-asr.magnetic_anisotropy.json
   gs.txt                         results-asr.magstate.json
   results-asr.gs@calculate.json  results-asr.structureinfo.json
   results-asr.gs.json            structure.json

Collect data to a database

.. code-block:: console

   $ asr run "database.fromtree ."
   $ ls
   ...
   database.db
   $ asr run "database.app database.db"

Open `<http://0.0.0.0:5000/>`_ to see result.
