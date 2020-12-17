==============
Data migration
==============

What about the data that we already have. How do we convert these into
those new RunRecords? Here is an example of MoS\ :subscript:`2`.

.. code-block:: console

   $ cd MoS2-b3b4685fb6e1

   $ asr cache ls
   name parameters
   
   $ asr cache migrate
   You have unapplied migrations:
   Migrate resultsfile results-asr.raman.json
   Migrate resultsfile results-asr.dimensionality.json
   Migrate resultsfile results-asr.polarizability.json
   ...
   Migrate resultsfile strains--1.0%-yy/results-asr.relax.json
   Migrate resultsfile strains--1.0%-yy/results-asr.formalpolarization.json
   
   Run
       $ asr cache migrate --apply
   to apply these migrations.

   $ asr cache migrate --apply

   $ asr cache migrate
   All records up to date. No migrations to apply.

   $ asr cache ls name=asr.gs::main -f "uid migrated_from migrated_to" -w 10
             uid migrated_from   migrated_to
   7679d7750b...          None e7b94d69fc...
   e7b94d69fc... 7679d7750b...          None

   
