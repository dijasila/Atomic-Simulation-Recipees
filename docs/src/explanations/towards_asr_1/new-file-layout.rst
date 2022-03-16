New data storage system: the cache
==================================

Data is now put into a hidden directory such as not to clutter the working
directory

.. code-block:: console

   $ mkdir Si_new
   $ cd Si_new
   $ ase build Si structure.json -x diamond

   $ asr run "gs --calculator {'name':'gpaw','mode':{'name':'pw','ecut':300},'kpts':{'density':2},'txt':None}"

   $ ls
   .  ..  .asr  structure.json
   
   $ tree .asr
   .asr
   ├── records
   │   ├── results-asr.gs::calculate-eb92b5bb5e.json
   │   ├── results-asr.gs::main-e74d516eaa.json
   │   ├── results-asr.magnetic_anisotropy::main-a11d444bf0.json
   │   ├── results-asr.magstate::main-45f46e9d2c.json
   │   └── run-data.json
   └── side_effects
       └── eb92b5bb5e24424gs.gpw

   $ asr cache ls
                            name                                  parameters
               asr.gs::calculate {'atoms': Atoms(symbols='Si2', pbc=True,...
                    asr.gs::main {'atoms': Atoms(symbols='Si2', pbc=True,...
   asr.magnetic_anisotropy::main {'atoms': Atoms(symbols='Si2', pbc=True,...
              asr.magstate::main {'atoms': Atoms(symbols='Si2', pbc=True,...


   $ asr cache graph --draw --labels


.. image:: deps_graph1.svg
   :align: center
