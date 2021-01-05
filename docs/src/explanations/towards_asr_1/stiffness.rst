=========
Stiffness
=========

Stiffness calculations are now also possible without artificial folders


.. code-block:: console

   $ ase build Ag structure.json -x fcc
   $ asr run "stiffness --calculator name='emt'"

   $ asr cache graph --draw --labels --saveto deps_graph3.svg


.. image:: deps_graph3.svg
   :align: center
