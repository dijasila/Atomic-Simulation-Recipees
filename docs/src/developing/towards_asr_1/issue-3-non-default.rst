Running a calculation with non default parameters
=================================================

The previous example highlights another deficiency


.. code-block:: console

   $ mkdir Si
   $ cd Si
   $ ase build Si structure.json -x diamond

   $ asr run "gs@calculate --calculator {'name':'gpaw','mode':{'name':'pw','ecut':300},'kpts':{'density':2},'txt':None}"
   $ asr run gs


Why can't we just say

.. code-block:: console

   $ asr run "gs --calculator {'name':'gpaw','mode':{'name':'pw','ecut':300},'kpts':{'density':2},'txt':None}"

?
