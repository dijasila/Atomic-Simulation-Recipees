======================
Databases from records
======================

Continuing the previous example. Collecting a database will now
collect all records into the row data

.. code-block::

   $ asr run structureinfo

   $ asr database fromtree .

   $ ase db database.db id=1 -l
   Si2:
   Unit cell in Ang:
   axis|periodic|          x|          y|          z|    length|     angle
      1|     yes|      0.000|      2.715|      2.715|     3.840|    60.000
      2|     yes|      2.715|      0.000|      2.715|     3.840|    60.000
      3|     yes|      2.715|      2.715|      0.000|     3.840|    60.000
   
   Data: structure.json, records

   ...
   
   $ asr database app database.db


Open `<http://0.0.0.0:5000/>`_ to see result.
