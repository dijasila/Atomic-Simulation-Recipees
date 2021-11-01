=================
Run Specification
=================

.. code-block:: python
   

   from asr.core import get_cache
   cache = get_cache()
   records = cache.select()
   record = records[0]
   print(record.run_specification)


.. code-block:: console

   RunSpec(
       name=asr.structureinfo::main,
       params={'atoms': Atoms(symbols='Si2', pbc=True, cell=[[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]])},
       version=0,
       codes=[version=0.4.1,git=63407ae5, version=3.21.0b1,git=323a9a71, version=20.10.1b1,git=cee4fbf8],
       uid=f13b110857a5466bbe433c89107b4c73
   )


- Extra information about code versions is also included.
- Note: A run specification is only matched against its name and
  parameters when the cache is looking for matches.
