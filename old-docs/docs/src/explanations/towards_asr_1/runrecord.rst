==========
Run record
==========

- A recipe always returns a record, ie. not only its results.

.. code-block:: python
   

   from asr.core import get_cache
   cache = get_cache()
   records = cache.select()
   record = records[0]
   print(record)


.. code-block:: console

   RunRecord(
       run_specification=RunSpec(
           name=asr.gs::main,
	   params={'atoms': Atoms(symbols='Si2', pbc=True, cell=[[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]),
	   'calculator': {'name': 'gpaw', 'kpts': {'density': 4}, 'mode': {'name': 'pw', 'ecut': 400}, 'txt': None}},
	   version=0,
	   codes=[version=0.4.1,git=63407ae5, version=3.21.0b1,git=323a9a71, version=20.10.1b1,git=cee4fbf8],
	   uid=67785f4d333f49c996502fb85f430525),
       result=Result(forces=[[-1.48671778e-1...,
       resources=Resources(time=70.4s, ncores=1),
       side_effects={},
       dependencies=['9a11232ff2a34f3d9374b9e6d731e561', 'ce450350a1654480afb3f810889d043f'],
       migration_id=None,
       migrated_from=None,
       migrated_to=None,
       tags=None)
