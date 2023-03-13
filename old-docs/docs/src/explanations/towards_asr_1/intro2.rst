What is a recipe?
=================

A recipe is a python module containing related python functions

asr.gs as an example
--------------------

.. code-block:: python

   ...

   @command(module='asr.gs',
            creates=['gs.gpw'],
            requires=['structure.json'],
            resources='8:10h')
   @option('-c', '--calculator', help='Calculator params.', type=DictStr())
   def calculate(calculator: dict = {
           'name': 'gpaw',
           'mode': {'name': 'pw', 'ecut': 800},
           'xc': 'PBE',
           'basis': 'dzp',
           'kpts': {'density': 12.0, 'gamma': True},
           'occupations': {'name': 'fermi-dirac',
                           'width': 0.05},
           'convergence': {'bands': 'CBM+3.0'},
           'nbands': '200%',
           'txt': 'gs.txt',
           'charge': 0}) -> ASRResult:

       ...
   
       return ASRResult()

   ...

.. code-block:: python

   ...

   @command(module='asr.gs',
   		requires=['gs.gpw', 'structure.json',
   		          'results-asr.magnetic_anisotropy.json'],
            dependencies=['asr.gs@calculate', 'asr.magnetic_anisotropy',
   	                  'asr.structureinfo'],
            returns=Result)
   def main() -> Result:
   
       ...
   	   
       return Result(etot=etot, forces=forces, ...)

   ...
