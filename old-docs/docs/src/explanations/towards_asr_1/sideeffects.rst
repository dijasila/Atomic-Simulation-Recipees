========================
Handling of side effects
========================

- A side effect in this context is simply a file that is being
  written by a recipe.
- We typically want to avoid them, but in some cases they are hard to
  avoid (gs and phonons).
- Identically named side effects can be overwritten upon multiple runs.
- The ground state recipe is an example.


.. code-block:: python

   def calculate(atoms, calculator):
       ...
       atoms.calc.write('gs.gpw')
       ...


The proposed solution is to create a separate folder where the recipe
is executed and register any files that the recipe has produced.
       
.. code-block:: python

   def calculate(atoms, calculator, asrcontrol):
       ...
       atoms.calc.write('gs.gpw')
       side_effect = asrcontrol.register_side_effect('gs.gpw')
       return side_effect


   ...
   
   def main(...):
       side_effect = calculate(...).result
       side_effect.restore()
       ...

Then the file will be copied to a "safe" and given a unique
identifier. Remember the earlier file layout.

.. code-block:: console
   :emphasize-lines: 9-10

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

All remaining files will be deleted automatically.
