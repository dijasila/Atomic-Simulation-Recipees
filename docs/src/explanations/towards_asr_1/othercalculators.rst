====================================
Compatibility with other calculators
====================================

The goal is that in time ASR will be compatible with muliple
calculators that are implemented in ASE

- EMT already works for relaxation and stiffness.

This is still an open problem. Some work has been done though


.. code-block:: python

   ...
   calc.write('gs.gpw')
   ...


.. code-block:: python

   from asr.calculators import get_calculator_class
   ...
   calc = get_calculator_class(name)(**parameters)
   ...
   calculation = calc.save(id='gs')
   ...
   return calculation
