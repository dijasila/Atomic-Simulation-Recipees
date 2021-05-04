.. _Recipe:

How to start writing recipes
============================

[Warning: This tutorial assumes knowledge of ASE.]

A "recipe" in ASR is a python module that contains instructions. This
concept is similar to cook-book recipes where a recipe could be "Make
a chocolate cake" and the recipe itself would contain a list of
instructions that you could follow to produce said chocolate cake.

To see how this looks in practice, let's write a recipe that
calculates the lowest energy crystal-structure of some well known
metals using the EMT calculator.

.. code-block:: python

   import asr
   from ase.calculator.calculator import get_calculator_class

   @asr.instruction()
   @asr.atomsopt
   def energy(atoms):
       """Calculate the total energy of atomic structure."""
       calculator = EMT()
       atoms.calc = calculator
       energy = atoms.get_potential_energy()
       return energy
