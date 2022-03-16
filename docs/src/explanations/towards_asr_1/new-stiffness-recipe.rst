How do the recipes change: stiffness-recipe
-------------------------------------------

- Takes ``atoms`` and ``calculator`` as arguments.

.. literalinclude:: stiffness_new.py
   :linenos:
   :lines: 192-210

- Forwards ``calculator`` to the relax-recipe.

.. literalinclude:: stiffness_new.py
   :linenos:
   :lines: 224-237
   :emphasize-lines: 9
