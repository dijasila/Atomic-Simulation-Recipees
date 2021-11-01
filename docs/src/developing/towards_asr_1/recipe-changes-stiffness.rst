Stiffness tensor recipe issues
==============================

The stiffness tensor recipe also has a bad smell originating from the
fact that relaxations have to be performed in separate subfolders

.. literalinclude:: stiffness.py
   :linenos:
   :lines: 216-237
   :emphasize-lines: 7-9
