How do the recipes change: GS-recipe
------------------------------------

- Dependencies are not declared explicitly
- Calculator, atoms are direct inputs.
- A new data structure: ``RunRecord`` has been introduced.

.. literalinclude:: gs_new.py
   :linenos:
   :lines: 602-623


In the future all recipes probably needs to take ``atoms`` as
input. Most recipes probably also needs the calculator as input.
