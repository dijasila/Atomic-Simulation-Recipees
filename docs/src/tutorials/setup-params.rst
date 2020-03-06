The `setup.params` recipe
=========================
All material folders can contain a `params.json`-file. This file can
changed to overwrite default settings in scripts. For example:

.. code-block:: json

   {
   "asr.gs": {"gpw": "otherfile.gpw",
              "ecut": 800},
   "asr.relax": {"d3": true}
   }


In this way all default parameters exposed through the CLI of a recipe
can be corrected.
