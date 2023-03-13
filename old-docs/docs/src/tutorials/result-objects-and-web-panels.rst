Specifying result objects and webpanels
=======================================

The :py:class:`asr.core.ASRResult` object is a way for you to document
the what might be contained in the results that you recipe-step returns.

A typical realistic example of a Result object declaration can be seen below

.. literalinclude:: result.py

There are several points to be made here

* The Result class takes some type hints like ``energy: float`` to
  indicate the type of the data contained in the object.
* The class also takes ``key_descriptions`` which eventually are used
  in the docstring for the class.
* The ``formats`` attribute can be set to include additional formatting
  functions and in this case we want to include an
  ``ase_webpanel``. The name is arbitrary but we are using this name
  in the ASR web application so it is important that you get it
  correct.
* The ``@prepare_result`` decorator sets up the result class by
  automatically settings the docstring and more.

The ``webpanel`` formatter takes a ``result`` object an
``ase.db.AtomsRow`` object and a ``key_descriptions`` dictionary.
