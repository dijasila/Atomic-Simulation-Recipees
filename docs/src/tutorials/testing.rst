.. _Testing tutorial:

=======================
Getting started testing
=======================

Testing is essential for any piece of software and in particular in
collaborative projects where the consequences of changes to your code
extends beyond yourself. This tutorial walks you through all the
important concepts that you need to know to write simple tests, but
leaves out non essential details. Thus this document is the only one
you need to read.

Pytest
======

As its test runner ASR uses PyTest_, which is a very popular python
framework for writing and running test suites. First install pytest

.. code-block:: console

   $ python3 -m pip install pytest --user

To invoke pytest and run all ASR tests change directory into your
``asr/`` folder and run pytest:

.. code-block:: console

  $ pytest --pyargs asr

This will locate all tests of ASR and evaluate them and test
summary. Pytest_ locates a test by searching for all files matching
``test_*`` and looking for functions also matching ``test_*``. In ASR
these can be found in ``asr/test/``. Let's try and write a simple test
to understand how it works:

.. code-block:: python
   :caption: asr/test/test_example.py

   def test_adding_numbers():
       a = 1
       b = 2
       assert a + b == 3

Save this in ``asr/test/test_example.py`` and run

.. code-block:: console

   $ pytest -k test_example


As you will see the test we jsut wrote ran and checked out
(hopefully). The option ``-k`` matches all tests with the given
pattern and only run those that match. More advanced logical
expressions like ``not test_example`` are also allowed. To see all
options of Pytest do:

.. code-block:: console

   $ pytest -h

Use this command as a reference in case you don't remember the meaning
of a specific option.

Pytest fixtures
---------------

Pytest has an important concept called ``fixtures`` which can be hard
to wrap your head around.


Tox
===


.. _PyTest: https://docs.pytest.org/en/latest/
