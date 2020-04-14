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
framework for writing test suites. To invoke pytest and run all ASR
tests change directory into your ``asr/`` folder and run pytest:

.. code-block:: console

  $ pytest --pyargs asr

This will locate all tests of ASR and evaluate them and test summary. 

Tox
===


.. _PyTest: https://docs.pytest.org/en/latest/
