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
framework for writing and running test suites. First install
``pytest`` and ``pytest-mock`` (don't worry about ``pytest-mock``
right now, we will need that for later)

.. code-block:: console

   $ python3 -m pip install pytest pytest-mock --user

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
to wrap your head around, so let's teach it by example. Don't worry,
once you know how they work they will be trivial to use.

Let's extend the previous example with the following

.. code-block:: python
   :caption: asr/test/test_example.py

   import pytest


   @pytest.fixture()
   def some_input_data():
       return 1

   def test_adding_numbers(some_input_data):
       b = 2
       assert some_input_data + b == 3

Now run the test (remember the command from before). It still checks
out! If you are not confused by this, take a minute to understand that
`somehow` the output of the function `some_input_data` was evaluated
and fed into our test. This is the magic of Pytest_. It matches the
input argument against all known fixtures and feeds into it the output
of that fixture, such that the output is available for the test.

This was a trivial example. Fixtures can in general be used to to
initialize tests, set up folders, mock up certain functions (see below
if you don't know what "mock" means), capture output etc.

ASR has its own set of fixtures that are available to all tests. They
are defined in :py:mod:`asr.test.fixtures`. Let's highlight a couple
of the most useful:

  - :py:func:`asr.test.fixtures.asr_tmpdir_w_params`: This sets up an
    empty temporary directory, changes directory into that directory,
    and puts in a parameter file containing a parameter-set that
    ensure fast execution. The temporary directory can be found in
    ``/tmp/pytest-of-username/test_example*``.
  - :py:func:`asr.test.fixtures.mockgpaw`: This substitues GPAW with a
    dummy calculator such that a full DFT calculation won't be needed
    when running a test. See the API documentation for a full
    explanation.
  - :py:func:`asr.test.fixtures.test_material`: A fixture that iterates
    over a set of test materials and runs your test on each material.

To use any of these fixtures in your test your only have to give them
as input arguments:

.. code-block::

   def test_example(asr_tmpdir_w_params, mockgpaw, test_material):
       ...

This will apply all the fixtures above to your test.

Mocks and pytest-mock
---------------------

The previous section mentioned the concept of mocking. Mocking
involves substituting some function, class or module with a `pretend`
version returns some artificial data that you have designed. The kinds
of function that we would like to mock is slow function/class calls
that are not important for the test. In ASR the most important example
of a mock is the mock of the GPAW calculator which can be found in
:py:mod:`asr.test.mocks.gpaw` and is applied by the
:py:func:`asr.test.fixtures.mockgpaw` fixture.

In the beginning of the turorial, we also installed ``pytest-mock``
which is a plugin to pytest that enables easy mocking. A common use
case is to modify a certain property returned by the Mocked gpaw calculator. :py:mod:`asr.test.mocks.gpaw` such that you can



Marks and conftest.py
---------------------

Tox
===

ASR Test sub-package
====================

.. _PyTest: https://docs.pytest.org/en/latest/
