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

A realistic test
================

We will now use our knowledge of Pytest and fixtures to write a
realistic test of the ground state recipe of ASR. Such as test already
exists, however, it will serve as a good learning experience to go
through each step. First open the existing
``asr/test/test_gs.py``.

.. note:: Notice the naming convention: We name the test after the module it's testing.

Here we create a new test by appending

.. code-block:: python
   :caption: asr/test/test_gs.py

   ...

   def test_gs_tutorial(asr_tmpdir_w_params, mockgpaw, test_material):
       from asr.gs import main

       main()
   

and we quickly check that the test is running by running:

.. code-block:: console

   $ pytest -k test_gs_tutorial

As you can see the test is running multiple times due to the
test_material fixture. At this point the test if of quite low quality
since the results aren't actually checked. We can improve this by
checking that the band gap is zero (which is the default setting of
the mocked up calculator):

.. code-block:: python
   :caption: asr/test/test_gs.py

   ...

   def test_gs_tutorial(asr_tmpdir_w_params, mockgpaw, test_material):
       from asr.gs import main

       results = main()

       assert results['gap'] == pytest.approx(0)

Here we use a utility function from pytest namely ``approx`` which is
useful when two floating point numbers are to be compared.


Mocks and pytest-mock
---------------------

The previous sections mentions the concept of mocking. Mocking involves
substituting some function, class or module with a `pretend` version
returns some artificial data that you have designed. The kinds of
functions that we would like to mock is slow function/class calls that
are not important for the test. In ASR the most important example of a
mock is the mock of the GPAW calculator which can be found in
:py:mod:`asr.test.mocks.gpaw` and is applied by the
:py:func:`asr.test.fixtures.mockgpaw` fixture.

In the beginning of the turorial, we installed ``pytest-mock`` which
is a plugin to pytest that enables easy mocking. A common use case is
to modify a certain property returned by the Mocked
calculator. :py:mod:`asr.test.mocks.gpaw` is designed such that you
can easily specify a band gap or a fermi level using the ``mocker``
fixture (which is provided by ``pytest-mock``), and check that the
corresponding results of yoru recipe are correct. For example let's
improve our ground state test by setting the band gap and fermi leve
to something non-trivial

.. code-block:: python
   :caption: asr/test/test_gs.py

   ...

   def test_gs_tutorial(asr_tmpdir_w_params, mockgpaw, mocker, test_material):
       from asr.gs import main
       from gpaw import GPAW

       mocker.patch.object(GPAW, '_get_band_gap')
       mocker.patch.object(GPAW, '_get_fermi_level')
       GPAW._get_fermi_level.return_value = 0.5
       GPAW._get_band_gap.return_value = 1
	     
       results = main()

       assert results['gap'] == pytest.approx(1)


As you can see in this concrete example ``mocker`` allows you to patch
objects and explicitly set the return values of the specified methods.

Parametrizing
-------------

We can improve our test even more by using pytest functionality for
parametrizing.

Marks and conftest.py
---------------------

Tox
===

ASR Test sub-package
====================

.. _PyTest: https://docs.pytest.org/en/latest/
