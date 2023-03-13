.. _Testing tutorial:

==============================
Introduction to testing in ASR
==============================

Testing is essential for any piece of software and in particular in
collaborative projects where the consequences of changes to your code
extend beyond yourself. This tutorial walks you through all the
important concepts and tools that you need to know to write tests of
your recipe.

This tutorial assumes that you have git-clone'd the ASR project and
that the clone is located in a directory named ``asr/``.

.. contents::
   :local:

PyTest
======

As its testing framework ASR uses pytest_ which is a very popular
python package for said purpose. First install ``pytest`` and
``pytest-mock`` (don't worry about ``pytest-mock`` right now, we will
need that for later)

.. code-block:: console

   $ python3 -m pip install pytest pytest-mock --user

To invoke pytest and run all ASR tests change directory into your
``asr/`` folder and run pytest:

.. code-block:: console
   :caption: In: asr/

   $ pytest  # Don't wait for this to finish: Ctrl-C to cancel

This will collect all tests of ASR, evaluate them and print a test
summary. pytest_ collects tests by searching for all files in the
current directory and child-directories matching ``test_*`` and
looking for functions in those files matching ``test_*``. In ASR these
can be found in ``asr/asr/test/``. Let's try and write a simple toy-model
test to understand how it works:

.. code-block:: python
   :caption: In: asr/asr/test/test_example.py

   def test_adding_numbers():
       a = 1
       b = 2
       assert a + b == 3

Save this in ``asr/asr/test/test_example.py`` and run

.. code-block:: console
   :caption: In: asr/

   $ pytest -k test_example
   ...
   asr/test/test_example.py .
   ...


Yay a dot! That means that the test ran successfully. A failed test
would be marked with "F". The option ``-k`` matches all tests with the
given pattern and only run those that match. More advanced logical
expressions like ``-k "not test_example"`` are also allowed. If we
want more verbose output we can also add the option ``-v``. To see all
options of pytest do:

.. code-block:: console

   $ pytest -h

Use this command as a reference in case you don't remember the meaning
of a specific option.

pytest fixtures
---------------

pytest has an important concept called ``fixtures`` which can be hard
to wrap your head around, so let's teach it by example. Don't worry,
once you know how they work they will be trivial to use.

Let's extend the previous example with the following

.. code-block:: python
   :caption: In: asr/asr/test/test_example.py

   import pytest


   @pytest.fixture()
   def some_input_data():
       return 1


   def test_adding_numbers(some_input_data):
       b = 2
       assert some_input_data + b == 3


Here we have created a function ``some_input_data`` which returns 1,
and decorated that with ``pytest.fixture``. At the same time we have
added an input argument to our original test identically named
``some_input_data`` and removed the definition ``a = 1``.

Now run the test (remember the command from before). It still checks
out?! If you are not confused by this, take a minute to appreciate
that *somehow* the output of the function ``some_input_data`` was
evaluated and fed into our test. This is the magic of pytest_. It
matches the input arguments of your test against all known fixtures
and feeds into it the output of that fixture, such that the output is
available for the test.

This was a trivial example. Fixtures can in general be used to
initialize tests, set up empty folders, set-up and tear-down tests,
mock up certain functions (see below if you don't know what "mock"
means), capture output etc.

ASR has its own set of fixtures that are automatically available to
all tests. They are defined in :py:mod:`asr.test.fixtures`. Let's
highlight a couple of the most useful:

  - :py:func:`asr.test.fixtures.asr_tmpdir_w_params`: This sets up an
    empty temporary directory, changes directory into that directory
    and puts in a parameter file ``params.json`` containing a default
    parameter-set that ensure fast execution. The temporary directory
    can be found in
    ``/tmp/pytest-of-username/pytest-current/test_example*``.
  - :py:func:`asr.test.fixtures.mockgpaw`: This substitues GPAW with a
    dummy calculator such that a full DFT performed won't be needed
    when running a test. See the API documentation for a full
    explanation :py:mod:`asr.test.mocks.gpaw.GPAW`.
  - :py:func:`asr.test.fixtures.test_material`: A fixture that
    iterates over a standard set of test materials and returns the
    atoms objects to your test one by one.

To use any of these fixtures in your test your only have to give them
as input arguments to your test function, you don't even have to
import them, and the order doesn't matter:

.. code-block:: console

   def test_example(asr_tmpdir_w_params, mockgpaw, test_material):
       ...

.. admonition:: Tip: Where are my tests running?

   When debugging it will be useful to check the actual output of your
   recipes, and to do this you need to know where pytest_ actually is
   running your tests. When you start pytest_ it will create a
   temporary directory and run all your tests in that folder. This
   folder can by default be found in
   ``/tmp/pytest-of-username/pytest-run_number``. The latest run can
   always be found under the symbolic link
   ``/tmp/pytest-of-username/pytest-current``.

A realistic test
================

We will now use our knowledge of pytest and fixtures to write a
realistic test of the ground state recipe of ASR. Such as test already
exists, however, it will serve as a good learning experience to go
through each step. First open the existing
``asr/asr/test/test_gs.py``.

.. note::

   Notice the naming convention: We name the test after the module
   it's testing.

We create a new test by appending the following to
``asr/asr/test/test_gs.py``

.. code-block:: python
   :caption: In: asr/asr/test/test_gs.py

   # ... Rest of test_gs.py

   def test_gs_tutorial(asr_tmpdir_w_params, mockgpaw, test_material):
       from asr.c2db.gs import main
       
       test_material.write('structure.json')
       main()
   

and we quickly check that the test works

.. code-block:: console
   :caption: In: asr/

   $ pytest -k test_gs_tutorial

As you can see the test is running multiple times (there are multiple
dots) due to the test_material fixture which feeds multiple different
test materials into the test as input. At this point the test is of
quite low quality since the results aren't actually checked against
anything. We can improve this by checking that the band gap is zero
(which is the default setting of the mocked-up/dummy calculator):

.. code-block:: python
   :caption: In: asr/asr/test/test_gs.py

   ...

   def test_gs_tutorial(asr_tmpdir_w_params, mockgpaw, test_material):
       from asr.c2db.gs import main

       test_material.write('structure.json')
       results = main()

       assert results['gap'] == pytest.approx(0)

Here we use a utility function from pytest namely ``approx`` which is
useful when two floating point numbers are to be compared.


Mocks and pytest-mock
---------------------

The previous sections mentions the concept of mocking. Mocking
involves substituting some function, class or module with a `pretend`
version which returns some artificial data that you have designed. The
kinds of functions that we would like to mock are slow function/class
calls that are not important for the test. In ASR the most important
example of a mock is the mock of the GPAW calculator which can be
found in :py:mod:`asr.test.mocks.gpaw` and is applied by the
:py:func:`asr.test.fixtures.mockgpaw` fixture.

In the beginning of the turorial, we installed ``pytest-mock`` which
is a plugin to pytest that enables easy mocking. A common use case is
to modify a certain physical property returned by the Mocked
calculator. :py:mod:`asr.test.mocks.gpaw` is designed such that you
can easily specify a band gap or a fermi level using the ``mocker``
fixture (which is provided by ``pytest-mock``), and check that the
corresponding results of your recipe are correct. For example let's
improve our ground state test by setting the band gap and Fermi level
to something non-trivial

.. code-block:: python
   :caption: In asr/asr/test/test_gs.py

   ...

   def test_gs_tutorial(asr_tmpdir_w_params, mockgpaw, mocker, test_material):
       from asr.c2db.gs import main
       from gpaw import GPAW

       mocker.patch.object(GPAW, '_get_band_gap')
       mocker.patch.object(GPAW, '_get_fermi_level')
       GPAW._get_fermi_level.return_value = 0.5
       GPAW._get_band_gap.return_value = 1

       test_material.write('structure.json')
       results = main()

       assert results['gap'] == pytest.approx(1)


As you can see in this concrete example ``mocker`` allows you to patch
objects and explicitly set the return values of the specified methods.

Parametrizing
-------------

We can improve our test even more by parametrizing over gaps and fermi
levels. The ``pytest.mark.parametrize`` decorator loops over each
entry in the supplied lists and assigns them to the specified
arguments of the test one-by-one.

.. code-block:: python
   :caption: In: asr/asr/test/test_gs.py

   ...

   @pytest.mark.parametrize('gap', [0, 1])
   @pytest.mark.parametrize('fermi_level', [0.5, 1.5])
   def test_gs_tutorial(asr_tmpdir_w_params, mockgpaw, mocker, test_material,
                        gap, fermi_level):
       from asr.c2db.gs import main
       from gpaw import GPAW

       mocker.patch.object(GPAW, '_get_band_gap')
       mocker.patch.object(GPAW, '_get_fermi_level')
       GPAW._get_fermi_level.return_value = fermi_level
       GPAW._get_band_gap.return_value = gap

       test_material.write('structure.json')
       results = main()

       assert results.get("efermi") == approx(fermi_level)
       if gap >= fermi_level:
           assert results.get("gap") == approx(gap)
       else:
           assert results.get("gap") == approx(0)

Testing web panels
------------------

To test the output of the web-panel you have implemented the
:py:func:`asr.test.fixtures.get_webcontent` fixture provides a
convenience function to return the content of your web-panel and below
we use this function to also check that the website data is consistent
with the input band gap

.. code-block:: python
   :caption: In asr/asr/test/test_gs.py

   ...

   @pytest.mark.parametrize('gap', [0, 1])
   @pytest.mark.parametrize('fermi_level', [0.5, 1.5])
   def test_gs_tutorial(asr_tmpdir_w_params, mockgpaw, mocker,
	                get_webcontent, test_material,
                        gap, fermi_level):
       from asr.c2db.gs import main
       from gpaw import GPAW

       mocker.patch.object(GPAW, '_get_band_gap')
       mocker.patch.object(GPAW, '_get_fermi_level')
       GPAW._get_fermi_level.return_value = fermi_level
       GPAW._get_band_gap.return_value = gap

       test_material.write('structure.json')
       results = main()

       assert results.get("efermi") == approx(fermi_level)
       if gap >= fermi_level:
           assert results.get("gap") == approx(gap)
       else:
           assert results.get("gap") == approx(0)

       content = get_webcontent()
       if gap >= fermi_level:
           assert f'<td>Bandgap</td><td>{gap:0.2f}eV</td>' in content
       else:
	   assert f'<td>Bandgap</td><td>0.00eV</td>' in content

Finally: Mark your test for CI execution
----------------------------------------

In software development continuous integration (CI) referes to the
practice of automatically and continuously running tests of your code
every time changes have been made. ASR utilizes Gitlab's CI runner for
this task. To register your test to be run in continuous integration
you will have to mark your test using the ``@pytest.mark.ci``
decorator. Then the test will be run along with all other tests in the
test suite when you push code to Gitlab. Mark your test with

.. code-block:: python
   :caption: In asr/asr/test/test_gs.py

   ...

   @pytest.mark.ci
   @pytest.mark.parametrize('gap', [0, 1])
   @pytest.mark.parametrize('fermi_level', [0.5, 1.5])
   def test_gs_tutorial(asr_tmpdir_w_params, mockgpaw, mocker,
	                get_webcontent, test_material,
                        gap, fermi_level):
       ...


Coverage
========

A very useful tool in guiding your focus when writing tests is how
well your tests cover your code, also known as test coverage or simply
coverage. Test coverage is usually displayed as a percentage which
represent fraction of source code that has actually been executed by
the tests. As such, coverage does not tell you anything about the
quality of the tests but it does tell you if nothing is being tested
at all!

The gitlab CI configuration (``.gitlab-ci.yml``)
includes the right incantations to generate coverage
statistics.


Summary
=======

Below you will find a list of the concepts you have been taught in
this tutorial:

  - pytest_: ``pytest.fixture``, ``pytest.mark.parametrize``,
    ``pytest.approx``, ``pytest.approx``, ``mocker``
  - ASR fixtures: ``mockgpaw``, ``asr_tmpdir_w_params``,
    ``test_material``, ``get_webcontent``


Where to go now?
================

Hopefully you will now be capable of writing and running tests for
your recipe. If you want more examples of tests we suggest looking at
the existing tests in ``asr/asr/test/test_*.py``. Additionally you can
take a look at the documentation of pytest_ itself.

.. _pytest: https://docs.pytest.org/en/latest/
