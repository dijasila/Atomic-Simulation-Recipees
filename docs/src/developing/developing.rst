Development
===========

.. toctree::
   :maxdepth: 1
   :hidden:

   intro-to-testing
   towards_asr_1/index

Run tests
---------

.. code-block:: console

   $ tox -e flake8 doctest py36 py36-gpaw

See :ref:`Testing tutorial` for a tutorial on writing tests in ASR.

Build documentation
-------------------

Build documentation:

.. code-block:: console

   $ tox -e docs

Update the shell code output of various .rst files:

.. code-block:: console

   $ tox -e updatedocs

For an explanation on how the division of the documentation into tutorials,
how-to guides, explanations and API reference should be understood, see Daniele
Procida's PyCon 2017 talk "`How documentation works...
<https://www.youtube.com/watch?v=azf6yzuJt54>`_".

Make a new release
------------------

Preferably make a new branch

.. code-block:: console

   $ git checkout -b release-VERSION

Make a new version:

  Make a minor release

  .. code-block:: console

     $ bumpversion minor

  Make a major release

  .. code-block:: console

     $ bumpversion major

Upload to PyPI

.. code-block:: console

   $ tox -e release

Towards ASR 1.0
---------------

In January 2021 we had a presentation about the direction that ASR is taking.
See the presentation here: :ref:`Towards ASR1`.