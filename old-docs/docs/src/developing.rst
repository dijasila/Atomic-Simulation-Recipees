Development
===========


Run tests
---------

.. code-block::

   $ pytest asr/test


Build documentation
-------------------

Build documentation:

.. code-block::

   $ sphinx-build docs/ docs/build/

Make a new release
------------------

Preferably make a new branch

.. code-block:: console

   $ git checkout -b release-DATE

Make a new version:

  Make a minor release

  .. code-block:: console

     $ bumpversion minor

  Make a major release

  .. code-block:: console

     $ bumpversion major

Upload to PyPI using twine.
