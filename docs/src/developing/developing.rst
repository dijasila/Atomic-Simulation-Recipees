Development
===========


Run tests
---------

.. code-block::

   $ tox -e flake8 doctest py36 py36-gpaw


Build documentation
-------------------

Build documentation:

.. code-block::

   $ tox -e docs

Update the shell code output of various .rst files:

.. code-block::

   $ tox -e updatedocs

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
