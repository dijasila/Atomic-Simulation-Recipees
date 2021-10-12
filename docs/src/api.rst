.. _API reference:

=============
API reference
=============

.. contents::
   :local:

Decorators
==========

Instruction
-----------
.. autofunction:: asr.instruction

CLI constructors
----------------
.. autofunction:: asr.option

.. autofunction:: asr.argument

Migration
---------
.. autofunction:: asr.migration

Dataclasses
===========

Record
------
.. autoclass:: asr.Record
   :members:

Run Specification
-----------------
.. autoclass:: asr.RunSpecification
   :members:

Resources
---------
.. autoclass:: asr.Resources
   :members:

Dependencies
------------
.. autoclass:: asr.Dependencies
   :members:

History
-------
.. autoclass:: asr.RevisionHistory
   :members:

Metadata
--------
.. autoclass:: asr.Metadata
   :members:


Database sub-package
====================

The database sub-package contains utilities for creating a web application that
browse multiple database projects. The basic workflow is to first create an
:class:`asr.database.DatabaseProject` with :func:`asr.database.make_project`. Then an
application can be created and started conveniently with :func:`asr.database.run_app`.
If more flexibility is needed and application object :func:`asr.database.App` can be
created manually.

Database project
----------------

.. autofunction:: asr.database.make_project_from_pyfile

.. autofunction:: asr.database.make_project

.. autoclass:: asr.database.DatabaseProject
   :members:

Run Application
---------------
.. autofunction:: asr.database.run_app


Application object
------------------
.. autoclass:: asr.database.App
   :members:


CLI
===

.. autofunction:: asr.core.cli.app