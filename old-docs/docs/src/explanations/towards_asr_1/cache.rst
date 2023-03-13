=====
Cache
=====

Caching works more or less like this

.. code-block:: python

   def call_recipe(run_specification):
       if cache.has(run_specification):
           run_record = cache.get(run_specification)
       else:
           run_record = func(run_specification)
           cache.add(run_record)
       return run_record
   

- The ``Cache`` is a kind of object that stores information about
  which functions+parameters have been evaluated, and their associated
  results, bundled into a ``RunRecord`` -object.
- The function+parameter set iself bundled into a ``RunSpecification`` -object.
- The cache is a "per folder" global object. Similar to MyQueue.
- When a function is called it looks at the current records in the
  cache to see if any of these match the requested function+parameter
  set. If it finds a match, it will simply return that record instead of
  evaluating the function again.

