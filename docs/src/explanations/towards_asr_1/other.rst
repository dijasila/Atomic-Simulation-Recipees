=============
Other changes
=============


- The asr database API will change. They will no longer be recipes.

  .. code-block:: console

     $ asr run "database.fromtree ."

     Becomes

     $ asr database fromtree .

     Similarly

     $ asr database app database.db
     $ asr database totree database.db


