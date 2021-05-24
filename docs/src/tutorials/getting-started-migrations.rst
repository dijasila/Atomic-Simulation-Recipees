.. _Getting started:

=====================================
Getting started - part 2 - migrations
=====================================

In this part of the getting started series we are going to discuss migrations.
It often happens that you want/have to make changes to an existing instruction.
For example, you want to add an additional argument, change the return type of
the result, change the implementation which requires thinking about what should
happen to existing Records in the cache.

ASR implements a revisioning system for Records for handling this problem which
revolves around defining functions (migrations) for updating existing records
to be compatible with the newest implementation of the instructions.

In the following we will continue with the example of calculating the most
stable crystal structure.
