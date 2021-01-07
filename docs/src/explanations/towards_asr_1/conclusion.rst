======================
Conclusion & Questions
======================

Thanks for listening.

- The core functionality of ASR has been reimplemented.
- Dependencies are no longer given explicitly.
- ``requires`` / ``creates`` keywords have been removed.
- All recipes must take whatever relevant arguments they need. This is
  important for the caching mechanism, but also for general usability.
- A new concept of ``SideEffects`` have been introduced.
- The ``asr.database`` functionality will no longer be recipes. They
  will have their own CLI interface, ie., ``asr database fromtree .``.
- These new changes have been implemented in ``asr.gs``,
  ``asr.magstate``, ``asr.magnetic_anisotropy``, ``asr.relax``,
  ``asr.structureinfo``, ``asr.stiffness``.

These are some questions you can think of

- There is now a greater disconnect between the users and the
  data. Before it was directly visible through the files in the
  folder, but now they are hidden.
- Do we want do delete all unregistered side-effects?
- Defect guys: We need to be able to migrate your data as well. Is
  there anything special about your data compared to the C2DB data?
- Multilayer guys: Same question to you.
- As you might have noticed I changed the ``@`` to ``::`` ie
  ``results-asr.gs@calculate.json``. It is easier (for me) to type
  ``::`` but we can skip this change if you'd like.

An outlook

- Me and JJ are thinking about how workflows are supposed to work in
  this new framework. The problem is that we used to depend on the
  existence of resultfiles. But they no longer exist.
- The calculator interface is not solved yet. Ask is currently
  thinking about/working on this.
