"""Shared library code for the benchmark tools in ``scripts/``.

Nothing here is a program. The modules hold the definitions that the generator
(``scripts/bench/``) and the verifier (``scripts/eval/``) have to agree on: if "closest"
meant one thing when a question was written and another when it was audited, the audit
would be worthless.

Consumers put the ``scripts/`` directory on ``sys.path`` and import ``utils.<module>``.
That path is the same on the host and inside the ai_module container, which bind-mounts
``scripts/`` at ``/home/docker/scripts``, so a consumer gets the same geometry either way
without the repo root being mounted at all.
"""
