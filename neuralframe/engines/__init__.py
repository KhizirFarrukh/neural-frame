"""Generative and face restoration engines.

These are research projects with their own (often conflicting) Python dependencies - diffusion and
transformer video models such as SeedVR2, FlashVSR, DOVE and SwiftVR, picture models such as OSEDiff,
HYPIR, SUPIR, LucidFlux and InvSR, NVIDIA RTX Video Super Resolution, and face restorers (GFPGAN,
RestoreFormer++, CodeFormer, KEEP). Each one runs in **its own environment** under
``<cache>/engines/<id>/``, created only when a user chooses it:

* ``catalog.py`` - what each engine needs: pinned upstream source (GitHub commit), Python version, PyTorch
  build, packages (training-only or Windows-incompatible requirements filtered out), our driver scripts;
* ``env.py`` - installs an engine with a pinned ``uv`` (which also provides the Python interpreter), with
  progress, cancellation and removal;
* ``adapters.py`` - turns "restore these frames to this size" into the engine's own command line;
* ``model.py`` - :class:`GenerativeModel`, which the temporal stage drives like a sequence model: each
  chunk's frames are written out, the engine runs, its frames are read back; the job then resizes them to
  the exact target, encodes them and muxes the original audio as for any other model;
* ``faces.py`` - an optional stage that restores faces in the finished frames.

Model weights are downloaded by the model library (progress, resume, checksums) where the files are
public; engines that fetch their own weights (gated or not published as plain files) do so during
installation.
"""
