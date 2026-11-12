# Component provenance and notices

This repository packages project-specific annotation code and adapters together with a selected historical preprocessing subset. It does not redistribute model weights, captured videos, real annotation datasets, or the OpenVINS / Fast-FoundationStereo source repositories.

- The DAAAM / egomemory-core preprocessing subset retains its existing BSD 3-Clause notice in [components/preprocessing/LICENSE](components/preprocessing/LICENSE), including the original Massachusetts Institute of Technology copyright. Local additions and portability edits are recorded in provenance/source_manifest.json and docs/PORTABILITY.md.
- OpenVINS is an external build dependency. Its source and license remain in the upstream checkout. The included compatibility patch records local changes; it does not replace upstream notices.
- Fast-FoundationStereo is an external source and checkpoint dependency. Its source and checkpoint terms remain applicable at their original distribution locations.
- Grounding DINO, FastSAM, Transformers, Ultralytics, PyTorch and other packages remain separate dependencies with their own notices. Their weights are excluded from this repository.
- The remaining semantic, annotation, additive and anchor modules were collected from the local project sources listed in the source manifest. This packaging operation does not assign a new repository-wide license or assert that every component shares the preprocessing license.

Recorded upstream revisions and model provenance are in provenance/external_dependencies.json. No keys or licensed model artifacts are embedded in those records.
