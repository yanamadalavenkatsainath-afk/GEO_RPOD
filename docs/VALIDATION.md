# Validation and provenance

## Evidence classes

1. Historical full single runs: seed 1 LEO and seed 1 GEO, raw trial records copied without alteration into evidence/. Both pass their saved top-level and terminal criteria. Original launch timestamps, package versions, source hashes and publication comparisons accompany each record.
2. Extraction checks: the local RPOD-only package passed ten selected component checks and four bounded far/terminal startup-and-stepping cases. Imports resolved to the extracted package. These are not end-to-end mission replays.
3. Publication checkout checks: see evidence/publication_checks.json and evidence/publication_checks.txt for the executed checks against this repository tree.

## Source comparison

The historical engine/configuration came from an isolated experimental workspace. The publication narrows five package export files, the sensor factory and the report helper. The rest of the compared original Python files are byte-identical to the historical snapshots; added entry points/tests are listed separately. No end-to-end equivalence claim is inferred from matching source hashes alone.

The original snapshots' complete source hash lists are included as provenance metadata; unrelated source code and large run artifacts are not bundled. The raw record SHA-256 values allow readers to detect changes to the supplied records. This is provenance, not an independent scientific validation.

## Acceptance interpretation

Top-level success requires far_handoff and terminal_capture. Terminal summary has its own eight criteria. Read all flags and metric definitions rather than using process exit status or the presence of a COMPLETE transition alone.

A separate legacy far evaluator can report rdv_success=false while the supervisor qualifies handoff. That difference is visible in the unmodified records. The final far summary shares the plant later used by terminal execution; its final_range is not an archived handoff measurement. Use terminal.handoff for the saved handoff state.

Reported elapsed_s excludes the initial 30-second coast; final.truth_elapsed_s includes it. Delta-v is a recorded translational diagnostic. Final contact measurements are post-attachment. Neither is a substitute for an independent acceptance assessment.

## Reproduction scope

The README provides seed-1 full replay commands for the extracted tree. They have not been executed end to end after extraction. A fresh replay should preserve its own source archive, dependencies, logs and criteria. Do not overwrite or relabel the historical results as results of a later source version.

A separately launched 300-trial local campaign is not included as completed evidence. Two favorable seed-1 cases cannot establish a success probability, broad robustness, realism of the physical assumptions or readiness for flight.
