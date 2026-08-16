# DDXPlus Runtime Metadata

`release_evidences.json` and `release_conditions.json` are metadata files from
the DDXPlus dataset. Please credit the DDXPlus authors and project when using
or redistributing these files. The files are obtained from the official DDXPlus
distribution and are released under the CC-BY license.

The large patient train, validation, and test files are intentionally not
bundled in this repository. Users who want to retrain the models must obtain
the full training data from the official DDXPlus source.

The tracked metadata is the runtime metadata used by the decoder, explanation,
severity, and display paths. It contains 223 evidence definitions and 49
condition definitions. The CSV splits are separate training/evaluation inputs
and are not required for normal inference.
