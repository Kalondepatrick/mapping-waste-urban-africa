# 08 — Annotation and review tools

The tooling used to digitise and verify waste polygons. Not needed to reproduce
the models — needed to reproduce the *labels*.

| item | what it is |
|---|---|
| `waste_review_app.R` | Shiny application for reviewing candidate waste objects and recording verdicts. The final of twelve iterations; the others are in `../NotNeeded/review_app_versions/`. |
| `shiny_app_legacy.R` | earlier standalone Shiny app |
| `qgis_plugin_waste_review/` | the project's own QGIS plugin, versions 0.1.2 to 0.3.1 |
| `qgis_plugins_external/` | third-party plugins used during digitising (GeoOSAM, guided digitizer) |

The review app reads and writes `data/annotations/waste_review_app/`: preview
chips, the review grid, the object-chip map and `review_log.sqlite`. Install the
newest QGIS plugin zip through Plugins > Manage and Install > Install from ZIP.

A QGIS project wired to the main layers is at
`../tools/qgis_project_wastemapping.qgz`.
