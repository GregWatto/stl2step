# stl2step development

For STL-to-STEP or editable CAD reconstruction, read and follow
`.agents/skills/stl-reconstruct/SKILL.md`. This adapts the bundled Claude skill
for the current Codex session and preserves geometric verification.

Read `TESTING.md` for the reproducible test pipeline. Run dependency checks and
existing regression cases before changing reconstruction algorithms; report
failures first. Keep failures visible rather than relaxing tolerances, bypassing
feature/occupancy/opening checks, or accepting a STEP just because it exports.
Keep generated artifacts under `out/`. Do not overwrite committed STL fixtures.
