# Raw adversarial-review journals, 2026-09-01

Preserved because a workflow's summary result can be **empty when every agent
died**, and because these journals are the only per-agent record of findings and
adversarial verdicts. Session storage is ephemeral; these are not.

- `wf-batch1-20260901.jsonl` — the four highest-risk dimensions of the beta96
  long-step change. 14 findings; `travel-gate-logic` fully verified,
  `linear-300-plumbing` and `arithmetic-soundness` findings produced but their
  verifiers all died on a spend limit, `long-window-downstream` never ran.
  Conclusions written up in `docs/SESSION-STATE-20260901-2000.md` §6-§7 and
  `docs/findings-2a-cannot-be-fixed-by-a-longer-step-20260901.md`.
- `wf-remaining-dims-20260901.jsonl` — the five never-reviewed dimensions against
  the current tree. ⚠️ **Launched but interrupted at a session limit; all five
  agents had STARTED and none returned.** Zero results. **These five dimensions
  remain unreviewed:** long-window-downstream, travel-guard-integrity,
  evio-scoring-larger-n, schema-yaml-consistency, regression-risk.

⚠️ **An empty result here means "nothing ran", NEVER "clean".** Check
`"type":"result"` line counts before drawing any conclusion from these files.
