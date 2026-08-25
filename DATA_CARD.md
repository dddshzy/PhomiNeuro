# Data card — what ships, where it came from, and what has to change before publication

Provenance and terms for every non-code artefact. Licence text and citations are in
[`NOTICE`](NOTICE); this file is the decision record: what was checked, what was found, and what
follows from it.

**Checked on 2026-08-13 by reading each source directly.** Two of the four upstream sources turned
out to have no licence statement at all — which is a finding, not a gap in the search. The SHARM
heads were removed the same day as a result; see below.

---

## Summary

| Artefact | Upstream | Upstream terms | Redistributable? |
|---|---|---|---|
| ~~`bw14`, `bw15`~~ | BrainWeb (McGill BIC) | **none stated**; citation required | ❌ **removed 2026-08-14** |
| `scb16` | scatterBrains | **scatterBrains License Agreement** (BSD-style) | ✅ yes, with the notice |
| ~~`sh001`, `sh027`~~ | SHARM → IXI | SHARM **none stated**; IXI **CC BY-SA 3.0** | ❌ **removed 2026-08-13** |
| (not shipped) | OASIS-3 | Data Use Agreement | ❌ no |
| `mc_reference/` | our mcx run on `scb16` | inherits scatterBrains | ✅ yes |
| `weights/inr_*` | ours | CC BY 4.0 | ✅ yes |
| `weights/fm_encoder/` | derivative of NV-Segment-CTMR | **NVIDIA OneWay Non-Commercial** | ⚠️ non-commercial only |

## What each finding rests on

**BrainWeb — no licence found, so it was dropped.** The BrainWeb site states a citation requirement
and nothing else; no copyright notice, licence or terms-of-use statement appears on the main page or
the data-format pages. Absence of a licence is not a permissive licence: the default is reserved
rights. BrainWeb derivatives have circulated for over two decades, but established practice is not
consent. Rather than depend on a permission that might not arrive, `bw14`, `bw15` and the `bw14`
Monte-Carlo field were removed on 2026-08-14.

**scatterBrains — clearly redistributable.** `LICENSE.txt` in the project repository is a BSD-style
"scatterBrains License Agreement" (Copyright © 2023 Melissa M. Wu, Stefan A. Carp) which permits
redistribution in source and binary form provided the copyright notice is retained. `NOTICE` retains
it. This is the only upstream source that unambiguously allows what this repository does.

**SHARM — two independent blockers.** First, the dataset is distributed through
`figshare.com/s/a4d9ba6f18a6b7f7ba2c`, a figshare **private share link** rather than a published item
with a DOI and a licence field; the paper calls the data "open-access" but names no licence. Second,
SHARM is generated from IXI, and IXI states: *"This data is made available under the Creative Commons
CC BY-SA 3.0 license."* ShareAlike propagates to adaptations, and our optical-property volumes are
adaptations. Either blocker alone is enough.

**OASIS-3.** Its Data Use Agreement restricts sharing derived data. Nothing OASIS-derived is here.

## Actions required before this repository becomes public

This is a **private staging repository**, so nothing has been distributed yet and the items below are
open, not breached. They are ordered by what blocks what.

**DONE — `sh001` and `sh027` removed (2026-08-13).** Deleted from the working tree and the history
rewritten with `git filter-repo`, then force-pushed. One residual to be aware of: GitHub keeps
unreferenced objects reachable by SHA until it garbage-collects, so on a repository that had ever
been public the only complete remedy is to delete and recreate it. This one has been private
throughout, so the exposure is nil.

**DECIDED — the release is academic-only.** No commercial licence will be sought from NVIDIA, and
the non-commercial restriction is accepted and documented rather than worked around.

**DONE — BrainWeb removed (2026-08-14).** `bw14`, `bw15` and the `bw14` Monte-Carlo field deleted and
the history rewritten. The demo moved to scatterBrains, whose licence explicitly permits
redistribution of derived models. **No permission request is outstanding: every shipped file is
under terms that allow redistribution.**

**Scope, set deliberately (2026-08-14).** This repository ships **one** head, `scb16`, with one
Monte-Carlo reference. It is a demonstration for the community — install it, run it, see the model
work. It is not an evaluation artefact, and the single shipped scene should not be read as one. The
evaluation is in the paper; the full head set goes to reviewers through a separate channel.

Still open:

1. **Deposit in Zenodo and cite the DOI.** Springer Nature's code policy states that *"providing a
   GitHub link only is not sufficient as it does not assign a permanent identifier"* — code must be
   deposited somewhere that mints one, and cited in the reference list.
3. **Optional: an unencumbered variant.** The no-pyramid arm needs no third-party weights and no
   872 MB download, at a measured cost of ΔR² = **−0.067** on the held-out set (−0.113 on dev-test).
   Not required now that academic-only is the chosen scope, but it is the only route to a version
   others could use commercially. The existing arm was trained at K=1000 and would need a K=3000 run
   to be a release artefact.

## Regenerating anything not shipped

`docs/HEAD_MODELS.md` documents the full chain from each of the four public sources to the
`(224, 256, 300, 4)` optical-property volumes, including the shared rigid-MNI step and the label →
optical-property mapping. Nothing about the pipeline is withheld; only the derived files are.
