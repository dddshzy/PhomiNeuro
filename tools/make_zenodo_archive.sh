#!/usr/bin/env bash
# Build the Zenodo deposit from a CLEAN CLONE of the pushed repository.
#
# WHY A CLONE AND NOT THE WORKING TREE. The deposit must be the thing that was published, not the
# thing that happens to be on this disk. Archiving the working tree would silently pick up untracked
# files, uncommitted edits, and the 4 GB pyramid_cache/ -- and nobody would notice until a reviewer
# downloaded it. Cloning makes the archive equal to the commit by construction, and the commit hash
# goes in the manifest so the two can be compared later.
#
#   ./tools/make_zenodo_archive.sh [output_dir]
set -euo pipefail

REPO="https://github.com/dddshzy/myV18h3temp.git"
OUT="${1:-$HOME/zenodo_myV18h3temp}"
STAMP="$(date +%Y%m%d)"
NAME="phomineuro-${STAMP}"

mkdir -p "$OUT"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "cloning $REPO ..."
git clone -q "$REPO" "$WORK/$NAME"
cd "$WORK/$NAME"
COMMIT="$(git rev-parse HEAD)"
SHORT="$(git rev-parse --short HEAD)"
NFILES="$(git ls-files | wc -l)"
echo "  HEAD $SHORT, $NFILES tracked files"

# --- refuse to ship anything that must not be shipped -----------------------------------------
# Same guards as sync.sh, for the same reason: these have each caught a real mistake before.
fail=0
# Every source whose terms do not clearly permit redistribution. scatterBrains is the only one that
# does, so this list is "everything except scb". Each name here was removed after checking the
# upstream terms directly -- see DATA_CARD.md -- and the guard exists so a later well-meaning `cp`
# cannot quietly put one back.
for pat in sh001 sh027 bw14 bw15 scb15 oas; do
  if git ls-files | grep -q "$pat"; then
    echo "  FAIL  '$pat' present -- only scatterBrains-derived data may be deposited"; fail=1
  fi
done
# The bracketed letters are deliberate. Written plainly, this pattern MATCHES ITSELF: grep walks the
# repository, reaches this very file, finds the literal, and the guard fails on a clean tree. Same
# family as `pkill -f pattern` killing its own shell. The brackets change the regex not at all and
# the literal text just enough.
if grep -rlE [c]laudeAiOaut[h]|gh[p]_[A-Za-z0-9]{20,}|s[k]-ant' . \
     --exclude-dir=.git --exclude-dir=weights --exclude-dir=demo_heads 2>/dev/null | grep -q .; then
  echo "  FAIL  token-shaped string present"; fail=1
fi
for f in LICENSE NOTICE DATA_CARD.md CITATION.cff README.md; do
  [ -f "$f" ] || { echo "  FAIL  $f missing"; fail=1; }
done
[ "$fail" -eq 0 ] || { echo "ABORTED"; exit 1; }
echo "  ok    only scatterBrains-derived data, no token, all required documents present"

# --- manifest ----------------------------------------------------------------------------------
# Checksums for every file, so a downloader can verify the archive without trusting the tarball.
git ls-files | sort | xargs sha256sum > MANIFEST.sha256
{
  echo "PhomiNeuro -- Zenodo deposit"
  echo
  echo "built            : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "source           : $REPO"
  echo "commit           : $COMMIT"
  echo "tracked files    : $NFILES"
  echo
  echo "Verify with:  sha256sum -c MANIFEST.sha256"
  echo
  echo "Licensing is NOT uniform across this deposit. Code is Apache-2.0 (LICENSE); the encoder"
  echo "adapter is a derivative of NVIDIA NV-Segment-CTMR and is non-commercial; derived head"
  echo "models carry their upstream terms. NOTICE gives each one, DATA_CARD.md the provenance."
  echo
  echo "NOT included, by design:"
  echo "  - the VISTA3D backbone (872 MB, third party; see README for the download)"
  echo "  - feature pyramids (4.1 GB per head; deterministic, rebuilt in ~13 s)"
  echo "  - BrainWeb-, SHARM- and OASIS-derived head models (see DATA_CARD.md)"
} > ARCHIVE_INFO.txt

rm -rf .git
cd "$WORK"
tar czf "$OUT/${NAME}.tar.gz" "$NAME"
sha256sum "$OUT/${NAME}.tar.gz" > "$OUT/${NAME}.tar.gz.sha256"

SZ=$(du -h "$OUT/${NAME}.tar.gz" | cut -f1)
echo
echo "wrote $OUT/${NAME}.tar.gz  ($SZ)"
echo "      $OUT/${NAME}.tar.gz.sha256"
echo
echo "Upload both to Zenodo, then:"
echo "  1. paste the DOI into CITATION.cff (the commented 'doi:' line) and into the paper's"
echo "     Code/Data availability statements"
echo "  2. cite the Zenodo record in the reference list -- Springer Nature's policy states a"
echo "     GitHub link alone is not sufficient, because it assigns no permanent identifier"
