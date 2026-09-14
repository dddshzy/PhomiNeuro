#!/usr/bin/env python3
"""Pre-flight for the input-aligned coordinate baselines. Run BEFORE training.

The claim being made in the paper is "the baselines see the same physical inputs as ours", and until
now that was false in BOTH directions: the coord baselines carried optical(4) and the full light(10)
that hero3 does not take, while their time input was a RAW SCALAR against hero3's 7-dim band-limited
PE. This checks the fix and, just as importantly, that nothing already scored has moved.

  python test_coord_align.py
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import make_coord_model

CK3 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inr_checkpoints_v3")
ok_all = True


def rep(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def first_linear(m):
    return next(x for x in m.modules() if isinstance(x, torch.nn.Linear))


print("T1  existing checkpoints rebuild and load unchanged (drop_feats absent -> ())")
for tag, want in (("inr_v17tr_coord_rff_s1.pt", 279), ("inr_v17tr_coord_siren_s1.pt", 26)):
    p = os.path.join(CK3, tag)
    if not os.path.exists(p):
        rep(tag, False, "missing"); continue
    ck = torch.load(p, map_location="cpu")
    m = make_coord_model(ck["arch"], path_dim=int(ck.get("path_dim", 0)),
                         time_dim=int(ck.get("time_dim", 0)), num_freq_t=ck.get("num_freq_t"),
                         drop_feats=ck.get("drop_feats", ()))
    m.load_state_dict(ck["model"])              # strict: a width change would raise here
    got = first_linear(m).in_features
    rep(tag, got == want and m.drop_feats == (), f"in_features={got} (want {want}) drop={m.drop_feats}")

print("\nT2  the aligned build has the intended width")
# hero3 non-pyramid = pos 27 + pos_t 7 + optical 0 + light 6 + srcfeat 2 + path 6 = 48
# aligned rff   = RFF 256 + light 6 + srcfeat 2 + path 6 + tPE 7 = 277
# aligned siren = xyz  3   + light 6 + srcfeat 2 + path 6 + tPE 7 =  24
for arch, want, lab in (("rff", 277, "Coord-RFF aligned"), ("siren", 24, "SIREN aligned")):
    m = make_coord_model(arch, path_dim=6, time_dim=1, num_freq_t=3,
                         drop_feats=("optical", "light:6:10"))
    got = first_linear(m).in_features
    rep(lab, got == want, f"in_features={got} (want {want})")
# and the OLD width, for the record
for arch, want, lab in (("rff", 279, "Coord-RFF as published"), ("siren", 26, "SIREN as published")):
    m = make_coord_model(arch, path_dim=6, time_dim=1, num_freq_t=None)
    rep(lab, first_linear(m).in_features == want, f"in_features={first_linear(m).in_features}")

print("\nT3  the dropped columns cannot influence the output, and the kept ones are the right ones")
for arch in ("rff", "siren"):
    m = make_coord_model(arch, path_dim=6, time_dim=1, num_freq_t=3,
                         drop_feats=("optical", "light:6:10")).eval()
    g = torch.Generator().manual_seed(0)
    n = 256
    xn = torch.rand(n, 3, generator=g) * 2 - 1
    opt = torch.rand(n, 4, generator=g); light = torch.rand(n, 10, generator=g)
    sf = torch.rand(n, 2, generator=g); path = torch.rand(n, 6, generator=g) * 50
    t = torch.rand(n, 1, generator=g) * 2 - 1
    with torch.no_grad():
        y1 = m.forward_feats(None, xn, opt, light, sf, path=path, t=t)
        o2 = torch.randn_like(opt) * 10
        l2 = light.clone(); l2[:, 6:10] = torch.randn_like(l2[:, 6:10]) * 10
        y2 = m.forward_feats(None, xn, o2, l2, sf, path=path, t=t)
        l3 = light.clone(); l3[:, 0:6] = torch.randn_like(l3[:, 0:6]) * 10   # KEPT -> must matter
        y3 = m.forward_feats(None, xn, opt, l3, sf, path=path, t=t)
    rep(f"{arch}: dropped columns inert", torch.equal(y1, y2),
        f"|dy|max={(y1-y2).abs().max().item():.3e}")
    rep(f"{arch}: kept light columns still active", (y1 - y3).abs().max().item() > 1e-6,
        f"|dy|max={(y1-y3).abs().max().item():.3e}")

print("\nT4  the temporal encoding is the 7-dim band-limited PE, not a raw scalar")
m_old = make_coord_model("rff", path_dim=6, time_dim=1, num_freq_t=None)
m_new = make_coord_model("rff", path_dim=6, time_dim=1, num_freq_t=3)
rep("published baseline tenc.out_dim == 1 (raw scalar t)", m_old.tenc.out_dim == 1,
    f"{m_old.tenc.out_dim}")
rep("aligned baseline tenc.out_dim == 7 (f = 1,2,4)", m_new.tenc.out_dim == 7,
    f"{m_new.tenc.out_dim}")

print("\n" + ("ALL TESTS PASSED" if ok_all else "SOME TESTS FAILED"))
sys.exit(0 if ok_all else 1)
