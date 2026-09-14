#!/usr/bin/env python3
"""V17 ENTRY-POINT features: recompute every source-dependent feature of an existing buffer with the
reference point moved from the (external) source point to the BEAM ENTRY POINT on the scalp.

WHY. `illumination.SRC_R_MARGIN = 15` parks the source 15 voxels OUTSIDE the scalp (mirroring the mcx
disc-source setup). Measured consequences, all with the source point as the feature reference:
  * the first ~15 mm of every ray is air -- 17.1% of all ray samples on sh001 -- and the CSF test
    `(mua<0.005) & (musp<0.5)` is TRUE for air (mua=0, musp=0), so 53-63% of the L_csf feature is
    air, not CSF (sh001 19.6/31.2 mm, bw14 62%, scb15 53%).
  * the surrogate acquires a non-physical sensitivity to sliding the source along its own axis
    (-0.129 decade/mm; a collimated beam must give exactly 0), which combined with the integer-voxel
    output of find_source_position turns J(A,B) into a 0.26-decade sawtooth.
The entry point does not depend on where along the beam axis the source sits, so referencing it makes
dJ/d(radial) identically zero BY CONSTRUCTION, and rays from it never cross air -- which removes the
L_csf contamination without touching the is_csf predicate.

WHAT IS EXACT HERE. entry = srcpos + SRC_R_MARGIN * srcdir (srcdir points INTO the head, = -n), so no
new ray-cast is needed: srcpos and srcdir are both already in each scene's meta. And LIGHT is
[srcpos_norm(3), srcdir(3), ang(4)] -- only the first three columns move, so the rest is COPIED from
the existing buffer rather than recomputed, which rules out any drift in the angle convention.

Writes side files next to the buffer; meta.pt is never modified, so the change is fully reversible:
    SF_entry.npy      (total,2)     LIGHT_entry.npy    (total,10)    PATH_entry.npy   (total,6)
    P_SF_entry.npy    (npin,7,2)    P_LIGHT_entry.npy  (npin,10)     P_PATH_entry.npy (npin,7,6)

  INR_SPLIT=v16_split INR_K=1000 python make_entryfeat.py inr_checkpoints_v11/bufmm_v16_k1000 --gpu 0
  ... --check-only          # run the pre-flight diagnostics on a few scenes, write nothing
"""
import os, sys, json, argparse
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "data_expansion"))
import train_inr_v11 as T
from train_inr_v3 import xyz_to_norm
from make_pathfeat import path_features, PATHDIM, NS
import illumination as ILL

K = int(os.environ.get("INR_K", 1000))
NSHARD = 8
MARGIN = ILL.SRC_R_MARGIN                    # 15 -- imported, never hard-coded


def replay(total):
    """The buffer's scene order: non-test, geom_key-sorted, sharded as heads[i % 8]."""
    scenes = [s for s in T.discover_scenes() if T.label_of(s["head"]) != "test"]
    scenes.sort(key=lambda s: s["geom_key"])
    heads = sorted(set(s["head"] for s in scenes))
    order, torder = [], []
    for sh in range(NSHARD):
        mine = {h for i, h in enumerate(heads) if i % NSHARD == sh}
        sh_scenes = [s for s in scenes if s["head"] in mine]
        order += sh_scenes
        torder += [s for s in sh_scenes if T.label_of(s["head"]) == "train"]
    assert len(order) * K == total, f"replay mismatch: {len(order)}*{K} != {total}"
    return order, torder


def entry_of(srcpos, srcdir, occ=None, vs=None, snap=True, max_step=40):
    """Beam entry point on the scalp. srcdir points INTO the head, so stepping MARGIN along it from
    the source undoes exactly the standoff find_source_position added.

    That lands one voxel SHORT of the tissue, because find_source_position stops at the FIRST AIR
    voxel and then adds the margin -- so srcpos - MARGIN is that air voxel, not the surface. With
    snap=True we walk the remaining sub-voxel gap: advance along srcdir in 0.5-voxel steps to the
    first occupied voxel. Cheap (a handful of steps) and makes the ray start IN tissue, which is the
    whole point of the reparameterisation.

    Walking must go BOTH ways. The forward-only version was invariant when the source moved AWAY from
    the head but drifted by |dr|-0.5 when it moved closer, because a source at a smaller standoff puts
    srcpos-MARGIN already inside the tissue and a forward-only walk then returns that interior point.
    A source at 11 mm standoff instead of 15 is perfectly physical, so the invariance has to hold
    there too. Note this changes NOTHING for the buffers: dataset srcpos is always at the nominal
    standoff, so e always starts in air and the backward branch never fires."""
    e = srcpos + MARGIN * srcdir
    if not snap or occ is None:
        return e

    def inside(p):
        i = p.round().long()
        if not bool((i >= 0).all() and (i < vs.long()).all()):
            return False
        return bool(occ[int(i[0]), int(i[1]), int(i[2])])

    if inside(e):                                    # started in tissue -> back out to the surface
        for k in range(1, max_step):
            p = e - 0.5 * k * srcdir
            if not inside(p):
                return p + 0.5 * srcdir              # first tissue point coming back in
        return e
    for k in range(max_step):                        # started in air -> advance to the surface
        p = e + 0.5 * k * srcdir
        i = p.round().long()
        if not bool((i >= 0).all() and (i < vs.long()).all()):
            break
        if inside(p):
            return p
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bufdir")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--nseg", type=int, default=0)
    ap.add_argument("--suffix", default="_entry")
    ap.add_argument("--nearest", action="store_true",
                    help="sample the path with the legacy nearest gather instead of trilinear "
                         "(trilinear is the default here: V17 wants a differentiable path)")
    ap.add_argument("--check-only", action="store_true",
                    help="pre-flight diagnostics on --check-n scenes, write nothing")
    ap.add_argument("--check-n", type=int, default=6)
    ap.add_argument("--no-snap", action="store_true",
                    help="do NOT snap the entry point onto the first tissue voxel (leaves it on the "
                         "air voxel find_source_position stopped at)")
    a = ap.parse_args()
    tri = not a.nearest
    PD = PATHDIM + a.nseg
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")

    meta = torch.load(os.path.join(a.bufdir, "meta.pt"), map_location="cpu")
    total = meta["meta"]["total"]; npin = meta["meta"]["npin"]
    XN = meta["XN"]; LIGHT = meta["LIGHT"]
    PXN = meta["P"]["XN"] if npin else None
    PLIGHT = meta["P"]["LIGHT"] if npin else None
    order, torder = replay(total)
    PER = npin // len(torder) if torder else 0
    print(f"[entryfeat] {len(order)} scenes x K={K} = {total} | {len(torder)} train x PER={PER} = {npin}"
          f" | margin={MARGIN} | path sampling={'trilinear' if tri else 'nearest'}", flush=True)

    import scipy.io as sio
    cache = {"key": None}

    def geom(sc):
        if cache["key"] != sc["geom_key"]:
            p = sio.loadmat(sc["prop"])[T.PROP_KEY].astype(np.float32)
            p = np.transpose(p, (3, 0, 1, 2))
            cache["prop"] = torch.from_numpy(p).unsqueeze(0).to(dev)
            cache["vs"] = torch.tensor(cache["prop"].shape[-3:], device=dev)
            cache["occ"] = (cache["prop"][0, 1] > 0)
            cache["key"] = sc["geom_key"]
        return cache["prop"], cache["vs"], cache["occ"]

    # ---------------- pre-flight diagnostics (S3) ----------------
    if a.check_only:
        print(f"\n{'head/tag':<16}{'r_src':>9}{'r_ent':>9}{'Δr均':>8}{'Δr标准差':>10}"
              f"{'空气%src':>10}{'空气%无snap':>10}{'空气%snap':>10}"
              f"{'Lcsf_src':>11}{'Lcsf无snap':>11}{'Lcsf_snap':>11}{'在组织':>10}")
        for sc in order[::max(1, len(order) // a.check_n)][:a.check_n]:
            prop, vs, occ = geom(sc)
            m = json.load(open(sc["meta"]))
            srcpos = torch.tensor(m["srcpos"], dtype=torch.float32, device=dev)
            srcdir = torch.tensor(m["srcdir"], dtype=torch.float32, device=dev)
            ent0 = entry_of(srcpos, srcdir, snap=False)
            ent = entry_of(srcpos, srcdir, occ, vs, snap=True)
            i = order.index(sc)
            xn = XN[i * K:(i + 1) * K].to(dev)
            xyz = (xn + 1.0) * 0.5 * (vs.float().view(1, 3) - 1.0)

            def air_and_lcsf(origin):
                s = torch.linspace(0., 1., NS, device=dev).view(1, NS, 1)
                pts = origin.view(1, 1, 3) + s * (xyz.view(-1, 1, 3) - origin.view(1, 1, 3))
                ii = pts.round().long()
                for d in range(3):
                    ii[..., d].clamp_(0, int(vs[d]) - 1)
                P = prop[0][:, ii[..., 0], ii[..., 1], ii[..., 2]]
                mua, musp = P[0], P[1] * (1 - P[2])
                air = (P[1] <= 0)
                r = (xyz - origin.view(1, 3)).norm(dim=1); ds = (r / (NS - 1)).view(-1, 1)
                lcsf = (((mua < 0.005) & (musp < 0.5)).float() * ds).sum(1)
                return float(air.float().mean() * 100), float(lcsf.mean()), float(r.mean())

            a_s, l_s, r_s = air_and_lcsf(srcpos)
            a_0, l_0, _ = air_and_lcsf(ent0)
            a_e, l_e, r_e = air_and_lcsf(ent)
            dr = (xyz - ent.view(1, 3)).norm(dim=1) - (xyz - srcpos.view(1, 3)).norm(dim=1)
            ei = ent.round().long().clamp_min(0)
            on_tissue = bool(occ[int(ei[0]), int(ei[1]), int(ei[2])])
            print(f"{sc['head']+'/'+sc['tag']:<16}{r_s:>9.1f}{r_e:>9.1f}{float(dr.mean()):>8.2f}"
                  f"{float(dr.std()):>10.2f}{a_s:>10.1f}{a_0:>10.1f}{a_e:>10.1f}"
                  f"{l_s:>11.2f}{l_0:>11.2f}{l_e:>11.2f}{str(on_tissue):>10}")
        print("\n  期望: 空气%(ent)≈0, L_csf(ent) 显著小于 L_csf(src), Δr标准差>0 说明偏移非常数")
        return

    # ---------------- build ----------------
    SFo = np.lib.format.open_memmap(os.path.join(a.bufdir, f"SF{a.suffix}.npy"), mode="w+",
                                    dtype=np.float32, shape=(total, 2))
    LIo = np.lib.format.open_memmap(os.path.join(a.bufdir, f"LIGHT{a.suffix}.npy"), mode="w+",
                                    dtype=np.float32, shape=(total, LIGHT.shape[1]))
    PAo = np.lib.format.open_memmap(os.path.join(a.bufdir, f"PATH{a.suffix}.npy"), mode="w+",
                                    dtype=np.float16, shape=(total, PD))
    if npin:
        PSFo = np.lib.format.open_memmap(os.path.join(a.bufdir, f"P_SF{a.suffix}.npy"), mode="w+",
                                         dtype=np.float32, shape=(npin, 7, 2))
        PLIo = np.lib.format.open_memmap(os.path.join(a.bufdir, f"P_LIGHT{a.suffix}.npy"), mode="w+",
                                         dtype=np.float32, shape=(npin, PLIGHT.shape[1]))
        PPAo = np.lib.format.open_memmap(os.path.join(a.bufdir, f"P_PATH{a.suffix}.npy"), mode="w+",
                                         dtype=np.float16, shape=(npin, 7, PD))

    tset = {id(s): i for i, s in enumerate(torder)}
    n_off_tissue = 0
    with torch.no_grad():
        for n, sc in enumerate(order):
            prop, vs, occ = geom(sc)
            m = json.load(open(sc["meta"]))
            srcpos = torch.tensor(m["srcpos"], dtype=torch.float32, device=dev)
            srcdir = torch.tensor(m["srcdir"], dtype=torch.float32, device=dev)
            ent = entry_of(srcpos, srcdir, occ, vs, snap=not a.no_snap)
            ei = ent.round().long().clamp(torch.zeros(3, dtype=torch.long, device=dev),
                                          vs.long() - 1)
            if not bool(occ[int(ei[0]), int(ei[1]), int(ei[2])]):
                n_off_tissue += 1
            ent_norm = (ent / (vs.float() - 1.0)).cpu()

            lo = n * K
            xn = XN[lo:lo + K].to(dev)
            xyz = (xn + 1.0) * 0.5 * (vs.float().view(1, 3) - 1.0)
            SFo[lo:lo + K] = T.src_features(xyz, ent, srcdir).cpu().numpy()
            li = LIGHT[lo:lo + K].clone()                 # keep srcdir + ang EXACTLY as built
            li[:, 0:3] = ent_norm.view(1, 3)
            LIo[lo:lo + K] = li.numpy()
            PAo[lo:lo + K] = path_features(xyz, ent, prop, vs, a.nseg, tri).half().cpu().numpy()

            j = tset.get(id(sc))
            if npin and j is not None:
                ps = slice(j * PER, (j + 1) * PER)
                pxyz = ((PXN[ps].to(dev).reshape(-1, 3) + 1.0) * 0.5
                        * (vs.float().view(1, 3) - 1.0))
                PSFo[ps] = T.src_features(pxyz, ent, srcdir).view(PER, 7, 2).cpu().numpy()
                pli = PLIGHT[ps].clone(); pli[:, 0:3] = ent_norm.view(1, 3)
                PLIo[ps] = pli.numpy()
                PPAo[ps] = path_features(pxyz, ent, prop, vs, a.nseg, tri).view(PER, 7, PD).half().cpu().numpy()
            if n % 1000 == 0:
                print(f"  scene {n}/{len(order)}", flush=True)

    for o in ([SFo, LIo, PAo] + ([PSFo, PLIo, PPAo] if npin else [])):
        o.flush()
    print(f"done -> {a.bufdir}/*{a.suffix}.npy | entry off-tissue in {n_off_tissue}/{len(order)} scenes",
          flush=True)


if __name__ == "__main__":
    main()
