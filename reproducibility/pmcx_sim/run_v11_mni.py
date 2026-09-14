#!/usr/bin/env python3
"""
v11 time-resolved MCX on the RIGID-MNI-normalized phantoms (dataset_v11_mni/).

Each phantom is already in the common MNI grid (224,256,288), so:
  - O = AC voxel (FIXED, same for every head) = mni_normalize.AC,
  - the 19 EEG 10-20 source directions are FIXED (inward = -normalize(MNI coords)),
  - per head, each direction is snapped to that head's scalp by the existing
    find_source_position (standoff in air), then time-resolved MCX is run
    (2 ns window / N_STEP gates, 810 nm), reusing the discrete-media core.

Examples
  python run_v11_mni.py --head sh001 --electrodes Cz O1 T3 --nphoton 1e7
  python run_v11_mni.py --head scb01 --nphoton 5e7            # all 19
  python run_v11_mni.py --heads sh001 bw01 scb01 --nphoton 5e7
"""
import os, sys, json, time, argparse
import numpy as np
import scipy.io as sio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data_expansion"))
sys.path.insert(0, os.path.join(ROOT, "pmcx_sim"))
import repro_config as RC

import mcx_fluence_simv2 as S
import mni_normalize as MN

PROP_KEY = MN.PROP_KEY
WL = "810"
RADIUS_MM = 5.0
WINDOW_DEFAULT = 2e-9
N_STEP_DEFAULT = 10
SEED = 29012392
V11_ROOT = os.path.dirname(os.path.dirname(RC.MNI_SIM_DIR))
V11_DATA = os.path.join(V11_ROOT, "data", "wl810")
V11_CACHE = os.path.join(V11_ROOT, "cache")


def get_cfg(head, force_rebuild=False):
    """Load a cached discretized MCX volume keyed by its source phantom.

    The cache records the source file's modification time and size and rebuilds
    whenever either value changes.
    """
    os.makedirs(V11_CACHE, exist_ok=True)
    path = os.path.join(V11_CACHE, f"{head}_cfg.npz")
    matp = os.path.join(MN.OUT_DATASET, f"{head}_v11mni_F810.mat")
    if not os.path.isfile(matp):
        raise FileNotFoundError(f"normalized phantom missing: {matp} (run mni_normalize --normalize {head})")
    st = os.stat(matp)
    key = np.array([int(st.st_mtime), int(st.st_size)], dtype=np.int64)

    if os.path.isfile(path) and not force_rebuild:
        d = np.load(path)
        if "src_key" in d and np.array_equal(d["src_key"], key):
            return d["cfg_vol"], d["cfg_prop"]
        print(f"  [cache STALE] {head}: phantom changed since the cached volume was built "
              f"-> rebuilding", flush=True)

    vol4d = sio.loadmat(matp)[PROP_KEY].astype(np.float64)
    mask = S.build_filled_mask(vol4d)
    cfg_vol, cfg_prop, _ = S.build_cfg_discrete(vol4d, mask, WL)
    np.savez_compressed(path, cfg_vol=cfg_vol, cfg_prop=cfg_prop, src_key=key)
    return cfg_vol, cfg_prop


def inward_dirs():
    """19 fixed inward srcdirs (toward AC) = -normalize(MNI electrode coords)."""
    M = MN._load_mni()
    return {n: -M["dirs"][n] for n in MN.NAMES}


# ---- canonical anatomical axes (voxel space): +X=right, +Y=anterior, +Z=superior
F_ANT = np.array([0.0, 1.0, 0.0]); U_SUP = np.array([0.0, 0.0, 1.0]); R_RIGHT = np.array([1.0, 0.0, 0.0])


def _outward_to_AB(n):
    """Outward scalp normal (AC->electrode) -> anatomical (A elevation, B azimuth) deg."""
    n = n / np.linalg.norm(n)
    A = np.degrees(np.arcsin(np.clip(n @ U_SUP, -1, 1)))
    B = np.degrees(np.arctan2(n @ R_RIGHT, n @ F_ANT))
    return float(A), float(B)


def AB_to_outward(A_deg, B_deg):
    """Inverse of _outward_to_AB: (elevation, azimuth) in degrees -> outward unit direction.

    A = asin(n.Z) and B = atan2(n.X, n.Y), so n = cosA sinB X + cosA cosB Y + sinA Z. The
    round trip is exact and is asserted at start-up rather than trusted.
    """
    a, b = np.radians(float(A_deg)), np.radians(float(B_deg))
    n = np.cos(a) * np.sin(b) * R_RIGHT + np.cos(a) * np.cos(b) * F_ANT + np.sin(a) * U_SUP
    return n / np.linalg.norm(n)


def ab_tag(A_deg, B_deg):
    """Filesystem- and glob-safe scene tag. p/m instead of +/- so the tag sorts and never needs
    quoting; the authoritative angles are the A_deg/B_deg fields written into the meta json."""
    s = lambda v: ("p" if v >= 0 else "m") + f"{abs(int(round(v))):0{3 if abs(v) >= 100 else 2}d}"
    return f"abA{s(A_deg)}B{('p' if B_deg >= 0 else 'm')}{abs(int(round(B_deg))):03d}"


def parse_ab(spec_list, spec_file):
    """Explicit (A,B) illumination angles, for sweeps that the 19 electrodes and the random
    augmentation cannot express -- in particular ANGLES OUTSIDE THE TRAINING DOMAIN, which is
    A in [-7.99 (F8), 84.77 (Cz)] for the electrodes and [-1.72, 73.95] for the 45 augmented
    directions actually simulated. `--n-aug` samples inside the convex hull of the electrodes and
    so can never leave it; this can.

    Accepts "A:B,A:B,..." and/or a file with one "A B" per line (# comments allowed).
    """
    out = []
    for s in (spec_list or "").split(","):
        s = s.strip()
        if s:
            A, B = s.split(":")
            out.append((float(A), float(B)))
    if spec_file:
        for line in open(spec_file):
            line = line.split("#", 1)[0].strip()
            if line:
                A, B = line.replace(",", " ").split()
                out.append((float(A), float(B)))
    jobs = []
    for A, B in out:
        d = AB_to_outward(A, B)
        A2, B2 = _outward_to_AB(d)                       # round-trip guard, not an assumption
        db = min(abs(B2 - B), 360 - abs(B2 - B))         # azimuth wraps
        if abs(A2 - A) > 1e-6 or (abs(np.cos(np.radians(A))) > 1e-9 and db > 1e-6):
            raise SystemExit(f"AB round-trip failed for A={A} B={B}: got A={A2} B={B2}")
        jobs.append((ab_tag(A, B), d, A, B))
    tags = [t for t, _, _, _ in jobs]
    if len(set(tags)) != len(tags):
        raise SystemExit("duplicate (A,B) after rounding to whole degrees -- tags would collide")
    return jobs


def augment_directions(n_aug, seed=0):
    """Sample n_aug OUTWARD directions inside the solid angle spanned by the 19
    standard 10-20 directions (O=AC-centred sphere), for A/B illumination-angle
    data augmentation (mirrors v10's angle-perturbation aug).

    Method: azimuthal-equidistant projection about the Cz (vertex) direction ->
    2D Delaunay of the 19 points -> area-weighted barycentric random sampling
    (stays inside the convex hull = the 10-20 scalp cap) -> unproject to sphere.
    Returns [(tag, outward_dir(3,), A_deg, B_deg)].
    """
    from scipy.spatial import Delaunay
    M = MN._load_mni()
    D = np.array([M["dirs"][n] for n in MN.NAMES])        # 19 outward unit dirs
    pole = M["dirs"]["Cz"] / np.linalg.norm(M["dirs"]["Cz"])
    e1 = F_ANT - (F_ANT @ pole) * pole; e1 /= np.linalg.norm(e1)     # tangent basis
    e2 = np.cross(pole, e1)

    def to2d(d):
        d = d / np.linalg.norm(d); th = np.arccos(np.clip(d @ pole, -1, 1))
        t = d - (d @ pole) * pole; nt = np.linalg.norm(t)
        if nt < 1e-9: return np.array([0.0, 0.0])
        t /= nt; return th * np.array([t @ e1, t @ e2])

    def from2d(p):
        th = np.linalg.norm(p)
        if th < 1e-9: return pole.copy()
        a = p / th; tang = a[0] * e1 + a[1] * e2
        return np.cos(th) * pole + np.sin(th) * tang

    P = np.array([to2d(d) for d in D])
    tri = Delaunay(P)
    simp = tri.simplices
    areas = np.array([abs(np.cross(P[t[1]] - P[t[0]], P[t[2]] - P[t[0]])) / 2 for t in simp])
    w = areas / areas.sum()
    rng = np.random.default_rng(seed)
    out = []
    for i in range(int(n_aug)):
        t = simp[rng.choice(len(simp), p=w)]
        r1, r2 = rng.random(), rng.random()
        if r1 + r2 > 1: r1, r2 = 1 - r1, 1 - r2
        p2 = P[t[0]] + r1 * (P[t[1]] - P[t[0]]) + r2 * (P[t[2]] - P[t[0]])
        d = from2d(p2); d /= np.linalg.norm(d)
        A, B = _outward_to_AB(d)
        out.append((f"aug{i:04d}", d, A, B))
    return out


def run_one(head, name, srcdir, nphoton, outdir, cfg, n_step=N_STEP_DEFAULT,
            tend=WINDOW_DEFAULT, radius_mm=RADIUS_MM, gpu_id=1, save=True, resume=False,
            seed=None):
    # LATE-BOUND ON PURPOSE. `seed=SEED` evaluated the module constant at DEF time, so every
    # caller that did `RV.SEED = s` before calling was silently running the default seed -- the
    # classic late-binding-default trap. MCX prints the seed it compiled with, and a probe
    # confirmed it: reassigning RV.SEED left `-Dgcfgseed=29012392` untouched and the two "seeds"
    # agreed to 6e-9 (GPU atomics jitter), while passing seed= explicitly moved the result by
    # 2.6e-2. Resolving None here restores the module-constant mechanism for the twelve existing
    # call sites AND keeps an explicit seed= working.
    if seed is None:
        seed = SEED
    base = f"{head}_F{WL}_{name}_r{int(radius_mm)}_t{n_step}"
    if resume and os.path.isfile(os.path.join(outdir, f"fluence_{base}.mat")):
        return None                                        # skip already-done sim
    t0 = time.time()
    cfg_vol, cfg_prop = cfg
    vol_size = cfg_vol.shape
    O = MN.AC.astype(float)
    src_pos = S.find_source_position(cfg_vol, vol_size, np.asarray(srcdir), O)
    tstep = float(tend) / int(n_step)
    mcfg = {
        "vol": cfg_vol, "prop": cfg_prop,
        "srcpos": src_pos.astype(float).tolist(), "srcdir": list(map(float, srcdir)),
        "srctype": "disk", "srcparam1": [float(radius_mm), 0.0, 0.0, 0.0],
        "tstart": 0.0, "tend": float(tend), "tstep": tstep,
        "issrcfrom0": 1, "seed": int(seed), "nphoton": int(nphoton), "unitinmm": 1.0,
        "outputtype": "fluence", "isreflect": 0, "isspecular": 0,
        "gpuid": gpu_id, "autopilot": 1,
    }
    flux = S.pmcx.run(mcfg)["flux"]
    if flux.ndim == 3:
        flux = flux[..., None]
    fluence = (flux * 1000.0).astype(np.float32)
    dt = time.time() - t0
    if save:
        os.makedirs(outdir, exist_ok=True)
        sio.savemat(os.path.join(outdir, f"fluence_{base}.mat"),
                    {"fluence": fluence, "srcpos": src_pos, "srcdir": np.asarray(srcdir),
                     "origin": O, "electrode": name, "n_step": n_step, "tstep": tstep,
                     "tend": float(tend), "vol_size": np.array(vol_size)}, do_compression=True)
        A, B = _outward_to_AB(-np.asarray(srcdir))       # outward = -srcdir
        meta = {"sample_id": head, "electrode": name, "wavelength": WL,
                "A_deg": A, "B_deg": B,
                "srcpos": src_pos.tolist(), "srcdir": list(map(float, srcdir)),
                "origin_AC": O.tolist(), "n_step": int(n_step), "tstart": 0.0,
                "tend": float(tend), "tstep": tstep, "nphoton": int(nphoton),
                # Store the random seed with each simulation for reproducibility.
                "seed": int(seed),
                "sim_seconds": round(dt, 1)}
        with open(os.path.join(outdir, f"meta_{base}.json"), "w") as f:
            json.dump(meta, f, indent=2)
    peaks = [float(fluence[..., k].max()) for k in range(fluence.shape[3])]
    print(f"  [{head} {name}] src={src_pos.tolist()} gates={fluence.shape[3]} "
          f"peak/gate={['%.2e'%p for p in peaks]} ({dt:.1f}s)", flush=True)
    return dict(fluence=fluence, srcpos=src_pos, sim_seconds=dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default=None)
    ap.add_argument("--heads", nargs="+", default=None)
    ap.add_argument("--electrodes", nargs="+", default=None)
    ap.add_argument("--n-aug", type=int, default=0,
                    help="A/B-augmentation sims/head for SHARM heads (solid-angle sampled)")
    ap.add_argument("--n-aug-scbbw", type=int, default=None,
                    help="augmentation sims/head for scb/bw heads (default = --n-aug); "
                         "lets small scb/bw sets get more aug for balance")
    ap.add_argument("--aug-seed", type=int, default=0)
    ap.add_argument("--ab-list", default=None,
                    help="explicit illumination angles 'A:B,A:B,...' in degrees (A elevation, "
                         "B azimuth). Unlike --n-aug, which samples inside the convex hull of the "
                         "19 electrodes, this can leave the training angle domain.")
    ap.add_argument("--ab-file", default=None, help="file with one 'A B' per line (# comments ok)")
    ap.add_argument("--standard", dest="standard", action="store_true", default=True)
    ap.add_argument("--no-standard", dest="standard", action="store_false",
                    help="skip the 19 standard electrodes (aug only)")
    ap.add_argument("--nphoton", type=float, default=1e7)
    ap.add_argument("--n-step", type=int, default=N_STEP_DEFAULT)
    ap.add_argument("--tend-ns", type=float, default=WINDOW_DEFAULT * 1e9)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--seed", type=int, default=SEED,
                    help="MCX RNG seed. Default reproduces every existing dataset. Give a NEW value "
                         "when the run must be an INDEPENDENT realisation of one already on disk "
                         "(e.g. a high-photon reference, or a replicate for the MC noise floor).")
    ap.add_argument("--out-dir", default=V11_DATA)
    ap.add_argument("--resume", action="store_true", help="skip sims whose output already exists")
    ap.add_argument("--force-rebuild", action="store_true")
    a = ap.parse_args()
    heads = a.heads or ([a.head] if a.head else [])
    if not heads:
        ap.error("give --head or --heads")
    dirs = inward_dirs()
    names = a.electrodes or MN.NAMES
    bad = set(names) - set(MN.NAMES)
    if bad:
        ap.error(f"unknown electrodes {sorted(bad)}; valid {MN.NAMES}")
    tend = a.tend_ns * 1e-9
    n_aug_scbbw = a.n_aug if a.n_aug_scbbw is None else a.n_aug_scbbw
    ab_jobs = parse_ab(a.ab_list, a.ab_file)
    # an explicit angle list is always the whole point of the run, so it turns the 19 standard
    # electrodes off unless they were asked for by name
    std_on = a.standard and not (ab_jobs and a.electrodes is None)
    std_jobs = [(n, dirs[n]) for n in names] if std_on else []
    # cache aug direction sets by count (deterministic given seed)
    aug_cache = {}
    def aug_for(head):
        na = n_aug_scbbw if MN.dataset_of(head) in ("scb", "bw") else a.n_aug
        if na <= 0:
            return []
        if na not in aug_cache:
            aug_cache[na] = augment_directions(na, seed=a.aug_seed)
        return aug_cache[na]
    print(f"[v11-mni] heads={len(heads)} standard={len(std_jobs)} ab-list={len(ab_jobs)} "
          f"aug SHARM={a.n_aug}/scb-bw={n_aug_scbbw} n_step={a.n_step} "
          f"window={a.tend_ns:.1f}ns nphoton={a.nphoton:.0e} seed={a.seed}"
          f"{' (DEFAULT)' if a.seed == SEED else ' (NEW -> independent realisation)'} "
          f"O=AC{MN.AC.tolist()} -> {a.out_dir}")
    for h in heads:
        cfg = get_cfg(h, force_rebuild=a.force_rebuild)
        for n, od in std_jobs:
            run_one(h, n, od, a.nphoton, a.out_dir, cfg, n_step=a.n_step, tend=tend,
                    gpu_id=a.gpu, resume=a.resume, seed=a.seed)
        for tag, od, A, B in aug_for(h):                   # od is OUTWARD; srcdir = -od
            run_one(h, tag, -od, a.nphoton, a.out_dir, cfg, n_step=a.n_step, tend=tend,
                    gpu_id=a.gpu, resume=a.resume, seed=a.seed)
        for tag, od, A, B in ab_jobs:                      # explicit angles; od is OUTWARD too
            run_one(h, tag, -od, a.nphoton, a.out_dir, cfg, n_step=a.n_step, tend=tend,
                    gpu_id=a.gpu, resume=a.resume, seed=a.seed)
    print("[v11-mni] DONE")


if __name__ == "__main__":
    main()
