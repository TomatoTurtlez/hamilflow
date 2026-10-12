"""Check whether ensemble members (and DFT) differ by a rigid energy shift c * S.

For a fixed overlap S, H - c S has the eigenvalues of H shifted by exactly -c and the same
eigenvectors. This script fits, for every structure, the shift that best removes a rigid offset
between two Hamiltonians sharing the same S,

    c = Re<H - H_ref, S> / <S, S>        (sums over all stored real-space entries),

and reports how much of the difference it explains:

    rms_before  = rms(H - H_ref)                  over all entries
    rms_after   = rms(H - H_ref - c S)
    residual    = |H - H_ref - c S| / |H - H_ref|  (0: purely rigid, 1: no rigid part)

Pairs checked:
  - every member against the ensemble mean (computed on the fly from the members);
  - with --dft-root: the ensemble mean and every member against DFT.

Rigidity checks: c is also fitted on on-site blocks only (R = 0, i = j), on off-site blocks
only, and on the on-site blocks of each species. For a rigid shift they all agree.

Standalone: needs only numpy and h5py. Entries are streamed in batches of whole atom-pair
blocks, so memory stays bounded. Spinful Hamiltonians (complex, 2x2 spin blocks) are handled
by expanding S onto the two diagonal spin blocks.

  python check_rigid_shift.py --structures structure_4234_300 structure_356
  python check_rigid_shift.py --all --dft-root /path/to/dft --n-procs 16 --out rigid_shift.json
"""

import argparse
import concurrent.futures
import json
from glob import glob
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-glob", default="train_{i}/infer/outputs_train/2026-10-06_09-41-0?/dft/",
                   help="glob for the member output dirs, '{i}' replaced by 1..n-models")
    p.add_argument("--n-models", type=int, default=5)
    p.add_argument("--structures", nargs="*", default=None)
    p.add_argument("--all", action="store_true", help="every structure directory in the first model dir")
    p.add_argument("--dft-root", default=None, help="DFT reference root, one dir per structure (optional)")
    p.add_argument("--hamiltonian-name", default="hamiltonian.h5")
    p.add_argument("--overlap-name", default="overlap.h5")
    p.add_argument("--batch-entries", type=int, default=4_000_000, help="entries per streamed batch")
    p.add_argument("--n-procs", type=int, default=1, help="structures processed in parallel")
    p.add_argument("--out", default="rigid_shift.json")
    return p.parse_args()


# ----------------------------------------------------------------------------- structure data

def read_layout(path):
    with h5py.File(path, "r") as f:
        return (np.array(f["atom_pairs"][:], dtype=np.int64),
                np.array(f["chunk_boundaries"][:], dtype=np.int64),
                np.array(f["chunk_shapes"][:], dtype=np.int64))


def read_species(structure_dir):
    """Element of every atom from a VASP-5 POSCAR (None if it cannot be parsed)."""
    try:
        lines = (Path(structure_dir) / "POSCAR").read_text().split("\n")
        names, counts = lines[5].split(), [int(x) for x in lines[6].split()]
        return [n for n, c in zip(names, counts) for _ in range(c)]
    except Exception:
        return None


def batches(bounds, batch_entries):
    """Ranges of atom pairs [p0, p1) holding about `batch_entries` entries each."""
    n_pairs = len(bounds) - 1
    p0 = 0
    while p0 < n_pairs:
        p1 = int(np.searchsorted(bounds, bounds[p0] + batch_entries, side="right")) - 1
        p1 = max(p1, p0 + 1)
        yield p0, min(p1, n_pairs)
        p0 = min(p1, n_pairs)


def s_in_h_layout(s_entries, s_shapes, h_shapes):
    """S entries arranged like H's blocks: identical when the shapes match, expanded onto the
    two diagonal spin blocks when H's blocks are twice as large (spinful)."""
    if np.array_equal(s_shapes, h_shapes):
        return s_entries
    out, k = [], 0
    for (n, m), (hn, hm) in zip(s_shapes, h_shapes):
        block = s_entries[k:k + n * m].reshape(n, m)
        k += n * m
        if (hn, hm) != (2 * n, 2 * m):
            raise ValueError(f"H block {hn}x{hm} is neither the S block {n}x{m} nor its spin expansion")
        big = np.zeros((hn, hm), dtype=block.dtype)
        big[:n, :m] = block
        big[n:, m:] = block
        out.append(big.reshape(-1))
    return np.concatenate(out) if out else np.zeros(0)


class Sums:
    """Accumulates Re<D, S>, <S, S>, |D|^2 and the entry count, overall and per subset."""

    def __init__(self, subsets):
        self.keys = ["all"] + list(subsets)
        self.ds = dict.fromkeys(self.keys, 0.0)
        self.ss = dict.fromkeys(self.keys, 0.0)
        self.dd = dict.fromkeys(self.keys, 0.0)
        self.n = dict.fromkeys(self.keys, 0)

    def add(self, d, s, masks):
        for k in self.keys:
            m = slice(None) if k == "all" else masks[k]
            dk, sk = d[m], s[m]
            self.ds[k] += float(np.sum(np.real(dk) * sk))
            self.ss[k] += float(np.sum(sk * sk))
            self.dd[k] += float(np.sum(np.abs(dk) ** 2))
            self.n[k] += dk.size

    def result(self):
        out = {}
        for k in self.keys:
            if self.ss[k] <= 0 or self.n[k] == 0:
                continue
            c = self.ds[k] / self.ss[k]
            dd_after = max(self.dd[k] - 2 * c * self.ds[k] + c * c * self.ss[k], 0.0)
            out[k] = {
                "c_eV": c,
                "rms_before_meV": 1e3 * np.sqrt(self.dd[k] / self.n[k]),
                "rms_after_meV": 1e3 * np.sqrt(dd_after / self.n[k]),
                "residual": float(np.sqrt(dd_after / self.dd[k])) if self.dd[k] > 0 else 0.0,
            }
        return out


# ----------------------------------------------------------------------------- one structure

def check_structure(structure, model_dirs, args):
    member_paths = [Path(d) / structure / args.hamiltonian_name for d in model_dirs]
    s_path = Path(model_dirs[0]) / structure / args.overlap_name
    dft_dir = Path(args.dft_root) / structure if args.dft_root else None

    atom_pairs, bounds, shapes = read_layout(member_paths[0])
    for p in member_paths[1:] + ([dft_dir / args.hamiltonian_name] if dft_dir else []):
        ap, b, sh = read_layout(p)
        if not (np.array_equal(ap, atom_pairs) and np.array_equal(sh, shapes)):
            raise ValueError(f"{structure}: sparse layout of {p} differs from {member_paths[0]}")
    s_ap, s_bounds, s_shapes = read_layout(s_path)
    if not np.array_equal(s_ap, atom_pairs):
        raise ValueError(f"{structure}: atom_pairs of {s_path} differ from the Hamiltonian's")

    species = read_species(Path(model_dirs[0]) / structure)
    onsite_pair = (atom_pairs[:, :3] == 0).all(axis=1) & (atom_pairs[:, 3] == atom_pairs[:, 4])
    subsets = ["onsite", "offsite"] + ([f"onsite_{el}" for el in dict.fromkeys(species)] if species else [])

    vs_mean = [Sums(subsets) for _ in member_paths]
    vs_dft = [Sums(subsets) for _ in member_paths] if dft_dir else None
    mean_vs_dft = Sums(subsets) if dft_dir else None
    s_dft_maxdiff = 0.0

    files = [h5py.File(p, "r") for p in member_paths]
    s_file = h5py.File(s_path, "r")
    dft_file = h5py.File(dft_dir / args.hamiltonian_name, "r") if dft_dir else None
    s_dft_file = h5py.File(dft_dir / args.overlap_name, "r") if dft_dir else None
    try:
        for p0, p1 in batches(bounds, args.batch_entries):
            e0, e1 = bounds[p0], bounds[p1]
            s = s_in_h_layout(np.asarray(s_file["entries"][s_bounds[p0]:s_bounds[p1]]), s_shapes[p0:p1], shapes[p0:p1])
            if s_dft_file is not None:
                s_dft = np.asarray(s_dft_file["entries"][s_bounds[p0]:s_bounds[p1]])
                s_dft_maxdiff = max(s_dft_maxdiff, float(np.max(np.abs(
                    s_dft - np.asarray(s_file["entries"][s_bounds[p0]:s_bounds[p1]])), initial=0.0)))

            sizes = shapes[p0:p1, 0] * shapes[p0:p1, 1]
            on = np.repeat(onsite_pair[p0:p1], sizes)
            masks = {"onsite": on, "offsite": ~on}
            if species:
                el_of_entry = np.repeat(np.array(species)[atom_pairs[p0:p1, 3]], sizes)
                for el in dict.fromkeys(species):
                    masks[f"onsite_{el}"] = on & (el_of_entry == el)

            hs = [np.asarray(f["entries"][e0:e1]) for f in files]
            mean = np.mean(hs, axis=0)
            for k, h in enumerate(hs):
                vs_mean[k].add(h - mean, s, masks)
            if dft_file is not None:
                h_dft = np.asarray(dft_file["entries"][e0:e1])
                mean_vs_dft.add(mean - h_dft, s, masks)
                for k, h in enumerate(hs):
                    vs_dft[k].add(h - h_dft, s, masks)
    finally:
        for f in files + [s_file, dft_file, s_dft_file]:
            if f is not None:
                f.close()

    out = {"members_vs_mean": [x.result() for x in vs_mean], "n_entries": int(bounds[-1])}
    if dft_dir:
        out["mean_vs_dft"] = mean_vs_dft.result()
        out["members_vs_dft"] = [x.result() for x in vs_dft]
        out["overlap_dft_vs_members_maxdiff"] = s_dft_maxdiff
    return structure, out


def report(structure, res):
    lines = [f"\n{'=' * 96}\n{structure}  ({res['n_entries']} entries)\n{'=' * 96}"]
    head = (f"{'':14s}{'c (eV)':>10s}{'rms before':>13s}{'rms after':>12s}{'residual':>10s}"
            f"{'c on-site':>11s}{'c off-site':>11s}  c on-site per species")

    def row(label, r):
        a = r["all"]
        per_el = "  ".join(f"{k[7:]} {v['c_eV']:+.4f}" for k, v in r.items() if k.startswith("onsite_"))
        return (f"{label:14s}{a['c_eV']:>+10.4f}{a['rms_before_meV']:>10.1f} meV{a['rms_after_meV']:>9.1f} meV"
                f"{a['residual']:>10.3f}{r.get('onsite', {}).get('c_eV', float('nan')):>+11.4f}"
                f"{r.get('offsite', {}).get('c_eV', float('nan')):>+11.4f}  {per_el}")

    lines += ["members vs ensemble mean", head]
    lines += [row(f"member {i + 1}", r) for i, r in enumerate(res["members_vs_mean"])]
    if "mean_vs_dft" in res:
        lines += ["", f"vs DFT (max |S_DFT - S_members| = {res['overlap_dft_vs_members_maxdiff']:.2e})", head,
                  row("mean", res["mean_vs_dft"])]
        lines += [row(f"member {i + 1}", r) for i, r in enumerate(res["members_vs_dft"])]
    return "\n".join(lines)


def _worker(structure, model_dirs, args):
    try:
        name, res = check_structure(structure, model_dirs, args)
        return name, res, report(name, res)
    except Exception as e:
        return structure, {"error": f"{type(e).__name__}: {e}"}, f"\n{structure}: FAILED: {type(e).__name__}: {e}"


def main():
    args = parse_args()
    model_dirs = []
    for i in range(1, args.n_models + 1):
        model_dirs.extend(Path(s) for s in sorted(glob(args.model_glob.format(i=i))))
    print("model dirs:")
    for i, d in enumerate(model_dirs, 1):
        print(f"  member {i}: {d}")
    if len(model_dirs) != args.n_models:
        print(f"WARNING: expected {args.n_models} model dirs, found {len(model_dirs)}")

    if args.structures:
        structures = args.structures
    elif args.all:
        structures = sorted(p.name for p in Path(model_dirs[0]).iterdir() if p.is_dir())
    else:
        raise SystemExit("pass --structures or --all")

    results = {}
    if args.n_procs <= 1:
        for s in structures:
            name, res, text = _worker(s, model_dirs, args)
            print(text, flush=True)
            results[name] = res
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.n_procs, len(structures))) as ex:
            futures = [ex.submit(_worker, s, model_dirs, args) for s in structures]
            for fut in concurrent.futures.as_completed(futures):
                name, res, text = fut.result()
                print(text, flush=True)
                results[name] = res

    with open(args.out, "w") as f:
        json.dump({s: results[s] for s in structures}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
