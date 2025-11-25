import os
import json
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"


class SweepPlot:
    def __init__(
        self,
        routine: str,
        variants: List[str],
        n_max: int = 2048,
        hide_spread_for: Optional[str] = None,
        # optional overrides; you can ignore these in normal use
        group: Optional[str] = None,
        title: Optional[str] = None,
        out_dir: Optional[str] = None,
        out_name: Optional[str] = None,
    ) -> None:
        """
        crit_root: /Users/vinland/coral/target/criterion
        routine:  e.g. "sasum", "sgemm_nn", "ssyr2_upper"
        variants: e.g. ["safe", "neon", "openblas", "accelerate"]

        By default we assume:
        - group    = f"{routine}_contiguous_sweep"
        - title    = routine.upper()
        - out_dir  = <project_root>/benches/plots
                    where project_root = dirname(dirname(crit_root))
        - out_name = f"{routine}.png"
        """
        self.crit_root = "/Users/vinland/coral/target/criterion"
        self.routine = routine
        self.variants = variants
        self.n_max = n_max
        self.hide_spread_for = hide_spread_for  # compare to *label*, see _variant_paths

        # auto-compute group if not given
        self.group = group if group is not None else f"{routine}_contiguous_sweep"

        # auto-compute title if not given
        self.title = title if title is not None else routine.upper()

        # auto-compute out_dir if not given
        if out_dir is not None:
            self.out_dir = out_dir.rstrip("/")
        else:
            # crit_root: /.../coral/target/criterion
            # project_root: /.../coral
            project_root = os.path.dirname(os.path.dirname(self.crit_root))
            self.out_dir = os.path.join(project_root, "plots")

        # auto-compute out_name if not given
        self.out_name = out_name if out_name is not None else f"{routine}.png"

    def _variant_paths(self, v: str):
        """
        Map a logical variant to (impl_dir, run_dir, label).
        Label is what shows in the legend, e.g. "coral-safe".
        """
        base = self.routine
        v_low = v.lower()

        if v_low == "safe":
            return f"{base}_coral_safe", "new", "coral-safe"
        if v_low == "neon":
            return f"{base}_coral_neon", "new", "coral-neon"
        if v_low == "openblas":
            return f"{base}_cblas", "openblas", "openblas"
        if v_low == "accelerate":
            return f"{base}_cblas", "accelerate", "accelerate"
        if v_low == "faer":
            return f"{base}_faer", "new", "faer"

        # fallback: treat as impl dir with run "new"
        return v, "new", v

    def _raw_samples(self, sample_path: str) -> np.ndarray:
        with open(sample_path, "r") as f:
            data = json.load(f)
        iters = np.asarray(data["iters"], dtype=float)
        times = np.asarray(data["times"], dtype=float)    # ns cumulative
        per_iter_ns = times / iters                       # ns/op
        return per_iter_ns * 1e-9                         # s/op

    def _load_variant(self, impl_dir: str, run_dir: str):
        """
        Return (n, med, low, high, vals_2d, unit) for this impl + run.

        vals_2d has shape (num_ns, num_samples) and holds the per-sample
        GiB/s or GFLOP/s values (trimmed to a common length). This is
        used both for quantiles and for drawing the faint background lines.
        """
        root = os.path.join(self.crit_root, self.group, impl_dir)
        if not os.path.isdir(root):
            print(f"[WARN] impl dir missing: {root}")
            return None, None, None, None, None, None

        ns = []
        vals_per_n = []
        unit = None

        for entry in sorted(os.listdir(root), key=lambda x: int(x) if x.isdigit() else 10**9):
            if not entry.isdigit():
                continue
            n = int(entry)
            if n > self.n_max:
                continue

            run_path = os.path.join(root, entry, run_dir)
            if not os.path.isdir(run_path):
                continue

            sample_path = os.path.join(run_path, "sample.json")
            bench_path = os.path.join(run_path, "benchmark.json")
            if not (os.path.exists(sample_path) and os.path.exists(bench_path)):
                continue

            samples_s = self._raw_samples(sample_path)

            with open(bench_path, "r") as f:
                bench = json.load(f)

            thr = bench.get("throughput")
            if not thr:
                continue

            # criterion 0.5 style: {"Bytes": size} or {"Elements": size}
            if "Bytes" in thr:
                thr_type = "Bytes"
                size = thr["Bytes"]
            elif "Elements" in thr:
                thr_type = "Elements"
                size = thr["Elements"]
            else:
                continue

            rate = size / samples_s  # Bytes/s or Elements/s

            if thr_type == "Bytes":
                vals = rate / (1024.0 ** 3)  # GiB/s
                this_unit = "GiB/s"
            else:  # Elements
                vals = rate / 1e9            # GFLOP/s
                this_unit = "GFLOP/s"

            if unit is None:
                unit = this_unit
            elif unit != this_unit:
                # shouldn't happen within one sweep
                continue

            ns.append(n)
            vals_per_n.append(np.asarray(vals, dtype=float))

        if not ns:
            return None, None, None, None, None, None

        # Stack to (num_ns, num_samples_common)
        min_len = min(len(v) for v in vals_per_n)
        vals_2d = np.stack([v[:min_len] for v in vals_per_n], axis=0)

        med = np.median(vals_2d, axis=1)
        low = np.percentile(vals_2d, 10, axis=1)
        high = np.percentile(vals_2d, 90, axis=1)

        ns = np.asarray(ns)
        order = np.argsort(ns)

        return (
            ns[order],
            med[order],
            low[order],
            high[order],
            vals_2d[order, :],
            unit,
        )

    def _color_for_label(self, label: str) -> str:
        l = label.lower()
        if "safe" in l:
            return "tab:cyan"
        if "neon" in l:
            return "tab:purple"
        if "openblas" in l:
            return "tab:blue"
        if "faer" in l:
            return "tab:orange"
        if "accelerate" in l: 
            return "mediumaquamarine"
        return "tab:black"

    def render(self, show: bool = True):
        """
        Render the sweep plot.

        Uses:
        - title   = self.title
        - outpath = os.path.join(self.out_dir, self.out_name)
        """
        series = []
        units = set()

        for v in self.variants:
            impl_dir, run_dir, label = self._variant_paths(v)
            ns, med, low, high, vals_2d, unit = self._load_variant(impl_dir, run_dir)
            if ns is None:
                print(f"[WARN] no data for variant {v} (impl={impl_dir}, run={run_dir})")
                continue
            series.append((label, ns, med, low, high, vals_2d))
            units.add(unit)

        if not series:
            print("[ERROR] no data for any variant")
            return

        y_unit = units.pop() if len(units) == 1 else "/".join(sorted(units))

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        for label, ns, med, low, high, vals_2d in series:
            color = self._color_for_label(label)

            # 10–90% band
            if label != self.hide_spread_for:
                ax.fill_between(
                    ns,
                    low,
                    high,
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                    label=f"{label} 10–90%",
                )
                # faint spiky per-sample lines (like the old plots)
                if vals_2d is not None:
                    for col in vals_2d.T:
                        ax.plot(
                            ns,
                            col,
                            color=color,
                            alpha=0.02,
                            linewidth=1.0,
                        )

            # median line
            ax.plot(
                ns,
                med,
                linewidth=2.2,
                color=color,
                label=f"{label} median",
            )

        ax.set_xlabel(r"Matrix dimension $n$", fontsize=12)
        ax.set_ylabel(y_unit, fontsize=12)
        ax.set_title(self.title, fontsize=13)
        ax.grid(True, linestyle=":", linewidth=0.6)

        # Apple M4 annotation like the old version
        ax.text(
            0.02,
            0.02,
            "Apple M4 Pro \nsingle-threaded",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=12,
            alpha=0.7,
        )

        ax.legend(
            frameon=True,
            fontsize=12,
            loc="lower right",
            facecolor="white",
            framealpha=0.7,
            edgecolor="none",
        )

        fig.tight_layout()
        os.makedirs(self.out_dir, exist_ok=True)
        out_path = os.path.join(self.out_dir, self.out_name)
        fig.savefig(out_path, dpi=300)
        if show:
            plt.show()
        plt.close(fig)
