"""check_paperfig_paths.py — which fiducial-ensemble member files exist for the paper figures, per ensemble scheme.
Run on a compute node with the v2 trees: MUCHISIMOCKS_DATA/RESULTS from env_v2.sh."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
import ensemble_kmax_utils as eku
import plotter_figs as pf

print("train:", pf.TAG_PARAMS_TRAIN, pf.TAG_BIASPARAMS_TRAIN, pf.TAG_NOISE_TRAIN, "| results:", pf.paths.DIR_RESULTS)
for scheme in eku.ENSEMBLE_SCHEMES:
    eku.set_ensemble_scheme(scheme)
    print(f"\n=== {scheme}")
    for stats, mask in eku.FIDUCIAL_CONFIGS:
        for nth in range(eku.N_ENSEMBLE_K):
            d = eku.model_dir(stats, mask, nth)
            row = []
            for mode, mock in [("shame", "_nbar0.00011"), ("shame", "_nbar0.00022"), ("shame", "_nbar0.00054"),
                               ("cvmean", None), ("cvindiv", None), ("coverage", None)]:
                kw = {"tag_mock": mock} if mock else {}
                p, kind = eku.resolve_member_samples_path(stats, mask, nth, test_mode=mode, **kw)
                row.append(f"{mode}{mock or ''}:{'-' if p is None else kind}")
            print(f"{'_'.join(stats):15s} m{nth} dir={'Y' if d.is_dir() else 'N'} " + " ".join(row))
print("\nexample dir:", eku.model_dir(*eku.FIDUCIAL_CONFIGS[3], 0).name)
print("example cvmean:", eku.member_samples_path(*eku.FIDUCIAL_CONFIGS[1], 0, test_mode="cvmean").name)
