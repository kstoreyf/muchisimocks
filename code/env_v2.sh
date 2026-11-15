# Source this in every job of the v2 rerun (independent test-set phases, per-draw noise,
# P_gm/covariance/validation fixes). Outputs land in parallel *_v2 trees; old outputs untouched.
# New test-set mock libraries (_coveragev2_p5_n1000, _shamev2_p0_n1000) are written to the
# default scratch lib root (/scratch/kstoreyf/muchisimocks/muchisimocks_lib<tag>).
# NOTE: the training fields (_p5_n10000) live in /data/kstoreyf/muchisimocks/; set
# MUCHISIMOCKS_LIB=/data/kstoreyf/muchisimocks/muchisimocks_lib when computing training stats.
export MUCHISIMOCKS_DATA=/scratch/kstoreyf/muchisimocks/data_v2
export MUCHISIMOCKS_RESULTS=/scratch/kstoreyf/muchisimocks/results_v2
