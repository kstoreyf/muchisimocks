"""Auxiliary / legacy helpers for muchisimocks.

This module collects older helpers that are not part of the main
workflow anymore (e.g. emu- and moment-network specific utilities),
so ``code.utils`` can stay focused and smaller.

These are kept here in case they are useful for future experiments
or for reproducing older results.
"""

from __future__ import annotations

import numpy as np
import os

from code.utils import (  # type: ignore[import-not-found]
    load_emu,
    get_posterior_maxes,
    get_moments_test_sbi,
    get_moments_test_mn,
    get_samples_mn,
    get_samples_sbi,
    get_samples_emcee,
    get_samples_dynesty,
    get_samples_fisher,
    repeat_arr_rlzs,
)

__all__ = [
    "load_emu",
    "get_posterior_maxes",
    "get_moments_test_sbi",
    "get_moments_test_mn",
    "get_samples_mn",
    "get_samples_sbi",
    "get_samples_emcee",
    "get_samples_dynesty",
    "get_samples_fisher",
    "repeat_arr_rlzs",
]

