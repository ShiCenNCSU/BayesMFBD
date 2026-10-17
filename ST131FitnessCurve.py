#!/usr/bin/env python3
"""Summarize the fitted ST131 response curve for any candidate model.

Run: python ST131FitnessCurve.py --fit-dir FIT_DIR --output-dir OUTPUT --no-save-draws
Plot: Rscript PlotST131GPFitnessCurve.R OUTPUT
"""

from ST131GPFitnessCurve import fitness_at_latent, latent_additive_scores, summarize_curve
from ST131GPFitnessCurve import main as _main


def main(argv=None):
    return _main(argv, gp_only=False)


if __name__ == "__main__":
    main()
