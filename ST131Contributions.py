"""Compute fitted-variation contributions for any ST131 candidate family.

Uses the same draw-paired decomposition as the existing GP analysis, with
the fitted linear, spline, or GP response supplied by ST131MFE.
"""

from ST131GPContributions import (
    analysis_rows, calculate_contributions, component_contributions,
    summarize_contributions, main as _main,
)


def main(argv=None):
    return _main(argv, gp_only=False)


if __name__ == "__main__":
    main()
