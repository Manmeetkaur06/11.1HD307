"""Experiment 1: Part 1 reproduction of Bhagat et al. (Protocol A, seed 42).

Six base learners and the five-fold stacking reconstruction on one shuffled 80/20 row
split. Writes measured metrics, predictions, OOF scores, figures and a published-vs-measured
comparison to results/01_reproduction/.
"""
from common import ROOT, Timer, fresh_dir

from src.reproduction import run_reproduction


def main():
    out = fresh_dir("01_reproduction")
    out.rmdir()  # run_reproduction creates the directory itself and refuses to overwrite
    with Timer("exp01 reproduction"):
        run_reproduction(ROOT, output_dir=out)


if __name__ == "__main__":
    main()
