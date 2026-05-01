from __future__ import annotations
import logging
import sys
import os

# Ensure the project root is in sys.path so we can import from the 'scripts' package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts import run_phase1, run_phase2, run_phase3

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s"
)
logger = logging.getLogger("pipeline")


def main():
    logger.info("Starting Phase 1: Clustering Analysis")
    run_phase1.main()

    logger.info("Starting Phase 2: Opportunity Scoring")
    run_phase2.main()

    logger.info("Starting Phase 3: Risk and Ablation Analysis")
    run_phase3.main()

    logger.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
