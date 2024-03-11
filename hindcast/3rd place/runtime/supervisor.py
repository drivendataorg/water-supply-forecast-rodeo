import os
from pathlib import Path
from time import sleep
from typing import Any

from loguru import logger
import pandas as pd
from tqdm import tqdm

import wsfr_read.config

import src.solution

IS_SMOKE = bool(os.getenv("IS_SMOKE", ""))

src_directory = Path("/code_execution/src")
data_directory = Path("/code_execution/data")
preprocessed_directory = Path("/code_execution/preprocessed")


def validate_prediction(predictions: Any):
    try:
        assert len(predictions) == 3
        assert all(pd.api.types.is_float(y) for y in predictions)
        assert all(not pd.isnull(y) for y in predictions)
    except AssertionError:
        logger.error(f"Validation failed for predictions {predictions}")
        raise


def main():
    assert hasattr(src.solution, "predict"), "Your solution.py must have a 'predict' function."

    logger.info("Beginning code execution...")

    logger.info("IS_SMOKE: {}", IS_SMOKE)
    logger.info("src_directory: {}", src_directory)
    logger.info("data_directory: {}", data_directory)
    logger.info("preprocessed_directory: {}", preprocessed_directory)

    # Check that DATA_ROOT is consistent
    assert wsfr_read.config.DATA_ROOT == data_directory

    # Check that data drive is fully mounted so that scanning returns data
    for i in range(30):
        try:
            next(data_directory.iterdir())
        except StopIteration:
            sleep(1)
        else:
            break
    try:
        next(data_directory.iterdir())
    except StopIteration:
        logger.error("Data directory not properly mounted after waiting 30 seconds.")
        raise
    else:
        logger.info("data_directory.iterdir returned results after waiting {} seconds.", i)

    # Create preprocessed directory
    try:
        preprocessed_directory.mkdir(parents=True)
    except FileExistsError:
        try:
            next(preprocessed_directory.iterdir())
            logger.warning(f"{preprocessed_directory} is not empty. This is not a clean run.")
        except StopIteration:
            pass

    if hasattr(src.solution, "preprocess"):
        logger.info("Running function 'preprocess'")
        assets = src.solution.preprocess(
            src_dir=src_directory,
            data_dir=data_directory,
            preprocessed_dir=preprocessed_directory,
        )
        logger.success("preprocess complete")
        test_features = pd.read_csv(preprocessed_directory / 'test_features.csv')
        test_features.to_csv("/code_execution/submission/test_features.csv", index=False)
        logger.info("preprocess features file saved")
        logger.info(
            "Loaded assets with keys: {}",
            ", ".join(repr(k) for k in assets.keys()),
        )
    else:
        logger.info("No 'preprocess' function found in solution.py. Skipping...")
        assets = {}

    if IS_SMOKE:
        submission_format_path = data_directory / "smoke_submission_format.csv"
    else:
        submission_format_path = data_directory / "submission_format.csv"
    submission_format_df = pd.read_csv(submission_format_path, index_col=["site_id", "issue_date"])

    logger.info("Beginning predictions...")
    update_iters = min(100, int(submission_format_df.shape[0] / 10))
    with open(os.devnull, "w") as devnull:
        pbar = tqdm(
            enumerate(submission_format_df.itertuples()),
            total=submission_format_df.shape[0],
            miniters=update_iters,
            file=devnull,
        )
        for i, row in pbar:
            if (i % update_iters) == 0:
                logger.info(str(pbar))
            site_id, issue_date = row.Index
            try:
                prediction = src.solution.predict(
                    site_id=site_id,
                    issue_date=issue_date,
                    assets=assets,
                    src_dir=src_directory,
                    data_dir=data_directory,
                    preprocessed_dir=preprocessed_directory,
                )

                validate_prediction(prediction)

                submission_format_df.loc[
                    row.Index, ["volume_10", "volume_50", "volume_90"]
                ] = prediction

            except Exception as exc:
                logger.error("Error predicting {}", row.Index)
                raise exc

    logger.success("Predictions complete.")

    logger.info("Saving predictions to file")
    submission_format_df.to_csv("/code_execution/submission/submission.csv")

    logger.success("Code execution run complete.")


if __name__ == "__main__":
    main()
