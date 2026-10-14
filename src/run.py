import sys
import traceback
import pandas as pd
from pathlib import Path
import os

SRC_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC_ROOT))
os.chdir(SRC_ROOT)


from src.logger.logger import get_logger
from src.config.config_loader import ConfigLoader

from src.data.load_data import DataLoader
from src.data.validate_data import DataValidator
from src.data.time_filter import TimeFilter
from src.features.build_features import FeatureBuilder

from src.training.train import Trainer
from src.schema.data_schema import DataConfig


logger = get_logger(__name__)
logger.info("current working directory: " + str(Path.cwd()))
CONFIG_PATH = "src/config/config.yaml"



class PipelineRunner:
    """
    Main pipeline orchestrator.

    Executes:
        1. Load data
        2. Time filter
        3. Validate data
        4. Feature engineering
        5. Train model
        6. Predict
        7. Evaluate
        8. Save artifacts
    """

    def __init__(self, config_path: str):

        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.get_config()
        self.data_config = DataConfig(**self.config["data"])

        logger.info("PipelineRunner initialized")

    # ---------------------------------------------------------
    # Pipeline Execution
    # ---------------------------------------------------------
    def run(self):

        try:
            logger.info("========== PIPELINE STARTED ==========")

            # =====================================================
            # STEP 1 — LOAD DATA
            # =====================================================
            logger.info("STEP 1: Loading raw data")

            data_loader = DataLoader(self.data_config)
            raw_df = data_loader.load()

            logger.info(f"Raw data shape: {raw_df.shape}")

            # =====================================================
            # STEP 2 — TIME FILTER
            # =====================================================
            logger.info("STEP 2: Applying time filter")

            time_filter = TimeFilter(self.data_config)
            filtered_df = time_filter.filter(raw_df)

            logger.info(f"Filtered data shape: {filtered_df.shape}")

            # =====================================================
            # STEP 3 — VALIDATE DATA
            # =====================================================
            logger.info("STEP 3: Validating data")

            validator = DataValidator(self.data_config)
            validator.validate(filtered_df)

            # =====================================================
            # STEP 4 — BUILD FEATURES
            # =====================================================
            logger.info("STEP 4: Feature engineering")

            feature_builder = FeatureBuilder(self.data_config)
            processed_df = feature_builder.build(filtered_df)

            logger.info(
                f"Processed data shape: {processed_df.shape}"
            )

            # =====================================================
            # STEP 5 — TRAIN MODEL
            # =====================================================
            logger.info("STEP 5: Training model")

            trainer = Trainer(self.config)
            model_path, _ = trainer.train(processed_df)

            logger.info(f"Model saved at: {model_path}")

            logger.info("========== PIPELINE COMPLETED ==========")

        except Exception as e:
            logger.error("Pipeline execution failed")
            logger.error(str(e))
            traceback.print_exc()
            sys.exit(1)


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    runner = PipelineRunner(CONFIG_PATH)
    runner.run()

