import io
import json
import logging
import time
import traceback
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from app.helpers.boltz_yaml_helper import BoltzYamlHelper
from app.helpers.fold_storage_manager import FoldStorageManager
from app.helpers.jobs_util import (
    LoggingRecorder,
    _live_update_tail,
    _psql_tail,
)
from app.helpers.sequence_util import (
    get_loci_set,
    process_and_validate_evolve_input_files,
)
from app.models import Evolution, Fold, Invokation
from sklearn.ensemble import RandomForestRegressor
from werkzeug.exceptions import BadRequest
from folde.few_shot_models import is_valid_few_shot_model_name, get_few_shot_model


def run_evolvepro(evolve_id: int):
    """Run the evolvepro workflow."""
    evolve = Evolution.get_by_id(evolve_id)
    if not evolve:
        raise BadRequest(f"Evolution {evolve_id} not found")
    fold = Fold.get_by_id(evolve.fold_id)
    if not fold:
        raise BadRequest(f"Fold {evolve.fold_id} not found")
    invokation = Invokation.get_by_id(evolve.invokation_id)
    if not invokation:
        raise BadRequest(f"Invokation {evolve.invokation_id} not found")

    with LoggingRecorder(invokation):
        """Helper function to run evolvepro with a logger."""
        if not fold.yaml_config:
            raise ValueError("Fold does not have a YAML config!")
        boltz_yaml_helper = BoltzYamlHelper(fold.yaml_config)
        if len(boltz_yaml_helper.get_protein_sequences()) != 1:
            raise ValueError(
                f"Fold has {len(boltz_yaml_helper.get_protein_sequences())} protein sequences, which is not supported for evolvepro yet."
            )
        wt_aa_seq = boltz_yaml_helper.get_protein_sequences()[0][1]

        mode = evolve.mode or "randomforest"

        fsm = FoldStorageManager()
        fsm.setup()

        # 1. Get the activity file.
        evolve_directory = Path("evolve") / evolve.name
        activity_file_path = evolve_directory / "activity.xlsx"
        logging.info(f"Getting the activity file {activity_file_path}")
        activity_file = fsm.storage_manager.get_binary(evolve.fold_id, str(activity_file_path))
        raw_activity_df = pd.read_excel(BytesIO(activity_file))

        # 2. Read and merge all embedding CSVs
        if not evolve.embedding_files or not evolve.naturalness_files:
            raise ValueError(f"These days, evolve jobs must specify both embedding files (found {evolve.embedding_files}) and naturalness files (found {evolve.naturalness_files})")
        
        if not evolve.few_shot_params:
            raise ValueError(f"These days, few shot params are required, got {evolve.few_shot_params}")

        embedding_files = evolve.embedding_files.split(",")
        naturalness_files = evolve.naturalness_files.split(",")
        logging.info(f"Reading {len(embedding_files)} embedding files and {len(naturalness_files)} naturalness files")
        embedding_dfs = []
        naturalness_dfs = []
        chunk_size = 10000  # Adjust based on memory constraints

        for path in embedding_files:
            # Get the CSV content as a string
            csv_blob = fsm.storage_manager.get_blob(evolve.fold_id, path)

            with csv_blob.open("r") as csv_f:
                # Create chunks iterator
                chunks = pd.read_csv(csv_f, chunksize=chunk_size)

                # Process each chunk
                path_dfs = []
                for chunk in chunks:
                    path_dfs.append(chunk)

                # Combine chunks for this path
                if path_dfs:
                    embedding_dfs.append(pd.concat(path_dfs, ignore_index=True))
        
        for path in naturalness_files:
            csv_blob = fsm.storage_manager.get_blob(evolve.fold_id, path)
            with csv_blob.open("r") as csv_f:
                naturalness_dfs.append(pd.read_csv(csv_f))

        # Combine all embeddings
        raw_embedding_df = pd.concat(embedding_dfs, ignore_index=True)
        raw_naturalness_df = pd.concat(naturalness_dfs, ignore_index=True)
        logging.info(f"Found {raw_embedding_df.shape[0]} embeddings and {raw_naturalness_df.shape[0]} naturalness values")

        # 3. Process the activity and embedding data.
        activity_df, embedding_df, naturalness_df = process_and_validate_evolve_input_files(
            wt_aa_seq, raw_activity_df, raw_embedding_df, raw_naturalness_df
        )
        logging.info(
            f"Found {activity_df.shape[0]} activity measurements among {activity_df.index.unique().shape[0]} mutants"
        )

        if not is_valid_few_shot_model_name(mode):
            raise BadRequest(f'Old modes such as {mode} are no longer supported.')

        # Compute naturalness for all mutants, where possible.
        def get_naturalness_of_multi_mutant(seq_id) -> float:
            if seq_id == 'WT':
                return 1.0
            try:
                return naturalness_df.wt_marginal.loc[seq_id.split('_')].prod()
            except Exception as e:
                raise BadRequest(f'Failure computing naturalness for {seq_id}: {e}')

        augmented_naturalness_series = pd.Series(
            embedding_df.index.map(get_naturalness_of_multi_mutant),
            index=embedding_df.index
        )

        for seq_id in activity_df.index:
            if seq_id not in embedding_df.index:
                raise ValueError(f"Activity seq id {seq_id} is missing either an embedding or naturalness value")

        params = json.loads(evolve.few_shot_params)

        few_shot_model = get_few_shot_model(
            mode,
            random_state=42,
            **params,
        )

        few_shot_model.fit(
            augmented_naturalness_series,
            embedding_df.embedding,
            activity_df.activity,
        )

        top_seq_ids, predicted_activity_ensemble = few_shot_model.get_top_n(
            24,
            augmented_naturalness_series,
            embedding_df.embedding,
        )

        predicted_activity_df = pd.DataFrame(
            {f"model_{ii}": predicted_activity_ensemble[ii] for ii in range(len(predicted_activity_ensemble))},
            index=predicted_activity_ensemble[0].index,
        )
        fsm.storage_manager.write_file(
            evolve.fold_id,
            str(evolve_directory / "predicted_activity.csv"),
            predicted_activity_df.to_csv(),
        )
