import io
import json
import logging
import time
import traceback
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from werkzeug.exceptions import BadRequest

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
from folde.few_shot_models import get_few_shot_model, is_valid_few_shot_model_name


def get_embedding_df_from_file(fold_id: int, fsm: FoldStorageManager, embedding_files: list[str]) -> pd.DataFrame:
    logging.info(f"Reading {len(embedding_files)} embedding files")

    embedding_dfs = []
    chunk_size = 10000  # Adjust based on memory constraints

    for path in embedding_files:
        # Get the CSV content as a string
        assert fsm.storage_manager is not None, "Storage manager not set up"
        csv_blob = fsm.storage_manager.get_blob(fold_id, path)

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
    raw_embedding_df = pd.concat(embedding_dfs, ignore_index=True)
    return raw_embedding_df


def get_naturalness_df_from_file(fold_id: int, fsm: FoldStorageManager, naturalness_files: list[str]) -> pd.DataFrame:
    logging.info(f"Reading {len(naturalness_files)} naturalness files")
    naturalness_dfs = []

    for path in naturalness_files:
        assert fsm.storage_manager is not None, "Storage manager not set up"
        csv_blob = fsm.storage_manager.get_blob(fold_id, path)
        with csv_blob.open("r") as csv_f:
            naturalness_dfs.append(pd.read_csv(csv_f))
    raw_naturalness_df = pd.concat(naturalness_dfs, ignore_index=True)
    return raw_naturalness_df

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
        # REQUIRED SETUP #######################################################
        fsm = FoldStorageManager()
        fsm.setup()

        # INPUT VALIDATION #####################################################
        if not fold.yaml_config:
            raise ValueError("Fold does not have a YAML config!")
        if not evolve.embedding_files or not evolve.naturalness_files:
            raise ValueError(f"These days, evolve jobs must specify both embedding files (found {evolve.embedding_files}) and naturalness files (found {evolve.naturalness_files})")
        if not evolve.few_shot_params:
            raise ValueError(f"These days, few shot params are required, got {evolve.few_shot_params}")
        if not evolve.mode or not is_valid_few_shot_model_name(evolve.mode):
            raise BadRequest(f'Old modes such as {evolve.mode} are no longer supported.')
        if evolve.num_mutants is None or evolve.num_mutants <= 0:
            raise ValueError(f"Evolve job must specify a positive number of mutants, got {evolve.num_mutants}")

        # LOAD INPUTS #########################################################
        boltz_yaml_helper = BoltzYamlHelper(fold.yaml_config)
        if len(boltz_yaml_helper.get_protein_sequences()) != 1:
            raise ValueError(
                f"Fold has {len(boltz_yaml_helper.get_protein_sequences())} protein sequences, which is not supported for evolvepro yet."
            )

        wt_aa_seq = boltz_yaml_helper.get_protein_sequences()[0][1]
        evolve_directory = Path("evolve") / evolve.name
        try:
            few_shot_params = json.loads(evolve.few_shot_params)
        except Exception as e:
            raise BadRequest(f"Failed to parse few shot params: {e}")

        # 2. Read and merge all embedding CSVs
        activity_file_contents = fsm.storage_manager.get_binary(evolve.fold_id, str(evolve_directory / "activity.xlsx"))

        raw_activity_df = pd.read_excel(BytesIO(activity_file_contents))
        raw_embedding_df = get_embedding_df_from_file(evolve.fold_id, fsm, evolve.embedding_files.split(','))
        raw_naturalness_df = get_naturalness_df_from_file(evolve.fold_id, fsm, evolve.naturalness_files.split(','))

        logging.info(f"Found {raw_embedding_df.shape[0]} embeddings and {raw_naturalness_df.shape[0]} naturalness values")

        # PROCESS INPUTS #######################################################
        activity_df, embedding_df, incomplete_naturalness_df = process_and_validate_evolve_input_files(
            wt_aa_seq, raw_activity_df, raw_embedding_df, raw_naturalness_df
        )
        assert embedding_df is not None
        assert incomplete_naturalness_df is not None
        logging.info(
            f"Found {activity_df.shape[0]} activity measurements among {activity_df.index.unique().shape[0]} mutants"
        )

        # AUGMENT SINGLE MUTANT NATURALNESS FOR MULTI MUTANTS ##################
        def get_naturalness_of_multi_mutant(seq_id) -> float:
            if seq_id == 'WT':
                return 1.0
            try:
                return incomplete_naturalness_df.wt_marginal.loc[seq_id.split('_')].prod()
            except Exception as e:
                raise BadRequest(f'Failure computing naturalness for {seq_id}: {e}')
        augmented_naturalness_series = pd.Series(
            embedding_df.index.map(get_naturalness_of_multi_mutant),
            index=embedding_df.index
        )

        # VALIDATE FINAL INPUT SEQUENCES #######################################
        for seq_id in activity_df.index:
            if seq_id not in embedding_df.index:
                raise ValueError(f"Activity seq id {seq_id} is missing either an embedding or naturalness value")

        few_shot_model = get_few_shot_model(
            evolve.mode,
            random_state=42,
            wt_aa_seq=wt_aa_seq,
            **few_shot_params,
        )

        few_shot_model.pretrain(
            augmented_naturalness_series,
            embedding_df.embedding,
        )

        few_shot_model.fit(
            augmented_naturalness_series,
            embedding_df.embedding,
            activity_df.activity,
        )

        top_seq_ids, predicted_activity_ensemble = few_shot_model.get_top_n(
            evolve.num_mutants,
            augmented_naturalness_series,
            embedding_df.embedding,
        )

        predicted_activity_df = pd.DataFrame(
            {f"model_{ii}": predicted_activity_ensemble[ii] for ii in range(len(predicted_activity_ensemble))},
            index=predicted_activity_ensemble[0].index,
        )

        logging.info(f"Top seq ids: {top_seq_ids}")

        # def get_selected_idx_or_none(seq_id):
        #     try:
        #         return top_seq_ids.index(seq_id)
        #     except ValueError as e:
        #         return None
        predicted_activity_df['selected'] = predicted_activity_df.index.isin(top_seq_ids)

        # predicted_activity_df[~predicted_activity_df.selected_idx.isna()]

        try:
            loci_to_measured_mutants = defaultdict(list)
            for measured_seq_id in activity_df.index.unique():
                loci = get_loci_set(measured_seq_id)
                for locus in loci:
                    loci_to_measured_mutants[locus].append(measured_seq_id)
            def get_relevant_measured_mutants(seq_id) -> str:
                return ", ".join(sorted(sum([loci_to_measured_mutants.get(locus, []) for locus in get_loci_set(seq_id)], [])))
            predicted_activity_df['relevant_measured_mutants'] = predicted_activity_df.index.map(
                get_relevant_measured_mutants
            )
        except Exception as e:
            logging.error(f"Error computing relevant measured mutants: {e}")

        # Save debug info to file
        try:
            fsm.storage_manager.write_file(
                evolve.fold_id,
                str(evolve_directory / "debug_info.json"),
                json.dumps(few_shot_model.get_debug_info()),
            )
        except Exception as e:
            logging.error(f"Error saving debug info to file: {e}")
        fsm.storage_manager.write_file(
            evolve.fold_id,
            str(evolve_directory / "predicted_activity.csv"),
            predicted_activity_df.to_csv(),
        )
