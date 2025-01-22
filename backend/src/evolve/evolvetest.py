import time
from io import BytesIO
from datetime import datetime, UTC, timedelta
import traceback
import json
import io
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from werkzeug.exceptions import BadRequest
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import statistics
import os
from datetime import date

#from app.helpers.fold_storage_manager import FoldStorageManager
from app.helpers.sequence_util import (
    get_measured_and_unmeasured_mutant_seq_ids,
    get_loci_set,
    process_and_validate_evolve_input_files,
)


def train_model(wt_aa_seq,raw_activity_df,raw_embedding_df,model):
    """
    Train a machine learning model on protein sequence data.
    
    Args:
        wt_aa_seq: Wild-type amino acid sequence
        raw_activity_df: DataFrame containing activity measurements
        raw_embedding_df: DataFrame containing protein embeddings
        model: Machine learning model instance
        
    Returns:
        DataFrame with predicted activities
    """
    activity_df, embedding_df = process_and_validate_evolve_input_files(
                wt_aa_seq, raw_activity_df, raw_embedding_df
            )
    measured_mutants, unmeasured_mutants = (
                get_measured_and_unmeasured_mutant_seq_ids(activity_df, embedding_df)
            )
    X_train = np.vstack(
                [json.loads(x) for x in embedding_df.loc[activity_df.index].embedding]
            )
    y_train = activity_df.activity.to_numpy()
    
    model.fit(X_train, y_train)
    try:
        all_mutants_embedding_array = np.vstack(
            [
                json.loads(x)
                for x in embedding_df.loc[
                    measured_mutants + unmeasured_mutants
                ].embedding
            ]
        )
        #print(all_mutants_embedding_array.shape)
        y_all_pred = model.predict(all_mutants_embedding_array)
        predicted_activity_df = pd.DataFrame(
            {
                "seq_id": measured_mutants + unmeasured_mutants,
                "predicted_activity": y_all_pred,
            }
        )
        predicted_activity_df.index = predicted_activity_df.seq_id
        predicted_activity_df["relevant_measured_mutants"] = (
            predicted_activity_df.seq_id.apply(
                lambda seq_id: " ".join(
                    [
                        m
                        for m in measured_mutants
                        if get_loci_set(m) & get_loci_set(seq_id)
                    ]
                )
            )
        )
        predicted_activity_df["actual_activity"] = predicted_activity_df.join(
            activity_df.groupby(level=0).activity.mean(), how="left"
        ).activity
        predicted_activity_df = predicted_activity_df.sort_values(
            "predicted_activity", ascending=False
        )
    except Exception as e:
        print(f"Failed to predict activities: {e}")
        raise
    predicted_activity_df.reset_index(drop=True,inplace=True)
    #predicted_activity_csv_path = evolve_directory / f"Round_{round_num}_predicted_activity.csv"
    #print(f"Storing predicted activities in {predicted_activity_csv_path}")
    #try:
    #    predicted_activity_df.to_csv(predicted_activity_csv_path, index=False)
    #except Exception as e:
    #    print(f"Failed to store predicted activities: {e}")
    #    raise
    return predicted_activity_df

def evaluate_predictions(predicted_activity_df,exp_activity_df,round_activity_df,num_var):
    """
    Evaluate model predictions and select top variants.
    
    Args:
        predicted_activity_df: DataFrame with model predictions
        exp_activity_df: DataFrame with experimental activities
        round_activity_df: DataFrame with current round activities
        num_var: Number of variants to select
        round_num: Current round number
        evolve_directory: Directory to save results
        
    Returns:
        DataFrame with next round activities
    """
    try:
        predict = predicted_activity_df["actual_activity"].isna()
        extract_predict = predicted_activity_df[predict]
        
        top_var = extract_predict.head(num_var)
        top_var_real = pd.merge(top_var,exp_activity_df,on='seq_id', how='left')
        top_var_real = top_var_real[['seq_id','activity']]
        next_round_activity = pd.concat([round_activity_df,top_var_real], ignore_index=True)
        print(f'The next round has {len(next_round_activity)} variants')
    except Exception as e:
        print(f"Failed to Evaluate Predictions")
        raise
    return next_round_activity
def evolve_simulation(wt_aa_seq,initial_round_activity,raw_embedding_df,exp_activity_df,top_percent_df,target_num_top_variants,model_type):
    try:
        round_num=1
        round_variants = {}
        predicted_activity_df = train_model(wt_aa_seq,initial_round_activity,raw_embedding_df,model_type)
        #print(f'predicted_activity is {predicted_activity_df.shape}')
        current_round_activity_df = evaluate_predictions(predicted_activity_df,exp_activity_df,initial_round_activity,num_var)
        num_top_var_df = pd.merge(top_percent_df,current_round_activity_df, on='seq_id',how='inner')
        #print(f'current_round_activity_df is {current_round_activity_df.shape}')
        print(f'Found {len(num_top_var_df)} top variants')    
        while True:
            round_variants[round_num] = num_top_var_df
            round_num += 1
            print(round_num)
            predicted_activity_df = train_model(wt_aa_seq,current_round_activity_df,raw_embedding_df,model_type)
            current_round_activity_df = evaluate_predictions(predicted_activity_df,exp_activity_df,current_round_activity_df,num_var)
            num_top_var_df = pd.merge(top_percent_df,current_round_activity_df, on='seq_id',how='inner')
            round_variants
            if len(num_top_var_df) >= target_num_top_variants:
                break
            print(f'Found {len(num_top_var_df)} top variants')
            #print(num_top_var_df)
        print(f'Found {len(num_top_var_df)} top variants after {round_num} rounds')
    except Exception as e:
        print(f"Evolution simulation failed: {str(e)}")
        traceback.print_exc()
        raise

def digivolve(wt_aa_seq,exp_activity_file_path,embeddings_dir, embeddings_paths,num_var,percent_top,rounds_evo,quantile,model_type):
    #Set up dataframes
    exp_activity_df = pd.read_excel(exp_activity_file_path)
    for path in embeddings_paths:
        embeddings_path = os.path.join(embeddings_dir, path)
        raw_embedding_df = pd.read_csv(embeddings_path)
        #sample for initial round

        #print(f'There are {len(initial_round_activity)} variants')
        #top variant data frame for comparison
        top_percent_activity = exp_activity_df['activity'].quantile(quantile)
        top_percent_df = exp_activity_df[exp_activity_df['activity'] >= top_percent_activity]
        top_percent_df = top_percent_df.sort_values('activity',ascending=False)
        target_num_top_variants = round(len(top_percent_df)*percent_top)
        print(f'The target number of variants is {target_num_top_variants} out of {len(top_percent_df)}')
        for rounds in range(rounds_evo):
            initial_round_var_df = raw_embedding_df.sample(num_var)
            initial_round_activity = pd.merge(exp_activity_df,initial_round_var_df,on='seq_id', how='inner')
            initial_round_activity = initial_round_activity[['seq_id','activity']]
            evolve_simulation(wt_aa_seq,initial_round_activity,raw_embedding_df,exp_activity_df,top_percent_df,target_num_top_variants,model_type)


def plot_evolution(evo_imgs, embeddings_paths, figsize=(14, 5), colors=None):
    """
    Plot evolution results for multiple embedding models.
    
    Args:
        evo_imgs: List of (means, std_errors) tuples for each model
        embeddings_paths: List of embedding model names/paths
        figsize: Figure size tuple
        colors: Optional list of colors for bars
    """
    if colors is None:
        colors = plt.cm.tab20(np.linspace(0, 1, len(evo_imgs)))
        
    num_imgs = len(evo_imgs)
    num_cols = 2
    num_rows = (num_imgs + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, 
                            figsize=(figsize[0], figsize[1] * num_rows))
    axes = np.array(axes).reshape(-1)
    
    max_y_value = max(max(means) + max(std_errors) 
                     for means, std_errors in evo_imgs)
    
    for i, ((means, std_errors), label, color) in enumerate(zip(evo_imgs, 
                                                              embeddings_paths,
                                                              colors)):
        ax = axes[i]
        x_indices = np.arange(1, len(means) + 1)
        
        ax.bar(x_indices, means, yerr=std_errors, 
               capsize=5, label=label, color=color, 
               edgecolor='black')
        
        ax.set_xlabel('Rounds')
        ax.set_ylabel('Top Variants')
        ax.set_title(f'Embedding Model: {label}')
        ax.set_xticks(x_indices)
        ax.set_xticklabels([f'Round {k}' for k in x_indices])
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(0, max_y_value * 1.1)
    
    for j in range(num_imgs, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()

def plot_evolution_line(evo_imgs, embeddings_paths, figsize=(14, 5), colors=None, markersize=8):
    """
    Plot evolution results for multiple embedding models on the same graph.
    
    Args:
        evo_imgs: List of (means, std_errors) tuples for each model
        embeddings_paths: List of embedding model names/paths
        figsize: Figure size tuple
        colors: Optional list of colors for bars
        
    Returns:
        fig: Matplotlib figure object
    """
    if not evo_imgs or not embeddings_paths:
        raise ValueError("evo_imgs and embeddings_paths must not be empty")
    
    if len(evo_imgs) != len(embeddings_paths):
        raise ValueError("evo_imgs and embeddings_paths must have the same length")
    
    if colors is None:
        colors = plt.cm.tab20(np.linspace(0, 1, len(evo_imgs)))
        
    fig, ax = plt.subplots(figsize=figsize)
    
    max_y_value = 0
    
    for i, ((means, std_errors), label, color) in enumerate(zip(evo_imgs, embeddings_paths, colors)):
        x_indices = np.arange(1, len(means) + 1)
        
        ax.errorbar(x_indices, means, yerr=std_errors, capsize=5, label=label, color=color, marker='o', linestyle='-',markersize=markersize)
        
        # Update max_y_value for setting y-axis limit
        max_y_value = max(max_y_value, max(means) + max(std_errors))
    
    ax.set_xlabel('Rounds')
    ax.set_ylabel('Top Variants')
    ax.set_title('Evolution Results for Multiple Embedding Models')
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f'Round {k}' for k in x_indices])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, max_y_value * 1.1)
    
    plt.tight_layout()

