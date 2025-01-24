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
from IPython.display import display
from scipy.stats import sem

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
        #print(f'The next round has {len(next_round_activity)} variants')
    except Exception as e:
        print(f"Failed to Evaluate Predictions")
        raise
    return next_round_activity
def evolve_simulation(wt_aa_seq,initial_round_activity,raw_embedding_df,exp_activity_df,ninety_percent_df,ninetyfive_percent_df,top_ten_df,model_type,num_var):
        round_num=0
        round_variants = []
        ninety_found = False
        ninetyfive_found = False
                     
        #num_top_var_df = pd.merge(top_percent_df,current_round_activity_df, on='seq_id',how='inner')
        #print(f'current_round_activity_df is {current_round_activity_df.shape}')    
        while True:
            if round_num == 0:
                round_num += 1
                predicted_activity_df = train_model(wt_aa_seq,initial_round_activity,raw_embedding_df,model_type)
                current_round_activity_df = evaluate_predictions(predicted_activity_df,exp_activity_df,initial_round_activity,num_var)
                if ninety_percent_df['seq_id'].isin(current_round_activity_df.values.flatten()).any() and not ninety_found:
                    round_variants.append(round_num)
                    ninety_found = True
                if ninetyfive_percent_df['seq_id'].isin(current_round_activity_df.values.flatten()).any() and not ninetyfive_found:
                    round_variants.append(round_num)
                    ninetyfive_found = True
                if top_ten_df['seq_id'].isin(current_round_activity_df.values.flatten()).any():
                    round_variants.append(round_num)
                    break                   
            else:
                round_num += 1
                #print(round_num)
                predicted_activity_df = train_model(wt_aa_seq,current_round_activity_df,raw_embedding_df,model_type)
                current_round_activity_df = evaluate_predictions(predicted_activity_df,exp_activity_df,current_round_activity_df,num_var)
                if ninety_percent_df['seq_id'].isin(current_round_activity_df.values.flatten()).any() and not ninety_found:
                    round_variants.append(round_num)
                    ninety_found = True
                if ninetyfive_percent_df['seq_id'].isin(current_round_activity_df.values.flatten()).any() and not ninetyfive_found:
                    round_variants.append(round_num)
                    ninetyfive_found = True
                if top_ten_df['seq_id'].isin(current_round_activity_df.values.flatten()).any():
                    round_variants.append(round_num)
                    break
        top_ten_variants_found =  pd.merge(top_ten_df,current_round_activity_df, on='seq_id',how='inner')
        top_ten_variants_found = top_ten_variants_found[['seq_id']]
        #round_variants = pd.DataFrame(round_variants)
        #print(f'Found {len(top_ten_variants_found)} top variants after {round_num} rounds')
        return round_variants, top_ten_variants_found

def clean_embeddings_df(dms_df,embeddings_df):
    clean_embeddings_df = pd.merge(embeddings_df,dms_df,on='seq_id',how='inner')
    clean_embeddings_df = clean_embeddings_df[['seq_id','embedding']]
    return clean_embeddings_df

def digivolve(wt_aa_seq,dataset,exp_activity_file_path,embeddings_dir, embeddings_paths,num_var,rounds_evo,model_type):
    #Set up dataframes
    exp_activity_df = pd.read_excel(exp_activity_file_path)
    ninety_percent_activity = exp_activity_df['activity'].quantile(.9)
    ninetyfive_percent_activity = exp_activity_df['activity'].quantile(.95)
    ninety_percent_df = exp_activity_df[exp_activity_df['activity'] >= ninety_percent_activity]
    ninetyfive_percent_df = exp_activity_df[exp_activity_df['activity'] >= ninetyfive_percent_activity]
    top_ten_df = ninety_percent_df.sort_values('activity',ascending=False)
    top_ten_df = top_ten_df.head(10)
    
    evolve_df = pd.DataFrame()
    benchmarks = ['90 percentile', '95 percentile', 'top ten']
    #display('Top Ten', top_ten_df)
    for path in embeddings_paths:
        print(path)
        embeddings_path = os.path.join(embeddings_dir, path)
        raw_embedding_df = pd.read_csv(embeddings_path)
        #sample for initial round

        #print(f'There are {len(initial_round_activity)} variants')
        #top variant data frame for comparison

        path_results = {}
        round_results =[]
        for rounds in range(rounds_evo):
            if rounds % 10 == 0:
                print(f"Starting {path} round {rounds}")
            while True:
                initial_round_var_df = raw_embedding_df.sample(num_var)
                if not initial_round_var_df['seq_id'].isin(top_ten_df.values.flatten()).any():
                    break
            initial_round_activity = pd.merge(exp_activity_df,initial_round_var_df,on='seq_id', how='inner')
            initial_round_activity = initial_round_activity[['seq_id','activity']]

            round_variants, top_ten_variants_found = evolve_simulation(
                wt_aa_seq,initial_round_activity,
                raw_embedding_df,exp_activity_df,
                ninety_percent_df,
                ninetyfive_percent_df,
                top_ten_df,
                model_type,
                num_var)
            round_results.append(round_variants)
            #print(f'This is {round_variants[2]} to top ten and {round_variants} is the output, {top_ten_variants_found} was found')
        path_results[path] = round_results

        path_results_df = pd.DataFrame(path_results)
        evolve_df = pd.concat([evolve_df,path_results_df], axis = 1) 
        #print(f'{path} embeddings gave {path_results}')
    #display(evolve_df)
    def average_lists_across_rows(column):
        # Transpose the lists to group elements by their positions
        """
        column = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        to
        [(1, 4, 7), (2, 5, 8), (3, 6, 9)]
        """
        transposed_lists = list(zip(*column))
        # Calculate the average for each group
        averaged_lists = [np.mean(sublist) for sublist in transposed_lists]
        std_error = [sem(sublist) for sublist in transposed_lists]
        return averaged_lists, std_error

    averaged_data = {}
    sem_data = {}

    for col in evolve_df.columns:
        averaged_lists, sem_lists = average_lists_across_rows(evolve_df[col])
        averaged_data[col] = averaged_lists
        sem_data[col] = sem_lists

    # Create a new DataFrame with the averaged lists and SEMs
    result_df = pd.DataFrame([averaged_data, sem_data])

    #print(result_df)
    plot_evolution(result_df,benchmarks)
    results_path = os.path.join(embeddings_dir,dataset)
    try:
        os.mkdir(results_path)
    except OSError as error:
        print(error)
    comp_dir = os.path.join(embeddings_dir, dataset, f'{num_var}_{dataset}_Model_Comparisons_{datetime.now().strftime('%d%H%M%S')}.xlsx')
    result_df.to_excel(comp_dir, index=False)




def plot_evolution(result_df,benchmarks):
    """
    Plot evolution results for multiple embedding models.
    
    Args:
        evo_imgs: List of (means, std_errors) tuples for each model
        embeddings_paths: List of embedding model names/paths
        figsize: Figure size tuple
        colors: Optional list of colors for bars
    """
    averaged_lists = result_df.iloc[0]
    sem_lists = result_df.iloc[1]

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Number of bars per column
    num_bars = len(averaged_lists.iloc[0])

    # Width of the bars
    width = 0.2

    # X-axis positions
    x = np.arange(num_bars)

    # Plot each column
    for i, (col, avg_list) in enumerate(averaged_lists.items()):
        sem_list = sem_lists[col]
        ax.bar(x + i * width, avg_list, width, yerr=sem_list, label=col, capsize=5)

    # Add labels and title
    ax.set_xlabel('Benchmarks')
    ax.set_ylabel('Average Round to find variant')
    ax.set_title('Embeddings Comparison')
    ax.set_xticks(x + (len(averaged_lists) - 1) * width / 2)
    ax.set_xticklabels(benchmarks)
    ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

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

