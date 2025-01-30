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
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import ast
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
def evolve_simulation(wt_aa_seq,initial_round_activity,raw_embedding_df,exp_activity_df,benchmarks,top_benchmark,model_type,num_var):
        round_num=0
        round_variants = []
        benchmarks_hit = {benchmark: False for benchmark in benchmarks}
        #print(benchmarks_hit)
        evolution_done = False             
        #num_top_var_df = pd.merge(top_percent_df,current_round_activity_df, on='seq_id',how='inner')
        #print(f'current_round_activity_df is {current_round_activity_df.shape}')    
        while not evolution_done:
            #print(benchmarks)
            round_num += 1
            print(round_num,end="\r",flush=True)
            if round_num == 1:
                predicted_activity_df = train_model(wt_aa_seq,initial_round_activity,raw_embedding_df,model_type)
                current_round_activity_df = evaluate_predictions(predicted_activity_df,exp_activity_df,initial_round_activity,num_var)   
            else:
                predicted_activity_df = train_model(wt_aa_seq,current_round_activity_df,raw_embedding_df,model_type)
                current_round_activity_df = evaluate_predictions(predicted_activity_df,exp_activity_df,current_round_activity_df,num_var)
            for benchmark in benchmarks:
                #print(benchmark)
                #print(benchmarks[benchmark])
                benchmark_df = benchmarks[benchmark]
                #print(f'First round {benchmark}')
                if benchmark_df['seq_id'].isin(current_round_activity_df['seq_id']).any() and not benchmarks_hit[benchmark]:
                    #print(f'{benchmark} and round {round_num}')
                    round_variants.append(round_num)
                    benchmarks_hit[benchmark] = True
                    #print(f'{benchmark} and round {round_num}')
                    if benchmark == top_benchmark:
                        #print(f'{benchmark} and round {round_num}')
                        evolution_done = True
        #round_variants = pd.DataFrame(round_variants)
        #print(round_variants)
        #print(f'Found {len(top_ten_variants_found)} top variants after {round_num} rounds')
        return round_variants

def clean_embeddings_df(dms_df,embeddings_df):
    clean_embeddings_df = pd.merge(embeddings_df,dms_df,on='seq_id',how='inner')
    clean_embeddings_df = clean_embeddings_df[['seq_id','embedding']]
    return clean_embeddings_df

def digivolve(wt_aa_seq,prot_name,wt_activity,benchmarks,dataset,exp_activity_file_path,embeddings_dir, embeddings_paths,num_var,rounds_evo,model_type):
    #Set up dataframes
    exp_activity_df = pd.read_excel(exp_activity_file_path)
    above_wt_df = exp_activity_df[exp_activity_df['activity'] > wt_activity]
    #print(above_wt_df)
    for benchmark in benchmarks:
        if benchmark == 'Above WT':
            benchmarks[benchmark] = above_wt_df
            #print(f'above WT{benchmark[benchmark]}')
            continue
        benchmark_activity = above_wt_df['activity'].quantile(benchmarks[benchmark])
        #print(f'{benchmark} has cutoff {benchmark_activity}')
        df = exp_activity_df[exp_activity_df['activity'] >= benchmark_activity]
        benchmarks[benchmark] = df
    #display(benchmarks)
    evolve_df = pd.DataFrame()
    top_benchmark = list(benchmarks)[-1]
    top_benchmark_df = benchmarks[top_benchmark]
    #print(f'This is the top benchmark {top_benchmark}')
    #display('Top Ten', top_ten_df)
    for path in embeddings_paths:
        try:
            embeddings_path = os.path.join(embeddings_dir, path)
            raw_embedding_df = pd.read_csv(embeddings_path)
            cleaned_embeddings_df = clean_embeddings_df(exp_activity_df,raw_embedding_df)
            #sample for initial round

            #print(f'There are {len(initial_round_activity)} variants')
            #top variant data frame for comparison

            path_results = {}
            round_results =[]
            for rounds in range(rounds_evo):
                print(f"Starting {path} round {(rounds+1)}")
                while True:
                    initial_round_var_df = cleaned_embeddings_df.sample(num_var)
                    if not initial_round_var_df['seq_id'].isin(top_benchmark_df.values.flatten()).any():
                        break
                initial_round_activity = pd.merge(exp_activity_df,initial_round_var_df,on='seq_id', how='inner')
                initial_round_activity = initial_round_activity[['seq_id','activity']]

                round_variants = evolve_simulation(
                    wt_aa_seq,initial_round_activity,
                    cleaned_embeddings_df,exp_activity_df,
                    benchmarks,
                    top_benchmark,
                    model_type,
                    num_var)
                round_results.append(round_variants)
                #print(f'This is {round_variants[2]} to top ten and {round_variants} is the output, {top_ten_variants_found} was found')
            path_results[path] = round_results

            path_results_df = pd.DataFrame(path_results)
            evolve_df = pd.concat([evolve_df,path_results_df], axis = 1)
            #display (evolve_df)
        except:
            continue
            #print(f'{path} embeddings gave {path_results}')
    #display(evolve_df)
    results_path = os.path.join(embeddings_dir,dataset)
    try:
        os.mkdir(results_path)
    except OSError as error:
        print()
    raw_comp_dir = os.path.join(embeddings_dir,dataset, f'raw_{num_var}_{dataset}_Model_Comparisons.xlsx')
    evolve_df.to_excel(raw_comp_dir, index=False)
    plot_evolution_boxplot(evolve_df,benchmarks,(results_path+"/"+prot_name))
    #def average_lists_across_rows(column):
    #    # Transpose the lists to group elements by their positions
    #    """
    #    column = [
    #        [1, 2, 3],
    #        [4, 5, 6],
    #        [7, 8, 9]
    #    ]
    #    to
    #    [(1, 4, 7), (2, 5, 8), (3, 6, 9)]
    #    """
    #    transposed_lists = list(zip(*column))
    #    # Calculate the average for each group
    #    averaged_lists = [np.mean(sublist) for sublist in transposed_lists]
    #    std_error = [sem(sublist) for sublist in transposed_lists]
    #    return averaged_lists, std_error
#
    #averaged_data = {}
    #sem_data = {}
#
    #for col in evolve_df.columns:
    #    averaged_lists, sem_lists = average_lists_across_rows(evolve_df[col])
    #    averaged_data[col] = averaged_lists
    #    sem_data[col] = sem_lists
#
    ## Create a new DataFrame with the averaged lists and SEMs
    #result_df = pd.DataFrame([averaged_data, sem_data])
    ##display(result_df)
    ##print(result_df)
#
    #plot_evolution_boxplot(evolve_df,benchmarks,(results_path+"/"+prot_name))
#
    #comp_dir = os.path.join(embeddings_dir,dataset, f'{num_var}_{dataset}_Model_Comparisons.xlsx')
    #result_df.to_excel(comp_dir, index=False)
    return evolve_df
def plot_evolution(result_df,benchmarks,save_path=None):
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

    num_bars = len(benchmarks.keys())


    # Width of the bars
    width = .8 / result_df.shape[1]

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
    ax.set_xticklabels(benchmarks.keys())

    ax.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    # Show the plot
    plt.tight_layout()
    plt.show()
def plot_evolution_boxplot(df,benchmarks,save_path=None):
    def convert_to_list(value):
        if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value
        return value

    # Convert all string representations of lists to actual lists
    df = df.applymap(convert_to_list)
    # Print the modified DataFrame
    #print(df)
        # Create empty lists for the new structure
    def reshape_for_violin(df,benchmarks):
        # Create empty lists for the new structure
        values = []
        positions = []
        columns = []

        # Iterate through each column
        for col in df.columns:
            # Iterate through each row's list
            for row in df[col]:
                # Iterate through each position in the list
                try:
                    for i, value in enumerate(row):
                        values.append(value)
                        positions.append(list(benchmarks.keys())[i])
                        columns.append(col)
                except:
                    continue
        print(positions)
        # Create a new dataframe with the restructured data
        return pd.DataFrame({
            'Rounds': values,
            'Benchmark': positions,
            'Embeddings': columns
        })

    # Reshape the data
    reshaped_df = reshape_for_violin(df,benchmarks)
    print(reshaped_df)
    # Create the violin plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=reshaped_df, showmeans=True,
                meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"},
                x='Benchmark', y='Rounds', hue='Embeddings')
    plt.title('Embeddings Comparison')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()
    #sns.violinplot(data=reshaped_df, x='Benchmark', y='Rounds', hue='Embeddings')
    #plt.title('Violinplot Comparing List Elements Across Columns')