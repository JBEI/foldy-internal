import time
from io import BytesIO
from datetime import datetime, UTC, timedelta
import traceback
import json
from tqdm.notebook import tqdm
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

def evaluate_predictions(predicted_activity_df,exp_activity_df,round_activity_df,num_var,strat="topn"):
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
        if strat == "topn":
            top_var = extract_predict.head(num_var)
        '''    
        elif strat =='top_uni':
            #print(f'strat is {strat}')
            extract_predict_copy = extract_predict.copy()
            extract_predict_copy['unique_residues'] = extract_predict['seq_id'].str.extract(r'([A-Z]\d{1,})')
            unique_pred = extract_predict_copy.drop_duplicates(subset='unique_residues', keep='first')
            unique_pred = unique_pred.sort_values('predicted_activity',ascending=False)
            top_var = unique_pred.head(num_var)
        elif strat == 'topn_uni':
            extract_predict_copy = extract_predict.copy()
            extract_predict_copy['unique_residues'] = extract_predict['seq_id'].str.extract(r'([A-Z]\d{1,})')
            # Group by unique residues and get top N for each group
            unique_pred = extract_predict_copy.groupby('unique_residues').head(3)
            unique_pred = unique_pred.sort_values('predicted_activity',ascending=False)
            top_var = unique_pred.head(num_var)'''
        top_var_real = pd.merge(top_var,exp_activity_df,on='seq_id', how='left')
        top_var_real = top_var_real[['seq_id','activity']]
        next_round_activity = pd.concat([round_activity_df,top_var_real], ignore_index=True)
        #print(f'The next round has {len(next_round_activity)} variants')
    except Exception as e:
        print(f"Failed to Evaluate Predictions")
        raise
    return next_round_activity
def evolve_simulation(wt_aa_seq, initial_round_activity, raw_embedding_df, exp_activity_df, 
                     benchmarks, top_benchmark, model_type, num_var, max_rounds=100):
    round_num = 0
    round_variants = []
    benchmarks_hit = {benchmark: False for benchmark in benchmarks}
    evolution_done = False
    
    # Check initial round
    for benchmark in benchmarks:
        benchmark_df = benchmarks[benchmark]
        if benchmark_df['seq_id'].isin(initial_round_activity['seq_id']).any() and not benchmarks_hit[benchmark]:
            round_variants.append(round_num)
            benchmarks_hit[benchmark] = True
            if benchmark == top_benchmark:
                evolution_done = True
    
    current_round_activity_df = initial_round_activity
    
    try:
        while not evolution_done and round_num < max_rounds:
            round_num += 1
            
            predicted_activity_df = train_model(wt_aa_seq, current_round_activity_df,
                                              raw_embedding_df, model_type)
            
            current_round_activity_df = evaluate_predictions(predicted_activity_df,
                                                          exp_activity_df,
                                                          current_round_activity_df,
                                                          num_var)
            
            for benchmark in benchmarks:
                benchmark_df = benchmarks[benchmark]
                if benchmark_df['seq_id'].isin(current_round_activity_df['seq_id']).any() and not benchmarks_hit[benchmark]:
                    round_variants.append(round_num)
                    benchmarks_hit[benchmark] = True                    
                    if benchmark == top_benchmark:
                        evolution_done = True
                        break
    
    finally:
        # Make sure to close the progress bar
        pbar.close()
    
    return round_variants

def clean_embeddings_df(dms_df,embeddings_df):
    clean_embeddings_df = pd.merge(embeddings_df,dms_df,on='seq_id',how='inner')
    clean_embeddings_df = clean_embeddings_df[['seq_id','embedding']]
    #print(f'clean embeddings {clean_embeddings_df.shape}')
    return clean_embeddings_df

def digivolve(wt_aa_seq,prot_name,wt_activity,benchmarks,dataset,exp_activity_file_path,embeddings_dir, embeddings_paths,num_var,rounds_evo,model_type):
    #Set up dataframes
    exp_activity_df = pd.read_excel(exp_activity_file_path)
    print(f'Exp activity df {exp_activity_df.shape}')
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
    
    top_benchmark = list(benchmarks)[-1]
    top_benchmark_df = benchmarks[top_benchmark]
    #print(f'This is the top benchmark {top_benchmark}')
    #display('Top Ten', top_ten_df)
    
    evolve_df = pd.DataFrame()
    print(f'Strategy: {strat}')
    all_path_results = []  # Store results for all paths
    
    for path in embeddings_paths:
        try:
            # Load and process embeddings
            embeddings_path = os.path.join(embeddings_dir, path)
            raw_embedding_df = pd.read_csv(embeddings_path)
            #print(f'raw embeddings {raw_embedding_df.shape}')
            cleaned_embeddings_df = clean_embeddings_df(exp_activity_df, raw_embedding_df)
            
            full_data_df = pd.merge(exp_activity_df,raw_embedding_df, on='seq_id', how='inner')
            #print(f'full data shape {full_data_df.shape}')
            # Store results for each round
            round_results = []
            ''''''
            for round_num in range(rounds_evo):
                print(f"Starting {path} round {(round_num+1)}")
                initial_round_activity = full_data_df.sample(num_var)
                initial_round_activity = initial_round_activity[['seq_id', 'activity']]
                
                round_variants = evolve_simulation(
                    wt_aa_seq, initial_round_activity,
                    cleaned_embeddings_df, exp_activity_df,
                    benchmarks, top_benchmark,
                    model_type, num_var
                )
                #print(round_variants)
                round_results.append(round_variants)
            
            # Create a dictionary with consistent length data
            path_results = {
                path: round_results
            }
            all_path_results.append(pd.DataFrame(path_results))
            
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            continue
    
    # Combine results only if we have data
    if all_path_results:
        strategy_df = pd.concat(all_path_results, axis=1)
        evolve_df = pd.concat([evolve_df, strategy_df], axis=1)
    
    # Save results
    results_path = os.path.join(embeddings_dir, dataset)
    os.makedirs(results_path, exist_ok=True)
    
    if not evolve_df.empty:
        raw_comp_dir = os.path.join(results_path, 
                                  f'raw_{num_var}_{dataset}_Model_Comparisons.xlsx')
        evolve_df.to_excel(raw_comp_dir, index=False)
        plot_evolution_boxplot(evolve_df, benchmarks, 
                             (results_path+"/"+prot_name))
    else:
        print(f"No results to save for strategy {strat}")
    return evolve_df
def optimize_memory(df):
    # Downcast numeric columns
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['integer']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df
def optimize_memory_with_checks(df):
    original_size = df.memory_usage(deep=True).sum()
    
    # Only optimize if potential benefit exists
    if original_size > 1e6:  # Only optimize if > 1MB
        df = optimize_memory(df)
        new_size = df.memory_usage(deep=True).sum()
        print(f"Memory reduced from {original_size/1e6:.2f}MB to {new_size/1e6:.2f}MB")
    
    return df
def digivolveZS(wt_aa_seq,prot_name,wt_activity,benchmarks,dataset,exp_activity_file_path,embeddings_dir, embeddings_paths,num_var,rounds_evo,model_type,logits_path,zero_shot=None):
    #Set up dataframes
    raw_exp_activity_df = optimize_memory_with_checks(pd.read_excel(exp_activity_file_path))
    logits_df = optimize_memory_with_checks(pd.read_csv(logits_path))
    start_benchmarks = benchmarks.copy()
    evolve_df = pd.DataFrame()
    all_path_results = []  # Store results for all paths
    #print(f'raw exp activity {raw_exp_activity_df.shape}')
    for path in tqdm(embeddings_paths, desc="Processing embedding paths"):
        try:
            # Load and process embeddings
            embeddings_path = os.path.join(embeddings_dir, path)
            raw_embedding_df = pd.read_csv(embeddings_path)
            round_results = []
            for round_num in tqdm(range(rounds_evo), desc=f"Evolution rounds for {path}", leave=False):
                benchmark_reset = start_benchmarks.copy()
                exp_activity_size = len(raw_exp_activity_df)
                #print(exp_activity_size)
                exp_activity_df = raw_exp_activity_df.sample((exp_activity_size // 2 ))
                #print(f'round{round_num} exp df = {exp_activity_df.shape}')

                cleaned_embeddings_df = clean_embeddings_df(exp_activity_df, raw_embedding_df)
                #print(f'Exp activity df {exp_activity_df.shape}')
                #print(wt_activity)
                #print(exp_activity_df)
                above_wt_df = exp_activity_df[exp_activity_df['activity'] >= wt_activity]
                #print(above_wt_df.shape)
                #display(above_wt_df)
                #print(benchmark_reset)
                for benchmark in benchmark_reset:
                    #print(benchmark)
                    if benchmark == 'Above WT':
                        benchmark_reset[benchmark] = above_wt_df
                        #print(f'above WT{benchmark_reset[benchmark]}')
                        continue
                    #print(f'{benchmark}{benchmark_reset[benchmark]}')
                    benchmark_activity = above_wt_df['activity'].quantile(benchmark_reset[benchmark])
                    #print(f'{benchmark} is {benchmark_activity}')
                    #print(f'{benchmark} has cutoff {benchmark_activity}')
                    df = exp_activity_df[exp_activity_df['activity'] >= benchmark_activity]
                    benchmark_reset[benchmark] = df


                top_benchmark = list(benchmark_reset)[-1]
                #print(f"Starting {path} round {(round_num+1)}")
                clean_logits_df = pd.merge(logits_df, cleaned_embeddings_df, on='seq_id', how='inner')
                logits_WTM_df = clean_logits_df.sort_values('wt_marginal', ascending=False)
                logits_nWT_df = clean_logits_df.sort_values('probability', ascending=False)
                WTM_vars = logits_WTM_df.head(zero_shot[0])
                nWT_vars = logits_nWT_df.head(zero_shot[1])
                ran_vars = exp_activity_df.sample(zero_shot[2])
                initial_round_activity = pd.concat([WTM_vars,nWT_vars])
                initial_round_activity = pd.merge(initial_round_activity, exp_activity_df, on='seq_id')
                initial_round_activity = initial_round_activity[['seq_id', 'activity']]
                #print(f'initial round {initial_round_activity.shape}')
                
                round_variants = evolve_simulation(
                    wt_aa_seq, initial_round_activity,
                    cleaned_embeddings_df, exp_activity_df,
                    benchmark_reset, top_benchmark,
                    model_type, num_var
                )
                round_results.append(round_variants)
            
            # Create a dictionary with consistent length data
            path_results = {
                path: round_results
            }
            all_path_results.append(pd.DataFrame(path_results))
            
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            #traceback.print_exc()
            continue
    # Combine results only if we have data
    if all_path_results:
        strategy_df = pd.concat(all_path_results, axis=1)
        evolve_df = pd.concat([evolve_df, strategy_df], axis=1)
    
    # Save results
    results_path = os.path.join(embeddings_dir, dataset)
    os.makedirs(results_path, exist_ok=True)
    
    if not evolve_df.empty:
        raw_comp_dir = os.path.join(results_path, 
                                  f'{zero_shot}_{dataset}.xlsx')
        evolve_df.to_excel(raw_comp_dir, index=False)
        
        zero_shot_string = str(zero_shot)
        print(zero_shot_string)
        plot_evolution_boxplot(evolve_df, benchmarks, 
                             (results_path+"/"+prot_name+ zero_shot_string),zero_shot_string)
    else:
        print(f"No results to save for strategy")
    return evolve_df
def plot_evolution_boxplot(df, benchmarks, save_path=None,zeroshot=None):
    def convert_to_list(value):
        if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value
        return value

    # Convert string representations of lists to actual lists
    #print(df.head())
    #print(benchmarks)
    df = df.map(convert_to_list)
    print(df)
    def reshape_for_violin(df, benchmarks):
        values = []
        positions = []
        columns = []
        
        # Get benchmark keys for reference
        benchmark_keys = list(benchmarks.keys())
        
        # Iterate through each column
        for col in df.columns:
            # Iterate through each row
            for row in df[col].dropna():  # Skip NaN values
                # Ensure row is a list or similar iterable
                if isinstance(row, (list, tuple, np.ndarray)):
                    # Check if length matches benchmarks
                    if len(row) == len(benchmark_keys):
                        for i, value in enumerate(row):
                            if value is not None:  # Skip None values
                                values.append(float(value))  # Convert to float
                                positions.append(benchmark_keys[i])
                                columns.append(col)
        
        # Create DataFrame only if we have data
        if values:
            return pd.DataFrame({
                'Rounds': values,
                'Benchmark': positions,
                'Embeddings': columns
            })
        else:
            print("No valid data for plotting")
            return None

    # Reshape the data
    reshaped_df = reshape_for_violin(df, benchmarks)
    
    # Only plot if we have valid data
    if reshaped_df is not None and not reshaped_df.empty:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=reshaped_df, 
                   x='Benchmark', 
                   y='Rounds', 
                   hue='Embeddings',
                   showmeans=True,
                   meanprops={"marker":"s",
                             "markerfacecolor":"white", 
                             "markeredgecolor":"black"})
        
        plt.title(f'Embeddings {zeroshot} Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path + '_boxplot.png', bbox_inches='tight')
            print(f"Plot saved to {save_path}_boxplot.png")
        
        plt.show()
    else:
        print("Unable to create plot: insufficient or invalid data")

    # Debug information
    #print("\nDataFrame shape:", df.shape)
    #print("\nSample of input DataFrame:")
    #print(df.head())
    #if reshaped_df is not None:
    #    print("\nReshaped DataFrame shape:", reshaped_df.shape)
    #    print("\nSample of reshaped DataFrame:")
    #    print(reshaped_df.head())