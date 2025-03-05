import time
from io import BytesIO
from datetime import datetime, UTC, timedelta
import traceback
import json
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import os
from IPython.display import display
from scipy.stats import sem
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from scipy.interpolate import griddata
from IPython.display import clear_output
import re
from scipy.stats import spearmanr
from sklearn.neural_network import MLPRegressor

#from app.helpers.fold_storage_manager import FoldStorageManager
from app.helpers.sequence_util import (
    get_measured_and_unmeasured_mutant_seq_ids,
    get_loci_set,
    process_and_validate_evolve_input_files,
)

def train_model(wt_aa_seq,raw_activity_df,raw_embedding_df,model,plot=False):
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
    # Get the minimum loss
    if plot and hasattr(model, 'loss_curve_') and len(model.loss_curve_) > 0:
        plt.figure(figsize=(5, 3))
        plt.plot(model.loss_curve_)
        plt.title('Loss Curve during Training')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()
    else:
        test = 1
        #print("No loss curve available")
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
    current_round_activity_df = initial_round_activity
    try:
        # Check initial round
        for benchmark in benchmarks:
            benchmark_df = benchmarks[benchmark]
            if benchmark_df['seq_id'].isin(current_round_activity_df['seq_id']).any() and not benchmarks_hit[benchmark]:
                round_variants.append(round_num)
                benchmarks_hit[benchmark] = True
                if benchmark == top_benchmark:
                    good_var_num = len(pd.merge(current_round_activity_df,benchmarks['Above WT'], on='seq_id'))
                    round_variants.append(good_var_num)
                    round_variants.append(10*good_var_num)
                    evolution_done = True

        with tqdm(total=max_rounds, desc="Evolution Progress", leave=False) as pbar:
            while not evolution_done and round_num < max_rounds:
                round_num += 1
                pbar.update(1)
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
                            good_var_num = len(pd.merge(current_round_activity_df,benchmarks['Above WT'], on='seq_id'))
                            round_variants.append(good_var_num)
                            round_variants.append(10*(good_var_num/round_num))
                            evolution_done = True
                            break
    finally:
        pbar.close()
        print('\033[1A\033[K', end='')
    return round_variants

def clean_embeddings_df(dms_df,embeddings_df):
    clean_embeddings_df = pd.merge(embeddings_df,dms_df,on='seq_id',how='inner')
    #clean_embeddings_df = clean_embeddings_df[['seq_id','embedding']]
    #print(f'clean embeddings {clean_embeddings_df.shape}')
    return clean_embeddings_df
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
        #print(f"Memory reduced from {original_size/1e6:.2f}MB to {new_size/1e6:.2f}MB")
    
    return df
def digivolveZS(wt_aa_seq,prot_name,
                wt_activity,dataset,exp_activity_file_path,
                embeddings_dir, embeddings_paths,
                num_var,rounds_evo,model_type,
                logits_path,zero_shot,new_variable,complete=None):
    #Set up dataframes
    raw_exp_activity_df = optimize_memory_with_checks(pd.read_excel(exp_activity_file_path))
    exp_activity_size = len(raw_exp_activity_df)
    logits_df = optimize_memory_with_checks(pd.read_csv(logits_path))

    evolve_df = pd.DataFrame()
    all_path_results = []  # Store results for all paths
    for path in tqdm(embeddings_paths, desc="Processing embedding paths"):
        start_benchmarks = {
         'Above WT' : wt_activity,
         '50th Percentile' : 0.5,
         '90th Percentile' : 0.9,
         }
        try:
            # Load and process embeddings
            embeddings_path = os.path.join(embeddings_dir, path)
            raw_embedding_df = pd.read_csv(embeddings_path)
            round_results = []
            if complete:
                print("full data")
                rounds_evo = 1
            for round_num in tqdm(range(rounds_evo), desc=f"Evolution rounds for {path}", leave=False):
                if not complete:
                    exp_activity_df = raw_exp_activity_df.sample((exp_activity_size // 2 ))
                    #print(f"not compelete is {exp_activity_df.shape}")
                else:
                    exp_activity_df = raw_exp_activity_df.copy()
                    #print(f"compelete is {exp_activity_df.shape}")
                cleaned_embeddings_df = clean_embeddings_df(exp_activity_df, raw_embedding_df)
                cleaned_embeddings_df = cleaned_embeddings_df[['seq_id','embedding']]
                above_wt_df = exp_activity_df[exp_activity_df['activity'] >= wt_activity]

                if len(above_wt_df) > 100:
                    start_benchmarks.update({'95th Percentile' : 0.95,})

                benchmark_reset = start_benchmarks.copy()

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
                clean_logits_df = pd.merge(logits_df, exp_activity_df, on='seq_id', how='inner')
                logits_WTM_df = clean_logits_df.sort_values('wt_marginal', ascending=False)
                logits_nWT_df = clean_logits_df.sort_values('probability', ascending=False)
                def extract_locus(mutation):
                    # Use regex to extract the number part
                    match = re.search(r'[A-Z](\d+)[A-Z]', mutation)
                    if match:
                        return int(match.group(1))
                    return None
                logits_WTM_df['locus'] = logits_WTM_df['seq_id'].apply(extract_locus)
                WTM_vars = logits_WTM_df.head(zero_shot[0])
                top_n_locus = logits_WTM_df.groupby('locus').head(3)
                top_n_locus = top_n_locus.sort_values('wt_marginal', ascending=False)
                nWT_vars = top_n_locus.head(zero_shot[1])
                #nWT_vars = logits_nWT_df.head(zero_shot[1])
                ran_vars = exp_activity_df.sample(zero_shot[2])
                initial_round_activity = pd.concat([WTM_vars,nWT_vars,ran_vars])
                initial_round_activity = initial_round_activity[['seq_id', 'activity']]
                
                round_variants = evolve_simulation(
                    wt_aa_seq, initial_round_activity,
                    cleaned_embeddings_df, exp_activity_df,
                    benchmark_reset, top_benchmark,
                    model_type, num_var
                )
                round_results.append(round_variants)
            
            # Create a dictionary with consistent length data
            path_results = {
                (path, new_variable): round_results
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
        start_benchmarks.update({'Number of Var>WT': 0})
        start_benchmarks.update({'Avg var per round(*10)': 0})
    
    # Save results
    results_path = os.path.join(embeddings_dir, dataset)
    os.makedirs(results_path, exist_ok=True)
    
    if not evolve_df.empty:
        raw_comp_dir = os.path.join(results_path, 
                                  f'{new_variable}_{dataset}.xlsx')
        evolve_df.to_excel(raw_comp_dir)
        
        variable_string = str(new_variable)
        #print(zero_shot_string)
        plot_evolution_boxplot(evolve_df, start_benchmarks,variable_string,
                             (results_path+"/"+prot_name+ variable_string))
    else:
        print(f"No results to save for strategy")
    clear_output(wait=False)
    return evolve_df
def evolve_multi(wt_aa_seq, initial_round_activity, mutant_dfs, exp_activity_df, 
                 benchmarks, top_benchmark, layers,
                 num_var, max_rounds, scope, depth, progressive):
    """
    Performs evolutionary optimization of protein sequences.
    
    Args:
        wt_aa_seq: Wild-type amino acid sequence
        initial_round_activity: DataFrame with initial activity data
        mutant_dfs: Dictionary of DataFrames containing mutant information
        exp_activity_df: DataFrame with experimental activity data
        benchmarks: Dictionary of benchmark DataFrames
        top_benchmark: Key of the benchmark that triggers termination when found
        model_type: Type of model to use for prediction
        num_var: Number of variants to evaluate per round
        max_rounds: Maximum number of rounds to run
        scope: Maximum mutation scope to consider
        depth: Depth of mutations to include in combined dataset
        progressive: If integer, gradually increases scope every N rounds
        
    Returns:
        tuple: (round_variants, var_found, spearman_df)
    """
    # Initialize variables
    round_num = 0
    round_variants = []
    benchmarks_hit = {benchmark: False for benchmark in benchmarks}
    evolution_done = False
    current_round_activity_df = initial_round_activity
    
    # Initialize DataFrames
    scope_mutants = pd.DataFrame()
    multi_activities = pd.DataFrame()
    combined_df = pd.DataFrame()
    var_found = pd.DataFrame()
    spearman_correlations = []
    current_layers = layers.copy()
    
    # Set up mutation scope
    if progressive:
        max_rounds = max_rounds + (progressive * (depth - 1))
        rounds = progressive
        current_scope = 1
        scope_mutants = pd.concat([scope_mutants, mutant_dfs["mut_1"]])
        #current_layers = [len(scope_mutants)] + layers
    else:
        for i in range(1, scope + 1):
            scope_mutants = pd.concat([scope_mutants, mutant_dfs[f"mut_{i}"]])
    
    # Prepare combined DataFrame for validation
    if depth == scope:
        combined_df = scope_mutants.copy()
    else:
        for i in range(1, depth + 1):
            combined_df = pd.concat([combined_df, mutant_dfs[f"mut_{i}"]])
    
    try:
        with tqdm(total=max_rounds, desc="Evolution Progress", leave=False) as pbar:
            while not evolution_done and round_num < max_rounds:
                round_num += 1
                pbar.update(1)
                
                # Train model and predict activities
                model_type = MLPRegressor(random_state=1,max_iter=5000, hidden_layer_sizes=current_layers)
                predicted_activity_df = train_model(wt_aa_seq, current_round_activity_df,
                                                  scope_mutants, model_type)
                
                # Handle progressive scope increase if enabled
                if progressive:
                    if current_scope < depth and not(round_num % rounds):                      
                        current_scope += 1
                        scope_mutants = pd.concat([scope_mutants, mutant_dfs[f"mut_{current_scope}"]])
                        #current_layers[0] = len(scope_mutants)

                    
                    validation_df = pd.merge(
                        predicted_activity_df[['seq_id', 'predicted_activity']], 
                        exp_activity_df[['seq_id', 'activity']], 
                        on='seq_id', 
                        how='inner'
                    )
                    
                    actual_activities = predicted_activity_df[predicted_activity_df["actual_activity"].isna()].head(num_var)
                    found_var_activities = pd.merge(actual_activities, combined_df, on='seq_id', how='inner')
                else:
                    # Train on combined dataset for validation
                    multi_predicted_activity_df = train_model(wt_aa_seq, current_round_activity_df,
                                                    combined_df, model_type)
                    
                    validation_df = pd.merge(
                        multi_predicted_activity_df[['seq_id', 'predicted_activity']], 
                        exp_activity_df[['seq_id', 'activity']], 
                        on='seq_id', 
                        how='inner'
                    )
                    
                    multi_activities = multi_predicted_activity_df[multi_predicted_activity_df['actual_activity'].isna()].head(num_var)
                    found_var_activities = pd.merge(multi_activities, combined_df, on='seq_id', how='inner')
                
                # Evaluate predictions and update current activity dataframe
                current_round_activity_df = evaluate_predictions(
                    predicted_activity_df,
                    exp_activity_df,
                    current_round_activity_df,
                    num_var
                )

                # Calculate Spearman correlation if enough data points
                if len(validation_df) > 2:
                    correlation, p_value = spearmanr(
                        validation_df['predicted_activity'], 
                        validation_df['activity']
                    )
                    spearman_correlations.append({
                        'round': round_num,
                        'correlation': correlation,
                        'p_value': p_value,
                        'n_samples': len(validation_df)
                    })
                    #print(f"Round {round_num} - Spearman correlation: {correlation:.3f} (p={p_value:.3e}, n={len(validation_df)})")

                # Track found variants
                found_var_activities['round_found'] = round_num
                var_found = pd.concat([var_found, found_var_activities], ignore_index=True)
                
                # Check if benchmarks are hit
                for benchmark in benchmarks:
                    benchmark_df = benchmarks[benchmark]
                    if benchmark_df['seq_id'].isin(found_var_activities['seq_id']).any() and not benchmarks_hit[benchmark]:
                        #print(f"{benchmark} in round {round_num}") 
                        round_variants.append(round_num)
                        benchmarks_hit[benchmark] = True          
                        if benchmark == top_benchmark:
                            pbar.update(max_rounds)
                            pbar.clear()
                            pbar.close()
                            evolution_done = True
                            
                            break
    finally:
        # Process results
        for benchmark in benchmarks_hit.keys():
            if not benchmarks_hit[benchmark]:
                round_variants.append(None)
        
        # Group found variants by seq_id and round
        round_info = var_found.groupby('seq_id')['round_found'].apply(list).reset_index()
        other_cols = var_found.drop(columns=['round_found']).drop_duplicates(subset='seq_id', keep='first')
        var_found = pd.merge(other_cols, round_info, on='seq_id')
        
        # Calculate performance metrics
        good_var = len(benchmarks['90th Percentile'].merge(var_found, on='seq_id'))
        round_variants.append(good_var)
        
        # Create Spearman correlation DataFrame
        spearman_df = pd.DataFrame(spearman_correlations)
        
        # Calculate and report best variant and correlation
        best_var = round(var_found['percentile'].max(), 3)
        #print(f"best var = {best_var}")
        
        max_spearman = round(spearman_df['correlation'].max(), 3) if not spearman_df.empty else 0
        #print(f"maxSpear = {max_spearman}")
        
        round_variants.append(best_var)
        round_variants.append(round(var_found['activity'].max(), 3))
        round_variants.append(max_spearman)
        

        #print('\033[1A\033[K', end='')
        
    return round_variants, var_found, spearman_df
def digivolve_multi(wt_aa_seq, prot_name, wt_activity, dataset, exp_activity_file_path,
                   embeddings_dir, embeddings_paths, num_var, layers, logits_path, 
                   scope, depth, progressive=None, full=None, show=None,):
    """
    Process protein sequences for directed evolution using multiple embedding paths.
    
    Args:
        wt_aa_seq: Wild-type amino acid sequence
        prot_name: Protein name
        wt_activity: Wild-type activity value
        dataset: Dataset name
        exp_activity_file_path: Path to experimental activity data
        embeddings_dir: Directory containing embeddings
        embeddings_paths: List of paths to embedding files
        num_var: Number of variants to consider
        layers: Network layer configuration
        logits_path: Path to logits file
        scope: Scope parameter for evolution
        depth: Depth parameter for evolution
        progressive: Progressive evolution flag
        show: Show visualization flag
    
    Returns:
        DataFrame containing evolution results
    """
    # Load and preprocess experimental activity data
    raw_exp_activity_df = optimize_memory_with_checks(pd.read_excel(exp_activity_file_path))
    #exp_activity_df = raw_exp_activity_df.sort_values('activity', ascending=False)
    raw_exp_activity_df['activity'] = normalize_dataset(raw_exp_activity_df['activity'], wt_activity)
    
    # Load logits data
    logits_df = optimize_memory_with_checks(pd.read_csv(logits_path))
    
    # Initialize containers for results
    mutant_dfs = {}
    round_dict = {}
    spearman_round_dict ={}
    path_results = {}
    evolve_df = pd.DataFrame()
    round_results = []  # Store results for all paths
    rel_wt_activity = wt_activity/abs(wt_activity)
    # Define benchmarks for evaluation
    start_benchmarks = {
        'Above Median': (rel_wt_activity),
        '50th Percentile': 0.5,
        '90th Percentile': 0.9,
        '95th Percentile': 0.95,
        '99th Percentile': 0.99,
    }
    
    # Process each embedding path
    for path in tqdm(embeddings_paths, desc=f"Processing {dataset}"):
            # Load and process embeddings
            for round_num in tqdm(range(10), desc=f"Evolution rounds for {dataset}", leave=False):

                #print(f"{dataset} round {round_num}")
                embeddings_path = os.path.join(embeddings_dir, path)
                raw_embedding_df = pd.read_csv(embeddings_path)
                if full:
                    exp_activity_sample = raw_exp_activity_df.copy()
                else:
                    exp_activity_sample = raw_exp_activity_df.sample(int((len(raw_exp_activity_df) // 2 )))
                # Filter for variants above wild-type activity
                above_wt_df = exp_activity_sample[exp_activity_sample['activity'] >= rel_wt_activity]

                # Set up benchmark datasets
                benchmarks_reset = start_benchmarks.copy()
                for benchmark in benchmarks_reset:
                    if benchmark == 'Above Median':
                        benchmarks_reset[benchmark] = above_wt_df
                        continue
                    benchmark_activity = above_wt_df['activity'].quantile(benchmarks_reset[benchmark])
                    df = exp_activity_sample[exp_activity_sample['activity'] >= benchmark_activity]
                    benchmarks_reset[benchmark] = df

                top_benchmark = list(benchmarks_reset)[-1]

                # Clean and prepare embeddings dataframe
                cleaned_embeddings_df = clean_embeddings_df(exp_activity_sample, raw_embedding_df)
                cleaned_embeddings_df['rank'] = cleaned_embeddings_df['activity'].rank(method='min', ascending=False)
                cleaned_embeddings_df['percentile'] = cleaned_embeddings_df['activity'].rank(pct=True)
                cleaned_embeddings_df['mut_count'] = cleaned_embeddings_df['seq_id'].str.count('_') + 1

                # Organize variants by mutation count
                for i in range(1, cleaned_embeddings_df['mut_count'].max() + 1):
                    mutants = cleaned_embeddings_df[cleaned_embeddings_df['mut_count'] == i].copy()
                    mutant_dfs[f"mut_{i}"] = mutants

                # Prepare for evolution
                max_rounds = 10

                # Process logits for single mutants
                clean_logits_df = pd.merge(logits_df, mutant_dfs['mut_1'], on='seq_id', how='inner')
                logits_WTM_df = clean_logits_df.sort_values('wt_marginal', ascending=False)
                WTM_vars = logits_WTM_df.head(num_var)
                initial_round_activity = WTM_vars[['seq_id', 'activity']]

                # Run evolution
                round_variants, top_variants, spearman_df = evolve_multi(
                    wt_aa_seq, initial_round_activity, mutant_dfs, exp_activity_sample, 
                    benchmarks_reset, top_benchmark, layers,
                    num_var, max_rounds, scope, depth, progressive
                )
                #print(round_variants)
                round_results.append(round_variants)
                
                # Process top variants
                top_variants = top_variants.sort_values('percentile', ascending=False)
                top_variants = top_variants[['seq_id', 'percentile', 'round_found', 
                                             'predicted_activity', 'relevant_measured_mutants', 'activity']]
                round_dict.update({f'round_{round_num+1}_variants': list(zip(top_variants['seq_id'], top_variants['relevant_measured_mutants']))})
                spearman_round_dict.update({f'round_{round_num+1}_spearman' : spearman_df})
    # Store results for this path
    path_results[dataset] = round_results
    # Combine results if we have data
    if path_results:
        evolve_df = pd.DataFrame(path_results)
        
        # Update benchmarks with additional metrics
        start_benchmarks.update({
            'Number of 90% Variants': 0,
            'Best Variant Percentile': 0,
            'Best Variant Fold Better': 0,
            'Max Spearman': 0
        })
    
    # Save results
    results_path = os.path.join(embeddings_dir, dataset)
    os.makedirs(results_path, exist_ok=True)
    
    if not evolve_df.empty:
        # Save to Excel

        raw_comp_dir = os.path.join(results_path, f'FolDE_{dataset}.xlsx')
        try:
            df = pd.DataFrame(round_dict)
            df.to_excel(os.path.join(results_path, f'VarsFound_{dataset}.xlsx'))
        except Exception as e:
            print(f"Error saving variant DataFrames: {str(e)}")
        evolve_df.to_excel(raw_comp_dir, index=False)
        top_variants.to_excel(os.path.join(results_path, f'TopVar_{dataset}.xlsx'))
        #spearman_df.to_excel(os.path.join(results_path, f'Spearman_{dataset}.xlsx'))
        
        variable_string = 'FolDE'
        
        # Plot Spearman correlation if we have multiple rounds
        graph_and_save_spearman(spearman_round_dict,results_path,prot_name,dataset)
        
        # Plot evolution boxplot
        plot_evolution_boxplot(
            evolve_df, 
            start_benchmarks,
            variable_string,
            os.path.join(results_path, f"{prot_name}_{variable_string}"),
            show,color_by_count=True
        )
    else:
        print("No results to save for strategy")
    
    return evolve_df
def graph_and_save_spearman(spearman_dict, results_path, prot_name, dataset,save = None):
    if spearman_dict:
        # Prepare data for plotting
        all_spearman_dfs = []

        # Collect all Spearman DataFrames
        for key, df in spearman_dict.items():
            # Make a copy with the round key as an identifier
            df_copy = df.copy()
            df_copy['source'] = key
            all_spearman_dfs.append(df_copy)

        # Combine all DataFrames
        combined_spearman = pd.concat(all_spearman_dfs, ignore_index=True)

        # Calculate statistics by round
        spearman_stats = combined_spearman.groupby('round').agg(
            mean_correlation=('correlation', 'mean'),
            std_correlation=('correlation', 'std'),
            count=('correlation', 'count')
        ).reset_index()

        # Calculate standard error
        spearman_stats['stderr'] = spearman_stats['std_correlation'] / np.sqrt(spearman_stats['count'])
        if not save:

            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 7))
    
            # Calculate opacity based on count
            min_count = spearman_stats['count'].min()
            max_count = spearman_stats['count'].max()
            
            # Function to calculate opacity - minimum 0.3, maximum 1.0
            def get_opacity(count):
                if max_count == min_count:  # Avoid division by zero
                    return 0.8  # Default opacity if all counts are the same
                return 0.3 + 0.7 * (count - min_count) / (max_count - min_count)
            
            # Add opacity column to the dataframe
            spearman_stats['opacity'] = spearman_stats['count'].apply(get_opacity)
            
            # Main color for all elements
            main_color = 'blue'
            
            # Plot points with opacity based on count
            for i, row in spearman_stats.iterrows():
                x = row['round']
                y = row['mean_correlation']
                opacity = row['opacity']
                
                # Plot the point
                ax.scatter(x, y, s=100, color=main_color, alpha=opacity, 
                          edgecolor='black', linewidth=1, zorder=5)
            
            # Connect points with lines (opacity based on the average count)
            for i in range(len(spearman_stats) - 1):
                # Get the two points
                x1, y1 = spearman_stats.loc[i, 'round'], spearman_stats.loc[i, 'mean_correlation']
                x2, y2 = spearman_stats.loc[i+1, 'round'], spearman_stats.loc[i+1, 'mean_correlation']
                
                # Get the opacities
                opacity1 = spearman_stats.loc[i, 'opacity']
                opacity2 = spearman_stats.loc[i+1, 'opacity']
                avg_opacity = (opacity1 + opacity2) / 2
                
                # Draw the line with opacity based on average count
                ax.plot([x1, x2], [y1, y2], '-', color=main_color, 
                       alpha=avg_opacity, linewidth=2, zorder=4)
    
            # Add error bars with opacity
            for i, row in spearman_stats.iterrows():
                x = row['round']
                y = row['mean_correlation']
                yerr = row['stderr']
                opacity = row['opacity'] * 0.7  # Slightly more transparent than the points
                
                # Plot error bars
                ax.fill_between(
                    [x-0.2, x+0.2],  # Make error bars a bit wider for visibility
                    [y-yerr, y-yerr],
                    [y+yerr, y+yerr],
                    alpha=opacity, 
                    color=main_color,
                    zorder=3
                )
    
            # Add reference line
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.3, zorder=2)
    
            # Add count annotations to each point
            for i, row in spearman_stats.iterrows():
                ax.annotate(
                    f"n={int(row['count'])}",
                    (row['round'], row['mean_correlation']),
                    xytext=(0, 10),  # Offset text 10 points above
                    textcoords='offset points',
                    ha='center',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7, edgecolor='gray')
                )
    
            # Add a legend explaining opacity
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=main_color, alpha=get_opacity(max_count), 
                     edgecolor='black', label=f'High sample count (n={int(max_count)})'),
                Patch(facecolor=main_color, alpha=get_opacity(min_count), 
                     edgecolor='black', label=f'Low sample count (n={int(min_count)})')
            ]
            
            # Add legend for error bars if they're different from the main elements
            legend_elements.append(
                Patch(facecolor=main_color, alpha=0.5, 
                     edgecolor=None, label='Standard Error')
            )
            
            ax.legend(handles=legend_elements, loc='best')
    
            # Formatting
            ax.set_xlabel('Round')
            ax.set_ylabel('Spearman Correlation')
            ax.set_title('Average Model Prediction Quality Over Evolution Rounds')
            ax.grid(True, alpha=0.3, zorder=1)
            
            # Set y-axis limits with some padding
            y_min = min(0, spearman_stats['mean_correlation'].min() - spearman_stats['stderr'].max()) - 0.05
            y_max = spearman_stats['mean_correlation'].max() + spearman_stats['stderr'].max() + 0.05
            ax.set_ylim(y_min, y_max)
            
            # Adjust x-axis to be integers if rounds are integers
            if all(spearman_stats['round'].apply(lambda x: x.is_integer() if isinstance(x, float) else True)):
                ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
            plt.tight_layout()
    
            # Also save a high-quality vector version
            plt.savefig(os.path.join(results_path, f"{prot_name}_avg_spearman.svg"), bbox_inches='tight', format='svg')
            #print(f"Saved vector version to {os.path.join(results_path, f'{prot_name}_avg_spearman.svg')}")
    
        # Also save the statistics DataFrame
        spearman_stats.to_excel(os.path.join(results_path, f"SpearmanStats_{dataset}.xlsx"), index=False)
        #print(f"Saved statistics to {os.path.join(results_path, f'SpearmanStats_{dataset}.xlsx')}")
        
        #plt.close()
        
        return spearman_stats
    else:
        print("No Spearman correlation data available for plotting.")
        return None
def convert_to_list(value):
    if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value
def plot_evolution_boxplot(df_input, benchmarks, new_variable, save_path=None, show=None, color_by_count=False):
    """
    Create boxplots comparing different embeddings across benchmarks.
    
    Parameters:
    -----------
    df_input : pandas.DataFrame or list of pandas.DataFrame
        DataFrame(s) containing the data to plot
    benchmarks : dict
        Dictionary of benchmarks
    new_variable : str
        Variable name for the title
    save_path : str, optional
        Path to save the plot
    show : bool, optional
        Whether to display the plot
    color_by_count : bool, optional
        Whether to use opacity based on the number of data points (default: False)
    """
    # Handle both single DataFrame and list of DataFrames
    if isinstance(df_input, list):
        # Process each DataFrame in the list
        processed_dfs = []
        for i, df in enumerate(df_input):
            # Apply convert_to_list to each DataFrame
            processed_df = df.map(convert_to_list)
            processed_dfs.append(processed_df)
    else:
        # Single DataFrame case
        processed_dfs = [df_input.map(convert_to_list)]
    
    def reshape_for_violin(dfs, benchmarks):
        values = []
        positions = []
        columns = []
        sources = []  # To track which DataFrame each point came from
        
        # Get benchmark keys for reference - maintain original order
        benchmark_keys = list(benchmarks.keys())
        
        for df_idx, df in enumerate(dfs):
            # Iterate through each column
            for col in df.columns:
                # Iterate through each row
                for row in df[col]:  # Skip NaN values
                    # Ensure row is a list or similar iterable
                    if isinstance(row, (list, tuple, np.ndarray)):
                        # Check if length matches benchmarks
                        if len(row) == len(benchmark_keys):
                            for i, value in enumerate(row):
                                if value is not None:  # Skip None values
                                    values.append(float(value))  # Convert to float
                                    positions.append(benchmark_keys[i])
                                    columns.append(col)
                                    sources.append(df_idx)  # Record which DataFrame this came from
        
        # Create DataFrame only if we have data
        if values:
            result_df = pd.DataFrame({
                'Rounds': values,
                'Benchmark': positions,
                'Embeddings': columns,
                'Source': sources
            })
                
            # Convert Benchmark to categorical to preserve order
            result_df['Benchmark'] = pd.Categorical(
                result_df['Benchmark'], 
                categories=benchmark_keys,
                ordered=True
            )
                
            return result_df
        else:
            print("No valid data for plotting")
            return None

    # Reshape the data
    reshaped_df = reshape_for_violin(processed_dfs, benchmarks)
    
    # Only plot if we have valid data
    if reshaped_df is not None and not reshaped_df.empty:
        # Create figure with proper layout
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get benchmark keys in the original order
        benchmark_keys = list(benchmarks.keys())
        
        # Determine grouping columns
        groupby_cols = ['Benchmark', 'Embeddings', 'Source']
                
        # Count the number of data points for each combination
        count_df = reshaped_df.groupby(groupby_cols, observed=False).size().reset_index(name='Count')
        
        # Get count statistics
        min_count = count_df['Count'].min()
        max_count = count_df['Count'].max()
        
        # Get unique values for each dimension
        # Use the original benchmark order instead of unique values
        benchmarks_list = benchmark_keys
        embeddings_list = reshaped_df['Embeddings'].unique()
        sources_list = sorted(reshaped_df['Source'].unique())  # Ensure sorted order
        n_sources = len(sources_list)
        
        # Create a colormap for different sources
        source_colors = plt.cm.tab10(np.linspace(0, 1, max(n_sources, 10)))
        
        # Plot each group separately with opacity based on count
        for i, benchmark in enumerate(benchmarks_list):
            # Skip benchmarks that have no data at all
            if benchmark not in reshaped_df['Benchmark'].values:
                continue
                
            for j, embedding in enumerate(embeddings_list):
                for k, source in enumerate(sources_list):
                    # Filter data for this combination
                    subset = reshaped_df[(reshaped_df['Benchmark'] == benchmark) & 
                                       (reshaped_df['Embeddings'] == embedding) &
                                       (reshaped_df['Source'] == source)]
                    
                    # Skip if no data for this specific combination
                    if subset.empty:
                        continue
                        
                    # Get count for this combination
                    count_filter = (count_df['Benchmark'] == benchmark) & \
                                  (count_df['Embeddings'] == embedding) & \
                                  (count_df['Source'] == source)
                    
                    # Get count
                    count_rows = count_df[count_filter]
                    if len(count_rows) == 0:
                        continue  # Skip if no count data
                        
                    count = count_rows['Count'].values[0]
                    
                    # Calculate opacity based on count
                    if color_by_count:
                        # Normalize count to get opacity (range 0.3 to 1.0)
                        if max_count > min_count:  # Avoid division by zero
                            opacity = 0.3 + 0.7 * (count - min_count) / (max_count - min_count)
                        else:
                            opacity = 0.8
                    else:
                        opacity = 0.8  # Default opacity if not coloring by count
                    
                    # Get base color for this source - use k as index to source_colors
                    base_color = source_colors[k % len(source_colors)]
                    
                    # Position for this box (offset by embedding and source)
                    # Adjust the position calculation to account for multiple sources
                    width_factor = 0.8 / (len(embeddings_list) * n_sources)
                    offset = j * n_sources + k
                    total_groups = len(embeddings_list) * n_sources
                    pos = i + (offset - total_groups/2 + 0.5) * width_factor
                    
                    # Create boxplot
                    boxplot = ax.boxplot(subset['Rounds'], positions=[pos], widths=width_factor*0.8, 
                                        patch_artist=True, showmeans=True,
                                        meanprops={"marker":"s", "markerfacecolor":"white", 
                                                  "markeredgecolor":"black"})
                    
                    # Set the color with appropriate opacity
                    for patch in boxplot['boxes']:
                        patch.set_facecolor((*base_color[:3], opacity))
                    
                    # Set other elements with appropriate opacity
                    for element in ['whiskers', 'caps', 'medians']:
                        for line in boxplot[element]:
                            line.set_alpha(opacity)
                            line.set_color(base_color[:3])  # Use only RGB components
        
        # Set x-ticks for all benchmarks, even those without data
        ax.set_xticks(range(len(benchmarks_list)))
        ax.set_xticklabels(benchmarks_list, rotation=45)
        
        # Create a custom legend for embeddings and sources
        from matplotlib.patches import Patch
        
        # Create legend for embeddings
        embedding_legend = [Patch(facecolor='white', edgecolor='black', 
                                label=f"{emb}") for emb in embeddings_list]
        
        # Only add source legend if there are multiple sources
        all_legend_elements = embedding_legend.copy()
        
        if len(sources_list) > 1:
            # Create legend for sources - use the actual source indices
            source_legend = []
            for k, source in enumerate(sources_list):
                color = source_colors[k % len(source_colors)]
                # Use "Input 1", "Input 2" instead of "Source 0", "Source 1"
                source_legend.append(Patch(facecolor=color, edgecolor='black', 
                                          label=f"Input {source+1}"))
            
            # Add source legend elements
            all_legend_elements.extend(source_legend)
        
        ax.legend(handles=all_legend_elements, title="Legend", loc='upper right')
        
        if color_by_count:
            # Create a custom colorbar to show opacity scale
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            
            # Create a separate axis for the opacity legend
            opacity_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])  # [left, bottom, width, height]
            
            # Create a gradient image
            gradient = np.linspace(0, 1, 256).reshape(256, 1)
            gradient = np.vstack((gradient, gradient))
            
            # Display the gradient image
            opacity_ax.imshow(gradient, aspect='auto', cmap=plt.cm.gray_r)
            
            # Configure the opacity legend
            opacity_ax.set_title('Count')
            opacity_ax.set_xticks([])
            opacity_ax.set_yticks([0, 255])
            opacity_ax.set_yticklabels([0, max_count])
        
        ax.set_title(f'Embeddings {new_variable} Comparison')
        ax.set_xlabel('Benchmark')
        ax.set_ylabel('Rounds')
        ax.set_ylim(-1, 20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path + '_boxplot.png', bbox_inches='tight')
            print(f"Plot saved to {save_path}_boxplot.png")
        if show:
            plt.show()
    else:
        print("Unable to create plot: insufficient or invalid data")
def string_to_array(embedding_string):
    return np.array(ast.literal_eval(embedding_string))
def plot_combined_landscape(activity,embeddings,logits,protname,embedding_dir):
    activity = optimize_memory_with_checks(pd.read_excel(activity))
    logits = optimize_memory_with_checks(pd.read_csv(os.path.join(embedding_dir,'embeddings',logits)))
    embeddings_df = pd.read_csv(os.path.join(embeddings))
    # Load and prepare data as before
    datasets = ['activity','wt_marginal']

    df = pd.merge(activity, embeddings_df, on='seq_id')
    df = pd.merge(df, logits, on='seq_id')

    X = np.stack(df['embedding'].apply(string_to_array).values)

    reducer = TSNE(n_components=2,
                    random_state=42,
                    perplexity=5, # Lower perplexity (default 30) focuses more on local structure
                    early_exaggeration=80.0, # Higher value (default 12.0) increases space between clusters
                    learning_rate=200, # Higher value can help spread points more (default 200)
                    max_iter=2000, # More iterations for better convergence
                    metric='euclidean')  
        
    X_2d = reducer.fit_transform(X)

    fig = go.Figure()
    normalized_marginal = pd.DataFrame()
    grid_size = 100
    x_range = np.linspace(X_2d[:, 0].min(), X_2d[:, 0].max(), grid_size)
    y_range = np.linspace(X_2d[:, 1].min(), X_2d[:, 1].max(), grid_size)
    x_mesh, y_mesh = np.meshgrid(x_range, y_range)
    normalized_marginal['wt_marginal'] = normalize_dataset(df['wt_marginal'])
    
    # Multiply all datasets
    z_mesh = griddata(
        points=(X_2d[:, 0], X_2d[:, 1]),
        values=normalized_marginal['wt_marginal'],
        xi=(x_mesh, y_mesh),
        method='linear'
    )
    # Split data into two groups based on activity
    high_activity = df['activity'] > 1
    low_activity = ~high_activity

    # Add surface plot
    fig.add_trace(go.Surface(
        x=x_range,
        y=y_range,
        z=z_mesh,
        colorscale='viridis',
        showscale=True,
        name='WT Marginal Surface'
    ))
    
    # Create hover text with original values
    hover_text = []
    for idx in range(len(df)):
        text = f"Sequence ID: {df['seq_id'].iloc[idx]}<br>"
        text += f"WT Marginal: {df['wt_marginal'].iloc[idx]:.3f}<br>"
        text += f"Activity: {df['activity'].iloc[idx]:.3f}<br>"
        hover_text.append(text)
    
       # Add scatter points with detailed hover information
    fig.add_trace(go.Scatter3d(
        x=X_2d[low_activity, 0],
        y=X_2d[low_activity, 1],
        z=normalized_marginal['wt_marginal'][low_activity],  # Only use low activity points
        mode='markers',
        marker=dict(
            size=1,
            color='blue',  # or use z_combined[low_activity] for gradient
            opacity=0.2
        ),
        text=[hover_text[i] for i in range(len(hover_text)) if not high_activity[i]],
        hoverinfo='text',
        name=''
    ))

    # Add highlighted high activity points
    fig.add_trace(go.Scatter3d(
        x=X_2d[high_activity, 0],
        y=X_2d[high_activity, 1],
        z=normalized_marginal['wt_marginal'][high_activity],  # Only use high activity points
        mode='markers',
        marker=dict(
            size=4,
            color=df['activity'][high_activity],
            colorscale='Reds',
            colorbar=dict(
                x=1.1
            ),
            symbol='diamond',
            opacity=1.0
        ),
        text=[hover_text[i] for i in range(len(hover_text)) if high_activity[i]],
        hoverinfo='text',
        name=''
    ))
    # Update layout
    method_name = type(reducer).__name__
    fig.update_layout(
        title=f'Combined Landscape ({" × ".join(datasets)}) using {method_name}',
        scene=dict(
            xaxis_title=f'{method_name}_1',
            yaxis_title=f'{method_name}_2',
            zaxis_title='WT Marginal',
            aspectratio=dict(x=1, y=1, z=0.5),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=900,
        height=900,
        showlegend=True,
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    output = os.path.join(embedding_dir,f'{protname}_landscape.html')
    fig.write_html(output)
    fig.show()
def plotembeddings(X_2d,df,reducer,data,protname):
    # Create figure
    fig = go.Figure()
    if data != 'activity':
            df[data] =normalize_dataset(df[data])
    else:
        ()
    # Add surface plot
    grid_size = 100
    x_range = np.linspace(X_2d[:, 0].min(), X_2d[:, 0].max(), grid_size)
    y_range = np.linspace(X_2d[:, 1].min(), X_2d[:, 1].max(), grid_size)
    x_mesh, y_mesh = np.meshgrid(x_range, y_range)

    
    z_mesh = griddata(
        points=(X_2d[:, 0], X_2d[:, 1]),
        values=df[data],
        xi=(x_mesh, y_mesh),
        method='linear'
    )

    fig.add_trace(go.Surface(
        x=x_range,
        y=y_range,
        z=z_mesh,
        opacity=0.7,
        colorscale='viridis',
        showscale=True
    ))
    if data == 'activity':
        # Split data into two groups based on activity
        high_activity = df[data] > 1
        low_activity = ~high_activity

        # Add regular points
        fig.add_trace(go.Scatter3d(
            x=X_2d[low_activity, 0],
            y=X_2d[low_activity, 1],
            z=df.loc[low_activity, data],
            mode='markers',
            marker=dict(
                size=1,
                color=df.loc[low_activity, data],
                colorscale='viridis',
                opacity=0.2
            ),
            text=df.loc[low_activity, 'seq_id'],
            name='Activity ≤ 1'
        ))

        # Add highlighted points
        fig.add_trace(go.Scatter3d(
            x=X_2d[high_activity, 0],
            y=X_2d[high_activity, 1],
            z=df.loc[high_activity, data],
            mode='markers',
            marker=dict(
                size=2,
                color='red',
                symbol='diamond',
                opacity=0.5
            ),
            text=df.loc[high_activity, 'seq_id'],
            name='Activity > 1'
        ))
    else:
        fig.add_trace(go.Scatter3d(
            x=X_2d[:, 0],
            y=X_2d[:, 1],
            z=df[data],
            mode='markers',
            marker=dict(
                size=1,
                color=df[data],
                colorscale='viridis',
                opacity=0.5
            ),
            text=df['seq_id'],
            name=f'{data}'
        ))

    # Update layout
    method_name = type(reducer).__name__
    fig.update_layout(
        title=f'3D Fitness Landscape using {method_name}',
        scene=dict(
            xaxis_title=f'{method_name}_1',
            yaxis_title=f'{method_name}_2',
            zaxis_title=data,
            aspectratio=dict(x=1, y=1, z=0.5),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=900,
        height=900,
        showlegend=True
    )
    fig.write_html(f'{protname}_{data}_landscape.html')
    #fig.show()
def normalize_dataset(data,wt_activity):
    """Normalize data to range [0,1] by dividing by max value"""
    new_data = data/abs(wt_activity)
    return (new_data)