'''
Author: Mitas Ray
Date: 2025-01-21
Last Editor: Gil 
Last Edit Date: 2025-07-24

Description: framework for training and evaluating yield spread models.

MAJOR UPDATE (July 2025): Extended from a training-only script to include:
- Historical model evaluation capabilities
- Production-style backtesting across all models
- RMSE metrics alongside MAE
- Enhanced BigQuery integration for prediction storage
- Comprehensive performance reporting

USAGE MODES:
1. train (default): Train a new model on recent data
   $ python train_model.py

2. list: Show all available models in GCS
   $ python train_model.py --mode list

3. evaluate: Test specific model(s) on specific date(s)
   $ python train_model.py --mode evaluate --model-dates 2025-04-03 --test-dates 2025-04-04

4. production-eval: Comprehensive backtesting - each model tested on all its production days
   $ python train_model.py --mode production-eval
   $ python train_model.py --mode production-eval --start-date 2025-03-01 --end-date 2025-03-31

**NOTE**: BACKGROUND EXECUTION:
$ nohup python -u train_model.py --mode production-eval >> output.txt 2>&1 &
This returns a process number (e.g., [1] 66581) that can be used to monitor or kill the process.
To run the procedure in the background, use the command: $ nohup python -u train_model.py >> output.txt 2>&1 &. This will return a process number such as [1] 66581, which can be used to kill the process.
Breakdown:
1. `nohup`: This allows the script to continue running even after you log out or close the terminal.
2. python -u train_model.py: This part is executing your Python script in unbuffered mode, forcing Python to write output immediately.
3. >> output.txt 2>&1:
    * >> output.txt appends the standard output (stdout) of the script to output.txt instead of overwriting it.
    * 2>&1 redirects standard error (stderr) to the same file as standard output, so both stdout and stderr go into output.txt.
4. &: This runs the command in the background.

To monitor: $ tail -f output.txt
To kill: $ kill 66581 (or kill -9 66581 to force)

OUTPUTS:
- Trained models: Saved locally and to GCS
- Predictions: Uploaded to BigQuery sandbox table
- Summary CSVs: Performance metrics by model and days since training
- Console output: Detailed evaluation results

See README.md for more detailed documentation.

To train a model with a processed data file: Heavily uses code from `automated_training/`. Note: update `auxiliary_functions.py::get_creds(...)` with the correct file path.

To redirect the error to a different file, you can use 2> error.txt. Note that just ignoring it (not including 2>...) will just output to std out in this case.

'''

import os
import sys
import pickle

import numpy as np
import pandas as pd

from google.cloud import storage
import re
from datetime import datetime

os.environ["google_application_credentials"] = '/home/gil/git/ficc_python/creds.json'


ficc_package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'automated_training'))    # get the directory containing the 'ficc_python/automated_training' directory
sys.path.append(ficc_package_dir)    # add the directory to sys.path


from auxiliary_variables import MODEL_TO_CUMULATIVE_DATA_PICKLE_FILENAME, BUCKET_NAME

import auxiliary_functions
auxiliary_functions.SAVE_MODEL_AND_DATA = True

from auxiliary_functions import train_model, setup_gpus, get_optional_arguments_for_process_data, get_data_and_last_trade_datetime    #, apply_exclusions
from clean_training_log import remove_lines_with_tensorflow_progress_bar


ficc_package_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ficc'))    # get the directory containing the 'ficc_python/ficc' directory
sys.path.append(ficc_package_dir)    # add the directory to sys.path


from utils.auxiliary_functions import function_timer, get_ys_trade_history_features, get_dp_trade_history_features


MODEL = 'yield_spread_with_similar_trades'
NUM_DAYS = 1

TESTING = False
if TESTING:
    auxiliary_functions.NUM_EPOCHS = 4
    NUM_DAYS = 1


def restrict_trades_by_trade_datetime(df: pd.DataFrame, 
                                      start_trade_datetime: str = None, 
                                      end_trade_datetime: str = None) -> pd.DataFrame:
    '''`start_trade_datetime` and `end_trade_datetime` can be string objects representing a date, e.g., '2025-01-17' 
    because numpy automatically converts the string into a datetime object before making the comparison.'''
    if start_trade_datetime is not None: df = df[df['trade_datetime'] >= start_trade_datetime]
    if end_trade_datetime is not None: df = df[df['trade_datetime'] <= end_trade_datetime]
    return df


@function_timer
def get_processed_data_pickle_file(model: str = MODEL) -> pd.DataFrame:
    file_name = "/home/gil/git/ficc_python/notebooks/train_model/processed_data_yield_spread_with_similar_trades_v2.pkl"  # MODEL_TO_CUMULATIVE_DATA_PICKLE_FILENAME[model]
    if os.path.isfile(file_name):
        print(f'Loading data from {file_name} which was found locally...')
        with open(file_name, 'rb') as file:
            data = pickle.load(file)
        most_recent_trade_datetime = data.trade_datetime.max()
    else:
        raise NotImplementedError
        print(f'Did not find {file_name} locally so downloading it from Google Cloud Storage...')
        data, most_recent_trade_datetime, _ = get_data_and_last_trade_datetime(BUCKET_NAME, file_name)
        with open(file_name, 'wb') as file:
            pickle.dump(data, file)
        file_name = f'gs://{BUCKET_NAME}/processed_data/{file_name}'    # used for print
    print(f'Loaded data from {file_name}. Most recent trade datetime: {most_recent_trade_datetime}')
    # data = data[data['trade_date'] <= '2025-04-07'] # *** restrict to trades before 2025-04-03
    return data


def get_num_features_for_each_trade_in_history(model: str = MODEL) -> int:
    optional_arguments = get_optional_arguments_for_process_data(model)
    use_treasury_spread = optional_arguments.get('use_treasury_spread', False)    # from `auxiliary_functions.py::update_data(...)`
    trade_history_features = get_ys_trade_history_features(use_treasury_spread) if 'yield_spread' in model else get_dp_trade_history_features()    # from `automated_training/auxiliary_functions.py::get_new_data(...)`
    return len(trade_history_features)    # from `auxiliary_functions.py::get_new_data(...)`


def train_model_from_data_file(data: pd.DataFrame, num_days: int, output_file_path: str = None, exclusions_function: callable = None):
    most_recent_dates = np.sort(data['trade_date'].unique())[::-1]    # sort the unique `trade_date`s in descending order (the descending order comes from the slice)
    most_recent_dates = most_recent_dates[:num_days + 1]    # restrict to `num_days` most recent dates
    for day_idx in range(num_days):
        date_for_test_set, most_recent_date_for_training_set = most_recent_dates[day_idx], most_recent_dates[day_idx + 1]
        data = data[data['trade_date'] <= date_for_test_set]    # iteratively remove the last date from `data`
        trained_model, _, _, _, _, mae, (mae_df, _), _ = train_model(data, most_recent_date_for_training_set, MODEL, get_num_features_for_each_trade_in_history(), exclusions_function=exclusions_function)
        if output_file_path is not None: remove_lines_with_tensorflow_progress_bar(output_file_path)
        trained_model.save(f'{MODEL}_{date_for_test_set}')
        
def list_similar_trades_models(bucket_name='automated_training', prefix='similar-trades-v2-model-'):
    """List all similar-trades-v2 models in GCS bucket"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    models = []
    seen_dates = set()  # To avoid duplicates
    
    for blob in bucket.list_blobs(prefix=prefix):
        # Extract date from model name (similar-trades-v2-model-YYYY-MM-DD/)
        match = re.search(r'similar-trades-v2-model-(\d{4}-\d{2}-\d{2})', blob.name)
        if match:
            date_str = match.group(1)
            if date_str not in seen_dates:
                seen_dates.add(date_str)
                models.append({
                    'date': date_str,
                    'path': f'gs://{bucket_name}/similar-trades-v2-model-{date_str}/',
                    'name': f'similar-trades-v2-model-{date_str}'
                })
    
    # Sort by date
    models.sort(key=lambda x: x['date'], reverse=True)
    
    print(f"Found {len(models)} similar-trades-v2 models:")
    for model in models[:10]:  # Show first 10
        print(f"  {model['date']}: {model['path']}")
    
    return models


def evaluate_model_on_single_day(
    model_date: str,
    test_date: str,
    data: pd.DataFrame = None,
    data_pkl_path: str = None,
    upload_to_bq: bool = True  # Add flag to control upload
):
    """Load a pre-trained model and evaluate it on a specific test date"""
    
    import tensorflow as tf
    from tensorflow import keras
    from datetime import datetime
    from pytz import timezone
    import gcsfs
    
    # 1. Load the data if not provided
    if data is None:
        if data_pkl_path is None:
            data_pkl_path = "/home/gil/git/ficc_python/notebooks/train_model/processed_data_yield_spread_with_similar_trades_v2.pkl"
        print(f"Loading data from {data_pkl_path}")
        with open(data_pkl_path, 'rb') as f:
            data = pickle.load(f)
    
    # 2. Filter to test date
    test_data = data[data['trade_date'] == test_date]
    print(f"Test data shape for {test_date}: {test_data.shape}")
    
    if len(test_data) == 0:
        print(f"No data found for test date {test_date}")
        return None
    
    # 3. Load the model
    model_path = f'gs://automated_training/similar-trades-v2-model-{model_date}'
    print(f"Loading model from {model_path}")
    model = keras.models.load_model(model_path)
    
    # 4. Load the encoders from GCS
    fs = gcsfs.GCSFileSystem()
    encoders_path = 'gs://automated_training/encoders_similar_trades.pkl'
    with fs.open(encoders_path, 'rb') as f:
        encoders = pickle.load(f)
    print(f"Loaded encoders from {encoders_path}")
    
    # 5. Create inputs
    from auxiliary_functions import create_input
    x_test, y_test = create_input(test_data, encoders, 'yield_spread_with_similar_trades')
    
    # 6. Generate predictions
    predictions = model.predict(x_test, batch_size=1000)
    
    # 7. Calculate metrics and segment results
    from auxiliary_functions import segment_results
    delta = np.abs(predictions.flatten() - y_test)
    result_df = segment_results(test_data, delta)
    
    print(f"\nResults for model {model_date} on test date {test_date}:")
    print(result_df.to_string())
    
    # 8. Prepare predictions dataframe
    test_data_copy = test_data.copy()
    test_data_copy['new_ys_prediction'] = predictions.flatten()
    test_data_copy['model_train_date'] = model_date
    
    # 9. Upload to BigQuery if requested
    if upload_to_bq and len(test_data_copy) > 0:
        print(f"\nUploading {len(test_data_copy)} predictions to BigQuery...")
        
        # Prepare data for upload
        upload_data = test_data_copy[['rtrs_control_number', 'cusip', 'trade_date', 
                                     'dollar_price', 'yield', 'new_ficc_ycl', 
                                     'new_ys', 'new_ys_prediction']].copy()
        
        EASTERN = timezone('US/Eastern')
        upload_data['prediction_datetime'] = pd.to_datetime(datetime.now(EASTERN).replace(microsecond=0))
        upload_data['trade_date'] = pd.to_datetime(upload_data['trade_date']).dt.date
        
        # Add extra fields with correct types
        upload_data['model_train_date'] = pd.to_datetime(model_date).date()  # Convert to DATE type
        upload_data['evaluation_mode'] = 'historical'
        upload_data['days_since_training'] = (pd.to_datetime(test_date) - pd.to_datetime(model_date)).days
        
        try:
            # Use the upload function from auxiliary_functions
            from auxiliary_functions import upload_predictions
            
            # Upload
            upload_predictions(upload_data, 'yield_spread_with_similar_trades')
            print(f"Successfully uploaded predictions for {test_date} using model {model_date}")
            
        except Exception as e:
            print(f"Failed to upload to BigQuery: {e}")
    
    return {
        'model_date': model_date,
        'test_date': test_date,
        'result_df': result_df,
        'predictions': test_data_copy,
        'mae': result_df.loc['Entire set', 'Mean Absolute Error'],
        'rmse': result_df.loc['Entire set', 'RMSE'] if 'RMSE' in result_df.columns else None
    }


def evaluate_models_on_production_dates(data: pd.DataFrame = None, upload_to_bq: bool = True, 
                                       start_date: str = None, end_date: str = None):
    """
    Evaluate each model on all dates it would have been in production.
    For each model, test from the day after it was trained until the next model was trained.
    
    Args:
        start_date: Only evaluate models from this date onwards
        end_date: Only evaluate models up to this date
    """
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Get all available models
    models = list_similar_trades_models()
    model_dates = [m['date'] for m in models]
    model_dates.sort()  # Ensure chronological order
    
    # Filter model dates if range specified
    if start_date:
        model_dates = [d for d in model_dates if d >= start_date]
    if end_date:
        model_dates = [d for d in model_dates if d <= end_date]
    
    print(f"Found {len(model_dates)} models in date range from {model_dates[0]} to {model_dates[-1]}")
    
    # Load data if not provided
    if data is None:
        data = get_processed_data_pickle_file(MODEL)
    
    # Get all available test dates
    all_dates = sorted(data['trade_date'].unique())
    
    results = []
    
    # For each model, test on all days until the next model
    for i, model_date in enumerate(model_dates):
        # Convert to datetime for comparison
        model_datetime = pd.to_datetime(model_date)
        
        # Find the next model date (or use last available date if this is the last model)
        if i < len(model_dates) - 1:
            next_model_date = pd.to_datetime(model_dates[i + 1])
        else:
            next_model_date = pd.to_datetime(all_dates[-1]) + timedelta(days=1)
        
        # Get all dates between this model and the next
        test_dates = [d for d in all_dates 
                     if pd.to_datetime(d) > model_datetime 
                     and pd.to_datetime(d) < next_model_date]
        
        if not test_dates:
            print(f"\nNo test dates for model {model_date}")
            continue
            
        print(f"\n{'='*80}")
        print(f"Model {model_date} will be tested on {len(test_dates)} days:")
        print(f"From {test_dates[0]} to {test_dates[-1]}")
        print(f"{'='*80}")
        
        # Evaluate this model on each of its production dates
        for test_date in test_dates:
            try:
                print(f"\n  Testing {model_date} on {test_date}...")
                result = evaluate_model_on_single_day(
                    model_date, 
                    test_date, 
                    data=data,
                    upload_to_bq=upload_to_bq
                )
                if result:
                    results.append(result)
                    print(f"  ✓ MAE: {result['mae']:.2f}, RMSE: {result['rmse']:.2f}")
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
    
    # Create summary DataFrame
    summary_data = []
    for r in results:
        summary_data.append({
            'model_date': r['model_date'],
            'test_date': r['test_date'],
            'days_since_training': (pd.to_datetime(r['test_date']) - pd.to_datetime(r['model_date'])).days,
            'mae': r['mae'],
            'rmse': r['rmse'],
            'ig_mae': r['result_df'].loc['Investment Grade', 'Mean Absolute Error'],
            'ig_rmse': r['result_df'].loc['Investment Grade', 'RMSE'] if 'RMSE' in r['result_df'].columns else None,
            'num_trades': int(r['result_df'].loc['Entire set', 'Trade Count'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Add some analysis
    print(f"\n\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    # Average performance by days since training
    days_perf = summary_df.groupby('days_since_training').agg({
        'mae': ['mean', 'std', 'count'],
        'rmse': ['mean', 'std']
    }).round(2)
    
    print("\nPerformance by Days Since Training:")
    print(days_perf)
    
    # Save complete results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f'production_evaluation_summary_{timestamp}.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved detailed results to {summary_file}")
    
    # Also save aggregated results
    model_summary = summary_df.groupby('model_date').agg({
        'mae': ['mean', 'std', 'min', 'max', 'count'],
        'rmse': ['mean', 'std', 'min', 'max'],
        'days_since_training': 'max'
    }).round(2)
    
    model_summary_file = f'model_performance_summary_{timestamp}.csv'
    model_summary.to_csv(model_summary_file)
    print(f"Saved model summary to {model_summary_file}")
    
    return summary_df, model_summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train or evaluate yield spread models')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'list', 'production-eval'], default='train',
                        help='Mode: train new models, evaluate existing models, list available models, or run production evaluation')
    parser.add_argument('--model-dates', nargs='+', help='Model dates to evaluate (YYYY-MM-DD)')
    parser.add_argument('--test-dates', nargs='+', help='Test dates to evaluate on (YYYY-MM-DD)')
    parser.add_argument('--start-date', help='Start date for production-eval (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for production-eval (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    if args.mode == 'list':
        list_similar_trades_models()
    
    elif args.mode == 'production-eval':
        # Run the comprehensive production evaluation
        setup_gpus(False)
        print("Starting comprehensive production evaluation...")
        
        if args.start_date or args.end_date:
            print(f"Date range: {args.start_date or 'beginning'} to {args.end_date or 'end'}")
        
        print("This will test each model on all days it was in production.")
        print("This may take several hours to complete.\n")
        
        summary_df, model_summary = evaluate_models_on_production_dates(
            upload_to_bq=True, 
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        print("\n\nEvaluation complete!")
        print(f"Tested {len(summary_df)} model-date combinations")
        print(f"Results uploaded to BigQuery sandbox table")
    
    elif args.mode == 'evaluate':
        # Example: python train_model.py --mode evaluate --model-dates 2025-04-03 --test-dates 2025-04-04
        if not args.model_dates or not args.test_dates:
            print("Please provide --model-dates and --test-dates for evaluation")
            sys.exit(1)
        
        setup_gpus(False)
        data = get_processed_data_pickle_file(MODEL)
        
        # For single evaluations, just loop through the combinations
        for model_date in args.model_dates:
            for test_date in args.test_dates:
                print(f"\nEvaluating model {model_date} on {test_date}")
                result = evaluate_model_on_single_day(model_date, test_date, data=data, upload_to_bq=True)
                if result:
                    print(f"MAE: {result['mae']:.3f}, RMSE: {result['rmse']:.3f}")
    
    else:  # train mode (default)
        setup_gpus(False)
        data = get_processed_data_pickle_file(MODEL)
        output_file_name = 'output.txt'
        train_model_from_data_file(data, NUM_DAYS, output_file_name)
