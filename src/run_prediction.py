import argparse
from datetime import datetime, timedelta
import pandas as pd
import os
from predict import GoldPricePredictor

def parse_args():
    parser = argparse.ArgumentParser(description='Predict Indonesia Gold Prices')
    parser.add_argument('--mode', choices=['next_day', 'days', 'specific_date', 'range', 'plot'], default='next_day',
                        help='Prediction mode: next_day, days ahead, specific date, date range, or create plots')
    parser.add_argument('--days', type=int, default=1, 
                        help='Number of days to predict ahead (for days mode)')
    parser.add_argument('--date', type=str, default=None, 
                        help='Target date in YYYY-MM-DD format (for specific_date mode)')
    parser.add_argument('--start_date', type=str, default=None, 
                        help='Start date in YYYY-MM-DD format (for range mode)')
    parser.add_argument('--end_date', type=str, default=None, 
                        help='End date in YYYY-MM-DD format (for range mode)')
    parser.add_argument('--model_path', type=str, default='models/gold_price_lstm.pth',
                        help='Path to the trained model')
    parser.add_argument('--data_path', type=str, default='data/processed/gold_prices_cleaned.csv',
                        help='Path to the processed CSV data')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save predictions as CSV (optional)')
    parser.add_argument('--plot_dir', type=str, default='plots',
                        help='Directory to save plot images (for plot mode)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        # Initialize the predictor
        predictor = GoldPricePredictor(args.model_path, args.data_path)
        
        results = None
        
        if args.mode == 'plot':
            print(f"\nGenerating prediction plots for multiple time periods...")
            print(f"Plots will be saved to: {args.plot_dir}")
            
            # Create plots directory if it doesn't exist
            if not os.path.exists(args.plot_dir):
                os.makedirs(args.plot_dir)
                
            # Generate plots for various time periods
            plot_paths = predictor.plot_predictions_for_periods(save_dir=args.plot_dir)
            
            print("\nGenerated the following prediction plots:")
            for period, path in plot_paths.items():
                print(f"- {os.path.basename(path)}")
            
            print(f"\nAll plots have been saved to: {os.path.abspath(args.plot_dir)}")
        
        elif args.mode == 'next_day':
            # Predict just the next day
            next_date, next_price = predictor.predict_next_day()
            results = pd.DataFrame({'date': [next_date], 'predicted_price': [next_price]})
            
        elif args.mode == 'days':
            # Predict for X days ahead
            results = predictor.predict_future(days=args.days)
            print(f"\nPredictions for the next {args.days} days:")
            if args.days > 10:
                # For many days, just show samples
                sample_indices = [0]
                step = max(1, args.days // 10)
                for i in range(step, args.days, step):
                    sample_indices.append(i)
                if args.days - 1 not in sample_indices:
                    sample_indices.append(args.days - 1)
                print(results.iloc[sample_indices].to_string(index=False))
                print(f"... (showing {len(sample_indices)} samples out of {args.days} days)")
            else:
                print(results.to_string(index=False))
            
        elif args.mode == 'specific_date':
            if not args.date:
                print("Error: You must provide a --date argument for specific_date mode")
                return
            
            try:
                target_date = datetime.strptime(args.date, '%Y-%m-%d')
                specific_date, specific_price = predictor.predict_specific_date(target_date)
                if specific_date:
                    results = pd.DataFrame({'date': [specific_date], 'predicted_price': [specific_price]})
            except ValueError:
                print("Error: Date format should be YYYY-MM-DD")
                return
            
        elif args.mode == 'range':
            if not args.start_date or not args.end_date:
                print("Error: You must provide --start_date and --end_date for range mode")
                return
                
            try:
                start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
                end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
                
                # Calculate days between the dates
                delta = (end_date - start_date).days + 1
                if delta <= 0:
                    print("Error: end_date must be after start_date")
                    return
                    
                # Get predictions for this range
                all_predictions = predictor.predict_future(days=delta)
                
                # Filter for the requested date range
                results = all_predictions[
                    (all_predictions['date'] >= start_date) & 
                    (all_predictions['date'] <= end_date)
                ]
                
                print(f"\nPredictions from {args.start_date} to {args.end_date}:")
                if len(results) > 10:
                    # For many dates, just show samples
                    indices = [0, len(results)//4, len(results)//2, 3*len(results)//4, len(results)-1]
                    print(results.iloc[indices].to_string(index=False))
                    print(f"... (showing {len(indices)} samples out of {len(results)} days)")
                else:
                    print(results.to_string(index=False))
                    
            except ValueError:
                print("Error: Date format should be YYYY-MM-DD")
                return
        
        # Save results to CSV if requested
        if results is not None and args.output:
            results.to_csv(args.output, index=False)
            print(f"\nPredictions saved to {args.output}")
    
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("Please make sure the model file and data CSV exist at the specified paths.")
    
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
    
if __name__ == "__main__":
    main()