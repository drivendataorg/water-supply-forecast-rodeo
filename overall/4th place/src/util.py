import pandas as pd
import numpy as np
from sklearn.metrics import mean_pinball_loss

def post_process_quantiles(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Post-processes quantile predictions by ensuring they are non-negative and sorted.

    This function applies an absolute value operation to the predictions to ensure 
    non-negativity and then sorts them along the specified axis. This is essential for 
    quantile predictions to maintain the correct order.

    Args:
        predictions (pd.DataFrame): A DataFrame of quantile predictions.

    Returns:
        pd.DataFrame: The post-processed predictions, which are non-negative and sorted.
    """
    # Ensure non-negativity
    predictions = predictions.abs()
    
    # Sort values along each row and preserve column structure
    sorted_predictions = pd.DataFrame(np.sort(predictions.values, axis=1), 
                                      index=predictions.index, 
                                      columns=predictions.columns)
    
    return sorted_predictions

def evaluate(actual, predicted, verbose=True):
    predicted = predicted.values
    pl10 = mean_pinball_loss(actual, predicted[:,0], alpha=0.1)
    pl50 = mean_pinball_loss(actual, predicted[:,1], alpha=0.5)
    pl90 = mean_pinball_loss(actual, predicted[:,2], alpha=0.9)
    avg_pl = 2 * (pl10 + pl50 + pl90) / 3
    
    # Calculate coverage
    coverage = np.mean((predicted[:,0] <= actual) & (actual <= predicted[:,2]))
    
    if verbose:
        print(f"Pinball Losses: Q10: {pl10:.1f}")
        print(f"                Q50: {pl50:.1f}")
        print(f"                Q90: {pl90:.1f}")
        print(f"Average Pinball Loss: {avg_pl:.2f} and coverage {coverage:.2f}")    
    return avg_pl
