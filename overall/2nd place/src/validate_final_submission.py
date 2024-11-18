import numpy as np
import pandas as pd
import argparse

def mql(y_true, y_pred, quantile):
    return 2*(quantile*np.maximum(y_true - y_pred, 0) + (1 - quantile)*np.maximum(y_pred - y_true, 0))


def mean_quantile_loss(y_true, y_pred10, y_pred50, y_pred90):
    return (mql(y_true, y_pred10, 0.1) + mql(y_true, y_pred50, 0.5) + mql(y_true, y_pred90, 0.9))/3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default="results/final/final_predict_submission_mlp_sumres.csv")
    args = parser.parse_args()

    submission_fn = args.submission
    submission = pd.read_csv(submission_fn)
    submission['year'] = submission['issue_date'].str[0:4].astype(int)
    target = pd.read_csv('data/final_stage/cross_validation_labels.csv')

    submission_format = pd.read_csv('data/final_stage/cross_validation_submission_format.csv')
    gt_pred = target.merge(submission, on=['site_id', 'year'], how='right')
    print(f'predicted rows: {len(gt_pred)}/{len(submission_format)} {(len(gt_pred)/len(submission_format)*100):.02f}%')

    pred_vs_sf = submission.merge(submission_format, on=['site_id', 'issue_date'], how='left')
    if len(pred_vs_sf) == len(pred_vs_sf):
        print('correct issued dates')
    else:
        print('!!!! Incorrect issued dates')

    score = np.mean(mean_quantile_loss(gt_pred['volume'].values,
                                       gt_pred['volume_10'].values,
                                       gt_pred['volume_50'].values,
                                       gt_pred['volume_90'].values))
    print(f'Score: {score:.03f}')

    try:
        if submission_format[['site_id', 'issue_date']].reset_index().equals(submission[['site_id', 'issue_date']].reset_index()):
            print('Correct order of site_id, issue_date in submission file')
        else:
            print('!!! Incorrect order of site_id, issue_date in submission file!')
    except:
        print('!!! Wrong count of rows in submission file!')