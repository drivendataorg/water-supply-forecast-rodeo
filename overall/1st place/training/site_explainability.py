#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[0]))

from loguru import logger


def main(data_dir: Path,
         preprocessed_dir: Path,
         site_id: str,
         issue_date: str):
    plt.style.use('ggplot')
    plot_dir = preprocessed_dir / f'plots/{site_id}/{issue_date}'
    plot_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Beginning model explainability for site_id {site_id} on issue_date {issue_date}. ' \
                f'Saving plots in {plot_dir}.')
    metadata = pd.read_csv(data_dir / 'metadata.csv', dtype={'usgs_id': 'string'})
    cv_features = pd.read_csv(preprocessed_dir / 'cv_features.csv')
    cv_test_features = pd.read_csv(preprocessed_dir / 'cv_test_features.csv')
    final_submission = pd.read_csv(preprocessed_dir / 'final_submission.csv')

    cv_labels = pd.read_csv(data_dir / 'cross_validation_labels.csv')
    prior_labels = pd.read_csv(data_dir / 'prior_historical_labels.csv')
    train = pd.concat([cv_labels, prior_labels])

    issue_year = pd.to_datetime(issue_date).year
    issue_month = pd.to_datetime(issue_date).month
    issue_day = pd.to_datetime(issue_date).day

    # Get previous issue_date
    if issue_day == 1 and issue_month == 1:
        logger.info("No previous predictions in this streamflow season")
        prev_issue_date = issue_date
    elif issue_day == 1:
        prev_issue_date = f'{issue_year}-{issue_month-1:02}-22'
    else:
        prev_issue_date = f'{issue_year}-{issue_month:02}-{issue_day-7:02}'
    logger.info(f'Previous issue_date: {prev_issue_date}')

    def quantile(n):
        def quantile_(x):
            return x.quantile(n)
        quantile_.__name__ = 'quantile_{:02.0f}'.format(n*100)
        return quantile_
    
    site_grouped = train.groupby(['site_id'])['volume'].agg([np.mean, np.std, quantile(.10), quantile(.50), quantile(.90)]).reset_index()
    site_mth_grouped = cv_features.groupby(['site_id', 'month'])['month_volume'].agg([np.mean, np.std, quantile(.10), quantile(.50), quantile(.90)]).reset_index()

    models = {
        'cat_10_mth': cb.CatBoostRegressor().load_model(preprocessed_dir / f'models/{issue_year}/cat_10_monthly_model.txt'),
        'cat_50_mth': cb.CatBoostRegressor().load_model(preprocessed_dir / f'models/{issue_year}/cat_50_monthly_model.txt'),
        'cat_90_mth': cb.CatBoostRegressor().load_model(preprocessed_dir / f'models/{issue_year}/cat_90_monthly_model.txt'),
        'cat_10_yr': cb.CatBoostRegressor().load_model(preprocessed_dir / f'models/{issue_year}/cat_10_yearly_model.txt'),
        'cat_50_yr': cb.CatBoostRegressor().load_model(preprocessed_dir / f'models/{issue_year}/cat_50_yearly_model.txt'),
        'cat_90_yr': cb.CatBoostRegressor().load_model(preprocessed_dir / f'models/{issue_year}/cat_90_yearly_model.txt'),
        'lgb_10_mth': lgb.Booster(model_file=str(preprocessed_dir / f'models/{issue_year}/lgb_10_monthly_model.txt')),
        'lgb_50_mth': lgb.Booster(model_file=str(preprocessed_dir / f'models/{issue_year}/lgb_50_monthly_model.txt')),
        'lgb_90_mth': lgb.Booster(model_file=str(preprocessed_dir / f'models/{issue_year}/lgb_90_monthly_model.txt')),
        'lgb_10_yr':lgb.Booster(model_file=str(preprocessed_dir / f'models/{issue_year}/lgb_10_yearly_model.txt')),
        'lgb_50_yr': lgb.Booster(model_file=str(preprocessed_dir / f'models/{issue_year}/lgb_50_yearly_model.txt')),
        'lgb_90_yr': lgb.Booster(model_file=str(preprocessed_dir / f'models/{issue_year}/lgb_90_yearly_model.txt')),
    }

    site_test_features = cv_test_features[
        (cv_test_features['year'] == issue_year) &
        (cv_test_features['issue_date'] <= issue_date) &
        (cv_test_features['site_id'] == site_id)
    ].set_index('issue_date')

    site_predictions = final_submission[
        (pd.to_datetime(final_submission['issue_date']).dt.year == issue_year) &
        (final_submission['issue_date'] <= issue_date) &
        (final_submission['site_id'] == site_id)
    ].set_index('issue_date')

    site_predictions.rename(columns={'volume_10': 'predicted_10',
                                    'volume_50': 'predicted_50',
                                    'volume_90': 'predicted_90'}, inplace=True)

    site_predictions['historical_10'] = site_grouped[site_grouped['site_id'] == site_id]['quantile_10'].values[0]
    site_predictions['historical_50'] = site_grouped[site_grouped['site_id'] == site_id]['quantile_50'].values[0]
    site_predictions['historical_90'] = site_grouped[site_grouped['site_id'] == site_id]['quantile_90'].values[0]

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 6))
    plt.suptitle('Fig 2: Predicted Volume Quantiles vs Historical Quantiles by Issue Date', fontsize=11)
    sns.lineplot(site_predictions[['predicted_10', 'predicted_50', 'predicted_90']], dashes=False, ax=axes[0])
    sns.lineplot(site_predictions[['historical_10', 'historical_50', 'historical_90']], dashes=[(2, 2), (2, 2), (2, 2)], ax=axes[0])
    plt.title('Fig 3: Feature Z-Score Value by Issue Date', fontsize=11)
    axes[0].fill_between(site_predictions.index, site_predictions['predicted_10'], site_predictions['predicted_90'], alpha=0.1, color='black')
    axes[0].set(ylabel='Volume (KAF)', xticklabels=[], xlabel=None)
    sns.move_legend(axes[0], "upper left", bbox_to_anchor=(0, 1))
    plt.setp(axes[0].get_legend().get_texts(), fontsize='10')
    dynamic_cols = ['streamflow_deviation_30_mean', 'combined_swe_deviation_20', 'precip_deviation_30', 'maxt_deviation_30', 'pdsi_deviation_2']
    feature_df = site_test_features[dynamic_cols]
    feature_df.columns = ['streamflow', 'swe', 'precipitation', 'max_temp', 'pdsi']
    axes[1] = sns.lineplot(feature_df, dashes=False, ax=axes[1])
    plt.xticks(rotation=45)
    axes[1].set(ylabel='Z-Score')
    sns.move_legend(axes[1], "upper left", bbox_to_anchor=(0, 1))
    plt.setp(axes[1].get_legend().get_texts(), fontsize='10')
    fig.tight_layout()
    plt.savefig(plot_dir / 'fig2-3.png', bbox_inches='tight')

    cat_mth_base_feature = ['site_id', 'pred_month', 'latitude', 'longitude', 'elevation', 'elevation_stds',
                        'day_of_year', 'streamflow_deviation_30_mean', 'snotel_tmax_deviation_10',
                        'precip_deviation_30', 'maxt_deviation_30', 'combined_swe_deviation_20',
                        'combined_swe_deviation_10', 'pdsi_deviation_5', 'acc_water_deviation',
                        'prev_month_volume']
    cat_mth_feature_cols = {
        '10': cat_mth_base_feature,
        '50': cat_mth_base_feature,
        '90': cat_mth_base_feature
    }

    lgb_mth_base_feature = ['site_cat', 'pred_month', 'latitude', 'longitude', 'elevation', 'elevation_stds',
                            'day_of_year', 'precip_deviation_30', 'streamflow_deviation_30_mean', 'maxt_deviation_30',
                            'snotel_tmax_deviation_10', 'acc_water_deviation',
                            'combined_swe_deviation_10', 'snotel_prec_deviation_10', 'snotel_prec_deviation_20']

    lgb_mth_feature_cols = {
        '10': lgb_mth_base_feature + ['precip_deviation_10', 'maxt_deviation_10', 'pdsi_deviation_2'],
        '50': lgb_mth_base_feature + ['prev_month_volume', 'combined_swe_deviation_20', 'pdsi_deviation_5'],
        '90': lgb_mth_base_feature + ['prev_month_volume', 'combined_swe_deviation_20', 'pdsi_deviation_2']
    }

    cat_yr_base_feature = ['site_id', 'latitude', 'longitude', 'elevation', 'elevation_stds',
                        'day_of_year', 'streamflow_deviation_30_mean', 'streamflow_deviation_season_mean',
                        'precip_deviation_180', 'combined_swe_deviation_180',
                        'prev_month_volume', 'combined_swe_deviation_20', 'combined_swe_deviation_10',
                        'maxt_deviation_30', 'maxt_deviation_180', 'precip_deviation_30',
                        'snotel_prec_deviation_20', 'pdsi_deviation_5', 'snotel_tmax_deviation_10']
    cat_yr_feature_cols = {
        '10': cat_yr_base_feature,
        '50': cat_yr_base_feature + ['acc_water_deviation'],
        '90': cat_yr_base_feature
    }

    lgb_yr_base_feature = ['site_cat', 'latitude', 'longitude', 'elevation', 'elevation_stds',
                        'day_of_year', 'streamflow_deviation_30_mean', 'streamflow_deviation_season_mean',
                        'combined_swe_deviation_180',
                        'prev_month_volume', 'combined_swe_deviation_10',
                        'maxt_deviation_30', 'precip_deviation_10', 'precip_deviation_30', 'precip_deviation_180',
                        'snotel_prec_deviation_20', 'snotel_tmax_deviation_10']

    lgb_yr_feature_cols = {
        '10': lgb_yr_base_feature + ['snotel_prec_deviation_180', 'pdsi_deviation_5', ],
        '50': lgb_yr_base_feature + ['combined_swe_deviation_20', 'pdsi_deviation_2', ],
        '90': lgb_yr_base_feature + ['pdsi_deviation_5', 'acc_water_deviation',]
    }

    site_cv_features = cv_test_features[(cv_test_features['year'] == issue_year) &
                     (cv_test_features['site_id'] == site_id)]
    
    def generate_monthly_features(train_features, no_monthly_data):
        monthly_train_features = train_features

        monthly_labels = monthly_train_features[
            ['site_id', 'year', 'month', 'month_volume', 'season_start_month', 'season_end_month']]
        monthly_labels = monthly_labels[(monthly_labels['month'] >= monthly_labels['season_start_month']) &
                                        (monthly_labels['month'] <= monthly_labels['season_end_month'])]
        monthly_labels = monthly_labels.groupby(['site_id', 'year', 'month'])['month_volume'].mean().reset_index().dropna()
        monthly_labels.columns = ['site_id', 'year', 'pred_month', 'month_volume_label']

        monthly_train_features = pd.merge(monthly_train_features, monthly_labels, on=['site_id', 'year'])
        monthly_train_features = monthly_train_features[~monthly_train_features['site_id'].isin(no_monthly_data)]
        train_features = monthly_train_features[monthly_train_features['season_start_month'] <= monthly_train_features['pred_month']]

        # Take log of volume
        train_features['month_volume_log'] = np.log(train_features['month_volume_label'])

        label = 'month_volume_log'

        train_labels = train_features[[label]]

        return train_features, train_labels

    no_monthly_data = ['american_river_folsom_lake', 'merced_river_yosemite_at_pohono_bridge']
    monthly_test_features, monthly_test_labels = generate_monthly_features(site_cv_features, no_monthly_data)
    monthly_test_features.reset_index(drop=True, inplace=True)

    mth_pct = {
        '10': {
            1: 0.4,
            2: 0.4,
            3: 0.4,
            4: 0.6,
            5: 0.7,
            6: 0.8,
            7: 1
            },
        '50': {
            1: 0.4,
            2: 0.4,
            3: 0.4,
            4: 0.4,
            5: 0.4,
            6: 0.7,
            7: 1
            },
        '90': {
            1: 0.4,
            2: 0.5,
            3: 0.6,
            4: 0.6,
            5: 0.6,
            6: 0.8,
            7: 1
            }
        }

    model_type_pct = {
        '10': 0.4, '50': 0.4, '90': 0.7
    }

    static_cols = ['site_cat', 'latitude', 'longitude', 'elevation', 'elevation_stds']

    dynamic_cols = ['day_of_year', 'streamflow_deviation_30_mean',
        'snotel_tmax_deviation_10', 'precip_deviation_30', 'maxt_deviation_30',
        'combined_swe_deviation_20', 'combined_swe_deviation_10',
        'pdsi_deviation_5', 'acc_water_deviation', 'prev_month_volume', 'snotel_prec_deviation_10', 'snotel_prec_deviation_20',
        'precip_deviation_10', 'maxt_deviation_10', 'pdsi_deviation_2',
        'streamflow_deviation_season_mean', 'precip_deviation_180',
        'combined_swe_deviation_180', 'maxt_deviation_180',
        'snotel_prec_deviation_180']

    model_names = models.keys()
    model_pct = {}

    for model_name in model_names:
        model_type, quantile, model_freq = model_name.split('_')
        monthly_pct = mth_pct[quantile][issue_month]
        if model_freq == 'yr':
            monthly_pct = 1 - monthly_pct
        type_pct = model_type_pct[quantile]
        if model_type == 'lgb':
            type_pct = 1 - type_pct

        model_pct[model_name] = monthly_pct * type_pct
    
    issue_date_mth_features = monthly_test_features[(monthly_test_features['issue_date'] == issue_date)]
    issue_date_yr_features = site_cv_features[site_cv_features['issue_date'] == issue_date]
    issue_date_predictions = site_predictions[site_predictions.index == issue_date]

    prev_issue_date_mth_features = monthly_test_features[(monthly_test_features['issue_date'] == prev_issue_date)]
    prev_issue_date_yr_features = site_cv_features[site_cv_features['issue_date'] == prev_issue_date]

    model_name_list = []
    quantile_list = []
    model_pred_contribution_list = []
    for quantile in ['10', '50', '90']:
        quant_pred = issue_date_predictions[f'predicted_{quantile}'].values[0]
        for model_name in model_pct.keys():
            model_type, model_quantile, model_freq = model_name.split('_')
            if model_quantile == quantile:
                model_name_list.append(model_name)
                quantile_list.append(quantile)
                model_pred_contribution_list.append(quant_pred*model_pct[model_name])
    
    fig = plt.figure(figsize=(1, 6))
    model_data = pd.DataFrame({'model_name': model_name_list,
                'quantile': quantile_list,
                'model_pred_contribution': model_pred_contribution_list})
    ax = sns.histplot(data=model_data, x='quantile', weights='model_pred_contribution', hue='model_name', multiple='stack')
    bar_labels = []
    bar_label = 0
    for idx, c in enumerate(ax.containers):
        for child in c.get_children():
            bar_label += child.get_height()
        if (idx + 1) % 4 == 0:
            bar_labels.append(bar_label)
            bar_label = 0
    bar_labels.reverse()
    for idx, label in enumerate(bar_labels):
        ax.text(idx, label, int(label), ha='center')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.setp(ax.get_legend().get_texts(), fontsize=11)
    plt.setp(ax.get_legend().get_title(), fontsize=11)
    ax.set(ylabel='Volume (KAF)')
    plt.title('Fig 1: Quantile Model Ensemble', fontsize=11)
    fig.tight_layout()
    plt.savefig(plot_dir / 'fig1.png', bbox_inches='tight')

    def use_month_volume_label_for_past(row, quantile):
        if row['pred_month'] < pd.to_datetime(row['issue_date']).month:
            return row['month_volume_label']
        else:
            return row[f'combined_pred_{quantile}']

    def add_mth_preds(issue_date_mth_features, quantile):
        cat_pred_mth = np.exp(models[f'cat_{quantile}_mth'].predict(issue_date_mth_features[cat_mth_feature_cols[quantile]]))
        lgb_pred_mth = np.exp(models[f'lgb_{quantile}_mth'].predict(issue_date_mth_features[lgb_mth_feature_cols[quantile]]))
        combined_pred_mth = cat_pred_mth * model_type_pct[quantile] + lgb_pred_mth * (1 - model_type_pct[quantile])
        issue_date_mth_features[f'combined_pred_{quantile}'] = combined_pred_mth
        issue_date_mth_features[f'mth_pred_{quantile}'] = issue_date_mth_features.apply(lambda rw: use_month_volume_label_for_past(rw, quantile), axis=1)
        return issue_date_mth_features

    if site_id in no_monthly_data:
        logger.info(f'No monthly data for site_id {site_id}, omitting Figure 4.')
    else:
        issue_date_mth_features = add_mth_preds(issue_date_mth_features, '10')
        issue_date_mth_features = add_mth_preds(issue_date_mth_features, '50')
        issue_date_mth_features = add_mth_preds(issue_date_mth_features, '90')

        prev_issue_date_mth_features = add_mth_preds(prev_issue_date_mth_features, '10')
        prev_issue_date_mth_features = add_mth_preds(prev_issue_date_mth_features, '50')
        prev_issue_date_mth_features = add_mth_preds(prev_issue_date_mth_features, '90')

        season_start_month = cv_features[cv_features['site_id'] == site_id].season_start_month.values[0]
        season_end_month = cv_features[cv_features['site_id'] == site_id].season_end_month.values[0]

        season_site_mth = site_mth_grouped[(site_mth_grouped['site_id'] == site_id) &
                                        (site_mth_grouped['month'] >= season_start_month) &
                                        (site_mth_grouped['month'] <= season_end_month)]

        x = issue_date_mth_features['pred_month']
        x_lin = np.linspace(x.min(), x.max(), 30)
        x_hist = season_site_mth['month']

        spline_10 = make_interp_spline(x, issue_date_mth_features['mth_pred_10'])
        y_10 = spline_10(x_lin)

        spline_50 = make_interp_spline(x, issue_date_mth_features['mth_pred_50'])
        x_50 = np.linspace(x.min(), x.max(), 30)
        y_50 = spline_50(x_lin)

        spline_90 = make_interp_spline(x, issue_date_mth_features['mth_pred_90'])
        x_90 = np.linspace(x.min(), x.max(), 30)
        y_90 = spline_90(x_lin)

        spline_10_hist = make_interp_spline(x_hist, season_site_mth['quantile_10'])
        y_10_hist = spline_10_hist(x_lin)

        spline_50_hist = make_interp_spline(x_hist, season_site_mth['quantile_50'])
        y_50_hist = spline_50_hist(x_lin)

        spline_90_hist = make_interp_spline(x_hist, season_site_mth['quantile_90'])
        y_90_hist = spline_90_hist(x_lin)

        pal = sns.color_palette()
        hex_pal = pal.as_hex()
        fig = plt.figure(figsize=(2.5, 6))
        ax = fig.add_subplot(111)

        sns.lineplot(x=x_lin, y=y_10, dashes=False, ax=ax, color=hex_pal[0], label='predicted_10')
        sns.lineplot(x=x_lin, y=y_50, dashes=False, ax=ax, color=hex_pal[1], label='predicted_50')
        sns.lineplot(x=x_lin, y=y_90, dashes=False, ax=ax, color=hex_pal[2], label='predicted_90')
        sns.lineplot(x=x_lin, y=y_10_hist, dashes=(2, 2), ax=ax, color=hex_pal[0], label='historical_10')
        sns.lineplot(x=x_lin, y=y_50_hist, dashes=(2, 2), ax=ax, color=hex_pal[1], label='historical_50')
        sns.lineplot(x=x_lin, y=y_90_hist, dashes=(2, 2), ax=ax, color=hex_pal[2], label='historical_90')
        ax.fill_between(x_lin, y_10, y_90, alpha=0.1, color='black')
        plt.legend()
        plt.title('Fig 4: Monthly Predicted Quantiles vs. Historical', fontsize=11, wrap=True)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('Month')
        ax.set_ylabel('Volume (KAF)')
        plt.setp(ax.get_legend().get_texts(), fontsize=11)
        plt.setp(ax.get_legend().get_title(), fontsize=11)
        plt.savefig(plot_dir / 'fig4.png', bbox_inches='tight')

    # Explanation of ensemble of yearly models
    def get_model_shap_df(model, data, feature_cols, model_name):
        model_type, quantile, model_freq = model_name.split('_')
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(data[feature_cols])
        monthly_pct = mth_pct[quantile][issue_month]
        if model_freq == 'yr':
            monthly_pct = 1 - monthly_pct
        type_pct = model_type_pct[quantile]
        if model_type == 'lgb':
            type_pct = 1 - type_pct
        shap_values.values = shap_values.values * type_pct * monthly_pct
        shap_df = pd.DataFrame(shap_values.values, columns=shap_values.feature_names)
        shap_df['model'] = model_name
        if model_freq == 'mth':
            shap_df['pred_month_index'] = pd.DataFrame(shap_values.data, columns=shap_values.feature_names)['pred_month'].astype(int)
        return shap_df

    def agg_static_and_dynamic_shap_yr(all_shap, quantile, issue_date_yr_features):
        static_contribution_sum = all_shap[static_cols].sum().sum()
        dynamic_set = list(set(all_shap.columns).intersection(set(dynamic_cols)))
        dynamic_contributions = all_shap[dynamic_set].sum(axis=0)
        dynamic_contributions.index = [i.split('_deviation')[0] for i in dynamic_contributions.index]
        dynamic_values = issue_date_yr_features[dynamic_cols].iloc[0]
        dynamic_values.index = [i.split('_deviation')[0] for i in dynamic_values.index]
        dynamic_values = dynamic_values.to_frame(name='feature_values').groupby(level=0).mean()
        dynamic_contributions = dynamic_contributions.to_frame(name='shap_contribution').groupby(level=0).sum()
        dynamic_contributions = dynamic_contributions.join(dynamic_values).sort_values('shap_contribution', ascending=False)
        static_contributions = pd.DataFrame([[static_contribution_sum, 0]], columns=['shap_contribution', 'feature_values'], index=['static_site_features'])
        total_contributions = pd.concat([static_contributions, dynamic_contributions])

        # This is a placeholder explainer that we will fill with the ensembled shap values
        explainer = shap.TreeExplainer(models[f'lgb_{quantile}_yr'])
        shap_values = explainer(issue_date_yr_features[lgb_yr_feature_cols[quantile]])
        shap_values.values = (total_contributions['shap_contribution']).values.T.reshape(1, -1)
        shap_values.data = total_contributions.values.T[1].reshape(1, -1)
        shap_values.feature_names = list(total_contributions.index)

        return shap_values, dynamic_values

    def agg_static_and_dynamic_shap_mth(all_shap, quantile, issue_date_mth_features):
        static_contribution_agg = all_shap[static_cols].sum(axis=1).to_frame().groupby(level=0).sum()
        static_contribution_agg.columns = ['static_site_features']
        dynamic_set = list(set(all_shap.columns).intersection(set(dynamic_cols))) + ['pred_month']
        dynamic_contributions = all_shap[dynamic_set].groupby(level=0).sum().T
        dynamic_contributions.index = [i.split('_deviation')[0] for i in dynamic_contributions.index]
        dynamic_contributions = dynamic_contributions.groupby(level=0).sum()
        total_contributions = pd.concat([dynamic_contributions, static_contribution_agg.T]).T

        # This is a placeholder explainer that we will fill with the ensembled shap values
        explainer = shap.TreeExplainer(models[f'lgb_{quantile}_mth'])
        shap_values = explainer(issue_date_mth_features[lgb_mth_feature_cols[quantile]])
        shap_values.values = total_contributions.values
        shap_values.data = np.concatenate([[np.ones_like(rw.values) * idx] for idx, rw in total_contributions.iterrows()])
        shap_values.feature_names = list(total_contributions.T.index)
        return shap_values

    def get_ensemble_yr_shap_values(quantile, issue_date_yr_features):
        quantile = str(quantile)
        cat_yr_shap_df = get_model_shap_df(models[f'cat_{quantile}_yr'], issue_date_yr_features, cat_yr_feature_cols[quantile], f'cat_{quantile}_yr')
        lgb_yr_shap_df = get_model_shap_df(models[f'lgb_{quantile}_yr'], issue_date_yr_features, lgb_yr_feature_cols[quantile], f'lgb_{quantile}_yr')

        all_shap = pd.concat([cat_yr_shap_df, lgb_yr_shap_df]).set_index('model')
        shap_values, dynamic_values = agg_static_and_dynamic_shap_yr(all_shap, quantile, issue_date_yr_features)
        return shap_values, dynamic_values

    def get_ensemble_mth_shap_values(quantile, issue_date_mth_features):
        quantile = str(quantile)
        cat_mth_shap_df = get_model_shap_df(models[f'cat_{quantile}_mth'], issue_date_mth_features, cat_mth_feature_cols[quantile], f'cat_{quantile}_mth')
        lgb_mth_shap_df = get_model_shap_df(models[f'lgb_{quantile}_mth'], issue_date_mth_features, lgb_mth_feature_cols[quantile], f'lgb_{quantile}_mth')

        all_shap = pd.concat([cat_mth_shap_df, lgb_mth_shap_df]).set_index('model')
        all_shap.set_index('pred_month_index', inplace=True)
        shap_values = agg_static_and_dynamic_shap_mth(all_shap, quantile, issue_date_mth_features)
        return shap_values
    
    # dynamic_values are input features and are the same regardless of the quantile
    shap_values_10_yr, dynamic_values = get_ensemble_yr_shap_values(10, issue_date_yr_features)
    shap_values_50_yr, _ = get_ensemble_yr_shap_values(50, issue_date_yr_features)
    shap_values_90_yr, _ = get_ensemble_yr_shap_values(90, issue_date_yr_features)

    _, prev_dynamic_values = get_ensemble_yr_shap_values(10, prev_issue_date_yr_features)

    dynamic_drop_cols = ['day_of_year', 'prev_month_volume']
    diff_dynamic_values = np.round(dynamic_values - prev_dynamic_values, 3).drop(dynamic_drop_cols)
    diff_dynamic_values.columns = ['feature_values_change']
    curr_dynamic_values = dynamic_values.drop(dynamic_drop_cols)

    fig = plt.figure()
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.3)
    ax1 = plt.subplot(gs[0])
    sns.heatmap(data=diff_dynamic_values, annot=True, cmap="RdYlBu", ax=ax1, yticklabels=True, vmin=-1, vmax=1)
    plt.title('Fig 5: Z-Score Change from Previous Issue Date', fontsize=10)
    ax2 = plt.subplot(gs[1])
    sns.heatmap(data=curr_dynamic_values,
                annot=True, cmap="RdYlBu", ax=ax2, yticklabels=False, vmin=-2.5, vmax=2.5)
    plt.title('Fig 6: Z-Score Value on Issue Date', fontsize=10)
    plt.gcf().set_size_inches(8, 1.4)
    plt.savefig(plot_dir / 'fig5-6.png', bbox_inches='tight')

    if site_id in no_monthly_data:
        logger.info(f'No monthly data for site_id {site_id}, omitting Figures 7-9.')
    else:
        shap_values_10_mth = get_ensemble_mth_shap_values(10, issue_date_mth_features)
        shap_values_50_mth = get_ensemble_mth_shap_values(50, issue_date_mth_features)
        shap_values_90_mth = get_ensemble_mth_shap_values(90, issue_date_mth_features)

        prev_shap_values_10_mth = get_ensemble_mth_shap_values(10, prev_issue_date_mth_features)
        prev_shap_values_50_mth = get_ensemble_mth_shap_values(50, prev_issue_date_mth_features)
        prev_shap_values_90_mth = get_ensemble_mth_shap_values(90, prev_issue_date_mth_features)

        diff_shap_values_10_mth = copy.copy(prev_shap_values_10_mth)
        diff_shap_values_10_mth.values = shap_values_10_mth.values - prev_shap_values_10_mth.values

        diff_shap_values_50_mth = copy.copy(prev_shap_values_50_mth)
        diff_shap_values_50_mth.values = shap_values_50_mth.values - prev_shap_values_50_mth.values

        diff_shap_values_90_mth = copy.copy(prev_shap_values_90_mth)
        diff_shap_values_90_mth.values = shap_values_90_mth.values - prev_shap_values_90_mth.values

        fig = plt.figure()
        fig.suptitle('Fig 7: Changes from Previous Issue Date - Monthly Model Ensemble - 10th Quantile', fontsize=11)
        gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 0.25], hspace=0.2)
        ax1 = plt.subplot(gs[0])
        shap.plots.beeswarm(diff_shap_values_10_mth, max_display=20, color_bar_label='pred_month', show=False, color="viridis", s=70, alpha=0.7)
        plt.title('Shap Value Change from Previous Issue Date', fontsize=10)
        for child in ax1.get_children():
            if isinstance(child, matplotlib.collections.PathCollection):
                child.set_color('black')
                child.set_linewidth(1)
        ax2 = plt.subplot(gs[1])
        sns.barplot(data=issue_date_mth_features, x='pred_month', y='mth_pred_10', hue='pred_month', alpha=0.8, palette="viridis", ax=ax2)
        sns.stripplot(data=prev_issue_date_mth_features, x='pred_month', y='mth_pred_10', hue='pred_month', alpha=0.8,
                    palette="magma", ax=ax2, size=12, edgecolor='black', linewidth=1)
        plt.title('Monthly Predictions', fontsize=10)
        ax2.legend(bbox_to_anchor=(2.3, 0.975))
        plt.gcf().text(1.08, 0.79, 'current', fontsize=10)
        plt.gcf().text(1.08, 0.53, 'previous', fontsize=10)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        plt.gcf().set_size_inches(8, 3.4)
        plt.savefig(plot_dir / 'fig7.png', bbox_inches='tight')

        fig = plt.figure()
        fig.suptitle('Fig 8: Changes from Previous Issue Date - Monthly Model Ensemble - 50th Quantile', fontsize=11)
        gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 0.25], hspace=0.2)
        ax1 = plt.subplot(gs[0])
        shap.plots.beeswarm(diff_shap_values_50_mth, max_display=20, color_bar_label='pred_month', show=False, color="viridis", s=70, alpha=0.7)
        plt.title('Shap Value Change from Previous Issue Date', fontsize=10)
        for child in ax1.get_children():
            if isinstance(child, matplotlib.collections.PathCollection):
                child.set_color('black')
                child.set_linewidth(1)
        ax2 = plt.subplot(gs[1])
        sns.barplot(data=issue_date_mth_features, x='pred_month', y='mth_pred_50', hue='pred_month', alpha=0.8, palette="viridis", ax=ax2)
        sns.stripplot(data=prev_issue_date_mth_features, x='pred_month', y='mth_pred_50', hue='pred_month', alpha=0.8,
                    palette="magma", ax=ax2, size=12, edgecolor='black', linewidth=1)
        plt.title('Monthly Predictions', fontsize=10)
        ax2.legend(bbox_to_anchor=(2.3, 0.975))
        plt.gcf().text(1.08, 0.79, 'current', fontsize=10)
        plt.gcf().text(1.08, 0.53, 'previous', fontsize=10)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        plt.gcf().set_size_inches(8, 3.4)
        plt.savefig(plot_dir / 'fig8.png', bbox_inches='tight')

        fig = plt.figure()
        fig.suptitle('Fig 9: Changes from Previous Issue Date - Monthly Model Ensemble - 90th Quantile', fontsize=11)
        gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 0.25], hspace=0.2)
        ax1 = plt.subplot(gs[0])
        shap.plots.beeswarm(diff_shap_values_90_mth, max_display=20, color_bar_label='pred_month', show=False, color="viridis", s=70, alpha=0.7)
        plt.title('Shap Value Change from Previous Issue Date', fontsize=10)
        for child in ax1.get_children():
            if isinstance(child, matplotlib.collections.PathCollection):
                child.set_color('black')
                child.set_linewidth(1)
        ax2 = plt.subplot(gs[1])
        sns.barplot(data=issue_date_mth_features, x='pred_month', y='mth_pred_90', hue='pred_month', alpha=0.8, palette="viridis", ax=ax2)
        sns.stripplot(data=prev_issue_date_mth_features, x='pred_month', y='mth_pred_90', hue='pred_month', alpha=0.8,
                    palette="magma", ax=ax2, size=12, edgecolor='black', linewidth=1)
        plt.title('Monthly Predictions', fontsize=10)
        ax2.legend(bbox_to_anchor=(2.3, 0.975))
        plt.gcf().text(1.08, 0.79, 'current', fontsize=10)
        plt.gcf().text(1.08, 0.53, 'previous', fontsize=10)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        plt.gcf().set_size_inches(8, 3.4)
        plt.savefig(plot_dir / 'fig9.png', bbox_inches='tight')

    # Generate plots for Fig 10-12 related to Yearly Model Shap Values
    fig = plt.figure()
    shap.plots.waterfall(shap_values_10_yr[0], max_display=20, show=False)
    plt.title('Fig 10: Yearly Model Shap Values - 10th Quantile (log Volume KAF)')
    plt.savefig(plot_dir / 'fig10.png', bbox_inches='tight')

    fig = plt.figure()
    shap.plots.waterfall(shap_values_50_yr[0], max_display=20, show=False)
    plt.title('Fig 11: Yearly Model Shap Values - 50th Quantile (log Volume KAF)')
    plt.savefig(plot_dir / 'fig11.png', bbox_inches='tight')

    fig = plt.figure()
    shap.plots.waterfall(shap_values_90_yr[0], max_display=20, show=False)
    plt.title('Fig 12: Yearly Model Shap Values - 90th Quantile (log Volume KAF)')
    plt.savefig(plot_dir / 'fig12.png', bbox_inches='tight')

    # Plotting the most correlated Snotel Sites
    feature_corr_dir = preprocessed_dir / 'feature_corrs'

    feature = 'snotel_wteq_deviation'
    snotel_wteq_stations = pd.read_csv(feature_corr_dir / f'{issue_year}/{site_id}/{feature}_corr.csv')
    # Read in the metadata
    station_metadata = pd.read_csv(data_dir / 'snotel/station_metadata.csv')
    station_metadata['station_triplet'] = station_metadata['stationTriplet'].str.replace(':','_')
    corr_stations = pd.merge(snotel_wteq_stations, station_metadata, on='station_triplet', how='left')

    train_snotel = pd.read_csv(preprocessed_dir / 'train_snotel.csv')
    site_snotel = train_snotel[(train_snotel['station_triplet'].isin(corr_stations['station_triplet'])) &
                            (train_snotel['month'].isin([1,2,3,4,5,6,7]))]
    snotel_grouped = site_snotel.groupby(['station_triplet', 'month_day'])['wteq'].agg([np.mean, np.std]).reset_index()
    snotel_deviation = pd.merge(site_snotel, snotel_grouped, on=['station_triplet', 'month_day'], how='left')
    snotel_deviation['snotel_wteq_deviation'] = (snotel_deviation['wteq'] - snotel_deviation['mean'])/snotel_deviation['std']
    snotel_deviation['snotel_wteq_deviation'] = snotel_deviation['snotel_wteq_deviation'].replace(np.inf, np.nan).replace(-np.inf, np.nan)
    snotel_deviation = snotel_deviation[['station_triplet', 'date', 'wteq', 'snotel_wteq_deviation']]
    station_deviation = pd.merge(snotel_deviation[snotel_deviation['date'] == issue_date], corr_stations, on='station_triplet', how='left')

    gdf = gpd.GeoDataFrame(station_deviation, geometry=gpd.points_from_xy(station_deviation.longitude, station_deviation.latitude))
    # USA State Shape File Available at - https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip
    us_map = gpd.read_file(data_dir / 'cb_2018_us_state_500k/cb_2018_us_state_500k.shp')
    site_meta = metadata[metadata['site_id'] == site_id]

    site_mask = us_map.apply(lambda row: Point(site_meta[['longitude', 'latitude']].values[0]).within(row['geometry']), axis=1)
    site_states = set()
    site_states.add(us_map[site_mask]['STUSPS'].values[0])
    for idx, rw in corr_stations.iterrows():
        station_mask = us_map.apply(lambda row: Point(rw[['longitude', 'latitude']].values).within(row['geometry']), axis=1)
        site_states.add(us_map[station_mask]['STUSPS'].values[0])
    
    site_map = us_map[us_map['STUSPS'].isin(site_states)]
    site_map = site_map.to_crs("EPSG:3395")
    gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:3395")
    meta_gdf = gpd.GeoDataFrame(metadata, geometry=gpd.points_from_xy(metadata.longitude, metadata.latitude))
    meta_gdf = meta_gdf.set_crs("EPSG:4326")
    meta_gdf = meta_gdf.to_crs("EPSG:3395")
    drainage_basins = gpd.read_file(data_dir / 'geospatial.gpkg')
    drainage_basins = drainage_basins.to_crs("EPSG:3395")

    fig, ax = plt.subplots(figsize=(10,8))
    site_map.boundary.plot(ax=ax, color='black')
    drainage_basins[drainage_basins['site_id'] == site_id].plot(ax=ax, alpha=0.4, color='black')
    drainage_basins[drainage_basins['site_id'] == site_id].boundary.plot(ax=ax, alpha=0.9, color='black')
    cmap = plt.get_cmap('RdYlBu')
    pc = gdf[gdf['date'] == issue_date].plot(
        ax=ax,
        alpha=0.9,
        column='snotel_wteq_deviation',
        legend_kwds={"shrink": 0.65},
        legend=True,
        cmap=cmap,
        vmin=-3,
        vmax=3,
        markersize=150,
        edgecolors='black'
    )
    meta_gdf[meta_gdf['site_id'] == site_id].plot(ax=ax, color='yellow', edgecolor='black', markersize=250, marker='X', zorder=3, alpha=0.8)
    plt.title(f'Fig 13: Snotel SWE Deviation (Correlated Stations) on {issue_date}')
    ax.set_yticks([])
    ax.set_xticks([])
    for idx, row in site_map.iterrows():
        plt.annotate(text=row['NAME'],
                    xy=(row['geometry'].centroid.xy[0][0],
                        row['geometry'].centroid.xy[1][0]),
                    horizontalalignment='center',
                    alpha=0.7)
    plt.savefig(plot_dir / 'fig13.png', bbox_inches='tight')


if __name__ == '__main__':
    from pathlib import Path

    import argparse
    import copy
    import matplotlib
    import sys
    import warnings
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from matplotlib import gridspec
    import lightgbm as lgb
    import catboost as cb
    import seaborn as sns
    import shap
    from scipy.interpolate import make_interp_spline
    from shapely.geometry import Point, Polygon

    sys.path.append(str(Path(__file__).parent.resolve()))

    PREPROCESSED_DIR = Path.cwd() / 'training/preprocessed_data'
    DATA_DIR = Path.cwd() / 'training/train_data'

    # Initiate the parser
    parser = argparse.ArgumentParser(description='Generate explainability summary for previous forecast')

    # Add arguments
    parser.add_argument('-s', '--site-id', type=str, default=None,
                        help='The site-id you would like to run explainability for.')
    parser.add_argument('-d', '--issue-date', type=str, default=None,
                        help='The issue-date you would like to run explainbility for.')
    
    # Read arguments from the command line
    args = parser.parse_args()

    site_id = args.site_id
    issue_date = args.issue_date

    warnings.filterwarnings('ignore')

    main(DATA_DIR, PREPROCESSED_DIR, site_id, issue_date)
