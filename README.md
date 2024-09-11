[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

[<img src='https://drivendata-public-assets.s3.amazonaws.com/watersupply-hungry-horse-dam-banner.jpg'>](https://www.drivendata.org/competitions/group/reclamation-water-supply-forecast/)

# Water Supply Forecast Rodeo

## Goal of the Challenge

Accurate seasonal water supply forecasts are crucial for effective water resources management in the Western United States. This region faces dry conditions and high demand for water, and these forecasts are essential for making informed decisions. They guide everything from water supply management and flood control to hydropower generation and environmental objectives. In this challenge, sponsored by the [U.S. Bureau of Reclamation](https://www.usbr.gov/), solvers developed probabilistic forecasting models to predict the future cumulative streamflow across 26 different monitoring sites.

## What's in this Repository

This repository contains code from winning competitors in the [Water Supply Forecast Rodeo](https://www.drivendata.org/competitions/group/reclamation-water-supply-forecast/) DrivenData challenge. Code for all winning solutions are open source under the MIT License.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

This competition was held across multiple stages, each with their own prizes:

- **Hindcast Stage**—models were evaluated with hold-out validation against 10 years of historical data.
- **Forecast Stage**—models were frozen in January 2024 and were used to make live predictions through July 2024.
- **Final Prize Stage**—models were evaluated with cross-validation over a 20-year historical period.
- **Explainability Bonus Track**—solvers produced forecast summary documents to communicate and explain forecasts to operational water resource managers.

> [!NOTE]
> The Water Supply Forecast Rodeo is still ongoing! Results will be added as they become finalized.

## Hindcast Stage

In this stage, solvers made code submissions that ran inference on a hold-out set of 10 years. They also submitted model reports detailing their modeling methodology. Winners were selected by a judging panel based on their quantitative performance and their methodology's rigor.

Place | Team or User | Score | Summary of Model
--- | --- | ---   | ---
1   | rasyidstat | 87.82 | Ensemble of LightGBM models with models per target quantile, trained using Tweedie loss for 50th quantile and quantile loss for 10th and 90th. Data sources were SNOTEL snow water equivalent, cumulative precipitation, USGS and USBR observed streamflow, and basin geographic attributes. Generated synthetic data during training.
2   | ck-ua | 90.78 | Ensemble of multilayer perceptron models with four layers with multiple outputs for the 3 quantiles, trained with quantile loss. Data sources were antecedent monthly flow, USGS observed streamflow, SNOTEL snow water equivalent, and precipitation. Snow water equivalent and precipitation aggregated and normalized across stations with per-site RANSAC linear models.
3   | oshbocker | 101.59 | Ensemble of CatBoost models targetting both monthly and seasonal streamflow, with models per quantile trained using quantile loss. Data sources were antecedent monthly flow, USGS observed streamflow, SNOTEL and CDEC snow water equivalent, Copernicus GLO elevations, and ACIS observed temperature and precipitation.

Code and reports for the Hindcast Stage can be found in the [`hindcast/`](./hindcast/) subdirectory. For each winner, see the `reports/` subdirectory for their model report and additional solution documentation.

**Winners Announcement: ["Meet the Winners of the Water Supply Forecast Rodeo Hindcast Stage"](https://drivendata.co/blog/water-supply-hindcast-winners)**

## Forecast Stage

In this stage, solvers submitted code submissions that DrivenData then executed on to issue forecasts for the 2024 season on four scheduled issue dates each month from January through July 2024. Winners were selected based on the lowest [averaged mean quantile loss](https://www.drivendata.org/competitions/259/reclamation-water-supply-forecast/page/827/#primary-metric-quantile-loss) of their forecasts.

Place | Team or User | Score | Summary of Model
--- | --- | ---   | ---
1   | oshbocker | 56.83 | Ensemble of CatBoost models targetting both monthly and seasonal streamflow, with models per quantile trained using quantile loss. Data sources were antecedent monthly flow, USGS observed streamflow, SNOTEL and CDEC snow water equivalent, Copernicus GLO elevations, and ACIS observed temperature and precipitation.
2   | ck-ua | 56.91 | Ensemble of multilayer perceptron models with four layers with multiple outputs for the 3 quantiles, trained with quantile loss. Data sources were antecedent monthly flow, USGS observed streamflow, SNOTEL snow water equivalent, and precipitation. Snow water equivalent and precipitation aggregated and normalized across stations with per-site RANSAC linear models.
3   | rasyidstat | 59.16 | Ensemble of LightGBM models with models per target quantile, trained using Tweedie loss for 50th quantile and quantile loss for 10th and 90th. Data sources were SNOTEL snow water equivalent, cumulative precipitation, USGS and USBR observed streamflow, and basin geographic attributes. Generated synthetic data during training.

Code and reports for the Hindcast Stage can be found in the [`forecast/`](./forecast/) subdirectory. For each winner, see the `reports/` subdirectory for their model report and additional solution documentation.

**Winners Announcement: _coming soon_
