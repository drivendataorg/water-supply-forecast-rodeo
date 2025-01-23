[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

[<img src='https://drivendata-public-assets.s3.amazonaws.com/watersupply-hungry-horse-dam-banner.jpg'>](https://www.drivendata.org/competitions/group/reclamation-water-supply-forecast/)

# Water Supply Forecast Rodeo

## Goal of the Challenge

Accurate seasonal water supply forecasts are crucial for effective water resources management in the Western United States. This region faces dry conditions and high demand for water, and these forecasts are essential for making informed decisions. They guide everything from water supply management and flood control to hydropower generation and environmental objectives. In this challenge, sponsored by the [U.S. Bureau of Reclamation](https://www.usbr.gov/), solvers developed probabilistic forecasting models to predict the future cumulative streamflow across 26 different monitoring sites.

## What's in this Repository

This repository contains code from winning competitors in the [Water Supply Forecast Rodeo](https://www.drivendata.org/competitions/group/reclamation-water-supply-forecast/) DrivenData challenge. Code for all winning solutions are open source under the MIT License.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Challenge structure

This competition was held across multiple stages, each with their own prizes:

- [**Hindcast Stage**](#hindcast-stage-winners)—models were evaluated with hold-out validation against 10 years of historical data.
- [**Forecast Stage**](#forecast-stage-winners)—models were frozen in January 2024 and were used to make live predictions through July 2024.
- [**Final Prize Stage**](#overall-prizes-winners)—models were evaluated with cross-validation over a 20-year historical period.
- [**Explainability and Communication Bonus Track**](#explainability-and-communication-bonus-track-winners)—solvers produced forecast summary documents to communicate and explain forecasts to operational water resource managers.

You can find summaries for the winners of each stage below.

## Hindcast Stage winners

In [this stage](https://www.drivendata.org/competitions/257/reclamation-water-supply-forecast-hindcast/), solvers made code submissions that ran inference on a hold-out set of 10 years. They also submitted model reports detailing their modeling methodology. Winners were selected by a judging panel based on their quantitative performance and their methodology's rigor.

Place | Team or User | Score | Summary of Model
--- | --- | ---   | ---
1   | rasyidstat | 87.82 | Ensemble of LightGBM models with models per target quantile, trained using Tweedie loss for 50th quantile and quantile loss for 10th and 90th. Data sources were SNOTEL snow water equivalent, cumulative precipitation, USGS and USBR observed streamflow, and basin geographic attributes. Generated synthetic data during training.
2   | ck-ua | 90.78 | Ensemble of multilayer perceptron models with four layers with multiple outputs for the 3 quantiles, trained with quantile loss. Data sources were antecedent monthly flow, USGS observed streamflow, SNOTEL snow water equivalent, and precipitation. Snow water equivalent and precipitation aggregated and normalized across stations with per-site RANSAC linear models.
3   | oshbocker | 101.59 | Ensemble of CatBoost models targetting both monthly and seasonal streamflow, with models per quantile trained using quantile loss. Data sources were antecedent monthly flow, USGS observed streamflow, SNOTEL and CDEC snow water equivalent, Copernicus GLO elevations, and ACIS observed temperature and precipitation.

Code and reports for the Hindcast Stage can be found in the [`hindcast/`](./hindcast/) subdirectory. For each winner, see the `reports/` subdirectory for their model report and additional solution documentation.

**Winners Announcement: ["Meet the Winners of the Water Supply Forecast Rodeo Hindcast Stage"](https://drivendata.co/blog/water-supply-hindcast-winners)**

## Forecast Stage winners

In [this stage](https://www.drivendata.org/competitions/259/reclamation-water-supply-forecast/), solvers submitted code submissions that DrivenData then executed on to issue forecasts for the 2024 season on four scheduled issue dates each month from January through July 2024. Winners were selected based on the lowest [averaged mean quantile loss](https://www.drivendata.org/competitions/259/reclamation-water-supply-forecast/page/827/#primary-metric-quantile-loss) of their forecasts.

Place | Team or User | Score | Summary of Model
--- | --- | ---   | ---
1   | oshbocker | 56.83 | Ensemble of CatBoost models targetting both monthly and seasonal streamflow, with models per quantile trained using quantile loss. Data sources were: antecedent monthly flow; USGS observed daily streamflow; snow water equivalent from SNOTEL, CDEC and SWANN; Copernicus GLO elevations; ACIS observed temperature and precipitation.
2   | ck-ua | 56.91 | Ensemble of multilayer perceptron models with four layers with multiple outputs for the 3 quantiles, trained with quantile loss. Data sources were antecedent monthly flow, USGS observed streamflow, SNOTEL snow water equivalent, and precipitation. Snow water equivalent and precipitation aggregated and normalized across stations with per-site RANSAC linear models.
3   | rasyidstat | 59.16 | Ensemble of LightGBM models with models per target quantile, trained using Tweedie loss for 50th quantile and quantile loss for 10th and 90th. Data sources were SNOTEL snow water equivalent, cumulative precipitation, USGS and USBR observed streamflow, basin geographic attributes, and Palmer Drought Severity Index (PDSI). Generated synthetic data during training.

Code and reports for the Forecast Stage can be found in the [`forecast/`](./forecast/) subdirectory. For each winner, see the `reports/` subdirectory for their model report and additional solution documentation.

**Winners Announcement: ["Meet the winners of the Forecast and Final Prize Stages of the Water Supply Forecast Rodeo"](https://drivendata.co/blog/water-supply-forecast-and-final-winners)**

## Overall Prizes winners

In the [Final Stage](https://www.drivendata.org/competitions/262/reclamation-water-supply-forecast-final/), solvers submitted predictions for a leave-one-out cross-validation over the 20-year period from 2004 through 2023. Additionally, they submitted a final model report detailing their modeling methodology. Winners were selected by a judging panel based on their cross-validation performance, their Forecast Stage performance, and additional criteria including rigor, innovation, generalizability, and efficiency and scalability, and clarity.

Place | Team or User | CV Score | Summary of Model
--- | --- | ---   | ---
1   | oshbocker | 85.84 | Ensemble of CatBoost models targetting both monthly and seasonal streamflow, with models per quantile trained using quantile loss. Data sources were: antecedent monthly flow; observed daily streamflow; snow station data from SNOTEL, CDEC; gridded snow water equivalent from SWANN; Copernicus GLO elevations; ACIS observed temperature and precipitation; Palmer Drought Severity Index (PDSI).
2   | ck-ua | 90.11 | Ensemble of multilayer perceptron models with four layers with multiple outputs for the 3 quantiles, trained with quantile loss. Data sources were antecedent monthly flow, observed daily streamflow, snow water equivalent and precipitation from SNOTEL stations. Snow water equivalent and precipitation aggregated and normalized across stations with per-site RANSAC linear models.
3   | rasyidstat | 79.49 | Ensemble of LightGBM models with models per target quantile, trained using Tweedie loss for 50th quantile and quantile loss for 10th and 90th. Data sources were antecedent monthly flow; daily observed streamflow; snow water equivalent and cumulative precipitation from SNOTEL and CDEC; gridded snow water equivalent from SWANN; Palmer Drought Severity Index (PDSI); seasonal forecasts from Copernicus; ERA5-Land reanalysis data. Generated synthetic data during training.
4   | kurisu | 88.95 | Ensemble of XGBoost models with models per quantile trained using quantile loss. Data sources were antecedent monthly flow; snow water equivalent from SNOTEL stations; gridded snow water equivalent from SWANN; gridded accumulated precipitation.
5   | progin | 86.78 | LightGBM models per issue month and per tarquet quantile, trained using quantile loss. Data sources were monthly antecedent flow; daily observed streamflow; snow station data from SNOTEL; Palmer Drought Severity Index (PDSI); seasonal forecasts from Copernicus; ERA5-Land reanalysis data.

Code and reports for the Overall Prize can be found in the [`overall/`](./overall/) subdirectory. For each winner, see the `reports/` subdirectory for their model report and additional solution documentation.

**Winners Announcement: ["Meet the winners of the Forecast and Final Prize Stages of the Water Supply Forecast Rodeo"](https://drivendata.co/blog/water-supply-forecast-and-final-winners)**

## Explainability and Communication Bonus Track winners

In [this bonus track](https://www.drivendata.org/competitions/262/reclamation-water-supply-forecast-final/page/880/), solvers developed approaches to explaining and communicating forecasts to water resource managers. Solvers submitted four example forecast summaries—short documents that are representative of publications that a forecast agency would issue—as well as a report detailing their explainability methodology. Winners were selected by a judging panel.

Place | Team or User | Summary of Approach
--- | --- | ---
1   | kurisu | Uses Forecast Plots for visualizing water supply forecasts at the 10%, 50%, and 90% quantiles, along with Context Plots showing watershed conditions (SWE, PPT, Antecedent Flow) throughout the year compared to historical conditions (2004-2022). Explanations of forecasts and uncertainties are provided through What-If Plots and SHAP Waterfall Plots, offering insights into the impact of the features on forecasts and uncertainties.
2   | kamarain | Forecast shown with previous forecasts and historical observed quantiles for context. Individual model features – the spatio-temporal principal components of each dataset – are explored with the SHAP analysis to find out which ones are contributing the most to the predictions of each issue date and target site. After that the identified features are visualized and narrated together with the other features and datasets.
3   | rasyidstat | We use SHAP (SHapley Additive exPlanations) to calculate the percentage of feature contribution and relative contribution as explainability metrics for a given location and issue date. We present forecast predictions starting from the initial forecast update in January, along with historical median, average, minimum, and maximum values. In the explainability summary, we describe explainability metrics with the latest two consecutive issue dates as a comparison, including values for each feature, typically expressed as percent of normal.
4   | oshbocker | The explainability report includes visualizations that offer insights from individual components of the model ensemble and visualizations that show how the contributions of the ensemble come together. We benefited from existing work that provides explanations of gradient boosted models using Shapley values and further customized these explanations to suit the ensemble of quantile regressors.
5   | iamo-team | For each issue date, the forecast summaries present predictions of seasonal streamflow volumes putting them in historical context, alongside current conditions for key hydrological variables like basin snowpack, accumulated precipitation, and antecedent river flow. We also address variations in the predictive power of these variables along all issue dates, accounting for factors like basin characteristics and issue date timing by including information on their relevance derived from the forecast model using a model-agnostic method.

Code and reports for the Explainability Bonus Track can be found in the [`explainability/`](./explainability/) subdirectory. For each winner, see the `reports/` subdirectory for their forecast summaries, methodology report, and additional solution documentation.

**Winners Announcement: ["Meet the winners of the Forecast and Final Prize Stages of the Water Supply Forecast Rodeo"](https://drivendata.co/blog/water-supply-forecast-and-final-winners)**
