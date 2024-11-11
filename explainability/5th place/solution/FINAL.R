

####### Installing necessary R packages for data reprocessing and interference ##############

packages <- c( "readr", "dplyr", "tidyr", "lubridate",'data.table',"terra", "reshape2",
               'scales','doParallel','caret', 'pls')

#installed_packages <- packages %in% rownames(installed.packages())
#if (any(installed_packages == FALSE)) {
#  install.packages(packages[!installed_packages])
#}

invisible(lapply(packages, library, character.only = TRUE))

rm(installed_packages, packages)


###### Creating Water Year function #####################################


# Create a function to assign/arrange timeseries to water years (credit to @Caner: https://stackoverflow.com/questions/27626533/r-create-function-to-add-water-year-column?rq=3 )
get_water_year <- function(date_x = NULL, start_month = 9) {
  # Convert dates into POSIXlt
  dates.posix = as.POSIXlt(date_x)
  # Year offset
  offset = ifelse(dates.posix$mon >= start_month - 1, 1, 0)
  # Water year
  adj.year = dates.posix$year + 1900 + offset
  # Return the water year
  adj.year
}


######## Create a schedule of forecast issue dates ###############################

# Create a full sequence of dates from 1969 to the present to determine forecast issue dates.
# Forecast issue dates refer to the scheduled dates when forecasts are generated (e.g., 1st, 8th, 15th, and 22nd of each month).
{
  date_forecasts<-as.data.frame(seq(as.Date('1969-10-01'),as.Date(Sys.Date()+38),'day')) 
  colnames(date_forecasts)<-'date'
  
  date_forecasts<-date_forecasts %>% 
    mutate(year=lubridate::year(date),
           month=lubridate::month(date),
           day=lubridate::day(date),
           day=ifelse(nchar(day)<2, paste("0",day, sep=""), day),
           forecast_issue=paste(month,"-",day, sep=""),
           forecast_issue=if_else(month <8 & (day== '01' | day== '08' | day== '15' | day== '22'), forecast_issue,NA)) %>% 
    arrange(date) %>% 
    # fill(forecast_issue, .direction = "up") %>%
    select(date,forecast_issue) 
  
  # Add forecast month for future use
  date_forecasts<-date_forecasts %>% 
    mutate(forecast_month=as.numeric(substr(forecast_issue,1,1))) 
  
}

#########################################################################
############### Input Data Preparation ################################

#######   Processing SNOTEL data (SWE and accumulated precipitation per each basin)  

# Set working directory to the location of the SNOTEL data
setwd(".")

# Read metadata of selected SNOTEL stations (e.g., station IDs, locations)
selected_snotel_meta <- read_csv('data/selected_snotel_meta.csv')
# maxSWEcor <- read.csv('data/maxSWEcor_meta.csv')

# Replace colons with underscores in stationTriplet (unique station identifiers), then select unique, non-NA stationTriplets
selected_snotel_meta$stationTriplet <- gsub(':', '_', selected_snotel_meta$stationTriplet)
snotel_files <- unique(selected_snotel_meta$stationTriplet)
snotel_files <- na.omit(snotel_files)


# List directories corresponding to different years of SNOTEL data
year_folders <- list.dirs('data/snotel', full.names = TRUE, recursive = FALSE)

# Initialize an empty list to store SNOTEL data for each year
snotel_update <- list()


# Loop over each folder (each corresponding to a year) and process CSV files inside each folder
for (year_folder in year_folders) {
  # List all CSV files in the current FY folder
  csv_files <- list.files(year_folder, pattern = "\\.csv$", full.names = TRUE)
  
  # Only keep files that match the stationTriplet
  matching_files <- csv_files[sapply(csv_files, function(x) gsub("\\.csv$", "", basename(x)) %in% snotel_files)]
  
  for (csv_file in matching_files) {
    # Read the CSV file
    file_update <- read_csv(csv_file)
    
    # Check if at least one of the required columns exists
    if (!("PREC_DAILY" %in% names(file_update)) && !("WTEQ_DAILY" %in% names(file_update))) {
      next  # Skip this file if neither column is present
    }
    
    # If PREC_DAILY is missing, create it with NA values
    if (!"PREC_DAILY" %in% names(file_update)) {
      file_update$PREC_DAILY <- NA
    }
    
    # If WTEQ_DAILY is missing, create it with NA values
    if (!"WTEQ_DAILY" %in% names(file_update)) {
      file_update$WTEQ_DAILY <- NA
    }
    
    # Extract the station name from the CSV file name (remove path and .csv extension)
    station_name <- gsub("\\.csv$", "", basename(csv_file))
    file_update$stationTriplet <- station_name
    file_update<-file_update %>% 
      mutate(date=as.Date(date)) %>% 
      select(date,WTEQ_DAILY,PREC_DAILY,stationTriplet)
    # Append the data to the list
    snotel_update[[paste(year_folder, station_name, sep = '_')]] <- file_update
  }
}


# Combine all the data stored in the list into a single data frame
snotel_df<- bind_rows(snotel_update)%>% 
  rename(precipitation_cumulative = PREC_DAILY,
         SWE = WTEQ_DAILY) %>% 
  mutate(date = as.Date(date),
         date_snotel = as.Date(date)) %>% 
  left_join(selected_snotel_meta) %>% 
  select(site_id, snotel_id, date, SWE, precipitation_cumulative, date_snotel)

# Assign water years to the SNOTEL data using the get_water_year function defined earlier
snotel_df$forecast_year<-get_water_year(snotel_df$date, start_month = 10)

# Join the forecast schedule with the SNOTEL data and fill missing forecast issue dates
snotel_df<-date_forecasts %>% 
  full_join(snotel_df) %>% 
  # group_by(snotel_id,forecast_year) %>%
  arrange(date) %>% 
  fill(forecast_issue, .direction = "up") %>% 
  fill(forecast_month, .direction = "up") %>%
  mutate(year=lubridate::year(date_snotel))%>% 
  group_by(snotel_id,forecast_year) %>% 
  filter(!(lubridate::month(date)>7 & lubridate::month(date)<12)) %>% 
  mutate(forecast_issue=if_else(lubridate::month(date)>7 & lubridate::month(date_snotel)<=12,'1-01',forecast_issue)) %>% 
  mutate(forecast_month=if_else(lubridate::month(date)>7 & lubridate::month(date_snotel)<=12,1,forecast_month)) %>% 
  # arrange(date) %>% 
  # fill(forecast_issue, .direction = "updown") %>% 
  # fill(forecast_month, .direction = "updown") %>%
  select(-year)

# Import seasonal flow volume data
prior_historical_labels <- read_csv("prior_historical_labels.csv") %>% 
  rename(forecast_year=year) %>%
  filter(forecast_year>1959)
cross_validation_labels <- read_csv("cross_validation_labels.csv") %>% 
  rename(forecast_year=year)

#Create a metadata on historical observations per each basin (min, mean, max observed seasonal flow)
meta_lowest_highest_volume<-rbind(prior_historical_labels, cross_validation_labels) %>% 
  group_by(site_id) %>% 
  summarise(nobs=length(!is.na(volume)),
            min=min(volume, na.rm=T),
            mean=median(volume,na.rm=T),
            max=max(volume,na.rm=T))

#Merge SNOTEL data with seasonal flow volume and metadata of the most representative SNOTEL stations per each basin
SWE_volume<-rbind(prior_historical_labels, cross_validation_labels) %>%
  right_join(snotel_df) %>% 
  distinct() 
SWE_volume<-rbind(prior_historical_labels, cross_validation_labels) %>%
  right_join(snotel_df) %>% 
  distinct()

# Flagging the (issue) date when accumulated snowpack generally exhibits highest correlation with volume (single snow station by site)
maxSWEcor<-SWE_volume %>%
  filter(forecast_year<2005) %>% # filtering years before the cross-validation period
  group_by(site_id,snotel_id,forecast_issue) %>% # Group by site, SNOTEL station, and forecast issue date
  mutate(maxSWEcor=cor(volume,SWE, use="na.or.complete"))%>% # Calculate the correlation between volume and SWE using available data
  summarise(maxSWEcor=max(maxSWEcor),
            snotel_id=median(snotel_id)) %>% 
  ungroup() %>%
  group_by(site_id) %>%
  top_n(1, maxSWEcor)# Keep only one station and issue date the highest correlation

# Display distribution of high correlation issue dates across all basins
table(maxSWEcor$forecast_issue) 

# Add new columns to maxSWEcor to store the best issue date and SNOTEL ID
maxSWEcor<-maxSWEcor %>% 
  mutate(maxSWEcor_issue=forecast_issue) %>% 
  mutate(maxSWEcor_snotel_id=snotel_id)

# Join the maxSWEcor data back to the original SWE_volume dataset
SWE_volume<-SWE_volume %>% 
  left_join(maxSWEcor)

# Fill missing data for representative SNOTEL stations
SWE_volume <- SWE_volume %>%
  group_by(site_id, snotel_id) %>%  # Group by site and SNOTEL station
  arrange(date) %>%  # Sort the data by date to ensure proper filling
  group_by(site_id) %>%
  fill(c(maxSWEcor_snotel_id, maxSWEcor, maxSWEcor_issue), .direction = "down")  # Fill down any missing values

# Identify rows where forecast_issue matches maxSWEcor_issue and fill missing values for those rows
SWE_volume<-SWE_volume %>%
  group_by(site_id) %>%
  mutate(matching_rows= maxSWEcor_issue==forecast_issue)%>%
  mutate(matching_rows=ifelse(matching_rows==TRUE, matching_rows,NA))%>%
  group_by(site_id, snotel_id, forecast_year) %>%
  arrange(forecast_issue) %>%
  fill(matching_rows, .direction = "down") %>%
  ungroup()%>%
  mutate(maxSWEcor=if_else(is.na(matching_rows),'NO','YES' )) %>%
  select(-c(matching_rows))

# Remove temporary variables used during SNOTEL data processing to free up memory
rm(snotel_update, snotel_files, year_folders, csv_files, csv_file,file_update, matching_files, station_name)
gc()
##### Processing monthly flow data ########################################

#Load naturalized monthly flow data
prior_historical_monthly_flow <- read_csv("prior_historical_monthly_flow.csv") %>% 
  rename(flow=volume) %>% 
  filter(forecast_year>1959)

cross_validation_monthly_flow <- read_csv("cross_validation_monthly_flow.csv") %>% 
  rename(flow=volume)

#Merge historical monthly and seasonal flow data
volume_flow<-prior_historical_labels %>% 
  left_join(prior_historical_monthly_flow)

# For each basin (site_id) and month, calculate the correlation between seasonal volume and monthly flow.
# Correlations below a threshold of 0.12 are filtered out.
negFLOWcor<-volume_flow %>%
  # filter(month<4 | month>6) %>%
  group_by(site_id, month) %>%
  summarise(FLOWcor=cor(volume,flow, use="na.or.complete")) %>%
  mutate(FLOWcor=if_else(FLOWcor>0.12,FLOWcor,NA))

# Monthly flow values with correlations below the threshold are replaced with NA.
volume_flow <- volume_flow%>%
  left_join(negFLOWcor, by = c("site_id", "month"))%>%
  mutate(flow=ifelse(is.na(FLOWcor),NA, flow)) %>%
  select(-FLOWcor)

# Determine the months with the highest correlation between monthly flow and seasonal volume
# This step focuses on months before April (or including July) to find the best predictor of seasonal flow.
maxFLOWcor<-volume_flow %>%
  filter(month<4 | month>7) %>%
  group_by(site_id, month) %>%
  mutate(maxFLOWcor=cor(volume,flow, use="na.or.complete"))%>%
  summarise(maxFLOWcor=max(maxFLOWcor))%>%
  top_n(1, maxFLOWcor)%>%
  filter(maxFLOWcor>0.2)

maxFLOWcor<-maxFLOWcor %>%
  rename(maxFLOWcor_month=month)

# Combine historical and cross-validation flow data and join with maxFLOWcor results
monthly_flow<-rbind(prior_historical_monthly_flow,cross_validation_monthly_flow) %>% 
  left_join(maxFLOWcor) %>% 
  mutate(date=as.Date(paste(year,month,'15',sep='-')),
         forecast_month=if_else(month<7,month+1,1)) 

# Join the monthly flow data with forecast issue dates and sort by date to ensure proper temporal alignment.
monthly_flow<-monthly_flow %>% 
  left_join(date_forecasts) %>% 
  group_by(forecast_year) %>%
  arrange(date) 

# For each basin, identify the months where the flow correlation is the highest, and mark them accordingly ('YES' or 'NO').
monthly_flow<-monthly_flow %>% 
  group_by(site_id,forecast_year) %>% 
  mutate(maxFLOWcor=if_else(maxFLOWcor_month<=month & month<7,'YES','NO')) %>% 
  mutate(maxFLOWcor=if_else(maxFLOWcor_month>=12 & (month<7 | month>=12),'YES',maxFLOWcor))%>% 
  mutate(maxFLOWcor=if_else((maxFLOWcor_month>=11 & maxFLOWcor_month<12) & (month<7 | month>=11),'YES',maxFLOWcor)) %>% 
  mutate(maxFLOWcor=if_else((maxFLOWcor_month>=10 & maxFLOWcor_month<11) & (month<7 | month>=10),'YES',maxFLOWcor)) %>% 
  mutate(maxFLOWcor_value = ifelse(maxFLOWcor_month == month, flow, NA)) %>% 
  group_by(forecast_issue) %>% 
  arrange(date) %>% 
  fill(maxFLOWcor_value, .direction = "down") %>%
  ungroup()%>% 
  mutate(date_flow=as.Date(paste(year,month,'15',sep='-'), format='%Y-%m-%d')) %>% 
  rename(month_flow=month)

# Adjust the flow dates to the end of the month
monthly_flow$date_flow<-ceiling_date(monthly_flow$date_flow, "month") - days(1)



#=========== Processing supplementary naturalized monthly flow from NRCS database ===============

# Load supplementary monthly flow from NRCS database
prior_historical_supplementary_nrcs_monthly_flow <- read_csv("data/prior_historical_supplementary_nrcs_monthly_flow.csv") %>% 
  rename(flow=volume) %>% 
  filter(forecast_year>1959)

cross_validation_supplementary_nrcs_monthly_flow <- read_csv("data/cross_validation_supplementary_nrcs_monthly_flow.csv") %>% 
  rename(flow=volume)

# Filter NRCS data for months from January to May (1-5)
filtered_nrcs <- prior_historical_supplementary_nrcs_monthly_flow %>%
  filter(month %in% 1:5)

# Join filtered NRCS data with prior_historical_labels on forecast_year
joined_data <- filtered_nrcs %>%
  inner_join(prior_historical_labels, by = "forecast_year")

# Calculate the correlation for each nrcs_id, site_id, and month
correlation_results <- joined_data %>%
  group_by(nrcs_id, site_id, month) %>%
  summarize(
    n_non_na = sum(!is.na(flow) & !is.na(volume)),  # Count non-NA pairs
    correlation = ifelse(n_non_na >= 2, cor(flow, volume, use = "complete.obs"), NA), # Only calculate correlation if enough data
    .groups = 'drop'
  ) %>%
  filter(!is.na(correlation) & n_non_na >= 10) %>% 
  filter(correlation>0)# Filter out groups with insufficient data

# Calculate the median correlation for each nrcs_id and site_id across the months (1-5)
median_correlations <- correlation_results %>%
  group_by(nrcs_id, site_id) %>%
  summarize(median_correlation = median(correlation), .groups = 'drop')

# For each site_id, select the nrcs_id with the highest median correlation
median_correlations<- median_correlations %>%
  group_by(site_id) %>%
  slice_max(median_correlation) %>%
  ungroup()

# Merge back to get the corresponding month with the highest correlation
maxFLOWcor_NRCS <- correlation_results %>%
  inner_join(median_correlations, by = c("nrcs_id", "site_id")) %>%
  group_by(site_id) %>%
  slice_max(correlation) %>%
  ungroup() %>%
  select(site_id, nrcs_id, month, correlation)%>%
  rename(maxFLOWcor = correlation,
         maxFLOWcor_month=month)

# Combine historical and cross-validation NRCS monthly flow data
NRCS_df<-rbind(prior_historical_supplementary_nrcs_monthly_flow,cross_validation_supplementary_nrcs_monthly_flow) %>% 
  distinct() %>% 
  right_join(maxFLOWcor_NRCS) %>% 
  mutate(date=as.Date(paste(year,month,'15',sep='-')),
         forecast_month=if_else(month<7,month+1,1)) %>% 
  filter(month!=7)

# Join NRCS data with forecast dates
NRCS_df<-NRCS_df%>% 
  left_join(date_forecasts) %>% 
  group_by(forecast_year) %>%
  arrange(date) 

# For each site and forecast year, flag whether a month has the maximum correlation based on the previously computed maxFLOWcor.
NRCS_df<-NRCS_df %>% 
  group_by(site_id,forecast_year) %>% 
  mutate(maxFLOWcor=if_else(maxFLOWcor_month<=month & month<7,'YES','NO')) %>% 
  mutate(maxFLOWcor=if_else(maxFLOWcor_month>=12 & (month<7 | month>=12),'YES',maxFLOWcor))%>% 
  mutate(maxFLOWcor=if_else((maxFLOWcor_month>=11 & maxFLOWcor_month<12) & (month<7 | month>=11),'YES',maxFLOWcor)) %>% 
  mutate(maxFLOWcor=if_else((maxFLOWcor_month>=10 & maxFLOWcor_month<11) & (month<7 | month>=10),'YES',maxFLOWcor)) %>% 
  mutate(maxFLOWcor_value = ifelse(maxFLOWcor_month == month, flow, NA)) %>% 
  group_by(forecast_issue) %>% 
  arrange(date) %>% 
  fill(maxFLOWcor_value, .direction = "down") %>%
  ungroup()%>% 
  mutate(date_flow=as.Date(paste(year,month,'15',sep='-'), format='%Y-%m-%d')) %>% 
  rename(month_flow=month)

# Adjust the flow dates to the end of the month
NRCS_df$date_flow<-ceiling_date(NRCS_df$date_flow, "month") - days(1)


# Remove temporary variables 
rm(median_correlations, correlation_results, joined_data)

########   Processing SWANN SWE data ########################

# Import SWANN data
files<-list.files('data/SWANN/', full.names = T, pattern = '\\.nc$')
swann<-rast(files)
layer_names <- names(swann)

# Use grep to find indices of layers that contain "SWE" in their name
swe_layers <- grep("SWE", layer_names)

# Select only the SWE layers from the SpatRaster
swann_swe <- swann[[swe_layers]]


# subset SWE layers that correspond to previous day of each forecast issue date
swann_dates<-time(swann_swe)

dates_to_subset<- date_forecasts %>%
  filter(!is.na(forecast_issue)) %>%
  mutate(date = date - 1)


indices <- which(swann_dates %in% dates_to_subset$date)

subset_swann <- swann_swe[[indices]]

head(time(subset_swann))

# Calculate basin averaged mean SWE per each day

#Load basin shapefiles
basins<-vect('data/geospatial.gpkg')
basins

basins<-project(basins, crs(swann_swe)) # aligning basins projection to that of the SWE data

plot(swann_swe[[1]])
plot(basins, add=T)

# Zonal statistics - aggregates all cells per basin to mean:
swann_df<-zonal(subset_swann, basins, "mean", na.rm = TRUE)

#Arrange the data and assign dates
swann_df<-as.data.frame(t(swann_df))
colnames(swann_df)<-basins$site_id
swann_df$date<-as.Date(time(subset_swann))


# Covert the dataframe from wide to long format
swann_swe <- melt(setDT(swann_df), id.vars = c("date"), variable.name = "site_id")
colnames(swann_swe)[3]<-'SWE_swann'

swann_swe <-swann_swe %>% 
  mutate(issue_date=as_date(date)+1,
         month=month(date),
         forecast_year=year(issue_date)) %>% 
  mutate(forecast_issue=substr(as.character(issue_date), 7,10))

rm(dates_to_subset,subset_swann,swann,swann_df, layer_names,files,indices,swann_dates,swe_layers)
gc()

################## Processing Climate Indices ###############################

# Read and preprocess Southern Oscillation Index (SOI) data
{
  SOI<-read_table2("data/teleconnections/soi.txt", skip=87)
  SOI<-head(SOI,-7)
  str(SOI)
  SOI<-data.frame(lapply(SOI, function(x) {gsub("-999.9", "", x)}))
  str(SOI)
  
  SOI<-lapply(SOI, as.numeric)
  SOI<-as.data.frame(SOI)
  SOI<-SOI[,c(1:13)]
  
  colnames(SOI)[-1]<-seq(1,12,by=1)
  
  SOI <-data.table:: melt(SOI, id.vars = c("YEAR"), variable.name = "month")
  
  SOI$month<-as.numeric(SOI$month)
  SOI<-SOI %>%
    rename(year = YEAR)
  SOI$date<-as.Date(paste('15',SOI$month,SOI$year, sep='-'), format = '%d-%m-%Y')
  SOI<-SOI %>% 
    arrange(date) %>% 
    mutate(SOI= data.table::frollmean(value,3,align='right')) %>% 
    select(year,month,SOI)
  
}
# Read and preprocess Pacific North American (PNA) Index data
{
  PNA<-read_table2("data/teleconnections/pna.txt", skip=1)
  colnames(PNA)[1]<-"year"
  colnames(PNA)[-1]<-seq(1,12,1)
  PNA <-data.table:: melt(PNA, id.vars = c("year"), variable.name = "month")
  PNA$date<-as.Date(paste('15',PNA$month,PNA$year, sep='-'), format = '%d-%m-%Y')
  PNA<-PNA %>% 
    arrange(date) %>% 
    mutate(PNA= data.table::frollmean(value,3,align='right')) %>% 
    select(year,month,PNA)
  
}
# Read and preprocess Pacific Decadal Oscillation (PDO) Index data
{
  PDO<-read_table2("data/teleconnections/pdo.txt", skip=1)
  colnames(PDO)[1]<-"year"
  
  colnames(PDO)[-1]<-seq(1,12,1)
  PDO<-data.table:: melt(PDO, id.vars = c("year"), variable.name = "month")
  PDO$value<-ifelse(PDO$value>10 | PDO$value< -10, NA, PDO$value)

  PDO$date<-as.Date(paste('15',PDO$month,PDO$year, sep='-'), format = '%d-%m-%Y')
  PDO<-PDO %>% 
    arrange(date) %>% 
    mutate(PDO= data.table::frollmean(value,3,align='right')) %>% 
    select(year,month,PDO)
  
}

# Merge SOI, PNA, and PDO data into one dataframe
ALL<-Reduce(function(...) merge(..., by=c('year','month'), all.x=TRUE), list(SOI,PNA,PDO))
ALL<-ALL[ALL$year>1983,]

# Convert the merged data from wide to long format
long<-data.table:: melt(ALL, id.vars = c("year","month"), variable.name = "INDEX")

# Generate forecast years and reformat the data
long<-long %>% 
  mutate(forecast_year=ifelse(month>7,year+1,year),
         month=ifelse(nchar(month)<2, paste("0",month, sep=""), month),
         date_index=as.Date(paste(year,'-', month, "-08", sep="")),
         month=as.numeric(month)) %>% 
  group_by(forecast_year) %>% 
  mutate(forecast_month=if_else(month<7, month+1,1)) %>%
  filter(!(month>6 & month<11)) %>% 
  rename(month_index=month)

# Restrict SOI values used in forecasting to the month of January
long<-long %>% 
  group_by(forecast_year) %>% 
  mutate(value=if_else(INDEX=='SOI'& forecast_month>1,NA,value)) %>% # Set SOI values to NA for forecast_month > 1
  group_by(forecast_year,INDEX) %>% 
  arrange(date_index) %>% 
  fill(value,.direction = 'down') %>% 
  ungroup()

# Restrict PNA values used in forecasting to the month of February
long<-long %>% 
  group_by(forecast_year) %>% 
  mutate(value=if_else(INDEX=='PNA'& forecast_month>2,NA,value)) %>% 
  group_by(forecast_year,INDEX) %>% 
  arrange(date_index) %>% 
  fill(value,.direction = 'down') %>% 
  ungroup()

# Restrict PDO values used in forecasting to the month of January
long<-long %>% 
  group_by(forecast_year) %>% 
  mutate(value=if_else(INDEX=='PDO'& forecast_month>1,NA,value)) %>%
  group_by(forecast_year,INDEX) %>% 
  arrange(date_index) %>% 
  fill(value,.direction = 'down') %>% 
  ungroup()

# Adjust the date to the last day of the month
long$date_index<-ceiling_date(long$date_index, "month") - days(1)

# Initialize an empty list to store correlation results for each site
cor_IND<-list()

# Loop through each unique site in the volume_flow dataset
for (i in unique(volume_flow$site_id)) {
  volume_df<-volume_flow[volume_flow$site_id==i,] # Subset the volume_flow data for the current site
  
  # Merge the volume data with the climate indices data by forecast year
  ddf<-volume_df %>% 
    filter(site_id==i) %>% 
    right_join(long,by=c('forecast_year'))
  
  # Initialize a list to store correlation results for each climate index
  inda<-list()
  
  # Loop through each unique climate index in the long dataset
  for (climind in unique(long$INDEX)) {
    
    # Calculate the correlation between the climate index values and volume for each month
    cor_indices<-ddf %>% 
      filter(INDEX==climind) %>% 
      filter(!(month_index>4 & month_index<10)) %>% 
      group_by(month_index) %>% 
      summarise( obs_volume=length(volume),
                 correlation = cor(value, volume, use="na.or.complete"))%>% 
      arrange(correlation)%>% 
      slice(1:1) %>% 
      mutate( site_id=i,
              INDEX=climind)
    
    inda[[climind]]<-cor_indices
  }
  
  
  cor_IND[[i]]<-inda
  
}

# Unlist and combine all correlation results into a single data frame
unlistIND<-unlist(cor_IND, recursive = FALSE)
meta_INDICES<-bind_rows(unlistIND)

# Filter out weak correlations (those between -0.1 and 0.1)
meta_INDICES<-meta_INDICES %>% 
  filter(!(correlation> -0.1 & correlation< 0.1)) %>%
  select(site_id,INDEX)

# Merge filtered correlations with the original 'long' dataset based on site_id and index
INDICES<-full_join(long,meta_INDICES) %>% 
  filter(forecast_year>1983)

# Remove temporary objects from the environment
rm(PDO,SOI,PNA, long, unlistIND,inda,ALL,ddf,cor_IND,cor_indices)

#################################################################################
#Save the key output data for later use (e.g. for explainability and visualization)

write.csv(SWE_volume, "explain/SWE_volume.csv", row.names = F)
write.csv(swann_swe, "explain/swann_swe.csv", row.names = F)
write.csv(INDICES, "explain/INDICES.csv", row.names = F)
write.csv(monthly_flow, "explain/monthly_flow.csv", row.names = F)
write.csv(NRCS_df, "explain/NRCS_df.csv", row.names = F)
write.csv(maxSWEcor, "explain/maxSWEcor.csv", row.names = F)
##################################################################################
#########################  INTERFERENCE ########################################## 

# This section sets up the model forecasting framework

# Read the submission format template, ensuring that the issue_date column is recognized as a date
submission_format <- read_csv("cross_validation_submission_format.csv", 
                              col_types = cols(issue_date = col_date(format = "%Y-%m-%d")))

# Extract unique basin identifiers and issue dates from the submission format
basins<-unique(submission_format$site_id)
issue_dates<-unique(submission_format$issue_date)
issue_year<-lubridate::year(Sys.time())

# Set up control parameters for model training using Leave-One-Out Cross-Validation (LOOCV)
fitControl <- trainControl(method = "LOOCV")

# Detect and register the number of CPU cores available for parallel processing
cores<-detectCores()
cl <- makeCluster(cores[1]-1)
registerDoParallel(cl)

# This loop performs parallel forecasting for each basin and each forecast issue date demanded in the submission template
  
PREDICTED<- foreach(i=1:length(basins),.combine='c',.packages = c("dplyr",'caret','tidyr','lubridate'))%dopar%{
    basin<-basins[i]  # Select current basin for processing
    
    ISSUES<-list()  # List to store results for each forecast issue date
    
    # Loop through each forecast issue date
    for(d in 1:length(issue_dates)){
      
      issue_date<-issue_dates[d] # Select current forecast issue date
      
      yesterday <- issue_date-3
      issue_month<-month(issue_date)
      month_current<-month(issue_date-3)
      
      year_forecast<- year(issue_date)
      year_current<-year(issue_date-3)
      
      issue<-substr(issue_date,7,10)
      
      # Filter SWE data for the current basin and forecast issue
      basin_SWE<-SWE_volume %>% 
        filter(site_id==basin & forecast_issue==issue)%>%
        group_by(site_id,snotel_id,forecast_year) %>% 
        arrange(desc(date_snotel))%>% # Sort by the date of the SWE measurement
        filter(!is.na(SWE)) %>%
        mutate(previous_date = lag(date_snotel)) %>%
        filter(!is.na(previous_date)) %>%
        slice_head(n = 1) %>% # Select the most recent SWE data (by the given forecast issue date)
        ungroup()
      
      # If there is a maximum SWE correlation station, append its data
      if('YES' %in% unique(basin_SWE$maxSWEcor)){
        maxSWE_issue<-unique(basin_SWE$maxSWEcor_issue)[1]
        maxSWE_station<-unique(basin_SWE$maxSWEcor_snotel_id)[1]
        
        maxSWE<-SWE_volume %>% 
          group_by(forecast_year) %>% 
          filter(site_id==basin & snotel_id==maxSWE_station & forecast_issue==maxSWE_issue) %>%
          arrange(desc(date_snotel)) %>% 
          filter(!is.na(SWE)) %>%
          mutate(previous_date = lag(date_snotel)) %>%
          filter(!is.na(previous_date)) %>%
          slice_head(n = 1) 
        
        maxSWE$snotel_id<-999
        basin_SWE<-rbind(basin_SWE,maxSWE)
      } 
      
      # Pivot SWE data to wide format
      basin_SWE <- basin_SWE %>%
        select(forecast_year,volume,snotel_id,SWE)%>% 
        pivot_wider(names_from = snotel_id, values_from = SWE, names_prefix = "snotel_")
      
      # --- Selecting SWANN SWE data ---
      basin_swann<-swann_swe %>%
        filter(site_id==basin & forecast_issue==issue) %>%
        select(forecast_year, SWE_swann)
      
      
      # --- Selecting precipitation data ---
      basin_PRCP<-SWE_volume %>% 
        filter(site_id==basin & forecast_issue==issue)%>%
        group_by(site_id,snotel_id,forecast_year) %>% 
        arrange(desc(date)) %>%
        filter(!is.na(precipitation_cumulative)) %>%
        mutate(previous_date = lag(date_snotel)) %>%
        filter(!is.na(previous_date)) %>%
        slice_head(n = 1) %>%
        ungroup()
      
      # Pivot precipitation data to wide format
      basin_PRCP <- basin_PRCP %>%
        select(forecast_year,snotel_id,precipitation_cumulative)%>% 
        pivot_wider(names_from = snotel_id, values_from = precipitation_cumulative, names_prefix = "prcp_")
      
      # --- Selecting climate indices ---
      basin_indices<-INDICES %>% 
        group_by(forecast_year, INDEX) %>%
        filter(forecast_month<=issue_month & site_id==basin)%>%
        arrange(desc(date_index)) %>%
        filter(!is.na(value)) %>%
        slice_head(n = 1) %>% 
        filter(!(forecast_month>3))
      
      n_indices<-length(unique(basin_indices$INDEX))
      
      # Pivot climate indices to wide format and filter out columns with many missing values
        basin_indices <- basin_indices %>%
        pivot_wider(names_from = c(INDEX,month_index), values_from = value, names_prefix = "") %>% 
        select_if(~ sum(is.na(.)) / length(.) < 0.3) %>% 
        select(forecast_year,tail(names(c(.)), n_indices))
      
      # Selecting monthly flow observations
        basin_flow<-monthly_flow %>% 
        group_by(forecast_year) %>% 
        filter(site_id==basin & forecast_month<=issue_month)%>% 
        arrange(desc(date_flow)) %>%
        filter(!is.na(flow)) %>%
        slice_head(n = 3 )  # Select the most recent 3 monthly flow records
      
        # If max flow correlation exists for the given issue date, append corresponding data
      if(nrow(basin_flow)>0){
        
        if('YES' %in% unique(basin_flow$maxFLOWcor)){
          maxFLOW_month<-unique(basin_flow$maxFLOWcor_month)
          
          maxFLOW<-monthly_flow %>% 
            group_by(forecast_year) %>% 
            filter(site_id==basin & month_flow==maxFLOW_month & maxFLOWcor=='YES')
          
          maxFLOW$month_flow<-999
          
          basin_flow<-rbind(basin_flow,maxFLOW)
        } 
        
      }
      
        # Pivot flow data to wide format
      basin_flow <- basin_flow %>%
        select(-forecast_month) %>% 
        pivot_wider(id_cols="forecast_year",names_from = c(month_flow), values_from = flow, names_prefix = "flow_") 
      
    
      # Extracting NRCS flow data for the given basin and given forecast issue date
      nrcs_flow<-NRCS_df %>% 
        group_by(forecast_year) %>% 
        filter(site_id==basin & forecast_month<=issue_month)%>% 
        arrange(desc(date_flow)) %>%
        filter(!is.na(flow)) %>%
        slice_head(n = 1 )
      
      if(nrow(nrcs_flow)>0){
        
        if('YES' %in% unique(nrcs_flow$maxFLOWcor)){
          maxFLOW_month<-unique(nrcs_flow$maxFLOWcor_month)
          
          maxFLOW<-nrcs_flow %>% 
            group_by(forecast_year) %>% 
            filter(site_id==basin & month_flow==maxFLOW_month & maxFLOWcor=='YES')
          
          maxFLOW$month_flow<-999
          
          nrcs_flow<-rbind(nrcs_flow,maxFLOW)
        } 
        
      }
      # Pivot NRCS flow data to wide format
      nrcs_flow <- nrcs_flow %>%
        rename(nrcs_month_flow=month_flow) %>% 
        select(-forecast_month) %>% 
        pivot_wider(id_cols="forecast_year",names_from = c(nrcs_month_flow), values_from = flow, names_prefix = "nrcs_flow_") 
      
      
      # Merging all data into a single DF
      DF<-Reduce(function(...) merge(..., by=c('forecast_year'), all.x=TRUE), 
                 list(basin_SWE, basin_swann, basin_PRCP,basin_flow, nrcs_flow, basin_indices))
      
      DF<-DF%>%
        filter(forecast_year>1983)
      
      VOLUME<-DF[c('forecast_year','volume')]
      
      # --- Splitting into test and train datasets ---
      
      test<-DF %>% 
        filter(forecast_year== year_forecast)%>%  # Filter rows corresponding to forecast year
        select_if(~ !any(is.na(.)))  # Select only columns without missing values in the test set
      
      # Prepare training set, removing highly missing columns and rows with missing values
      train<-DF[!DF$forecast_year %in% test$forecast_year,] # Filter out test year from the training set
      
      train<-subset(train, select=colnames(test)) 
      
      train<-train %>% 
        left_join(VOLUME)%>%
        select_if(~sum(!is.na(.)) > 0)%>%
        select_if(~ sum(is.na(.)) / length(.) < 0.5)%>% 
        na.omit()%>%
        select(-forecast_year)
      
      train<-train[colMeans(train == 0) <= 0.6]
      
      train<-train[!duplicated(as.list(train))]
      
      
      train_duplcate<-train  # Duplicate the train set for later use
      
      # --- Training the PLS model ---
      set.seed(1)
      
      
      
      PLS<-train(volume~., data=train ,method='pls',
                 trControl = fitControl,
                 tuneGrid=data.frame(ncomp=c(1,2,3,4)),
                 preProc = c("center","scale"))
      
      
      train_duplcate$simPLS<-predict(PLS,train_duplcate) # Predict on the training set
      
      
      
      test$simPLS<-predict(PLS,test)  # Predict on the test set
      
      
      RMSE<-min(PLS[["results"]][["RMSE"]])  # Extract RMSE of the best model
      test$volume_50<-predict(PLS,test) # Predict the 50th percentile (median) volume
      test$volume_10<-test$volume_50 - 1.2816*RMSE # Calculate 10th percentile volume
      test$volume_90<-test$volume_50+1.2816*RMSE # Calculate 90th percentile volume
      
      
      
      # Store results for the current issue date
      issue_results<-test[c('forecast_year','volume_10','volume_50','volume_90','simPLS')]
      issue_results$forecast_issue<-issue
      issue_results$site_id<-basin
      issue_results$RMSE<-RMSE
      issue_results$rmsePLS<-min(PLS[["results"]][["RMSE"]])/mean(train_duplcate$volume)
      issue_results$predictors<-paste(colnames(train),collapse=" ")
      issue_results$obs_train<-nrow(train)
      
      ISSUES[[d]]<-issue_results  # Append the results to the list for this basin and issue date
      
    }    
    ISSUES
    
    
  }


# Combine all the predicted results from the parallel processing into one dataframe
RESULTS<-bind_rows(PREDICTED)

# Join the RESULTS with metadata containing the lowest and highest volume values for each basin
RESULTS<-RESULTS %>% 
  left_join(meta_lowest_highest_volume)

# Adjust the predicted volumes to ensure no negative values and maintain logical consistency
RESULTS<-RESULTS %>%
  mutate(volume_10=if_else(volume_10<0,min,volume_10)) %>%
  mutate(volume_50=if_else(volume_50<min,min,volume_50))%>%
  mutate(volume_90=if_else(volume_90<volume_50,volume_50+1.2816*RMSE,volume_90))

# Extract the forecast issue month and day from the forecast_issue field
RESULTS$month<-as.numeric(substr(RESULTS$forecast_issue,1,1))
RESULTS$day<-as.numeric(substr(RESULTS$forecast_issue,3,4))

# Create the submission dataframe with required format, including issue date and predicted volumes
submission <- RESULTS %>%
  dplyr::mutate(issue_date = make_date(forecast_year, month, day)) %>% 
  dplyr::select(site_id,issue_date,volume_10, volume_50, volume_90) 

# Join the submission format with the generated predictions
final<-as.data.frame(plyr:: join(submission_format[,-c(3:5)], submission))

write.csv(final, "submission_cross_validation.csv", row.names=FALSE)

# # Ensure that the final dataset contains both the original submission format and the new predictions
# final<-submission_format %>% 
#   left_join(submission)

