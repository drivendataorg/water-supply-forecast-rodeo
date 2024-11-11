# Load necessary packages

packages <- c( "readr", "dplyr", "tidyr", "lubridate",'data.table',
               'scales','doParallel', 'purrr')

installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}

invisible(lapply(packages, library, character.only = TRUE))


#===Uploading and preprocessing metadata of basin specific SNOTEL stations

sites_to_snotel_stations <- read_csv("data/snotel/sites_to_snotel_stations.csv")

snotel_all<-as.data.frame(unique(sites_to_snotel_stations$stationTriplet))
colnames(snotel_all)<-colnames(sites_to_snotel_stations)[2]

snotel_all<-as.data.frame(unique(sites_to_snotel_stations$stationTriplet))
colnames(snotel_all)<-colnames(sites_to_snotel_stations)[2]

station_metadata <- read_csv("data/snotel/station_metadata.csv")
station_metadata$huc<-substr(station_metadata$huc,1,6)


snotel_all<-snotel_all %>% 
  left_join(station_metadata) %>% 
  filter(beginDate<'1985-12-20') %>% 
  filter(endDate>'2023-10-01')

snotel_all$snotel_id<- sub("^([0-9]{3,4}).*", "\\1", snotel_all$stationTriplet) %>% 
  as.numeric()

snotel_all<-snotel_all %>% 
  left_join(sites_to_snotel_stations) %>% 
  select(stationTriplet, snotel_id, elevation, latitude, longitude, site_id, in_basin, huc)


snotel_all$huc<-substr(snotel_all$huc,1,6)

#===========Downloading data for all SNOTEL stations=======================
# Defining the main directory containing SNOTEL  subfolders
main_dir <- 'data/snotel'  # Replace with the actual path

# Function to read and process each CSV file
read_and_label_csv <- function(file_path) {
  # Extract the station ID from the file name
  station_id <- tools::file_path_sans_ext(basename(file_path))
  
  # Read the CSV file
  data <- read_csv(file_path, col_types = cols(.default = "c"))  # Read all columns as characters
  
  # Add the station ID as a new column
  data <- data %>%
    mutate(station_id = station_id)
  
  return(data)
}

# List all CSV files in the SNOTEL subfolders
csv_files <- list.files(main_dir, pattern = "\\.csv$", full.names = TRUE, recursive = TRUE)

# Read and combine all CSV files into a single data frame

system.time(
  all_data <- map_dfr(csv_files, read_and_label_csv)
)





# Preprocessing the SNOTEL data

all_data$snotel_id<- sub("^([0-9]{3,4}).*", "\\1", all_data$station_id) %>% 
  as.numeric()

snotel_df<-all_data%>%
  mutate(date=as.Date(date)) %>% 
  filter(date>'1984-12-01')%>%
  select(snotel_id, date,WTEQ_DAILY,PREC_DAILY) %>%
  mutate(WTEQ_DAILY=as.numeric(WTEQ_DAILY),
         PREC_DAILY=as.numeric(PREC_DAILY)) %>% 
  rename(SWE=WTEQ_DAILY,
         precipitation_cumulative=PREC_DAILY) %>%
  mutate(date_snotel=date)

head(snotel_df)

# Water Year function
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

snotel_df$forecast_year<-get_water_year(snotel_df$date, start_month = 10)  

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
  
  date_forecasts<-date_forecasts %>% 
    mutate(forecast_month=as.numeric(substr(forecast_issue,1,1))) 
  
}


snotel_df<-snotel_df %>% 
  left_join(date_forecasts,by='date') %>% 
  mutate(year=lubridate::year(date_snotel))%>% 
  group_by(snotel_id,forecast_year) %>% 
  filter(!(lubridate::month(date)>7 & lubridate::month(date)<12)) %>% 
  mutate(forecast_issue=if_else(lubridate::month(date_snotel)==12,'1-01',forecast_issue)) %>% 
  mutate(forecast_month=if_else(lubridate::month(date_snotel)==12,1,forecast_month)) %>% 
  arrange(date) %>% 
  fill(forecast_issue, .direction = "updown") %>% 
  fill(forecast_month, .direction = "updown") %>%
  select(-year)

# Determining stations where SWE and PRCP exhibit highest correlation with seasonal streamflow

historical_labels <- read_csv("prior_historical_labels.csv") %>% 
  rename(forecast_year=year)
cv_labels <- read_csv("cross_validation_labels.csv") %>% 
  rename(forecast_year=year)
train_volume<-rbind(historical_labels,cv_labels)

cor_PRCP<-list()

for (i in unique(train_volume$site_id)) {
  volume_df<-train_volume[train_volume$site_id==i,]
  
  ddf<-snotel_all %>% 
    filter(site_id==i) %>% 
    select(site_id,snotel_id,in_basin)%>%
    left_join(volume_df)%>% 
    left_join(snotel_df) %>%
    filter(!is.na(date))
  
   correlations<-ddf %>% 
    filter(!is.na(forecast_issue)) %>% 
    group_by(snotel_id, forecast_issue, forecast_year) %>% 
    arrange(desc(date_snotel))%>%
    slice(1:1)  %>%
    group_by(snotel_id, forecast_issue) %>% 
    summarise( obs_SWE=length(precipitation_cumulative),
               obs_volume=length(volume),
               correlation = cor(precipitation_cumulative, volume, use="pairwise.complete.obs"),
               forecast_month=forecast_month) %>% 
    drop_na() %>% 
    distinct()
  
  cor_max_mean<-correlations %>%
    filter(obs_SWE>20) %>% 
    filter(forecast_month!=7) %>% 
    group_by(snotel_id) %>% 
    arrange(desc(correlation)) %>% 
    # arrange(missing) %>%
    summarise( obs=min(obs_SWE),
               max_cor=max(correlation, na.rm=T),
               min_cor=min(correlation,na.rm=T),
               mean_cor=median(correlation,na.rm=T)) %>% 
    mutate(season='middle') %>% 
    arrange(desc(mean_cor))%>% 
    arrange(desc(obs))%>% 
    slice(1:4) %>% 
    filter(mean_cor>0.3) %>% 
    drop_na()
  
  cor_max_min<-correlations %>%
    filter(forecast_issue=='1-01'| forecast_issue=='1-08' | forecast_issue=='1-15'|
             forecast_issue=='1-22'| forecast_issue=='2-01'|
             forecast_issue=='2-08') %>% 
    group_by(snotel_id) %>% 
    arrange(desc(correlation)) %>% 
    # arrange(missing) %>%
    summarise( obs=min(obs_SWE),
               max_cor=max(correlation, na.rm=T),
               min_cor=min(correlation,na.rm=T),
               mean_cor=median(correlation,na.rm=T)) %>% 
    mutate(season='start') %>% 
    # ungroup() %>% 
    # group_by(snotel_id) %>% 
    arrange(desc(mean_cor))%>% 
    arrange(desc(obs))%>% 
    slice(1:2) %>% 
    filter(mean_cor>0.3) %>% 
    drop_na()
  
  cor_summary<- rbind(cor_max_mean,cor_max_min) %>% 
    group_by(snotel_id) %>% 
    slice(1:1) %>% 
    mutate(site_id=i,
           source='PRCP')
  
  
  cor_PRCP[[i]]<-cor_summary
  
}


meta_selected_PRCP<-bind_rows(cor_PRCP)


cor_SWE<-list()

for (i in unique(train_volume$site_id)) {
  volume_df<-train_volume[train_volume$site_id==i,]
  
  ddf<-snotel_all %>% 
    filter(site_id==i) %>% 
    select(site_id,snotel_id,in_basin)%>%
    left_join(volume_df)%>%
    left_join(snotel_df) %>%
    filter(!is.na(date))
  
  
  correlations<-ddf %>% 
    filter(!is.na(forecast_issue)) %>% 
    group_by(snotel_id, forecast_issue, forecast_year) %>% 
    arrange(desc(date_snotel))%>%
    slice(1:1)  %>%
    group_by(snotel_id, forecast_issue) %>% 
    summarise( obs_SWE=length(SWE),
               obs_volume=length(volume),
               correlation = cor(SWE, volume, use="pairwise.complete.obs"),
               forecast_month=forecast_month) %>% 
    drop_na() %>% 
    distinct()
  
  cor_max_mean<-correlations %>%
    filter(obs_SWE>20) %>% 
    filter(forecast_month!=7) %>% 
    group_by(snotel_id) %>% 
    arrange(desc(correlation)) %>% 
    summarise( obs=min(obs_SWE),
               max_cor=max(correlation, na.rm=T),
               min_cor=min(correlation,na.rm=T),
               mean_cor=median(correlation,na.rm=T)) %>% 
    mutate(season='middle') %>% 
    arrange(desc(mean_cor))%>% 
    arrange(desc(obs))%>% 
    slice(1:3) %>% 
    filter(mean_cor>0.3) %>% 
    drop_na()
  
  cor_max_min<-correlations %>%
    filter(forecast_issue=='1-01'| forecast_issue=='1-08' | forecast_issue=='1-15'|
             forecast_issue=='1-22'| forecast_issue=='2-01'|
             forecast_issue=='2-08') %>% 
    group_by(snotel_id) %>% 
    arrange(desc(correlation)) %>% 
    summarise( obs=min(obs_SWE),
               max_cor=max(correlation, na.rm=T),
               min_cor=min(correlation,na.rm=T),
               mean_cor=median(correlation,na.rm=T)) %>% 
    mutate(season='start') %>% 
    arrange(desc(mean_cor))%>% 
    arrange(desc(obs))%>% 
    slice(1:2) %>% 
    filter(mean_cor>0.3) %>% 
    drop_na()
  
  cor_summary<- rbind(cor_max_mean,cor_max_min) %>% 
    group_by(snotel_id) %>% 
    slice(1:1) %>% 
    mutate(site_id=i,
           source="SWE")
  
  
  cor_SWE[[i]]<-cor_summary
  
}

meta_selected_SWE<-bind_rows(cor_SWE)

meta_selected_SNOTEL<-rbind(meta_selected_PRCP,meta_selected_SWE)%>% 
  group_by(site_id,snotel_id) %>% 
  arrange(desc(mean_cor)) %>% 
  slice(1:1)

meta_selected_SNOTEL<-meta_selected_SNOTEL%>% 
  group_by(site_id) %>% 
  arrange(desc(mean_cor)) %>% 
  slice(1:4)


table(meta_selected_SWE$obs) 

# Some basins left with fewer stations, this loop complement those with stations from donor basins
meta_insufficient_SNOTEL<-meta_selected_SNOTEL %>% 
  group_by(site_id) %>% 
  mutate(number=length(snotel_id)) %>% 
  filter(number<3)  

meta_insufficient_SNOTEL

cor_PRCPins<-list()
for (i in unique(meta_insufficient_SNOTEL$site_id)) {
  volume_df<-train_volume[train_volume$site_id==i,]
  
  huc_selected<-snotel_all %>% 
    filter(site_id==i) %>% 
    select(huc) %>% 
    distinct() %>% 
    as.character()
  
  ddf<-snotel_all %>% 
    filter(huc==huc_selected) %>% 
    mutate(site_id=i) %>%  
    distinct() %>% 
    select(site_id,snotel_id,huc)%>%
    left_join(volume_df)%>% 
    left_join(snotel_df,by=c('snotel_id','forecast_year')) %>% 
    filter(!is.na(date)) %>% 
    distinct()
 
  
  correlations<-ddf %>% 
    filter(!is.na(forecast_issue)) %>% 
    group_by(snotel_id, forecast_issue, forecast_year) %>% 
    arrange(desc(date_snotel))%>%
    slice(1:2)  %>%
    group_by(snotel_id, forecast_issue) %>% 
    summarise( obs_SWE=length(precipitation_cumulative),
               obs_volume=length(volume),
               correlation = cor(precipitation_cumulative, volume, use="pairwise.complete.obs"),
               forecast_month=forecast_month) %>% 
    drop_na() %>% 
    distinct()
  
  cor_max_mean<-correlations %>%
    filter(obs_SWE>20) %>% 
    filter(forecast_month!=7) %>% 
    group_by(snotel_id) %>% 
    arrange(desc(correlation)) %>% 
    summarise( obs=min(obs_SWE),
               max_cor=max(correlation, na.rm=T),
               min_cor=min(correlation,na.rm=T),
               mean_cor=median(correlation,na.rm=T)) %>% 
    mutate(season='middle') %>% 
    arrange(desc(mean_cor))%>% 
    arrange(desc(obs))%>% 
    slice(1:1) %>% 
    filter(mean_cor>0.3) %>% 
    drop_na()
  
 cor_summary<- cor_max_mean %>% 
    group_by(snotel_id) %>% 
    slice(1:1) %>% 
    mutate(site_id=i,
           source='PRCP')
  
  
  cor_PRCPins[[i]]<-cor_summary
  
}

meta_additional_SNOTEL<-bind_rows(cor_PRCPins)


#Merging metadata of selected snow stations per basin
meta_selected_SNOTEL<-bind_rows(meta_selected_SNOTEL,meta_additional_SNOTEL)%>% 
  select(site_id, snotel_id) %>% 
  # distinct(site_id, snotel_id) %>%
  left_join(snotel_all %>% select(-site_id)) %>% 
  distinct(site_id,snotel_id,stationTriplet,elevation,
           latitude,longitude,huc) %>% 
  left_join(snotel_all)



write.csv(meta_selected_SNOTEL, 'data/selected_snotel_meta.csv', row.names = F)
