
####### Installing necessary R packages for data reprocessing and interference ##############

packages <- c( "readr", "dplyr", "tidyr", "lubridate",'data.table',"terra", "reshape2",
               'scales','doParallel','caret','vip','ggplot2','knitr','grid','gridExtra')

#installed_packages <- packages %in% rownames(installed.packages())
#if (any(installed_packages == FALSE)) {
#  install.packages(packages[!installed_packages])
#}

invisible(lapply(packages, library, character.only = TRUE))

#rm(installed_packages, packages)


##### Loading preprocessed input data for the interference ######################

setwd(".")

SWE_volume<-read.csv('explain/SWE_volume.csv')
swann_swe<-read.csv('explain/swann_swe.csv')
monthly_flow<-read.csv('explain/monthly_flow.csv')
NRCS_df<-read.csv('explain/NRCS_df.csv')
INDICES<-read.csv('explain/INDICES.csv')

maxSWEcor<-read.csv('explain/maxSWEcor.csv')

##################################################################################
#########################  INTERFERENCE ########################################## 

# This section sets up the model forecasting framework, and simultaneously 

# Read the submission format template
submission_format <- read_csv("submission_format_explanability.csv", 
                              col_types = cols(issue_date = col_date(format = "%Y-%m-%d")))

# Extract unique basin identifiers and issue dates from the submission format
basins<-unique(submission_format$site_id)
issue_dates<-unique(submission_format$issue_date)
issue_year<-lubridate::year(Sys.time())

representative_snotel<-SWE_volume %>% 
  group_by(site_id) %>% 
  summarise(snotel=unique(maxSWEcor_snotel_id)) %>% 
  na.omit()

comb <- function(x, ...) {
  lapply(seq_along(x), function(i) c(x[[i]], lapply(list(...), function(y) y[[i]])))
}

# Set up control parameters for model training using Leave-One-Out Cross-Validation (LOOCV)
fitControl <- trainControl(method = "LOOCV")

# Detect and register the number of CPU cores available for parallel processing
cores<-detectCores()
cl <- makeCluster(cores[1]-1)
registerDoParallel(cl)

# This loop performs parallel forecasting for each basin and each forecast issue date demanded in the submission template

PREDICTED<- foreach(i=1:length(basins),.combine='c',.packages = c("dplyr",'caret','tidyr','lubridate','vip','ggplot2',"gridExtra", "grid"))%dopar%{
  basin<-basins[i]  # Select current basin for processing
  
  ISSUES<-list()  # List to store results for each forecast issue date
  PLOTS_S <- list() #List to store visuals explaining each forecast
  
  
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
    
    
    ######## Explainability ########
    
    # Calculate the historic median flow for 1990-2022
    issue_results$historic_mean<-DF %>% 
      filter(forecast_year>1989 & forecast_year<2023) %>% 
      summarise(mean_flow=median(volume)) %>% 
      as.numeric()
    
    # Retrieve previous Water Year (WY) volume
    issue_results$previous_WY<-DF %>% 
      filter(forecast_year==year_forecast-1) %>%
      select(volume) %>% 
      as.numeric()
    
    # Identify the main SNOTEL station for SWE correlation
    snotel_main<-maxSWEcor %>% 
      filter(site_id==basin) %>% 
      select(snotel_id) %>% 
      as.numeric()
    
    # Filter SWE data for the current basin and SNOTEL station
    status_swe<-SWE_volume %>% 
      filter(forecast_year>1989) %>% 
      filter(site_id==basin)%>% 
      filter(snotel_id==snotel_main) %>% 
      # distinct() %>% 
      arrange(date)
    
    # Calculate historic SWE and precipitation statistics (1990-2022)
    historical_wy<-status_swe %>%
      filter(forecast_year<2023) %>% 
      group_by(month = month(date), day = day(date)) %>%
      summarize(SWE_mean = median(SWE,use='na.or.complete'), SWE_max = max(SWE),SWE_min = min(SWE), 
                SWE_L=quantile(SWE, 0.25),SWE_H=quantile(SWE, 0.75),
                PRCP_mean=median(precipitation_cumulative),
                PRCP_max=max(precipitation_cumulative),
                PRCP_min=min(precipitation_cumulative),
                PRCP_L=quantile(precipitation_cumulative, 0.25),
                PRCP_H=quantile(precipitation_cumulative, 0.75),) %>% 
      filter(!(month>7 & month<10)) %>% 
      filter(!(month==2 & day==29))%>% 
      mutate(SWE_mean=if_else(SWE_mean==0,NA,SWE_mean))
    
    # Current Water Year (WY) SWE and precipitation values up to the forecast issue date
    current_wy<-status_swe %>% 
      filter(forecast_year==year_forecast) %>% 
      filter(date<issue_date) %>% 
      group_by(month = month(date), day = day(date)) %>% 
      select(month,day,SWE,precipitation_cumulative)%>% 
      rename(SWE_current=SWE,
             PRCP_current=precipitation_cumulative) %>% 
      mutate(SWE_current=if_else(SWE_current==0,NA,SWE_current))
    
    # Previous Water Year (WY) SWE and precipitation values
    last_wy<-status_swe %>% 
      group_by(forecast_year) %>% 
      filter(forecast_year==year_forecast-1) %>%  
      filter(forecast_year==max(unique(forecast_year))) %>% 
      group_by(month = month(date), day = day(date)) %>% 
      select(forecast_year,month,day,SWE,precipitation_cumulative)%>% 
      mutate(SWE_lastwy=if_else(SWE==0,NA,SWE),
             PRCP_lastwy=if_else(precipitation_cumulative==0,NA,precipitation_cumulative),
             lastwy=year_forecast) %>% 
      select(-c(forecast_year,SWE,precipitation_cumulative))
    
    # Merge historic, current, and last year SWE and precipitation data, and arrange by day of water year
    historical_wy<-historical_wy %>% 
      left_join(current_wy) %>% 
      left_join(last_wy) %>% 
      mutate(
        dowy = case_when(
          month >= 10 ~ (month - 10) * 30 + day,  # For October, November, December
          month < 10 ~ (month + 2) * 30 + day     # For January to September
        )
      ) %>% 
      arrange(dowy)
    
    # Rearrange data to align with water year day (dowy)
    # historical_wy<-rbind(historical_wy[c(213:304),],historical_wy[c(1:212),])
    # historical_wy$dowy<-seq(1,304,1)
    
    
    
    # Generate a plot of SWE data with historic range and current year comparison
    plot_snowpack<-ggplot(historical_wy, aes(dowy, SWE_mean)) +
      geom_ribbon(aes(ymin = SWE_min, ymax = SWE_max, fill = "historic max-min"), alpha = 0.75) +
      geom_ribbon(aes(ymin = SWE_L, ymax = SWE_H, fill = "IQR"), alpha = 0.5) +
      geom_smooth(aes(color = "1990-2022 median"), method = "loess", span = 0.2, linetype = "dashed") +
      geom_line(aes(y = SWE_lastwy, color = "previous WY"),lwd=1) +
      geom_line(aes(y = SWE_current, color = "current WY"),lwd=1) +
      labs(x = "Day of Water Year", y = "(in)") +
      scale_color_manual(values = c("previous WY" = "orange", "current WY" = "red", 
                                    "1990-2022 median" = "gray30"),
                         name = NULL) +
      scale_fill_manual(values = c("historic max-min" = "grey90", "historic IQR" = "grey70"),
                        name = NULL) +
      guides(color = guide_legend(override.aes = list(fill = NA)),
             linetype = guide_legend(override.aes = list(fill = NA))) +
      theme_minimal() +
      labs(title="Figure 1. Basin snowpack (SWE)",
           subtitle = paste(basin,',','issue date:',issue_date,sep=' '))+
      theme(
        legend.position = "none",
        legend.key = element_rect(fill = "white", color = 'white'),
        plot.title = element_text(size = 15),  
        plot.subtitle = element_text(size = 12), 
        axis.title = element_text(size = 14), 
        axis.text = element_text(size = 13))
    
    # Generate a plot of precipitation data with historic range and current year comparison
    plot_PRCP <- ggplot(historical_wy, aes(dowy, PRCP_mean)) +
      geom_ribbon(aes(ymin = PRCP_min, ymax = PRCP_max, fill = "historic max-min"), alpha = 0.75) +
      geom_ribbon(aes(ymin = PRCP_L, ymax = PRCP_H, fill = "historic IQR"), alpha = 1) +
      geom_smooth(aes(color = "1990-2022 median"), method = "loess", span = 0.2, linetype = "dashed") +
      geom_line(aes(y = PRCP_lastwy, color = "previous WY"), lwd = 1) +
      geom_line(aes(y = PRCP_current, color = "current WY"), lwd = 1) +
      labs(x = "Day of Water Year", y = "(in)") +
      scale_color_manual(values = c("previous WY" = "orange", "current WY" = "red", 
                                    "1990-2022 median" = "gray30"),
                         name = NULL) +
      scale_fill_manual(values = c("historic max-min" = "grey90", "historic IQR" = "grey70"),
                        name = NULL) +
      guides(color = guide_legend(override.aes = list(fill = NA)),
             linetype = guide_legend(override.aes = list(fill = NA))) +
      theme_minimal() +
      labs(title = "Figure 2. Accumulated precipitation",
           subtitle = paste(basin, ',', 'issue date:', issue_date, sep = ' ')) +
      theme(
        legend.position = "right",
        legend.key.size = unit(0.7, "lines"),  
        legend.text = element_text(size = 12),  
        legend.title = element_text(size = 10),  
        plot.title = element_text(size = 15),
        plot.subtitle = element_text(size = 12), 
        axis.title = element_text(size = 14), 
        axis.text = element_text(size = 13),
        plot.margin = margin(t = 10, r = 10, b = 10, l = 10)  
      )
    
    
    # Calculate SWE and precipitation normals
    issue_results$SWE_normals<-historical_wy %>% 
      summarise(SWE_normals=(SWE_current)/SWE_mean) %>%
      filter(month<=month(issue_date)) %>%
      na.omit() %>% 
      ungroup %>% 
      select(!month) %>% 
      slice_tail() %>% 
      as.numeric() %>% 
      round(2)
    
    issue_results$PRCP_normals<-historical_wy %>% 
      summarise(PRCP_normals=(PRCP_current)/PRCP_mean) %>%
      filter(month<=month(issue_date)) %>%
      na.omit() %>% 
      ungroup %>% 
      select(!month) %>% 
      slice_tail() %>% 
      as.numeric() %>% 
      round(2)
    
    # Gather flow data for antecedent analysis
    status_flow<-monthly_flow %>% 
      filter(forecast_year>1989) %>% 
      group_by(forecast_year) %>% 
      filter(site_id==basin )
    
    # Add current year flow
    FLOW<-status_flow %>%
      filter(forecast_year<2023) %>% 
      select(year, month_flow,flow) 
    
    current_flow<-status_flow %>% 
      filter(forecast_year==year_forecast) %>% 
      filter(month_flow<month(issue_date))%>% 
      select(month_flow,flow)%>% 
      rename(flow_current=flow) 
    
    FLOW<-FLOW %>% 
      full_join(current_flow)
    
    # Plot antecedent flow vs historic monthly flow boxplots
    plot_antecedent<-ggplot(FLOW, aes(x = factor(month_flow), y = flow)) +
      geom_boxplot() +
      labs(x = "", y = "(KAF)") +
      scale_x_discrete(labels = c("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")) +
      geom_point(data = current_flow, aes(x = factor(month_flow), y = flow_current), shape=19, color = "red", size=3)+
      ggtitle("Figure 3. Antecedent flow (red) vs\n historic monthly flow (boxplots)",
              subtitle = paste(basin,',','issue date:',issue_date,sep=' ')) +
      theme_minimal()+
      theme(
        # legend.position = "right",
        # legend.key = element_rect(fill = "white", color = 'white'),
        plot.title = element_text(size = 15),  
        plot.subtitle = element_text(size = 12), 
        axis.title = element_text(size = 14), 
        axis.text = element_text(size = 13))
    
    # Calculate flow normals
    last_month<-current_flow %>% 
      slice_tail() %>% 
      ungroup() %>% 
      select(month_flow) %>% 
      as.numeric()
    
    last_month_flow<-current_flow %>% 
      slice_tail() %>% 
      ungroup() %>% 
      select(flow_current) %>% 
      as.numeric()
    
    flows<-FLOW %>% 
      filter(month_flow==last_month) %>% 
      ungroup() 
    
    issue_results$FLOW_normals<-last_month_flow/median(flows$flow, na.rm = T)
    
    
    # Run variance-based feature importance analysis
    imp<-vip(PLS, method='firm',data=train)
    
    imp<-imp[["data"]] %>% 
      mutate(predictors='none')
    
    imp$predictors <- ifelse(grepl("flow", imp$Variable), "antecedent flow", 
                             ifelse(grepl("snotel|SWE", imp$Variable), "SWE", 
                                    ifelse(grepl("prcp", imp$Variable), "accumulated precipitation", 
                                           ifelse(grepl("SOI|PNA|PDO", imp$Variable), "climate indices", "none"))))
    
    # Aggregate and normalize importance scores
    imp <- imp %>%
      group_by(predictors) %>%
      summarise(Importance = sum(Importance)) %>% 
      mutate(Importance=Importance/sum(imp$Importance))%>%
      arrange(desc(Importance)) %>% 
      mutate(predictors=gsub(" ", "\n",predictors))
    
    # Plot predictor importance
    
    class_colors <- c("accumulated\nprecipitation" = "#a6cee3", "SWE" = "#1f78b4", 
                      "antecedent\nflow" = "#b2df8a", "climate\nindices" = "#33a02c")
    
    
    plot_importance<-ggplot(imp, aes(x = reorder(predictors, -Importance), y = Importance, fill = predictors)) +
      geom_bar(stat = "identity",show.legend = FALSE) +
      scale_fill_manual(values = class_colors) +  # Use custom colors
      labs(x = "", y = "") +
      ggtitle("Figure 4. Relative importance of model predictors",
              subtitle = paste(basin,',','issue date:',issue_date,sep=' ')) +
      theme_minimal() +
      theme(
        plot.title = element_text(size = 15),  
        plot.subtitle = element_text(size = 12), 
        axis.title = element_text(size = 14), 
        axis.text = element_text(size = 13))
    
    # Filter results to the desired columns
    results_table <- issue_results %>%
      select(site_id, volume_10, volume_50, volume_90, historic_mean, previous_WY)
    
    # Define the list of plots
    plots <- list(
      snowpack = plot_snowpack,
      precipitation = plot_PRCP,
      antecedent_flow = plot_antecedent,
      importance = plot_importance
    )
    
    # Writing each forecast as a pdf file 
    pdf_filename <- paste0("reports/", basin, "_", issue_date, ".pdf")
    
    
    pdf(pdf_filename, width = 11, height = 9.5)  
    
    #  Title 
    title_grob <- textGrob(
      paste("Seasonal water supply forecast for", basin, "\nIssue date:", issue_date),
      gp = gpar(fontsize = 16, fontface = "bold"),
      just = "top"
    )
    
    # Converting the results table to a grob
    table_grob <- tableGrob(results_table)
    
    # Arranging everything on one page
    grid.arrange(
      title_grob,
      arrangeGrob(
        plots$snowpack, plots$precipitation,
        plots$antecedent_flow, plots$importance,
        ncol = 2, nrow = 2
      ),
      table_grob,
      nrow = 3,
      heights = c(0.2, 0.6, 0.2)  
    )
    
    
    dev.off()
    
    
  }    
  
  ISSUES
  
}

