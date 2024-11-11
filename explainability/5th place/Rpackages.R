####### Installing necessary R packages for data preprocessing and inference ##############

packages <- c( "readr", "dplyr", "tidyr", "lubridate",'data.table',"terra", "reshape2",
               'scales','doParallel','caret','pls', 'vip','ggplot2','knitr', 'pdp',
               "gridExtra"
             )

installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages], repos = "http://cran.us.r-project.org")
}
