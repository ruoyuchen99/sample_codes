rm(list=ls())
setwd("/Users/chenruoyu/Desktop/Applied Quant Research Design/PLSC_final_project_Ruoyu_Chen/raw_data")

library(tidyverse)
library(ggplot2)
library(haven)
library(labelled)
library(expss)
library(Jmisc)


# 0. Importing and data, normalization of monetary values by exchange rate and
# GDP deflator, defining a function for variable transformation/creation
# procedure to be applied later to multiple versions of the data

DRC1 <- read_dta("DRC_2006_2010_panel.dta")
DRC2 <- read_dta("DRC_2010_2013_Panel.dta")
Ghana <- read_dta("Ghana_2007_2013_Panel.dta")
Kenya <- read_dta("Kenya_2007_2013_Panel.dta")
Malawi <- read_dta("Malawi_2009_2014.dta")
Senegal <- read_dta("Senegal_2007_2014_Panel.dta")
Tanzania <- read_dta("Tanzania_2006_2013_Panel.dta")
Uganda <- read_dta("Uganda_2006_2013_Panel.dta")
Zambia <- read_dta("Zambia_2007_2013_Panel.dta")

DRC2 <- DRC2 %>%rename(panel_id=idPANEL2010)
Ghana <- Ghana %>%rename(panel_id=idPANEL2007)
Kenya <- Kenya %>%rename(panel_id=idPANEL2007)
Malawi <- Malawi %>%rename(panel_id=id2009)
Senegal <- Senegal %>%rename(panel_id=idPANEL2007)
Tanzania <- Tanzania %>%rename(panel_id=idPANEL2006)
Uganda <- Uganda %>%rename(panel_id=idPANEL2006)
Zambia <- Zambia %>%rename(panel_id=idPANEL2007)

# importing and pre-processing data for exchange rate and GDP deflator
ex_deflator <- read_csv("ex_gdp.csv")
ex_deflator <- ex_deflator[complete.cases(ex_deflator),]

ex_deflator <- ex_deflator %>% 
  rename_with(~sub(" .*", "", .), starts_with('2')) %>% 
  mutate(country_code = gl(8,2))

ex_rate <- ex_deflator %>% 
  filter(`Series Name`=='Official exchange rate (LCU per US$, period average)')

deflator <- ex_deflator %>% 
  filter(`Series Name`=='GDP deflator (base year varies by country)')



# Defining the function for variable transformation/creation
clean_data <- function(df_recreate){
  
  df_recreate <- df_recreate[df_recreate$a6b!=0,] # cut out micro firms
  df_recreate <- df_recreate %>% 
    mutate(a6b = a6b-1) # subtract one so that dummy variable's value starts from 0
  
  df_recreate$l10<-replace(df_recreate$l10, df_recreate$l10==2, 0) # change dummy so no=0
  
  # convert variables to numerical
  df_recreate$d2 <-type.convert(df_recreate$d2)
  df_recreate$l1 <-type.convert(df_recreate$l1)
  df_recreate$n2a <-type.convert(df_recreate$n2a)
  
  # apply exchange rate and GDP deflator by country, year to monetary values
  for (x in unique(df_recreate$country)){
    country <- subset(df_recreate, df_recreate$country==x)
    
    for (y in unique(country$year)){
      year_str = as.character(y)
      ex_rate = ex_rate$year_str[which(ex_rate$country_code==x)]
      
      deflator_base = deflator$`2000`[which(deflator$country_code==x)]
      deflator_calc = deflator_base/deflator$year_str[which(deflator$country_code==x)]
      
      sales <- df_recreate$d2[which(df_recreate$country==x & df_recreate$year==y)]  
      costs <- df_recreate$n2a[which(df_recreate$country==x & df_recreate$year==y)] 
      capital <- df_recreate$n5a[which(df_recreate$country==x & df_recreate$year==y)] 
      
      df_recreate$d2[which(df_recreate$country==x & df_recreate$year==y)] <- sales/ex_rate*deflator_calc
      df_recreate$n2a[which(df_recreate$country==x & df_recreate$year==y)] <- costs/ex_rate*deflator_calc
      df_recreate$n5a[which(df_recreate$country==x & df_recreate$year==y)] <- capital/ex_rate*deflator_calc

    }
    
  }
  
  
  df_recreate <- df_recreate %>% 
    mutate(foreign=ifelse(b2b>=10, 1, 0)) %>% # define as foreign firm if foreign ownership > 10%
    mutate(export = ifelse(d3c>0, 1, 0)) %>% # define as exporting firm if % sales directly exported > 0
    mutate(age = year-b5) %>% # firm age: year of survey - year starting operation
    mutate(sector = as.integer(d1a2/100)) %>% # sector code: change from 4-digit to 2-digit
    mutate(LP = log(d2/l1)) %>%  # calculate log labor productivity
    mutate(capital_pw = log(n5a/l1)) %>% # calculate log capital per worker
    mutate(markup = (d2-n2a)/d2) %>% # calculate firm leval markup
    
    # initialize the following variables to be calculated:
    mutate(demo=0) %>% 
    mutate(tech_gap=0) %>% 
    mutate(tech_gap_alt=0) %>% 
    mutate(l_mobility_alt=0) %>% 
    mutate(comp_alt=0) %>% 
    mutate(LP_gap =0 ) %>% 
    
    # rename the following variables:
    rename(absorptive_cap = l10) %>% 
    rename(size = a6b) %>% 
    rename(num_employee = l1) 
    
  

  # calculate FDI demonstration effect, technology gap, foreign labor share 
  # and foreign mark-up share by country, sector, year
  for (y in unique(df_recreate$country)){
    in_country <- subset(df_recreate, country==y, select=c(year, sector, num_employee, markup, foreign, d2, LP))
    
    for (x in unique(in_country$sector)){
      in_sector <- subset(in_country, sector==x)
      
      for (t in unique(df_recreate$year)){
        in_year <- subset(in_sector, year==t)
        
        foreign <- subset(in_year, foreign==1, select=c(d2, LP, num_employee, markup))
        domestic <- subset(in_year, foreign==0, select=c(d2, LP, num_employee, markup))
        
        foreign_sales <- sum(foreign$d2)
        total_sales <- sum(in_year$d2)
        demo_effect <- foreign_sales/total_sales
        df_recreate$demo[which(df_recreate$sector==x & df_recreate$country==y & df_recreate$year==t)]<-demo_effect
        
        foreign_LP <- mean(foreign$LP)
        domestic_LP<- mean(domestic$LP)
        sector_LP <- mean(in_year$LP)
        TG <- foreign_LP/domestic_LP
        df_recreate$tech_gap[which(df_recreate$sector==x & df_recreate$country==y & df_recreate$year==t)]<-TG
        df_recreate$tech_gap_alt[which(df_recreate$sector==x & df_recreate$country==y & df_recreate$year==t)]<-sector_LP
        
        
        foreign_labor <- sum(foreign$num_employee)
        total_labor <- sum(in_year$num_employee)
        labor_alt <- foreign_labor/total_labor
        df_recreate$l_mobility_alt[which(df_recreate$sector==x & df_recreate$country==y & df_recreate$year==t)]<-labor_alt
        
        foreign_markup <- sum(foreign$markup)
        total_markup <- sum(in_year$markup)
        comp_effect <- foreign_markup/total_markup
        df_recreate$comp_alt[which(df_recreate$sector==x & df_recreate$country==y & df_recreate$year==t)]<-comp_effect
      }

    }
  }
  
  # Demean variable used to calculate interaction terms, per the original study
  df_recreate<-df_recreate %>% 
    mutate(demo = demean(demo)) %>% 
    mutate(num_employee = demean(num_employee)) %>% 
    mutate(l_mobility = demo*num_employee) 
  
  # For firms in sectors with no FDI, change the technology gap with FDI to zero
  df_recreate$tech_gap[is.na(df_recreate$tech_gap)]<-0
  
  # For firms with no capital, change capital per worker to zero
  df_recreate$capital_pw[is.infinite(df_recreate$capital_pw)]<-0
  
  # Define sectors in which domestic firm labor productivity is lower than foreign firms' as low tech
  df_recreate<-df_recreate %>% 
    mutate(tech_gap=ifelse(tech_gap>1,1,0)) %>% 
    mutate(tech_gap_alt = tech_gap_alt/LP) %>% 
    mutate(tech_gap_alt=ifelse(tech_gap_alt>1,1,0))
  
  return(df_recreate)
}



df.list <- list(DRC2, Ghana, Kenya, Malawi, Senegal, Tanzania, Uganda, Zambia)

# 1. Processing data for recreation, with a balanced panel

# select only firms that are observed for two periods
panel.list <- lapply(df.list, function(x) filter(x,panel==3))

country_list <- list(1,2,3,4,5,6,7,8)

# for each country, assign a country dummy, select relevant variables,
# delete observations with missing data, and add a country label to firms' panel id
for (x in 1:8){
  panel.list[[x]]<- mutate(panel.list[[x]], country=country_list[[x]])
  panel.list[[x]]<- select(panel.list[[x]], panel_id, year, country, d3c, d2, b2b, a6b, b5, l1, l10, d1a2, n2a, n5a)
  panel.list[[x]]<-na_if(panel.list[[x]],c(-9,-8))
  panel.list[[x]]<- panel.list[[x]][complete.cases(panel.list[[x]]),]
  panel.list[[x]]<- mutate(panel.list[[x]], panel_id=panel_id+country*100000)
}

# combine separate datasets into one dataframe
df <- panel.list[[1]]
for (x in 2:8){
  df <- bind_rows(df, panel.list[[x]])
}


# apply the previously defined data processing function
df_recreate <- clean_data(df)

# keep only observations with complete data for both periods
balanced <- df_recreate %>% count(panel_id) %>% filter(n >= 2)
df_recreate <- df_recreate[df_recreate$panel_id %in% balanced$panel_id, ]

# export data
write.csv(df_recreate, "df_recreate.csv")


# 2. Processing the data as an unbalanced panel

panel.list <- df.list

# apply similar procedure, except this time keep all observations with complete data, 
# resulting in an unbalanced panel
for (x in 1:8){
  panel.list[[x]]<- mutate(panel.list[[x]], country=country_list[[x]])
  panel.list[[x]]<- select(panel.list[[x]], panel_id, year, country, d3c, d2, b2b, a6b, b5, l1, l10, d1a2, n2a, n5a)
  panel.list[[x]]<-na_if(panel.list[[x]],c(-9,-8))
  panel.list[[x]]<- panel.list[[x]][complete.cases(panel.list[[x]]),]
  panel.list[[x]]<- mutate(panel.list[[x]], panel_id=panel_id+country*100000)
}


df <- panel.list[[1]]
for (x in 2:8){
  df <- bind_rows(df, panel.list[[x]])
}


df_analysis <- clean_data(df)

df_analysis<-df_analysis[complete.cases(df_analysis),]

write.csv(df_analysis, "df_analysis.csv")

# 3. Processing the data for probit drop-out test

# select observations that are either in both periods or in the first period only,
# and apply similar processing procedure as before
panel.list <- lapply(df.list, function(x) filter(x,panel==3|panel==2))

for (x in 1:8){
  panel.list[[x]]<- mutate(panel.list[[x]], country=country_list[[x]])
  panel.list[[x]]<- select(panel.list[[x]], panel, panel_id, year, country, d3c, d2, b2b, a6b, b5, l1, l10, d1a2, n2a, n5a, n3, l2)
  panel.list[[x]]<-na_if(panel.list[[x]],c(-9,-8))
  panel.list[[x]]<- panel.list[[x]][complete.cases(panel.list[[x]]),]
  panel.list[[x]]<- mutate(panel.list[[x]], panel_id=panel_id+country*100000)
}


df <- panel.list[[1]]
for (x in 2:8){
  df <- bind_rows(df, panel.list[[x]])
}


df_probit <- clean_data(df)
df_probit <- df_probit %>% 
  mutate(LP_lagged=log(n3/l2)) # add a new variable: lagged productivity (from three years before the survey)
df_probit<-df_probit[complete.cases(df_probit),]

# Change dummy variable value label to 0 and 1
df_probit$panel<-replace(df_probit$panel, df_probit$panel==2, 0)
df_probit$panel<-replace(df_probit$panel, df_probit$panel==3, 1)


write.csv(df_probit, "df_probit.csv")



