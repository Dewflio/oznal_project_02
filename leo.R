library(magrittr)
library(tidyverse)
library(dplyr) # mutate
library(caret) # confusion matrix
library(nortest) # Anderson-Darling test for normality
library(ROCit) # rocit() - optimal cut-off
library(pROC) # roc() - create ROC curve
library(MASS)


data <- read_csv("spotify_top_songs_audio_features.csv", col_names = TRUE, num_threads = 4)
head(data)

# EDA
numeric_columns <- names(data)[sapply(data, is.numeric)]
numeric_columns
numeric_data <- data %>% dplyr::select(numeric_columns)

numeric_correlation_matrix <- cor(numeric_data)
heatmap(numeric_correlation_matrix, 
        col = colorRampPalette(c("blue", "white", "red"))(100),
        symm = TRUE)


# Regression models

# Model 1:  acousticness ~ danceability + loudness + energy
hist(data$acousticness, col="green")

# Model 2:  energy ~ loudness + valence
hist(data$energy, col="green")