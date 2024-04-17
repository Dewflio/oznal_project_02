library(magrittr)
library(tidyverse)
library(dplyr) # mutate
library(caret) # confusion matrix
library(randomForest)

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


model1_data <- data[, c("acousticness", "loudness", "energy", "danceability")]

set.seed(1)
train_indices <- createDataPartition(model1_data$acousticness, p = 0.8, list = FALSE)
train_data <- model1_data[train_indices, ]
test_data <- model1_data[-train_indices, ]

model <- randomForest(acousticness ~ ., data = train_data)

predictions <- predict(model, newdata = test_data)
mse <- mean((test_data$acousticness - predictions)^2)
mse

varImpPlot(model)

residuals <- test_data$acousticness - predictions
plot(predictions, residuals, xlab = "Predicted", ylab = "Residuals", main = "Residual Plot")


#Improving the random forest

tuneGrid <- expand.grid(
  ntree = c(50, 100, 200),
  mtry = c(2, 3)
)
ctrl <- trainControl(method = "cv", number = 5)
ctrl

tuned_model <- train(
  acousticness ~ .,
  data = train_data,
  method = "rf",
  tuneGrid = tuneGrid,
  trControl = ctrl
)


