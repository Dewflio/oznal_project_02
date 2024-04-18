library(magrittr)
library(tidyverse)
library(dplyr) # mutate
library(caret) # confusion matrix
library(randomForest)
library(MASS)
library(gbm)

data <- read_csv("spotify_top_songs_audio_features.csv", col_names = TRUE, num_threads = 4)
head(data)

View(data)

# EDA
numeric_columns <- names(data)[sapply(data, is.numeric)]
numeric_columns
numeric_data <- data %>% dplyr::select(numeric_columns)

numeric_correlation_matrix <- cor(numeric_data)
heatmap(numeric_correlation_matrix, 
        col = colorRampPalette(c("blue", "white", "red"))(100),
        symm = TRUE)


min_max_scale <- function(x, min_val, max_val) {
  (x - min_val) / (max_val - min_val)
}

# Model 1:  acousticness ~ danceability + loudness + energy
hist(data$acousticness, col="green")
range(data$acousticness)
range(data$danceability)
range(data$loudness)
range(data$energy)


randomForest_data <- data[, c("acousticness", "loudness", "energy", "danceability")]
randomForest_data <- randomForest_data %>%
  mutate(
    loudness = min_max_scale(loudness, min(loudness), max(loudness)),
    energy = min_max_scale(energy, min(energy), max(energy)),
    danceability = min_max_scale(danceability, min(danceability), max(danceability))
  )

mean(randomForest_data$acousticness)
median(randomForest_data$acousticness)
range(randomForest_data$acousticness)
range(randomForest_data$danceability)
range(randomForest_data$loudness)
range(randomForest_data$energy)


set.seed(1)
randomForest_train_indices <- createDataPartition(randomForest_data$acousticness, p = 0.8, list = FALSE)
randomForest_train_data <- randomForest_data[randomForest_train_indices, ]
randomForest_test_data <- randomForest_data[-randomForest_train_indices, ]

randomForest_model <- randomForest(acousticness ~ ., data = randomForest_train_data, ntree=1000)
summary(randomForest_model)

randomForest_predictions <- predict(randomForest_model, newdata = randomForest_test_data)
randomForest_mse <- mean((randomForest_test_data$acousticness - randomForest_predictions)^2)
randomForest_mse

varImpPlot(randomForest_model)

randomForest_residuals <- randomForest_test_data$acousticness - randomForest_predictions
plot(randomForest_predictions, randomForest_residuals, xlab = "Predicted", ylab = "Residuals", main = "Residual Plot")


# Model 2:  energy ~ loudness + valence

hist(data$energy, col="green")
range(data$valence)

model2_data <- data[, c("energy", "loudness", "valence")]
model2_data <- model2_data %>%
  mutate(
    #energy = min_max_scale(energy, min(energy), max(energy)),
    loudness = min_max_scale(loudness, min(loudness), max(loudness)),
    valence = min_max_scale(valence, min(valence), max(valence))
  )

range(model2_data$energy)
range(model2_data$loudness)
range(model2_data$valence)


set.seed(1)
model2_train_indices <- createDataPartition(randomForest_data$acousticness, p = 0.8, list = FALSE)
model2_train_data <- model2_data[model2_train_indices, ]
model2_test_data <- model2_data[-model2_train_indices, ]

model2 <- randomForest(energy ~ ., data = model2_train_data, ntree=1000)
summary(model2)

model2_predictions <- predict(model2, newdata = model2_test_data)
model2_mse <- mean((model2_test_data$energy - model2_predictions)^2)
model2_mse
model2_rmse <- sqrt(model2_mse)
model2_rmse

varImpPlot(model2)

model2_residuals <- model2_test_data$energy - model2_predictions
plot(model2_predictions, model2_residuals, xlab = "Predicted", ylab = "Residuals", main = "Residual Plot")

# Model 3:  acousticness ~ danceability + loudness + energy
# Classification with QDA
hist(data$acousticness, col="green")
range(data$acousticness)
range(data$danceability)
range(data$loudness)
range(data$energy)

QDA_data <- data[, c("acousticness", "loudness", "energy", "danceability")]

mean_acousticness <- mean(QDA_data$acousticness)
mean_acousticness
median_acousticness <- median(QDA_data$acousticness)
median_acousticness

QDA_data <- QDA_data %>%
  mutate(
    acousticness = ifelse(acousticness >= mean_acousticness, 1, 0),
    loudness = min_max_scale(loudness, min(loudness), max(loudness)),
    energy = min_max_scale(energy, min(energy), max(energy)),
    danceability = min_max_scale(danceability, min(danceability), max(danceability))
  )

range(QDA_data$acousticness)
range(QDA_data$danceability)
range(QDA_data$loudness)
range(QDA_data$energy)

table(QDA_data$acousticness)


set.seed(1)
QDA_train_indices <- createDataPartition(QDA_data$acousticness, p = 0.8, list = FALSE)
QDA_train_data <- QDA_data[QDA_train_indices, ]
QDA_test_data <- QDA_data[-QDA_train_indices, ]

QDA_model <- qda(acousticness ~ ., data = QDA_train_data)
summary(QDA_model)

# Make predictions on the test data
QDA_predictions <- predict(QDA_model, newdata = QDA_test_data, decision.values = TRUE)
QDA_probabilities <- predict(QDA_model, newdata = QDA_test_data, type="probs")
QDA_decision_values <- QDA_probabilities$posterior[,"1"]
head(QDA_decision_values)

QDA_roc <- roc(QDA_test_data$acousticness, QDA_decision_values)
plot(QDA_roc, main = "ROC Curve for QDA Model", col = "green", lwd = 2)
legend("bottomright", legend = paste("AUC =", round(auc(QDA_roc), 5)), col = "green", lwd = 2, bty = "n")

# Calculate accuracy or other classification metrics
QDA_accuracy <- mean(QDA_predictions$class == QDA_test_data$acousticness)

# Print the accuracy
cat("Accuracy:", QDA_accuracy, "\n")

QDA_confusion_matrix <- caret::confusionMatrix(as.factor(QDA_predictions$class),
                                                  as.factor(QDA_test_data$acousticness))
QDA_ct <- QDA_confusion_matrix$table
QDA_TP <- QDA_ct[2, 2] # true positive
QDA_FP <- QDA_ct[2, 1] # false positive
QDA_TN <- QDA_ct[1, 1] # true negative
QDA_FN <- QDA_ct[1, 2] # false negative
QDA_P <- QDA_TP + QDA_FN # all positives (1)
QDA_N <- QDA_FP + QDA_TN # all negatives (0)

ggplot(as.data.frame(QDA_ct), aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "white", high = "green") +
  labs(x = "Actual", y = "Predicted", title = "Confusion Matrix Heatmap")


# QDA - median
QDA_median_data <- data[, c("acousticness", "loudness", "energy", "danceability")]
median_acousticness <- median(QDA_median_data$acousticness)
median_acousticness


QDA_median_data <- QDA_median_data %>%
  mutate(
    acousticness = ifelse(acousticness >= median_acousticness, 1, 0),
    loudness = min_max_scale(loudness, min(loudness), max(loudness)),
    energy = min_max_scale(energy, min(energy), max(energy)),
    danceability = min_max_scale(danceability, min(danceability), max(danceability))
  )


table(QDA_median_data$acousticness)

range(QDA_median_data$acousticness)
range(QDA_median_data$danceability)
range(QDA_median_data$loudness)
range(QDA_median_data$energy)


set.seed(1)
QDA_median_train_indices <- createDataPartition(QDA_median_data$acousticness, p = 0.8, list = FALSE)
QDA_median_train_data <- QDA_median_data[QDA_median_train_indices, ]
QDA_median_test_data <- QDA_median_data[-QDA_median_train_indices, ]

QDA_median_model <- qda(acousticness ~ ., data = QDA_median_train_data)
summary(QDA_median_model)

# Make predictions on the test data
QDA_median_predictions <- predict(QDA_median_model, newdata = QDA_median_test_data, decision.values = TRUE)
QDA_median_probabilities <- predict(QDA_median_model, newdata = QDA_median_test_data, type="probs")
QDA_median_decision_values <- QDA_median_probabilities$posterior[,"1"]
head(QDA_median_decision_values)

QDA_median_roc <- roc(QDA_median_test_data$acousticness, QDA_median_decision_values)
plot(QDA_median_roc, main = "ROC Curve for QDA Model", col = "green", lwd = 2)
legend("bottomright", legend = paste("AUC =", round(auc(QDA_median_roc), 5)), col = "green", lwd = 2, bty = "n")

# Calculate accuracy or other classification metrics
QDA_median_accuracy <- mean(QDA_median_predictions$class == QDA_median_test_data$acousticness)

# Print the accuracy
cat("Accuracy:", QDA_median_accuracy, "\n")

QDA_median_confusion_matrix <- caret::confusionMatrix(as.factor(QDA_median_predictions$class),
                                                  as.factor(QDA_median_test_data$acousticness))
QDA_median_ct <- QDA_median_confusion_matrix$table
QDA_median_TP <- QDA_median_ct[2, 2] # true positive
QDA_median_FP <- QDA_median_ct[2, 1] # false positive
QDA_median_TN <- QDA_median_ct[1, 1] # true negative
QDA_median_FN <- QDA_median_ct[1, 2] # false negative
QDA_median_P <- QDA_median_TP + QDA_median_FN # all positives (1)
QDA_median_N <- QDA_median_FP + QDA_median_TN # all negatives (0)

ggplot(as.data.frame(QDA_median_ct), aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "white", high = "green") +
  labs(x = "Actual", y = "Predicted", title = "Confusion Matrix Heatmap")

#Model 4: Classification with GBM

hist(data$acousticness, col="green")
range(data$acousticness)
range(data$danceability)
range(data$loudness)
range(data$energy)

GBM_data <- data[, c("acousticness", "loudness", "energy", "danceability")]

mean_acousticness <- mean(GBM_data$acousticness)
mean_acousticness
median_acousticness <- median(GBM_data$acousticness)
median_acousticness

GBM_data <- GBM_data %>%
  mutate(
    acousticness = ifelse(acousticness >= mean_acousticness, 1, 0),
    loudness = min_max_scale(loudness, min(loudness), max(loudness)),
    energy = min_max_scale(energy, min(energy), max(energy)),
    danceability = min_max_scale(danceability, min(danceability), max(danceability))
  )

range(GBM_data$acousticness)
range(GBM_data$danceability)
range(GBM_data$loudness)
range(GBM_data$energy)


set.seed(1)
GBM_train_indices <- createDataPartition(GBM_data$acousticness, p = 0.8, list = FALSE)
GBM_train_data <- GBM_data[GBM_train_indices, ]
GBM_test_data <- GBM_data[-GBM_train_indices, ]

GBM_model <- gbm(acousticness ~ ., data = GBM_train_data, distribution = "bernoulli", n.trees = 100, interaction.depth = 3)
summary(GBM_model)

# Make predictions on the test data
GBM_prediction_probabilities <- predict(GBM_model, newdata = GBM_test_data, n.trees = 100, type = "response")
head(GBM_prediction_probabilities)

GBM_predictions <- ifelse(GBM_prediction_probabilities >= 0.713, 1, 0)



GBM_roc <- roc(GBM_test_data$acousticness, GBM_prediction_probabilities)
plot(GBM_roc, main = "ROC Curve for QDA Model", col = "green", lwd = 2)
legend("bottomright", legend = paste("AUC =", round(auc(GBM_roc), 5)), col = "green", lwd = 2, bty = "n")

# Calculate accuracy or other classification metrics
GBM_accuracy <- mean(GBM_predictions == GBM_test_data$acousticness)

# Print the accuracy
cat("Accuracy:", GBM_accuracy, "\n")

GBM_confusion_matrix <- caret::confusionMatrix(as.factor(GBM_predictions),
                                               as.factor(GBM_test_data$acousticness))
GBM_ct <- GBM_confusion_matrix$table
GBM_TP <- GBM_ct[2, 2] # true positive
GBM_FP <- GBM_ct[2, 1] # false positive
GBM_TN <- GBM_ct[1, 1] # true negative
GBM_FN <- GBM_ct[1, 2] # false negative
GBM_P <- GBM_TP + GBM_FN # all positives (1)
GBM_N <- GBM_FP + GBM_TN # all negatives (0)

ggplot(as.data.frame(GBM_ct), aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "white", high = "green") +
  labs(x = "Actual", y = "Predicted", title = "Confusion Matrix Heatmap")





