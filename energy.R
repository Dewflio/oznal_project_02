library(magrittr)
library(tidyverse)
library(dplyr) # mutate
library(caret) # confusion matrix
library(randomForest)
library(MASS)
library(gbm)
library(pROC)

hist(data$energy, col="green")
range(data$valence)

QDA_data <- data[, c("energy", "loudness", "valence")]

mean_energy <- mean(QDA_data$energy)
mean_energy
median_energy <- median(QDA_data$energy)
median_energy

QDA_data <- QDA_data %>%
  mutate(
    energy = ifelse(energy >= mean_energy, 1, 0),
    loudness = min_max_scale(loudness, min(loudness), max(loudness)),
    valence = min_max_scale(valence, min(valence), max(valence))
  )

range(QDA_data$energy)
range(QDA_data$valence)
range(QDA_data$loudness)


table(QDA_data$energy)


set.seed(1)
QDA_train_indices <- createDataPartition(QDA_data$energy, p = 0.8, list = FALSE)
QDA_train_data <- QDA_data[QDA_train_indices, ]
QDA_test_data <- QDA_data[-QDA_train_indices, ]

QDA_model <- qda(energy ~ ., data = QDA_train_data)
summary(QDA_model)

# Make predictions on the test data
QDA_predictions <- predict(QDA_model, newdata = QDA_test_data, decision.values = TRUE)
QDA_probabilities <- predict(QDA_model, newdata = QDA_test_data, type="probs")
QDA_decision_values <- QDA_probabilities$posterior[,"1"]
head(QDA_decision_values)

QDA_roc <- roc(QDA_test_data$energy, QDA_decision_values)
plot(QDA_roc, main = "ROC Curve for QDA Model", col = "green", lwd = 2)
legend("bottomright", legend = paste("AUC =", round(auc(QDA_roc), 5)), col = "green", lwd = 2, bty = "n")

# Calculate accuracy or other classification metrics
QDA_accuracy <- mean(QDA_predictions$class == QDA_test_data$energy)


# Print the accuracy
cat("Accuracy:", QDA_accuracy, "\n")

QDA_confusion_matrix <- caret::confusionMatrix(as.factor(QDA_predictions$class),
                                               as.factor(QDA_test_data$energy))
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
QDA_median_data <- data[, c("energy", "loudness", "valence")]
median_energy <- median(QDA_median_data$energy)
median_energy


QDA_median_data <- QDA_median_data %>%
  mutate(
    energy = ifelse(energy >= median_energy, 1, 0),
    loudness = min_max_scale(loudness, min(loudness), max(loudness)),
    valence = min_max_scale(valence, min(valence), max(valence))
  )



range(QDA_median_data$energy)
range(QDA_median_data$valence)
range(QDA_median_data$loudness)
range(QDA_median_data$energy)


set.seed(1)
QDA_median_train_indices <- createDataPartition(QDA_median_data$energy, p = 0.8, list = FALSE)
QDA_median_train_data <- QDA_median_data[QDA_median_train_indices, ]
QDA_median_test_data <- QDA_median_data[-QDA_median_train_indices, ]

QDA_median_model <- qda(energy ~ ., data = QDA_median_train_data)
summary(QDA_median_model)

# Make predictions on the test data
QDA_median_predictions <- predict(QDA_median_model, newdata = QDA_median_test_data, decision.values = TRUE)
QDA_median_probabilities <- predict(QDA_median_model, newdata = QDA_median_test_data, type="probs")
QDA_median_decision_values <- QDA_median_probabilities$posterior[,"1"]
head(QDA_median_decision_values)

QDA_median_roc <- roc(QDA_median_test_data$energy, QDA_median_decision_values)
plot(QDA_median_roc, main = "ROC Curve for QDA Model", col = "green", lwd = 2)
legend("bottomright", legend = paste("AUC =", round(auc(QDA_median_roc), 5)), col = "green", lwd = 2, bty = "n")

# Calculate accuracy or other classification metrics
QDA_median_accuracy <- mean(QDA_median_predictions$class == QDA_median_test_data$energy)

# Print the accuracy
cat("Accuracy:", QDA_median_accuracy, "\n")

QDA_median_confusion_matrix <- caret::confusionMatrix(as.factor(QDA_median_predictions$class),
                                                      as.factor(QDA_median_test_data$energy))
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

#Model 4: Classification with GBM - Gradient Boosting Machine

hist(data$energy, col="green")
range(data$energy)
range(data$valence)
range(data$loudness)
range(data$energy)

GBM_data <- data[, c("energy", "loudness", "valence")]

mean_energy <- mean(GBM_data$energy)
mean_energy
median_energy <- median(GBM_data$energy)
median_energy

GBM_data <- GBM_data %>%
  mutate(
    energy = ifelse(energy >= median_energy, 1, 0),
    loudness = min_max_scale(loudness, min(loudness), max(loudness)),
    valence = min_max_scale(valence, min(valence), max(valence))
  )

range(GBM_data$valence)
range(GBM_data$loudness)



set.seed(1)
GBM_train_indices <- createDataPartition(GBM_data$energy, p = 0.8, list = FALSE)
GBM_train_data <- GBM_data[GBM_train_indices, ]
GBM_test_data <- GBM_data[-GBM_train_indices, ]

GBM_model <- gbm(energy ~ ., data = GBM_train_data, distribution = "bernoulli", n.trees = 100, interaction.depth = 3)
summary(GBM_model)

# Make predictions on the test data
GBM_prediction_probabilities <- predict(GBM_model, newdata = GBM_test_data, n.trees = 100, type = "response")
head(GBM_prediction_probabilities)


GBM_roc <- roc(GBM_test_data$energy, GBM_prediction_probabilities)
plot(GBM_roc, main = "ROC Curve for QDA Model", col = "green", lwd = 2)
legend("bottomright", legend = paste("AUC =", round(auc(GBM_roc), 5)), col = "green", lwd = 2, bty = "n")

GBM_predictions <- ifelse(GBM_prediction_probabilities >= 0.8576, 1, 0)

# Calculate accuracy or other classification metrics
GBM_accuracy <- mean(GBM_predictions == GBM_test_data$energy)

# Print the accuracy
cat("Accuracy:", GBM_accuracy, "\n")

GBM_confusion_matrix <- caret::confusionMatrix(as.factor(GBM_predictions),
                                               as.factor(GBM_test_data$energy))
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


# Define the parameter grid
param_grid <- expand.grid(
  n.trees = c(100, 200, 300),
  interaction.depth = c(3, 4, 5),
  shrinkage = c(0.01, 0.1, 0.3),
  n.minobsinnode = c(10, 20, 30)
)

# Specify the control parameters for training
ctrl <- trainControl(method = "cv", number = 5)

# Perform hyperparameter tuning
gbm_tuned <- train(
  energy ~ ., 
  data = GBM_train_data, 
  method = "gbm", 
  trControl = ctrl, 
  tuneGrid = param_grid,
  distribution="bernoulli",
)

GBM_tuned_model <- gbm_tuned$finalModel
str(GBM_tuned_model)

GBM_tuned_prediction_probabilities <- predict(GBM_tuned_model, newdata = GBM_test_data, type = "response")
head(GBM_tuned_prediction_probabilities)
GBM_tuned_predictions <- ifelse(GBM_tuned_prediction_probabilities >= 0.8, 1, 0)


GBM_tuned_roc <- roc(GBM_test_data$energy, GBM_tuned_prediction_probabilities)
plot(GBM_tuned_roc, main = "ROC Curve for GBM Model", col = "green", lwd = 2)
legend("bottomright", legend = paste("AUC =", round(auc(GBM_tuned_roc), 5)), col = "green", lwd = 2, bty = "n")

# Calculate accuracy or other classification metrics
GBM_tuned_accuracy <- mean(GBM_tuned_predictions == GBM_test_data$energy)

# Print the accuracy
cat("Accuracy:", GBM_tuned_accuracy, "\n")

GBM_tuned_confusion_matrix <- caret::confusionMatrix(as.factor(GBM_tuned_predictions),
                                               as.factor(GBM_test_data$energy))
GBM_tuned_ct <- GBM_tuned_confusion_matrix$table
GBM_tuned_TP <- GBM_tuned_ct[2, 2] # true positive
GBM_tuned_FP <- GBM_tuned_ct[2, 1] # false positive
GBM_tuned_TN <- GBM_tuned_ct[1, 1] # true negative
GBM_tuned_FN <- GBM_tuned_ct[1, 2] # false negative
GBM_tuned_P <- GBM_tuned_TP + GBM_tuned_FN # all positives (1)
GBM_tuned_N <- GBM_tuned_FP + GBM_tuned_TN # all negatives (0)

ggplot(as.data.frame(GBM_tuned_ct), aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "white", high = "green") +
  labs(x = "Actual", y = "Predicted", title = "Confusion Matrix Heatmap")


