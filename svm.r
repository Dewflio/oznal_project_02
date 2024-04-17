#' ---
#' title: Classification model - Support Vector Machines (SVM)
#' author: Ema Richnakova
#' ---

#' *There will be header, date, authors and dataset description like from 1st project*

#'
#'## Importing Libraries

library(magrittr)
library(tidyverse)
library(dplyr) # mutate
library(e1071) # for SVM
library(pROC) # roc() - create ROC curve
library(caret) # confusion matrix

#'
#'## Loading the Dataset

data <- read_csv("spotify_top_songs_audio_features.csv", col_names = TRUE, num_threads = 4)
head(data)

#'
#'## Data exploration
#'
#' From contructed correlation heatmap below, we can see strong positive
#' correlation between energy and loudness and valence. From this data,
#' hypothesis can be created.

heatmap(cor(data %>% select_if(is.numeric)),
        col = colorRampPalette(c("darkgreen", "white", "red"))(100),
        symm = TRUE)

#'
#'### Hypothesis 1
#'
#' *Songs classified as more energetic (>=0.64)(1) are more likely to be louder
#' and happier compared to songs classified as less energetic (<0.64)(class 0).*

mean_energy <- mean(data$energy) ; mean_energy
class_data <- data %>%
  mutate(energy = ifelse(energy >= mean_energy, 1, 0)) %>%
  select_if(is.numeric)
head(class_data)

#' Using energy mean as threshold for classification is beneficial for
#' top Spotify songs dataset, where higher energy levels are prevalent.
#' Classification classes are more balanced.

table(class_data$energy)

#'
#'## Classification model - Support Vector Machine (SVM)
#'
#'### Why SVM?
#'
#' SVM preforms well in high-dimensional spaces, making it suitable for dataset
#' with many features. Also it aims to find best decision boundary, that
#' maximizes margin between classes (reduces overfitting). Performs well by
#' focusing on the most informative features. It can also handle inbalanced
#' datasets, but our classification solves this issue already.
#'
#'### Prepare data for SVM
#'
#' SVM model does not require normally distributed data. However, it is crucial
#' to properly scale input data. Scale of song features can influence decision
#' boudnaries of SVM, thereby affecting its optimal performance.
#'
#' Common practice is to scale input to similar range (e.g. 0-1) before training.

hist(class_data$loudness + class_data$valence, col = "#99ff99") # hypergeometric distribution
#' Range of loudness:
range(class_data$loudness)
#' Range of valence:
range(class_data$valence)
#' Valence values are already normalized, but we scale it also with loudness, so
#' both have min=0 and max=1.
#'
#'#### Scaling

min_max_scale <- function(x, min_val, max_val) {
  (x - min_val) / (max_val - min_val)
}
class_data <- class_data %>%
  mutate(
    loudness = min_max_scale(loudness, min(loudness), max(loudness)),
    valence = min_max_scale(valence, min(valence), max(valence))
  )
#' New range of loudness:
range(class_data$loudness)
#' New range of valence:
range(class_data$valence)

#'
#'### Sampling

sample <- sample(c(TRUE, FALSE), nrow(class_data), replace = TRUE, prob = c(0.7, 0.3))
train_data <- class_data[sample, ]
test_data <- class_data[!sample, ]

#'
#'### Train SVM model

model_svm <- svm(as.factor(energy) ~ loudness + valence, data = train_data, kernel = "linear", cost = 10, scale = FALSE)
# higher cost -> more complex decision boundary -> penalizes misclassification
# scale = FALSE -> inputs are already scaled, so scalling in SVM can be disabled
summary(model_svm)
# number of support vectors -> support vectors are data points that lies closest
# to decision boundary and influence placement of decision boundary.
coefs <- coef(model_svm) ; coefs # weight of each feature to predict TP/N
model_svm$rho
# (+ coef - higher chance of class 1, - coef - opposite)
plot(model_svm, train_data, loudness ~ valence, col = c("#ffc0c9", "lightgrey"))
abline(-coefs[1] / coefs[2], (-coefs[3] / coefs[2]), col = "green")
abline((-coefs[1] - 1) / coefs[2], -coefs[3] / coefs[2], lty = 2, col = "green")
abline((-coefs[1] + 1) / coefs[2], -coefs[3] / coefs[2], lty = 2, col = "green")
# ablines from https://www.datacamp.com/tutorial/support-vector-machines-r

#'
#'### Test SVM model

predicted_classes <- predict(model_svm, newdata = test_data, decision.values = TRUE)
decision_values <- attr(predicted_classes, "decision.values")[, 1]
head(decision_values) # signed distance of each data point to the decision boundary
# confidence of the model in its prediction
#'
#'#### ROC curve
roc <- roc(test_data$energy, decision_values)
plot(roc, main = "ROC Curve for SVM Model", col = "green", lwd = 2)
legend("bottomright", legend = paste("AUC =", round(auc(roc), 5)), col = "green", lwd = 2, bty = "n")

#'## Evaluation of SVM
confusion_matrix <- caret::confusionMatrix(as.factor(predicted_classes), as_factor(test_data$energy), positive = "1")
ct <- confusion_matrix$table

TP <- ct[2, 2] # true positive
FP <- ct[2, 1] # false positive
TN <- ct[1, 1] # true negative
FN <- ct[1, 2] # false negative
P <- TP + FN # all positives (1)
N <- FP + TN # all negatives (0)

# confusion matrix heatmap
ggplot(as.data.frame(ct), aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "white", high = "green") +
  labs(x = "Actual", y = "Predicted", title = "Confusion Matrix Heatmap")

print(paste("Accuracy: ", (TP + TN) / (P + N)))
print(paste("Proportion of class 1: ", P / (P + N)))
print(paste("Proportion of class 0: ", N / (P + N)))
precision <- TP / (FP + TP)
print(paste("Precision: ", precision))
recall <- TP / P
print(paste("Recall: ", recall))
f1_score <- 2 * (precision * recall) / (precision + recall)
print(paste("F1 score: ", f1_score))
print(paste("Specificity: ", TN / (TN + FP)))