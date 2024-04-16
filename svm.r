#' ---
#' title: Classification model - Support Vector Machines (SVM)
#' author: Ema Richnakova
#' ---

### Importing Libraries

library(magrittr)
library(tidyverse)
library(dplyr) # mutate
library(nortest) # Anderson-Darling test for normality
library(e1071) # for SVM
library(caret) # confusion matrix

### Loading the Dataset

data <- read_csv("spotify_top_songs_audio_features.csv", col_names = TRUE, num_threads = 4)
head(data)

### Data exploration

heatmap(cor(data %>% select_if(is.numeric)),
        col = colorRampPalette(c("darkgreen", "white", "red"))(100),
        symm = TRUE)

class_data <- data %>%
  mutate(energy = ifelse(energy >= mean(data$energy), 1, 0)) %>%
  select_if(is.numeric)
head(class_data)
table(class_data$energy)
# TODO
### Why Support Vector Machines (SVM)
# algorithm finds hyperplane that best separates the classes in the feature space

### Hypothesis 1
# *Songs classified as more energetic (>=0.6)(class 1) are more likely to be happier and louder compared to songs classified as less energetic (<0.6)(class 0).*

### Check data normal distribution

ad.test(class_data$valence + class_data$loudness)$p.value
hist(class_data$valence + class_data$loudness, col = "green")

### Sampling
sample <- sample(c(TRUE, FALSE), nrow(class_data), replace = TRUE, prob = c(0.7, 0.3))
train_data <- class_data[sample, ]
test_data <- class_data[!sample, ]
head(train_data)

### Train model
model_svm <- svm(as.factor(energy) ~ valence + loudness, data = train_data, kernel = "linear", cost = 10, scale = FALSE)
summary(model_svm)
coefs <- coef(model_svm)
coefs
plot(model_svm, train_data, loudness ~ valence, col=c("#ffc0c9", "#9eff9e"))
abline(-coefs[1] / coefs[3], -coefs[2] / coefs[3], col="black")
abline(-(coefs[1] - 1) / coefs[3], -coefs[2] / coefs[3], lty = 2, col="black")
abline(-(coefs[1] + 1) / coefs[3], -coefs[2] / coefs[3], lty = 2, col="black")


predicted_classes <- predict(model_svm, newdata = test_data, type="response")


### Eval
confusion_matrix <- caret::confusionMatrix(as_factor(predicted_classes), as_factor(test_data$energy), positive = "1")

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

print(paste("Proportion of major mode (1): ", P / (P + N)))

print(paste("Proportion of minor mode (0): ", N / (P + N)))

precision <- TP / (FP + TP)

print(paste("Precision: ", precision))

recall <- TP / P

print(paste("Recall: ", recall))

f1_score <- 2 * (precision * recall) / (precision + recall)

print(paste("F1 score: ", f1_score))

print(paste("Specificity: ", TN / (TN + FP)))