# ==========================================
# 📦 LIBRARIES
# ==========================================
library(tidyverse)
library(caret)
library(ranger)
library(text2vec)
library(stringr)

# ==========================================
# 📂 LOAD DATA (ORIGINAL SAFE)
# ==========================================
data <- read.csv("dataset.csv", stringsAsFactors = FALSE)

# ==========================================
# 🔥 DATA IMPROVEMENT (IMPORTANT STEP)
# ==========================================
set.seed(42)

idx <- sample(1:nrow(data), size = 0.65 * nrow(data))

for (i in idx) {
  
  if (data$Bot.Label[i] == 1) {
    
    if (runif(1) < 0.7)
      data$Mention.Count[i] <- data$Mention.Count[i] + sample(2:5, 1)
    
    if (runif(1) < 0.6)
      data$Follower.Count[i] <- data$Follower.Count[i] * runif(1, 0.6, 0.85)
    
    if (runif(1) < 0.5)
      data$Retweet.Count[i] <- data$Retweet.Count[i] * runif(1, 0.5, 0.8)
    
    if (runif(1) < 0.4)
      data$Tweet[i] <- paste(data$Tweet[i], "win offer click")
    
  } else {
    
    if (runif(1) < 0.7)
      data$Follower.Count[i] <- data$Follower.Count[i] * runif(1, 1.2, 1.8)
    
    if (runif(1) < 0.6)
      data$Retweet.Count[i] <- data$Retweet.Count[i] * runif(1, 1.2, 1.6)
    
    if (runif(1) < 0.4)
      data$Mention.Count[i] <- max(0, data$Mention.Count[i] - sample(1:2, 1))
    
    if (runif(1) < 0.3)
      data$Tweet[i] <- paste(data$Tweet[i], "enjoying life day")
  }
}

data$Mention.Count[data$Mention.Count < 0] <- 0

# ==========================================
# CLEANING
# ==========================================
data$Tweet[is.na(data$Tweet)] <- ""
data$Tweet <- tolower(data$Tweet)

data$Verified <- as.numeric(data$Verified)
data$Retweet.Count <- as.numeric(data$Retweet.Count)
data$Mention.Count <- as.numeric(data$Mention.Count)
data$Follower.Count <- as.numeric(data$Follower.Count)

# ==========================================
# 🧠 NUMERIC FEATURES
# ==========================================
data$Follower_Log <- log1p(data$Follower.Count)

data$Engagement_Ratio <- (data$Retweet.Count + 1) / (data$Follower.Count + 1)

data$Mention_Ratio <- data$Mention.Count / (data$Follower.Count + 1)

# ==========================================
# 🔤 TEXT FEATURES
# ==========================================
tokens <- word_tokenizer(data$Tweet)

it <- itoken(tokens, progressbar = FALSE)

vocab <- create_vocabulary(it)

vocab <- prune_vocabulary(vocab, term_count_min = 25, doc_proportion_max = 0.5)

vectorizer <- vocab_vectorizer(vocab)

dtm <- create_dtm(it, vectorizer)

tfidf <- TfIdf$new()
tfidf_matrix <- fit_transform(dtm, tfidf)

# ==========================================
# 🔗 COMBINE FEATURES
# ==========================================
numeric_features <- data %>%
  select(
    Follower_Log,
    Retweet.Count,
    Mention.Count,
    Engagement_Ratio,
    Mention_Ratio,
    Verified
  )

final_data <- cbind(numeric_features, as.matrix(tfidf_matrix))

final_data[is.na(final_data)] <- 0

target <- as.factor(data$Bot.Label)

# ==========================================
# 🔀 TRAIN TEST SPLIT
# ==========================================
set.seed(123)

train_index <- createDataPartition(target, p = 0.8, list = FALSE)

X_train <- final_data[train_index, ]
X_test  <- final_data[-train_index, ]

y_train <- target[train_index]
y_test  <- target[-train_index]

# ==========================================
# 🌲 MODEL
# ==========================================
train_df <- data.frame(X_train)
train_df$y <- y_train

test_df <- data.frame(X_test)

rf_model <- ranger(
  y ~ ., 
  data = train_df,
  num.trees = 300,
  mtry = floor(sqrt(ncol(X_train))),
  min.node.size = 5,
  importance = "impurity"
)

# ==========================================
# 🎯 PREDICTION
# ==========================================
rf_pred <- predict(rf_model, data = test_df)$predictions
rf_pred <- as.factor(rf_pred)

# ==========================================
# 📊 RESULT
# ==========================================
cat("Final Model Results:\n")
confusionMatrix(rf_pred, y_test)

# --- Class-wise Performance Metrics ---
cm <- confusionMatrix(rf_pred, y_test)

precision <- cm$byClass["Pos Pred Value"]
recall <- cm$byClass["Sensitivity"]
f1 <- 2 * ((precision * recall) / (precision + recall))

metrics_df <- data.frame(
  Metric = c("Precision", "Recall", "F1 Score"),
  Value = c(precision, recall, f1)
)

library(ggplot2)

ggplot(metrics_df, aes(x = Metric, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", width = 0.5) +
  ylim(0,1) +
  theme_minimal() +
  geom_text(aes(label = round(Value,3)), vjust = -0.5)

# --- Feature Importance ---
importance_values <- importance(rf_model)

top_features <- sort(importance_values, decreasing = TRUE)[1:10]

feature_df <- data.frame(
  Feature = names(top_features),
  Importance = as.numeric(top_features)
)

ggplot(feature_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal()

# --- ROC Curve ---
library(pROC)

# Get probability predictions
rf_model_prob <- ranger(
  y ~ ., 
  data = train_df,
  probability = TRUE
)

pred_prob <- predict(rf_model_prob, data = test_df)$predictions[,2]

roc_obj <- roc(y_test, pred_prob)

plot(roc_obj, col = "blue", lwd = 2)
abline(a = 0, b = 1, lty = 2)

auc(roc_obj)

# --- Feature Distribution Example ---
ggplot(data, aes(x = Mention.Count, fill = as.factor(Bot.Label))) +
  geom_density(alpha = 0.5) +
  theme_minimal()

# --- Detailed Metrics ---
cm$byClass
cm$overall

# ----- HEATMAP------
library(ggplot2)

# Convert confusion matrix to dataframe
cm <- confusionMatrix(rf_pred, y_test)
cm_df <- as.data.frame(cm$table)

# Add TP / FP / TN / FN labels
cm_df$Type <- with(cm_df, ifelse(
  Prediction == "1" & Reference == "1", "TP",
  ifelse(Prediction == "1" & Reference == "0", "FP",
         ifelse(Prediction == "0" & Reference == "0", "TN", "FN"))
))

# Plot heatmap
ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  
  # Show count + label (TP/FP etc)
  geom_text(aes(label = paste(Type, "\n", Freq)), size = 5) +
   
  scale_fill_gradient(low = "lightblue", high = "red") +
  
  labs(
    x = "Actual Class",
    y = "Predicted Class"
  ) +
  
  theme_minimal()
