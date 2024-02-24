library(ggplot2)
library(e1071)
library(pROC)
library(future.apply)

generate_data <- function(delta) {

  # Number of samples per class
  n <- 500

  # Generate data for Class 1
  x1_class1 <- rnorm(n, mean = 0, sd = 1)
  x2_class1 <- rnorm(n, mean = 0, sd = 1)
  y_class1 <- rep(0, n)

  # Generate data for Class 2
  x1_class2 <- rnorm(n, mean = 0, sd = 1)
  x2_class2 <- rnorm(n, mean = delta, sd = 1)
  y_class2 <- rep(1, n)

  # Combine the data
  x1 <- c(x1_class1, x1_class2)
  x2 <- c(x2_class1, x2_class2)
  y <- c(y_class1, y_class2)

  # Create a data frame
  data <- data.frame(x1, x2, y)
  # Ensure the target variable is a factor
  data$y <- as.factor(data$y)

  # Shuffle the data
  data <- data[sample(nrow(data)), ]

  return(data)
}

plot_data <- function() {
  # Generate some random data
  data <- generate_data(2.0)

  # Scatter plot with points colored by class
  ggplot(data, aes(x = x1, y = x2, color = factor(y))) +
    geom_point() +
    theme_minimal() +
    labs(color = "Class", x = "Feature x1", y = "Feature x2", title = "Scatter Plot of the Synthetic Dataset") +
    scale_color_manual(values = c("blue", "red"))
}

get_model_auc <- function(data) {
  # Train model
  svm_model <- svm(y ~ ., data = data, type = "C-classification", kernel = "radial", probability = TRUE)

  # Predict probabilities
  prob_predictions <- predict(svm_model, data, probability = TRUE)
  probabilities <- attr(prob_predictions, "probabilities")[,2]

  # Calculate AUC
  roc_result <- roc(data$y, probabilities, quiet = TRUE)
  auc_value <- auc(roc_result)

  return(as.double(auc_value))
}

computational_experiment <- function(delta) get_model_auc(generate_data(delta))

run_experiments <- function(delta) {
  trials <- future_replicate(1000, computational_experiment(delta))
  return(trials)
}

make_dataset <- function() {
  plan(multisession, workers = 8)
  ds <- NULL
  for (delta in c(0, 3, 5)) {
    data <- run_experiments(delta)
    df1 <- data.frame(y = data, delta = delta, type = "AUC")

    logits <- qlogis(data)
    df2 <- data.frame(y = logits, delta = delta, type = "logit-AUC")
    ds <- rbind(ds, df1, df2)
  }
  ds$type <- as.factor(ds$type)
  return(ds)
}

ds <- make_dataset()

plot_histograms <- function() {
  par(mfrow = c(3, 2))
  for (delta in c(0, 3, 5)) {
    for (type in c("AUC", "logit-AUC")) {
      data <- ds[(ds$delta == delta) & (ds$type == type), "y"]
      hist(data,
           main = paste0(type, " (delta = ", delta, ")"),
           xlab = NULL, ylab = NULL)
    }
  }
}

plot_qq <- function() {
  par(mfrow = c(3, 2))
  for (delta in c(0, 3, 5)) {
    for (type in c("AUC", "logit-AUC")) {
      data <- ds[(ds$delta == delta) & (ds$type == type), "y"]
      data <- data[is.finite(data)]
      qqnorm(data,
           main = paste0(type, " (delta = ", delta, ")"),
           xlab = NULL, ylab = NULL)
      qqline(data,
             main = paste0(type, " (delta = ", delta, ")"),
             xlab = NULL, ylab = NULL)

    }
  }
}
