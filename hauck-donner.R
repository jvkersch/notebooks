# Dataset with fixed proportions
create_dataset <- function(n, p0, p1) {
  n0 <- as.integer(p0*n)
  n1 <- as.integer(p1*n)
  x <- c(rep(0, n), rep(1, n))
  y <- c(rep(1, n0), rep(0, n - n0),
         rep(1, n1), rep(0, n - n1))
  data.frame(x = x, y = y)
}

# Fit and extract logistic model coefficients
model_coefs <- function(df) {
  m <- glm(y ~ x, data = df, family = "binomial")
  beta <- summary(m)$coefficients["x", "Estimate"]
  z <- summary(m)$coefficients["x", "z value"]
  c(beta = beta, z = z)
}

# Re-run Hauck and Donner experiment
p0 <- 0.25
p1s <- c(seq(0.5, 0.95, by = 0.05),
         seq(0.95, 0.99, by = 0.01))

data <- NULL
for (p1 in p1s) {
  coefs <- model_coefs(create_dataset(100, p0, p1))
  data <- rbind(data, c(p1 = p1, coefs))
}
data <- as.data.frame(data)

plot(p1s, data$z, lty = "line")
points(p1s, data$z)
