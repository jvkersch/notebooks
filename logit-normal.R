set.seed(1234)
library(MASS)

samples <- rnorm(1000, mean = 0.7, sd = 0.1)
samples <- pmin(pmax(samples, 0.01), 0.99)
print(length(samples))

logit <- qlogis
logit_samples <- logit(samples)
logit_samples <- pmin(pmax(logit_samples, -3), 3)

fit <- fitdistr(samples, "normal")
fit_logit <- fitdistr(logit_samples, "normal")

par(mfrow = c(2, 1))
hist(samples, main = "Probabilities", prob = TRUE)
para <- fit$estimate
curve(dnorm(x, para[1], para[2]), col = 2, add = TRUE)

hist(logit_samples, main = "Logits", prob = TRUE)
para <- fit_logit$estimate
curve(dnorm(x, para[1], para[2]), col = 2, add = TRUE)

print(para)