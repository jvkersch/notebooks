theta <- c(0.5, 1)
sigma <- 0.1

n <- 10

sample.theta <- function() {
  xs <- runif(n)
  ys <- theta[1] + theta[2]*xs + rnorm(n, sd = sigma)
  m <- lm(ys ~ xs)
  
  m$coefficients
}

nsims <- 100
theta0 <- vector(length = nsims)
theta1 <- vector(length = nsims)
for (k in 1:nsims) {
  coefs <- sample.theta()
  theta0[k] = coefs[1]
  theta1[k] = coefs[2]
}

plot(theta0, theta1)
points(theta[1], theta[2], pch=4, col="red", lwd=2)
  
