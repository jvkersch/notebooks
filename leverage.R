set.seed(12345)

n <- 20
x <- 3*runif(n)
y <- x + 0.5*rnorm(n)

m <- lm(y ~ x)

par(mfrow=c(2, 1))
plot(x, y)
abline(coef(m))

plot(1, type="n", xlab="", ylab="", xlim=c(1, n), ylim=c(0, 1))
segments(1:n, 0, 1:n, hatvalues(m))