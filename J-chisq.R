draw_chisq <- function(n, k) {
  samples <- vector(length = k)
  for (i in 1:k) {
    samples[i] <- sum(rnorm(n)^2)
  }
  samples
}

p1 <- hist(draw_chisq(5, 300), breaks = 0:50)
p2 <- hist(draw_chisq(10, 300), breaks = 0:50)
plot(p1, col=rgb(0,0,1,1/4), xlim = c(0, 20), main="", xlab = "")
plot(p2, col=rgb(1,0,0,1/4), xlim = c(0, 20), add=T)
#legend("topright", c("nu = 5", "nu = 10"))