# Reproduce the power plots from "Power study of anova versus Kruskal-Wallis
# test", T. Van Hecke. Main findings: power is comparable between KW and ANOVA
# for normal distributions. For skewed distributions, KW has higher power.

p.aov <- function(formula) {
  m <- aov(formula)
  summary(m)[[1]][["Pr(>F)"]][[1]]
}

p.kw <- function(formula) {
  m <- kruskal.test(formula)
  m[[3]]
}

prepare.data <- function(dist, n, d) {
  index <- c(rep(0, n), rep(1, n), rep(2, n))
  samples <- dist(length(index)) + d*index
  samples ~ as.factor(index)
}

compute.power <- function(dist, n, d, ntrials = 500, alpha = 0.05) {
  n.aov <- 0
  n.kw <- 0
  
  for (k in 1:ntrials) {
    f <- prepare.data(dist, n, d)
    n.aov <- n.aov + (p.aov(f) < alpha)
    n.kw <- n.kw + (p.kw(f) < alpha)
  }

  c(n.aov, n.kw) / ntrials
}

power.plot <- function(dist) {
  n <- 20
  ds <- seq(0.1, 1.0, by = 0.1)
  powers <- c()
  for (d in ds) {
    powers <- c(powers, compute.power(dist, n, d))
  }
  powers <- matrix(powers, ncol = 2, byrow = TRUE)
  plot(ds, powers[,1], type="l", lty=1, 
       xlab = "d", ylab = "power", ylim = c(0, 1))
  points(ds, powers[,1], pch=16)
  lines(ds, powers[,2], type="l", lty=2)
  points(ds, powers[,2], pch=17)
}

# power.plot(rnorm)
# power.plot(rlnorm)
# power.plot(function(n) {rchisq(n, 3)})