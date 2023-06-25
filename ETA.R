alphas <- scan("~/Downloads/alphas.dat", what = 0)

alphas = matrix(as.numeric(alphas), ncol=3, byrow = TRUE)
colnames(alphas) = c("node", "feature", "alpha")
hist(alphas[,"alpha"])

quantile(alphas[,"alpha"], c (0,0.1,0.25,0.5,0.9,1))
mean(alphas[,"alpha"]>1)

for (ftr in 1:5){
  q = quantile(subset(alphas, alphas[,"feature"] == ftr-1)[,"alpha"], c (0,0.1,0.25,0.5,0.9,1))
  print(q)
}
