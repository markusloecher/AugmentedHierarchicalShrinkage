setwd("/Users/loecherm/Documents/GitHub/AugmentedHierarchicalShrinkage/markus/arne_copy/experiments/power_simulation")

alphas <- read.csv("~/Documents/GitHub/AugmentedHierarchicalShrinkage/markus/arne_copy/experiments/power_simulation/alphas.txt", header=FALSE)
colnames(alphas) = c("node", "feature", "alpha")
alphas[,"feature"] =alphas[,"feature"] +1
alphas[,"depth"] = cut(alphas[,"node"], breaks = c(-1,0,2,10,30,100), include.lowest = F)


library(ggplot2)

hist(log10(alphas[,"alpha"]))
hist(alphas[,"node"])

ggplot(alphas, aes(alpha, fill = depth)) + 
  geom_histogram(alpha = 0.75)+#, position = "dodge") + 
  #geom_density() +
  facet_wrap(~feature) +
  #scale_x_log10( ) +
  scale_y_log10( )
  #scale_y_sqrt( )
  
  
quantile(alphas[,"alpha"], c (0,0.1,0.25,0.5,0.9,1))
mean(alphas[,"alpha"]>1)

for (ftr in 1:5){
  q = quantile(subset(alphas, alphas[,"feature"] == ftr-1)[,"alpha"], c (0,0.1,0.25,0.5,0.9,1))
  print(q)
}
