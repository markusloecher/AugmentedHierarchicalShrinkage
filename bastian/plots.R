#plots
FI_no_hsc = read.table("FI_no_hsc")
FI_hsc = read.table("FI_hsc")
FI_ehsc = read.table("FI_ehsc")

DATA = read.table("OMICS.txt")
genes = colnames(DATA)

# Density Plot
plot(density(scale(FI_no_hsc )), col="black")
lines(density(scale(FI_hsc )), col="blue")
lines(density(scale(FI_ehsc )), col="green")

plot(scale(FI_no_hsc ), col="black", pch=19)
points(scale(FI_hsc ), col="blue", pch=19)
points(scale(FI_ehsc ), col="green", pch=19)

