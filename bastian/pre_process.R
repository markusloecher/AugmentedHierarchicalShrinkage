# R-script pre-process

# DNA
Mut = read.table("BRCA_mut.cbt", sep= "\t")
Mut_genes = as.character(Mut[,1][-1])
Mut_patients = as.character(Mut[1,][-1])
Mut = Mut[-1,-1]
Mut = t(Mut)
Mut = matrix(as.numeric(Mut), dim(Mut)[1], dim(Mut)[2])
colnames(Mut) = Mut_genes
rownames(Mut) = Mut_patients

# mRNA
mRNA = read.table("BRCA_mRNA.cct", sep= "\t")
mRNA_genes = as.character(mRNA[,1][-1])
mRNA_patients = as.character(mRNA[1,][-1])
mRNA = mRNA[-1,-1]
mRNA = t(mRNA)
mRNA = matrix(as.numeric(mRNA), dim(mRNA)[1], dim(mRNA)[2])
colnames(mRNA) = mRNA_genes
rownames(mRNA) = mRNA_patients

# Clin

# Clin
Clin = read.table("BRCA_clin.tsi", sep="\t")
Clin_patients = as.character(Clin[1,][-1])
Clin_features = as.character(Clin[,1][-1])
Clin = Clin[-1,-1]
Clin = t(Clin)
colnames(Clin) = Clin_features
rownames(Clin) = Clin_patients


genes = intersect(Mut_genes, mRNA_genes)
patients = intersect(intersect(Mut_patients, mRNA_patients),
                Clin_patients)

mRNA = mRNA[patients,genes]
Mut  = Mut[patients,genes]
colnames(mRNA) = paste(genes,"_","mRNA", sep="")
colnames(Mut) = paste(genes,"_","DNA", sep="")

DATA = cbind(mRNA, Mut)
CLIN = Clin[patients,]

# create target vector
target = CLIN[,"PAM50"]
lumA_ids = which(target=="LumA")
not_lumA_ids = which(target!="LumA")
target = numeric(length(target))
target[lumA_ids] = 1
names(target) = patients

# Final checks for NaNs
dim(DATA)
length(target)

na.ids = which(apply(DATA,1,function(x){any(is.na(x))}))
DATA = DATA[-na.ids,]
target = target[-na.ids]

write.table(DATA, file="OMICS.txt", sep="\t")
write.table(target, file="omics_target.txt", sep="\t")