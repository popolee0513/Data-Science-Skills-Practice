data<-read.csv(file.choose())
head(data)
data<-data[,-1]
head(data)
eigenval=eigen(cor(data))
eigenval$values
install.packages("psych")
library("psych")
fit <- principal(data, nfactors=3,rotate="varimax")
fit$loadings
fit$weights
reduced_data = cbind(data,fit$scores)
head(reduced_data)