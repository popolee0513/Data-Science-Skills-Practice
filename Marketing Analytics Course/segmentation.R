set.seed(1990)
library(NbClust)
library(mclust)
library(gmodels)
data <- read.csv(file.choose())
head(data)
data <- scale(data[,2:6])
head(data)
dist <- dist(data, method = "euclidean")
set.seed(1990)
clust <- hclust(dist, method = "ward.D2")
plot(clust)
h_cluster <- cutree(clust, 4)
rect.hclust(clust, k=4, border="red")
hclust_summary <- aggregate(data,by=list(h_cluster),FUN=mean)
hclust_summary
set.seed(1990)
NbClust(data=data, min.nc=2, max.nc=15, index="all", method="ward.D2")
h_cluster <- cutree(clust, 2)
rect.hclust(clust, k=2, border="red")
hclust_summary <- aggregate(data,by=list(h_cluster),FUN=mean)
hclust_summary
set.seed(1990)
Cluster3 <-kmeans(data, 3, iter.max=100,nstart=100)
table(Cluster3$cluster)
kmeans_summary <- aggregate(data,by=list(Cluster3$cluster),FUN=mean)
kmeans_summary
set.seed(1990)
NbClust(data=data, min.nc=2, max.nc=15, index="all", method="kmeans")
set.seed(1990)
lca_clust<-Mclust(data,verbose=F,modelNames ="EEI")
summary(lca_clust)
lca_clusters <- lca_clust$classification
lca_clust_summary <- aggregate(data,by=list(lca_clusters),FUN=mean)
lca_clust_summary




