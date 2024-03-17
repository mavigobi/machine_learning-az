# K-MEANS MODEL
# R - mavigobi

# K-MEANS CLUSTERING


# LOAD DATASET
dataset = read.csv('Mall_Customers.csv')
X = dataset[,4:5]


# METHOD OF ELBOW GRAPH REPRESENTATION FOR K-MEANS CLUSTERING
# The goal is to find the optimal point k
set.seed(6) #random state
wcss = vector()
for(i in 1:10){
  wcss[i] <- sum(kmeans(X,i)$withinss)
}


# VISUALITATION OF ELBOW GRAPH
plot(1:10,wcss, type ="b", main = "Método del codo",
     xlab = "Número de clusters",ylab = "WCSS")


# METHOD OF K-MEANS 
kmeans <- kmeans(X,5,iter.max = 300, nstart = 10)

# VISUALITATION OF DATA SET 
library(cluster)
clusplot(X,
         kmeans$cluster,
         lines=0, #not line
         shade = TRUE, # the ellipses are shaded in relation to their density
         color = TRUE,
         labels=2, #all points and ellipses are labelled in the plot
         plotchar = FALSE, # the plotting symbols not differ for points belonging to different clusters,
         span = TRUE, # each cluster is represented by the ellipse with smallest area containing all its points
         main = "Clustering of clients",
         xlab ="Annual Income (k$)",
         y_lab = "Spending Score (1-100)"
         )
      