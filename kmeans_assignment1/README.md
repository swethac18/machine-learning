# K Means clustering
Dataset: cars.csv <br>
Number of features: 6 <br>
Features: ['mpg', 'cylinders', 'cubicinches', 'hp', 'weightlbs', 'time-to-60'] <br>

Optimal Cluster count K  by Elbow method: 5 <br>
Elbow method plot shows that k-means-score steeply increases as we increase K value and then starts to flatten out at K=5 
Optimal Cluster count K  by Silhoutte score : 2 <br>
Silhoutte score is a function of how dense the clusters are and how far away clusters are
they range between -1 to 1
Silhoutte score promotes dense clusters that are well seperated

Silhoutte  plot shows that there are two peaks at K=2 and K =5 

We can choose either K =2 or K =5. Inorder to be more specific and be aligned with elbow method, we can assume that there are 5 distinct clusters in the dataset

