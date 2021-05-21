#import the dependencies
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

nba = pd.read_csv('data/players_stats.csv')# the nba_2013.csv data 
print(nba.shape)
#print(nba.mean())# Print the first 7 rows of data or first 7 players

#print(nba.loc[:,"FG%"].mean())

#Display pairwise scatte plot 
#sns.pairplot(nba[["FG%","AST", "REB"]])
#plt.savefig('save_as_a_png.png')

#heatmap
#correlation=nba[["FG%","AST", "REB"]].corr()
#sns.heatmap(correlation, annot = True)
#plt.savefig('heatmap.png')

#Make teh cluster of players using Kmeans
kmeans_model = KMeans(n_clusters=5, random_state=1) # creating KMeans model with 5 clusters
good_columns = nba._get_numeric_data().dropna(axis=1) # dropping any missing values
kmeans_model.fit(good_columns) # train the model
labels = kmeans_model.labels_ # Get the label for each player
#print(labels)

# plot players by cluster
pca_2 = PCA(2) # 2-dimmenstional
plot_columns = pca_2.fit_transform(good_columns)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
#plt.show()

#FInd Lebron James

LeBron = good_columns.loc[ nba['Name'] == 'LeBron James',: ]

#Find player Durant
Durant = good_columns.loc[ nba['Name'] == 'Kevin Durant',: ]

#print the players
print(LeBron)
print(Durant)

#Predictions
#Change the dataframes to a list 
Lebron_list = LeBron.values.tolist()
Durant_list = Durant.values.tolist()

#Predict which group LeBron James and Kevin Durant belongs
LeBron_Cluster_Label = kmeans_model.predict(Lebron_list)
Durant_Cluster_Label = kmeans_model.predict(Durant_list)

print(LeBron_Cluster_Label)
print(Durant_Cluster_Label)

#split data
x_train, x_test, y_train, y_test = train_test_split(nba[['FG%']], nba[['AST']], test_size=0.2, random_state=42)

#Create the Linear Regression Model

lr = LinearRegression() # Create the model
lr.fit(x_train, y_train) #Train the model
predictions = lr.predict(x_test) #Make predictions on the test data
#print(predictions)#Predict the numbers of assist 
#print(y_test) #print actual values

#Test Model: Score return the cofficient of determination R^2 of the prediction
lr_confidence = lr.score(x_test, y_test)
print("lr confidence (R^2): ", lr_confidence)