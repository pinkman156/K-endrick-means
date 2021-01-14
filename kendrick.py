import pandas as pd
import numpy as np
import plotly
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import lyricsgenius
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import plotly.offline as py
import plotly.graph_objs as go



genius = lyricsgenius.Genius("DLz5rjGa263arxNO1gin2tpQxqo2N5FHFSuKNswHB8hAFb6g3Aty794TMRnOQkvR")

genius.verbose = False # Turn off status messages
genius.remove_section_headers = True # Remove section headers (e.g. [Chorus]) from lyrics when searching
genius.skip_non_songs = False # Include hits thought to be non-songs (e.g. track lists)
genius.excluded_terms = ["(Live)"] # Exclude songs with these words in their title



filename = './kendrick.csv' 
kendrick_df = pd.read_csv(filename)
songs = kendrick_df['track_name']
print(songs.head())



#Create heatmap to see the relationship of each variable's relationship
plt.subplots(figsize=(12,9))
ax = plt.axes()
ax.set_title("Echo Nest Correlation")
corr = kendrick_df.corr()
sns.heatmap(corr,
            cmap="YlGnBu",
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
# plt.show() 

df_cluster_features = kendrick_df.drop(["track_name", "album_name", "album_release_year","album_release_date", "time_signature","liveness"], axis=1)
print(df_cluster_features.head(1))


scaler = MinMaxScaler()
print(scaler.fit(df_cluster_features))
print(scaler.transform(df_cluster_features))

k_means = KMeans(n_clusters=3, random_state=0)
kmeans = k_means.fit(scaler.transform(df_cluster_features))
kendrick_df['cluster'] = kmeans.labels_
# print(kendrick_df.head())



trace1 = go.Scatter3d(
    x=kendrick_df["tempo"],
    y=kendrick_df["danceability"],
    z=kendrick_df["speechiness"],
    mode='markers',
    text=kendrick_df["track_name"],
    marker=dict(
        size=12,
        color=kendrick_df["cluster"], 
        colorscale='Viridis',  
        opacity=0.8
    )
)
#print(trace1)
data = [trace1]
layout = go.Layout(
    showlegend=False,
    title="Kendrick Songs Clustering",
    scene = dict(
        xaxis = dict(title='Tempo Score'),
        yaxis = dict(title="Danceability Score"),
        zaxis = dict(title="Speechiness Score"),
    ),
    width=1000,
    height=900,
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='3d-scatter-colorscale')

#Seperate Clusters
cluster_1= kendrick_df.loc[kendrick_df['cluster'] == 1]
cluster_2= kendrick_df.loc[kendrick_df['cluster'] == 2]
cluster_0= kendrick_df.loc[kendrick_df['cluster'] == 0]

#cluster
print(cluster_0[['track_name', 'album_name']])
print(cluster_1[['track_name', 'album_name']])
print(cluster_2[['track_name', 'album_name']])