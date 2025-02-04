### Loading the Dataset
We will load in all our packages and our dataset all at once in the beginning.


```python
# Loading all packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
df = pd.read_csv('spotify_history.csv')

# Display first rows of code
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>spotify_track_uri</th>
      <th>ts</th>
      <th>platform</th>
      <th>ms_played</th>
      <th>track_name</th>
      <th>artist_name</th>
      <th>album_name</th>
      <th>reason_start</th>
      <th>reason_end</th>
      <th>shuffle</th>
      <th>skipped</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2J3n32GeLmMjwuAzyhcSNe</td>
      <td>2013-07-08 02:44:34</td>
      <td>web player</td>
      <td>3185</td>
      <td>Say It, Just Say It</td>
      <td>The Mowgli's</td>
      <td>Waiting For The Dawn</td>
      <td>autoplay</td>
      <td>clickrow</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1oHxIPqJyvAYHy0PVrDU98</td>
      <td>2013-07-08 02:45:37</td>
      <td>web player</td>
      <td>61865</td>
      <td>Drinking from the Bottle (feat. Tinie Tempah)</td>
      <td>Calvin Harris</td>
      <td>18 Months</td>
      <td>clickrow</td>
      <td>clickrow</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>487OPlneJNni3NWC8SYqhW</td>
      <td>2013-07-08 02:50:24</td>
      <td>web player</td>
      <td>285386</td>
      <td>Born To Die</td>
      <td>Lana Del Rey</td>
      <td>Born To Die - The Paradise Edition</td>
      <td>clickrow</td>
      <td>unknown</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5IyblF777jLZj1vGHG2UD3</td>
      <td>2013-07-08 02:52:40</td>
      <td>web player</td>
      <td>134022</td>
      <td>Off To The Races</td>
      <td>Lana Del Rey</td>
      <td>Born To Die - The Paradise Edition</td>
      <td>trackdone</td>
      <td>clickrow</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0GgAAB0ZMllFhbNc3mAodO</td>
      <td>2013-07-08 03:17:52</td>
      <td>web player</td>
      <td>0</td>
      <td>Half Mast</td>
      <td>Empire Of The Sun</td>
      <td>Walking On A Dream</td>
      <td>clickrow</td>
      <td>nextbtn</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



### Checking for Missing Data
Based on the first view of the dataset, this data is fairly simple in terms of columns as it explores basic data regarding people's Spotify listening history.
Now we check for null values in the dataset.


```python
df.isnull().sum()
```




    spotify_track_uri      0
    ts                     0
    platform               0
    ms_played              0
    track_name             0
    artist_name            0
    album_name             0
    reason_start         143
    reason_end           117
    shuffle                0
    skipped                0
    dtype: int64



### Summary Statistics
There is a decent number of null values for the reason columns, so before deciding whether or not to remove them we are going to check the summary statistics to see the numeric values.


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ms_played</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.498600e+05</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.283166e+05</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.178401e+05</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.795000e+03</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.388400e+05</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.185070e+05</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.561125e+06</td>
    </tr>
  </tbody>
</table>
</div>



### Cleaning the Duplicate Data
The only null values are all in the reason columns and there is thousands of rows in this dataset, so we are dropping all duplicates so that there are unique values.


```python
# Drop duplicates
df.drop_duplicates(inplace=True)

# Convert timestamps to simple datetime datatype
df['ts'] = pd.to_datetime(df['ts'])

df.info()

```

    <class 'pandas.core.frame.DataFrame'>
    Index: 148675 entries, 0 to 149859
    Data columns (total 11 columns):
     #   Column             Non-Null Count   Dtype         
    ---  ------             --------------   -----         
     0   spotify_track_uri  148675 non-null  object        
     1   ts                 148675 non-null  datetime64[ns]
     2   platform           148675 non-null  object        
     3   ms_played          148675 non-null  int64         
     4   track_name         148675 non-null  object        
     5   artist_name        148675 non-null  object        
     6   album_name         148675 non-null  object        
     7   reason_start       148532 non-null  object        
     8   reason_end         148558 non-null  object        
     9   shuffle            148675 non-null  bool          
     10  skipped            148675 non-null  bool          
    dtypes: bool(2), datetime64[ns](1), int64(1), object(7)
    memory usage: 11.6+ MB
    

### Creating Data Visualizations
Now, we will create some graphs to visually explore more about this Spotify dataset to see some potential trends.

#### Graph 1 (Top 10 Most Played Songs)
This is calculated by looking at the play count of the songs (number of times the song was played).


```python
top_songs = df['track_name'].value_counts().head(10)

plt.figure(figsize=(12,6))
top_songs.plot(kind='bar', color='blue')
plt.xlabel('Track Name')
plt.ylabel('Play Count')
plt.title('Top 10 Most Played Songs')
plt.xticks(rotation=45)
plt.show()

```


    
![png](output_10_0.png)
    


Ode To The Mets is the most played song in this dataset.

#### Graph 2 (Top Artists)
This is created by also using the play count, but for artists instead.


```python
top_artists = df['artist_name'].value_counts().head(10)

plt.figure(figsize=(12,6))
top_artists.plot(kind='bar', color='green')
plt.xlabel('Artist Name')
plt.ylabel('Play Count')
plt.title('Top 10 Most Played Artists')
plt.xticks(rotation=45)
plt.show()

```


    
![png](output_13_0.png)
    


The Beatles are the most played band (artist) in this dataset as well. (not very surprising)

#### Graph 3 (Hourly Listening Activity)
Now, let's see what hour in the day is the most popular for listening to music.

Since we already changed the ts (timestamp) column to datetime format, we will first need to extract the hour from the timestamp before continuing with the creation of the graph.


```python
# Extract hour from timestamp
df['hour'] = df['ts'].dt.hour


# Count the number of plays per hour
hourly_activity = df['hour'].value_counts().sort_index()

plt.figure(figsize=(10,5))
sns.lineplot(x=hourly_activity.index, y=hourly_activity.values, marker='o')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Songs Played')
plt.title('Listening Activity by Hour of the Day')
plt.xticks(range(0,24))
plt.grid(True)
plt.show()

```


    
![png](output_17_0.png)
    


People seem to love listening to music the most at night.

### Understanding the Dataset
This dataset contains people's Spotify listening history, including track names, artist names, album names, and other basic columns as seen above.

#### Key Findings
Based on the graphs we created, we can see that <b>Ode To The Mets<b/> is the most played track in the dataset.

<b>The Beatles<b/> are the most played band in the dataset.

And that <b>people like listening to music at night the most.<b/>


```python

```
