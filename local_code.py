"""
This folder contains python code too long to put into jupyter
"""
import os
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
import plotly.plotly as py
from plotly.graph_objs import *
import requests

topsize = 30
PLOTLY_API_KEY = "w1LROCX3bYA8amfuLA4g"

def read_folder(path, plot=True):
    dfs = list()
    plot_x = list()
    plot_y = list()
    for file in os.listdir("./data/"):
        if file.endswith(".zip"):
            zip_ref = zipfile.ZipFile(os.path.join(path, file), 'r')
            df = pd.read_csv(zip_ref.extract(zip_ref.filelist[0]))
            df["starttime"] = pd.to_datetime(df["starttime"], format="%Y/%m/%d %H:%M:%S")
            df.index = df["starttime"]
            del df["starttime"]
            dfs.append(df)

            plot_x.append(file.split("-")[0])
            plot_y.append(dfs[-1].shape[0])
            print("Reading {}MB file of {:,} trips recorded on {}"
                  .format(int(os.path.getsize(os.path.join(path, file)) / 1024 / 1024.0),
                          dfs[-1].shape[0], file.split("-")[0]))

    dfs = pd.concat(dfs)

    # delete unused data
    del dfs['stoptime']
    del dfs['start station name']
    del dfs['end station name']



    plt.bar([i for i, x in enumerate(plot_x)], plot_y)
    plt.xticks([i for i, x in enumerate(plot_x)], plot_y)
    plt.ylabel('number of bike rents')
    plt.ylabel('months')
    plt.show()


    return dfs


def make_stations(trips):

    st = pd.DataFrame(np.union1d(trips['start station id'].unique(),
                                 trips['end station id'].unique()),
                      columns=["station_id"])
    locs = list()
    sizes = list()
    for index, row in st.iterrows():
        locs.append(
            [trips[trips['start station id'] == row["station_id"]]['start station longitude'].mean(),
             trips[trips['start station id'] == row["station_id"]]['start station latitude'].mean()]
        )
        sizes.append(trips[trips['start station id'] == row["station_id"]].shape[0])
    st['pos'] = [Point(loc[0], loc[1]) for loc in locs]
    st['orig_trips'] = sizes
    st = st.set_index("station_id")

def get_RTI():
    json = requests.get("https://gbfs.citibikenyc.com/gbfs/en/station_information.json").json()
    data = json['data']['stations']
    station_information = pd.DataFrame(data)


def tt_mtx():
    o = 72
    d = 116
    perc = 0.95
    bins = 10

    matrix = trips.groupby(by=['start station id', 'end station id'])  # pivot the trips for the matrix
    durations = matrix.get_group((o, d)).tripduration  # access a single cell of the matrix
    durations = durations[durations < durations.quantile(perc)]  # filter below the given percentile




def plot_stations(stations):
    data = [Scattermapbox(
        lat=stations.lat, lon=stations.lon,
        mode='markers',
        marker=Marker(
            size=dfs.stations.capacity / 300 * stations.capacity.max(),
            opacity=0.7
        ),
        text="Station: " + str(stations.index),
        hoverinfo='text'
    )]

    layout = Layout(
        title='Stations of the system with their capacities',
        autosize=True,
        hovermode='closest',
        showlegend=False,
        mapbox=dict(
            accesstoken=PLOTLY_API_KEY,
            bearing=0,
            center=dict(
                lat=40.73,
                lon=-73.93
            ),
            pitch=0,
            zoom=10,
            style='light'
        ),
    )
    fig = dict(data=data, layout=layout)
    py.iplot(fig, filename='CITI stations')

if __name__ == "__main__":
    trips = read_folder("./data/")
    stations = make_stations(trips)
    plot_stations()