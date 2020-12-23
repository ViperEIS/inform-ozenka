import folium
from geopy.distance import geodesic
import json
# from kneed import KneeLocator
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import plotly.graph_objects as go
import re
import requests
from sklearn.cluster import KMeans


def __is_inside(obj, polygon):
    '''
    Is object inside the
    polygon or not
    '''
    obj_la = float(obj['Latitude'][:1])
    obj_lo = float(obj['Longitude'][:1])

    poly = Polygon(polygon[['Longitude', 'Latitude']], False)
    is_inside = poly.contains_point((obj_lo, obj_la))
    return is_inside


def check_distr(obj, city_name):
    '''
    Returns the district of the
    city in which the object
    is located
    '''
    js = open("{}.js".format(city_name), "r", encoding='utf-8').readlines()[:2]

    distrs = json.loads(js[0][re.search('"features', str(js)).end():-1])

    for distr in distrs:
        df_distr = pd.DataFrame(distr['geometry']['coordinates'][0], columns=['Longitude', 'Latitude'])
        if __is_inside(obj, df_distr):
            return distr['properties']['description']
    return 'Unknown'


def address_to_coord(address, city):
    '''
    Convert address to longitude and
    latitude (return tuple (long, lat))
    '''
    address = address if city in address else city + ' ' + address

    url = "http://search.maps.sputnik.ru/search"

    payload = ""
    headers = {'cookie': 'session_id=CvLubV%2BJmi5SZTO4igJwAg%3D%3D'}

    querystring = {"q": address}
    response = requests.request(
        "GET", url, data=payload, headers=headers, params=querystring)
    data = json.loads(response.text)

    try:
        return [data['result'][0]['position']['lat'], data['result'][0]['position']['lon']]
    except IndexError:
        return []


def convert_df_address_to_coord(df, city, type_):
    '''
    Changes addresses ("Address") to coordinates 
    ("Latitude" and "Longitude") in DataFrame
    and adds column named "City" and 
    "Type" (type of point of interest)
    '''
    arr = df['Address'].apply(address_to_coord, args=(city,))
    df['Latitude'] = [(lambda el: el[0] if len(el) != 0 else 0)(el) for el in arr]
    df['Longitude'] = [(lambda el: el[1] if len(el) != 0 else 0)(el) for el in arr]
    df['City'], df['Type'] = city, type_
    df.drop(df[(df['Latitude'] == 0.0) & (df['Longitude'] == 0.0)].index, inplace=True)
    df.reset_index(inplace=True)
    df.drop(columns='index', inplace=True)


def __walking_dist__time(la1, lo1, la2, lo2, dist_time):
    '''
    Returns walking distance or time
    '''
    r_text = requests.get('http://192.168.1.89:8989/route?point={},{}&point={},{}&vehicle=foot'.format(
        la1,
        lo1,
        la2,
        lo2)
    ).text

    data = json.loads(r_text)
    if dist_time:
        return round(data['paths'][0]['time'] / 60000, 2)
    else:
        return round(data['paths'][0]['distance'], 2)


def __street_map_plotting(is_current_cluster, cdv, colors, oc):
    '''
    Creating a street map with markers of
    clusters (all clusters or only object's cluster)
    '''

    street_map = folium.Map(
        location=[cdv['Latitude'].mean(
        ), cdv['Longitude'].mean()],
        zoom_start=10
    )

    if is_current_cluster:
        for _, obj_m in cdv.iterrows():
            if obj_m['cluster_label'] in {oc, 8}:
                folium.Marker(
                    location=[(obj_m['Latitude']), obj_m['Longitude']],
                    popup=obj_m['Title'],
                    icon=folium.Icon(color=colors[obj_m['cluster_label']])
                ).add_to(street_map)
    else:
        for _, obj_m in cdv.iterrows():
            folium.Marker(
                location=[(obj_m['Latitude']), obj_m['Longitude']],
                popup=obj_m['Title'],
                icon=folium.Icon(color=colors[obj_m['cluster_label']])
            ).add_to(street_map)
    return street_map


def __walking_dist__time__title(obj, poi_s):
    '''
    Returns walking dist, time, title of
    place of interest
    '''

    # Sample from data with id for merge ---------------------------------------------------------------------------------------------------
    poi_s['parcelid'] = pd.Series(range(poi_s.shape[0]))
    X = poi_s.loc[:, ['parcelid', 'Latitude', 'Longitude']]

    # Searching optimal nuber of clusters --------------------------------------------------------------------------------------------------
    # K_clusters = range(1, 20)
    # kmeans = [KMeans(n_clusters=i) for i in K_clusters]
    # lat_long = X[['Latitude', 'Longitude']]
    # score = [kmeans[i].fit(lat_long).score(lat_long)
    #         for i in range(len(kmeans))]
    # kn = KneeLocator(K_clusters, score, curve='concave',
    #                 direction='increasing').knee

    # Visualize the Results
    # centers = kmeans.cluster_centers_  # Coordinates of cluster centers.
    # labels = kmeans.predict(X[['Latitude', 'Longitude']])  # Labels of each point

    # X.plot.scatter(x = 'Latitude', y = 'Longitude', c=labels, s=50, cmap='viridis')
    # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

    # Model building and predicting --------------------------------------------------------------------------------------------------------
    # (kn instead of 8, when searching optimal nuber of clusters runs)
    kmeans = KMeans(n_clusters=8, init='k-means++')
    kmeans.fit(X[['Latitude', 'Longitude']])
    X['cluster_label'] = kmeans.predict(X[['Latitude', 'Longitude']])

    # Merging with general data ------------------------------------------------------------------------------------------------------------
    X = X[['parcelid', 'cluster_label']]
    clustered_data = poi_s.merge(X, left_on='parcelid', right_on='parcelid')

    clustered_data.drop(columns='parcelid', inplace=True)

    # Visualize the Results (without searching optimal number of clusters) -----------------------------------------------------------------
    clustered_data_for_visualisation = clustered_data.append(
        {'Longitude': obj['Longitude'], 'Latitude': obj['Latitude'], 'Title': 'Property object', 'cluster_label': 8},
        ignore_index=True)

    clst_colors = {0: 'gray', 1: 'green', 2: 'blue', 3: 'beige',
                   4: 'pink', 5: 'purple', 6: 'orange', 7: 'black', 8: 'red'}

    plt.figure(figsize=(20, 20))
    scatter = plt.scatter(clustered_data_for_visualisation['Latitude'], clustered_data_for_visualisation['Longitude'],
                          c=clustered_data_for_visualisation['cluster_label'].replace(clst_colors), s=50,
                          cmap='viridis')
    plt.grid(True)
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

    obj_la = float(obj['Latitude'][:1])
    obj_lo = float(obj['Longitude'][:1])

    for lat, lon in zip(centers[:, 0], centers[:, 1]):
        plt.plot([lat, obj_la], [lon, obj_lo], color='r')
        plt.annotate(round(geodesic([obj_la, obj_lo], [lat, lon]).meters, 2),
                     xy=(np.mean([lat, obj_la]), np.mean([lon, obj_lo])))

    obj_clst = kmeans.predict(obj[['Latitude', 'Longitude']])[
        0]
    street_map = (__street_map_plotting(False, clustered_data_for_visualisation, clst_colors, obj_clst),
                  __street_map_plotting(True, clustered_data_for_visualisation, clst_colors, obj_clst))

    # Getting cluster of the object --------------------------------------------------------------------------------------------------------
    clustered_data = clustered_data[clustered_data['cluster_label'] == obj_clst]

    # Adds a column with walking distance to object in poi_s set ---------------------------------------------------------------------------
    clustered_data['dist_to_obj_walking'] = list(
        map(
            lambda a: __walking_dist__time(
                obj_la,
                obj_lo, a[0], a[1], False),
            np.array(clustered_data[['Latitude', 'Longitude']])
        )
    )

    min_dist = min(
        clustered_data['dist_to_obj_walking']
    )

    return clustered_data[['Title', 'Latitude', 'Longitude', 'dist_to_obj_walking']][
               clustered_data['dist_to_obj_walking'] == min_dist], street_map


def nearest_poi(obj, poi_s):  # obj - DataFrame [1:2]; poi_s - DataFrame [:3]
    '''
    Returns distance from current porperty (obj)
    to point of interest (poi)
    (meters), walking distance (meters), amount of time to poi (minutes), title of poi,
    street map with markers (folium),
    object latitude, longitude
    poi latitude, longitude
    '''

    walking_dist__la__lo__time__title, street_map = __walking_dist__time__title(
        obj, poi_s)

    obj_la = float(obj['Latitude'][:1])
    obj_lo = float(obj['Longitude'][:1])
    poi_la = float(walking_dist__la__lo__time__title['Latitude'][:1])
    poi_lo = float(walking_dist__la__lo__time__title['Longitude'][:1])

    dist = round(geodesic([obj_la, obj_lo],
                          [poi_la, poi_lo]).meters, 2)

    time = __walking_dist__time(obj_la, obj_lo, poi_la, poi_lo, True)

    return dist, walking_dist__la__lo__time__title['dist_to_obj_walking'][:1], time, [obj_la, obj_lo], [poi_la, poi_lo], \
           walking_dist__la__lo__time__title['Title'][:1], street_map


def map_constructor_adjust(csv_poi, city, type_):
    '''
    Returns csv file from yandex
    map construcor in
    appropriate form
    '''
    csv_data = pd.read_csv(csv_poi)[['Описание', 'Долгота', 'Широта', 'Подпись']]
    csv_data['City'], csv_data['Type'] = city, type_
    csv_data['Описание'] = [el.split('\n')[1] for el in csv_data['Описание']]
    return pd.DataFrame(
        np.array(
            csv_data[['Описание', 'Долгота', 'Широта', 'Подпись', 'City', 'Type']]),
            columns=['Address', 'Longitude', 'Latitude', 'Title', 'City', 'Type']
        )
