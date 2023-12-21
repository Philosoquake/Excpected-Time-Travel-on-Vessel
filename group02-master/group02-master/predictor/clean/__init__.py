import numpy as np
import pandas as pd  # type: ignore
import datetime as dt
import math


def time_to_POSIX_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    '''
        Converts the time columns to POSIX timestamps

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to convert

        Returns
        -------
        pd.DataFrame
            The converted dataframe
    '''
    new_df = df.copy()
    labels = ['StartTime', 'EndTime', 'time']
    # Check if already converted
    if new_df[labels[0]].dtype == np.dtype('int64'):
        return new_df
    for label in labels:
        new_df[label] = new_df[label].apply(lambda x: x.timestamp())
    return new_df


def time_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()
    labels = ['StartTime', 'EndTime', 'time']
    # Check if already converted
    if new_df[labels[0]].dtype == np.dtype('datetime64[ns]'):
        return new_df
    for label in labels:
        new_df[label] = new_df[label].apply(
            lambda x: dt.datetime.fromtimestamp(x))
    return new_df


def print_change(original: int, new: int) -> None:
    print(
        f"removed {original - new} entries. Original: {original}, New: {new}")
    original = new


# pick necessary Attributes for Machine Learning
def pick_initial_features(df: pd.DataFrame) -> pd.DataFrame:
    features_to_drop = [
        'TripID', 'MMSI', 'ID',
        'StartPort', 'EndPort', 'Destination', 'Name', 'Callsign', 'AisSourcen',
        'StartLatitude', 'StartLongitude', 'StartTime'
    ]
    new_df = df.copy()
    new_df = new_df.drop(
        features_to_drop, axis=1)
    return new_df


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    new_df = df.copy()

    # new_df = prepare_COG_TH(new_df)
    # new_df = prepare_Length_Breadth_Draught(new_df)
    # new_df = prepare_distance(new_df)
    new_df = prepare_time(new_df)

    new_df = new_df.drop(['TripID', 'MMSI', 'ID'], axis=1)
    new_df = new_df.drop(['StartPort', 'EndPort', 'Destination',
                         'Name', 'Callsign', 'AisSourcen'], axis=1)

    new_df = new_df.drop(
        ['StartLatitude', 'StartLongitude', 'StartTime'], axis=1)

    # x features
    # y is to be predicted
    x_data = new_df.drop("time_remaining", axis=1)
    y_data = new_df["time_remaining"]
    return x_data, y_data
# 584, 342t, NaN, ?


def format_dates(df: pd.DataFrame) -> pd.DataFrame:
    strs = ['time', 'StartTime', 'EndTime']
    new_df = df.copy()
    for s in strs:
        new_df[s] = pd.to_datetime(df[s]).dt.strftime('%Y-%m-%d %H:%M')
        new_df[s] = pd.to_datetime(
            new_df[s], infer_datetime_format=True, errors="coerce")
    return new_df


def make_trip_consistent(df: pd.DataFrame, print_info=False):
    '''
        Makes a trip consistent by replacing inconsistent values with the most common value in the trip.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to make consistent.
        print_info : bool, optional
            Whether to print information about the consistency of the dataframe, by default False

        Returns
        -------
        pd.DataFrame
            The consistent dataframe.
    '''
    new_df = df.copy()
    labels = ['shiptype', 'Length', 'Breadth',
              'Draught',
              'Destination', 'Name',
              # 'AisSourcen',
              'StartPort', 'EndPort',
              'Callsign', 'MMSI', 'StartLatitude', 'StartLongitude', 'EndLatitude', 'EndLongitude']
    inconsistent_count: dict = {
        label: 0 for label in labels
    }
    trip_ids = new_df['TripID'].value_counts().index
    for trip_id in trip_ids:
        df_trip_slice: pd.DataFrame = new_df[new_df['TripID'] == trip_id]
        for label in labels:
            slice_value_counts = df_trip_slice[label].value_counts()
            if slice_value_counts.shape[0] > 1:
                inconsistent_count[label] += 1
                new_df.loc[new_df['TripID'] == trip_id,
                           label] = slice_value_counts.index[0]
    for label, count in inconsistent_count.items():
        if count > 0:
            print(f"{count} trips had inconsistent '{label}' values. Replaced with the most frequent value in a trip.") if print_info and count > 0 else None

    return new_df


def drop_entry_if_within_radius(df: pd.DataFrame, radius_in_km: int, print_info=False):
    '''
        Drops entries if the distance between the start and end coordinates is less than 5km.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to drop entries from.
        print_info : bool, optional
            Whether to print info about the amount of dropped entries. The default is False.

        Returns
        -------
        pd.DataFrame
            The dataframe with the dropped entries.
    '''

    new_df = df.copy()
    mask = new_df.apply(
        lambda row:
            haversine_distance(
                row['Latitude'],
                row['Longitude'],
                row['StartLatitude'],
                row['StartLongitude']
            ) < radius_in_km
            or
            haversine_distance(
                row['Latitude'],
                row['Longitude'],
                row['EndLatitude'],
                row['EndLongitude']
            ) < radius_in_km,
        axis=1
    )
    df_filtered = new_df[~mask]
    return df_filtered


def drop_entry_amount_per_trip_outliers(df: pd.DataFrame, print_info=False):
    '''
        Drops entries with a trip id that has an outlying amount of entries.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to drop the entries from.
        print_info : bool, optional
            Whether to print info about the amount of dropped entries. The default is False.

        Returns
        -------
        pd.DataFrame
            The dataframe with the dropped entries.
    '''
    new_df = df.copy()
    trip_id_val_counts = new_df['TripID'].value_counts()
    data = trip_id_val_counts.values
    Q1 = np.percentile(data, 25)
    Q2 = np.percentile(data, 50)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    factor = 1.5
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    data_min = np.min(data)
    data_max = np.max(data)

    new_df = new_df[new_df['TripID'].isin(
        trip_id_val_counts.index[trip_id_val_counts.values <= upper_bound])]
    new_df = new_df[new_df['TripID'].isin(
        trip_id_val_counts.index[trip_id_val_counts.values >= lower_bound])]
    new_df = new_df.reset_index(drop=True)

    if print_info:
        print('Minimum:', data_min)
        print('Lower Bound:', lower_bound)
        print('Q1:', Q1)
        print('Median:', Q2)
        print('Q3:', Q3)
        print('Upper Bound:', upper_bound)
        print('Maximum:', data_max)
        print('IQR:', IQR)

    return new_df


def deg_to_sin(deg: int) -> float:
    '''
        Converts degrees to sine.

        Parameters
        ----------
        deg : int
            The degrees to convert.

        Returns
        -------
        float
            The sine of the degrees.
    '''
    rad = np.deg2rad(deg)
    return np.sin(rad)


def deg_to_cos(deg: int) -> float:
    '''
        Converts degrees to cosine.

        Parameters
        ----------
        deg : int
            The degrees to convert.

        Returns
        -------
        float
            The cosine of the degrees.
    '''
    rad = np.deg2rad(deg)
    return np.cos(rad)


def haversine_distance(start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> float:
    '''
        This function takes two coordinates and calculates the distance between them using the Haversine formula.

        Parameters
        ----------
        start_lat: float
            The latitude of the starting point.
        start_lon : float
            The longitude of the starting point.
        end_lat : float
            The latitude of the ending point.
        end_lon : float
            The longitude of the ending point.

        Returns
        -------
        distance : float
            The distance between the two coordinates in kilometers.
    '''
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(
        math.radians, [start_lat, start_lon, end_lat, end_lon])

    # Earth radius in kilometers
    R = 6371.0

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = math.sin(dlat/2)**2 + math.cos(lat1) * \
        math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    # Distance in kilometers
    distance = R * c

    return distance


def prepare_distance(df: pd.DataFrame) -> pd.DataFrame:
    '''
        This function takes a dataframe with 'Latitude', 'Longitude', 'EndLatitude' and 'EndLongitude' columns.
        It calculates the distance between the two points and returns a dataframe with a new column 'distance'.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the labels 'Latitude', 'Longitude', 'EndLatitude' and 'EndLongitude'.

        Returns
        -------
        new_df : pd.DataFrame
            The dataframe with a new column 'distance' containing the distance between the two points.
    '''
    new_df = df.copy()
    labels = ['Latitude', 'Longitude', 'EndLatitude', 'EndLongitude']
    new_df['distance_remaining_in_km'] = new_df[labels].apply(
        lambda x: haversine_distance(*x), axis=1)
    return new_df


def prepare_time(df: pd.DataFrame) -> pd.DataFrame:
    '''
        This function takes a dataframe with 'StartTime', 'EndTime' and 'time' columns.
        It calculates the time remaining until the end of the trip and adds it as a new column as 'time_remaining'.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the labels 'StartTime', 'EndTime' and 'time'.

        Returns
        -------
        new_df : pd.DataFrame
            The dataframe with a new column 'time_remaining' containing the time remaining until the end of the trip.
    '''
    new_df = df.copy()
    new_df["time_remaining"] = (
        new_df["EndTime"] - new_df["StartTime"]) - (new_df["time"] - new_df["StartTime"])
    return new_df


def prepare_COG_TH(df: pd.DataFrame) -> pd.DataFrame:
    '''
        This function takes a dataframe with 'COG' and 'TH' columns.
        It takes the mean of the two columns and then converts it to sin and cos.

        The reasoning is that COG and TH are cyclic variables and also highly correlated.
        It is better to have only one label that represents both of them. In the end,
        we will have two labels, 'mean sin' and 'mean cos', but that is because this is the best
        way to represent cyclic variables.
    '''
    new_df = df.copy()
    labels = ['COG', 'TH']
    mean_direction = new_df[labels].mean(axis=1)
    new_df['mean_dir_sin'] = mean_direction.apply(lambda x: deg_to_sin(x))
    new_df['mean_dir_cos'] = mean_direction.apply(lambda x: deg_to_cos(x))
    new_df = new_df.drop(labels, axis=1)
    return new_df


def prepare_Length_Breadth_Draught(df: pd.DataFrame) -> pd.DataFrame:
    '''
        This function takes a dataframe with 'Length', 'Breadth' and 'Draught' columns.
        It takes the mean of the three columns and then puts it in a new column 'mean_size'.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe containing the labels 'Length', 'Breadth' and 'Draught'.

        Returns
        -------
        new_df : pd.DataFrame
            The dataframe with a new column 'mean_size' containing the mean of the three columns.
    '''

    new_df = df.copy()
    labels = [
        'Length',
        'Breadth',
        'Draught'
    ]
    mean_size = new_df[labels].mean(axis=1)
    new_df['mean_size'] = mean_size
    new_df = new_df.drop(labels, axis=1)
    return new_df


def clean_up(df: pd.DataFrame) -> pd.DataFrame:
    '''
        This function takes a dataframe and cleans it up.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to be cleaned.

        Returns
        -------
        new_df : pd.DataFrame
            The cleaned dataframe.
    '''
    new_df = df.copy()
    new_df = new_df.drop_duplicates()
    new_df = new_df.drop(columns=['d']) if 'd' in new_df else new_df
    new_df = new_df.drop_duplicates(subset=['ID'], keep='first')
    new_df = new_df.replace("?", np.nan)
    new_df['Draught'] = pd.to_numeric(new_df['Draught'])
    new_df['COG'] = new_df['COG'].apply(
        lambda x: np.nan if x >= 360 else x)
    new_df['TH'] = new_df['TH'].apply(
        lambda x: np.nan if x > 359 else x)

    time_columns = ['StartTime', 'EndTime', 'time']
    for time_column in time_columns:
        new_df[time_column] = pd.to_datetime(
            new_df[time_column],
            # infer_datetime_format=True,
            errors="coerce"
        ).apply(lambda x: x.timestamp())

    new_df = make_trip_consistent(new_df, print_info=False)

    new_df.loc[(new_df['shiptype'] >= 0) & (
        new_df['shiptype'] <= 19), 'shiptype'] = np.nan
    new_df.loc[(new_df['shiptype'] >= 99), 'shiptype'] = np.nan

    new_df = new_df.dropna(subset=['shiptype', 'Draught', 'COG', 'TH'])
    for i in range(1, 3):
        new_df = drop_entry_amount_per_trip_outliers(new_df, print_info=False)

    # drop entry if Latitude, Longitude is within 5km of StartLatitude,StartLongitude.
    new_df = drop_entry_if_within_radius(new_df, 20, print_info=False)

    labels = [
        'TripID', 'MMSI', 'ID',
        'StartLatitude', 'StartLongitude',
        'EndLatitude', 'EndLongitude',
        'Latitude', 'Longitude',
        'StartTime', 'EndTime', 'time',
        'StartPort', 'EndPort',
        'shiptype', 'Length', 'Breadth', 'Draught',
        'SOG', 'COG', 'TH',
        'Destination', 'Name', 'Callsign',
        'AisSourcen'
    ]
    new_df = new_df.reindex(columns=labels)

    return new_df
