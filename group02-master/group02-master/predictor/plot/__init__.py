import pandas as pd  # type: ignore
import numpy as np
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns # type: ignore
# import streamlit as st # type: ignore
# import tempfile

# Function to plot the distribution of a numerical variable
def plot_distribution(df: pd.DataFrame, column: str):
    """
    Plot the distribution of a numerical variable.

    Parameters:
    - df (DataFrame): The input data.
    - column (str): The column name of the numerical variable to plot.

    Returns:
    - None (displays a plot)
    """
    sns.histplot(df[column], kde=True)
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.title(f'Distribution of {column}')
    plt.show()

# Function to plot the relationship between two numerical variables
def plot_relationship(df: pd.DataFrame, x_column: str, y_column: str):
    """
    Plot the relationship between two numerical variables.

    Parameters:
    - df (DataFrame): The input data.
    - x_column (str): The column name of the x-axis variable.
    - y_column (str): The column name of the y-axis variable.

    Returns:
    - None (displays a plot)
    """
    sns.scatterplot(data=df, x=x_column, y=y_column)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f'{x_column} vs {y_column}')
    plt.show()


# Function to plot the trend of a numerical variable over time
def plot_trend_over_time(df: pd.DataFrame, time_column: str, variable_column: str):
    """
    Plot the trend of a numerical variable over time.

    Parameters:
    - df (DataFrame): The input data.
    - time_column (str): The column name of the time variable.
    - variable_column (str): The column name of the variable to plot.

    Returns:
    - None (displays a plot)
    """
    # Convert the time column to datetime type
    df[time_column] = pd.to_datetime(df[time_column])
    
    # Convert the variable column to numeric type
    df[variable_column] = pd.to_numeric(df[variable_column], errors='coerce')
    
    if pd.api.types.is_numeric_dtype(df[variable_column]):
        # Plot the trend over time
        sns.set_style('darkgrid')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=df, x=time_column, y=variable_column, ax=ax)
        ax.set_xlabel(time_column)
        ax.set_ylabel(variable_column)
        ax.set_title(f'{variable_column} over time')
        ax.tick_params(axis='x', rotation=45)
        return fig

    
# Function to plot the actual vs predicted values
def plot_actual_vs_predicted(actual_values: pd.Series, predicted_values: pd.Series) -> plt.Figure:
    """
    Plot the actual vs predicted values.

    Parameters:
    - actual_values (Series): The actual values.
    - predicted_values (Series): The predicted values.

    Returns:
    - None (displays a plot)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=actual_values, y=predicted_values, ax=ax)
    sns.lineplot(x=actual_values, y=actual_values, color='red', linestyle='--', ax=ax)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs Predicted')
    return fig
    
        
def plot_vessel_movement(data: pd.DataFrame) -> plt.Figure:
    """
    Plot vessel movement based on latitude and longitude data.

    Parameters:
    - data (DataFrame): The DataFrame containing latitude and longitude data.

    Returns:
    - None
    """

    # Check if the required columns are present in the data
    if 'Longitude' not in data.columns or 'Latitude' not in data.columns:
        raise ValueError('The selected CSV file does not contain the required columns.')

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=data, x='Longitude', y='Latitude', color='blue', ax=ax)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Vessel Movement')
    return fig

def polar_plot(df: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(8, 8), dpi=80)
    ax = fig.add_subplot(111, polar=True)

    data = df.values
    data_rad = np.deg2rad(data)
    hist, bin_edges = np.histogram(data_rad, bins=36, density=True)

    ax.bar(bin_edges[:-1], hist, width=0.15, color='b', alpha=0.7)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title('Directional data', size=20, color='black', y=1.1)
    return fig

def plot_vessel_speed(data: pd.DataFrame) -> plt.Figure:
    """
    Plot vessel speed based on time and speed data.

    Parameters:
    - data (DataFrame): The DataFrame containing time and speed data.

    Returns:
    - None
    """

    # Check if the required columns are present in the data
    if 'time' not in data.columns or 'SOG' not in data.columns:
        raise ValueError('The selected CSV file does not contain the required columns.')

    # Set the style using Seaborn
    sns.set_style('darkgrid')

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=data, x='time', y='SOG', color='blue', ax=ax)
    ax.set_xlabel('Time')
    ax.set_ylabel('Speed over Ground')
    ax.set_title('Vessel Speed')
    return fig

      
# Function to calculate the duration of each ship's journey
def calc_duration(df: pd.DataFrame) -> pd.DataFrame:
    df['Duration'] = df['EndTime'] - df['StartTime']
    return df

# Function to calculate the extents for plotting
def calc_extents(df: pd.DataFrame) -> tuple[float, float, float, float]:
    """
    Calculate the extents for longitude and latitude plotting.

    Parameters:
    - df (DataFrame): The DataFrame containing longitude and latitude data.

    Returns:
    - tuple[float, float, float, float]: The minimum and maximum values for latitude and longitude.
    """
    max_latitude = df['Latitude'].max() // 10 * 10 + 10
    max_longitude = df['Longitude'].max() // 10 * 10 + 10 + 3
    min_latitude = df['Latitude'].min() // 10 * 10
    min_longitude = df['Longitude'].min() // 10 * 10
    return min_latitude, max_latitude, min_longitude, max_longitude

# Function to calculate the extents for plotting
def plot_longitude_latitude(df: pd.DataFrame) -> plt.Figure: # tuple[plt.Figure, plt.Axes]:
    """
    Plot the ship locations based on longitude and latitude.

    Parameters:
    - df (DataFrame): The DataFrame containing longitude and latitude data.

    Returns:
    - None (displays a plot)
    """
    
     # Check if the required columns are present in the data
    if 'Longitude' not in df.columns or 'Latitude' not in df.columns:
        raise ValueError('The selected CSV file does not contain the required columns.')
    
    min_latitude, max_latitude, min_longitude, max_longitude = calc_extents(df)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x='Longitude', y='Latitude', alpha=0.01, s=1, label='Location', color='blue', ax=ax)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Ship Locations')
    ax.set_xlim(min_longitude, max_longitude)
    ax.set_ylim(min_latitude, max_latitude)
    return fig
#     sns.scatterplot(data=df, x='Longitude', y='Latitude', alpha=0.01, s=1, label='Location', color='blue')
#     plt.xlabel('Longitude')
#     plt.ylabel('Latitude')
#     plt.title('Ship Locations')
#     plt.xlim(min_longitude, max_longitude)
#     plt.ylim(min_latitude, max_latitude)
  
#    # Save the plot as a temporary image file
#     with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
#         filename = tmpfile.name
#         plt.savefig(filename)



def month_num_to_str(month_num: int) -> str:
    mapping = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December",
    }
    return mapping.get(month_num, "Unknown")


def shiptype_id_to_str(id: int) -> str:
    mapping = {
        0: "Not available (default)",
        1: "Reserved for future use",
        4: "Reserved for future use",
        9: "Reserved for future use",
        55: "Law enforcement",
        69: "Passenger, No additional information",
        70: "Cargo",
        71: "Cargo - Hazard A (Major)",
        72: "Cargo - Hazard B (Medium)",
        73: "Cargo - Hazard C (Minor)",
        74: "Cargo - Hazard D (Recognizable)",
        79: "Cargo",
        82: "Tanker - Hazard B (Medium)",
        142: "Unknown type",
        139: "Unknown type",
    }
    result = mapping.get(id, "Unknown")
    if result == "Unknown":
        result = f"Unknown shiptype id: {id}"

    return result


def print_counts(df: pd.DataFrame) -> None:
    print(f"Entry count   : {df.shape[0]}")
    print(f"TripID count  : {df['TripID'].nunique()}")
    print(f"Callsign count: {df['Callsign'].nunique()}")
    print(f"MSSI count    : {df['MMSI'].nunique()}")
    print(f"Name count    : {df['Name'].nunique()}")


def plot_on_shiptype(df: pd.DataFrame) -> None:
    
    """
    Plot ship locations based on ship type.

    Parameters:
    - df (DataFrame): The DataFrame containing ship type data.

    Returns:
    - None (displays a plot)
    """
    
    c = df['shiptype'].value_counts()
    types = c.index
    for i in range(types.shape[0]):
        print(f"plotting shiptype {types[i]} : {shiptype_id_to_str(types[i])}")
        temp_df = df.loc[df['shiptype'] == types[i]]
        print_counts(temp_df)

        plot_longitude_latitude(temp_df)

# Function to plot ship locations based on hour of the day
def plot_on_hour(df: pd.DataFrame, label: str) -> None:
    """
    Plot ship locations based on the hour of the day.

    Parameters:
    - df (DataFrame): The DataFrame containing the ship data.
    - label (str): The column label representing the time.

    Returns:
    - None (displays a plot)
    """
    dt = pd.to_datetime(df[label], format='%Y-%m-%d %H:%M')
    hours = dt.dt.hour.value_counts().sort_index()
    hours.plot(kind='bar', title='Amount of entries per hour')
    for hour_num, amount_in_hour in hours.items():
        print(f"plotting hour {hour_num}")
        temp_df = df.loc[pd.to_datetime(
            df['StartTime'], format='%Y-%m-%d %H:%M').dt.hour == hour_num]
        print_counts(temp_df)

        plot_longitude_latitude(temp_df)

# Function to plot ship locations based on day name
def plot_on_day_name(df: pd.DataFrame, label: str) -> None:
    dt = pd.to_datetime(df[label], format='%Y-%m-%d %H:%M')
    day_names = dt.dt.day_name().value_counts()

    # sort day_names
    day_names = day_names.reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    print(f"day_names: {day_names}")
    day_names.plot(kind='bar', title='Amount of entries per day name')
    plt.show()

    for day_name, amount_in_day in day_names.items():
        print(f"plotting day {day_name}")
        temp_df = df.loc[pd.to_datetime(
            df['StartTime'], format='%Y-%m-%d %H:%M').dt.day_name() == day_name]
        print_counts(temp_df)

        plot_longitude_latitude(temp_df)

# Function to plot ship locations based on day of the month
def plot_on_day(df: pd.DataFrame, label: str) -> None:
    dt = pd.to_datetime(df[label], format='%Y-%m-%d %H:%M')
    days = dt.dt.day.value_counts().sort_index()

    days.plot(kind='bar', title='Amount of entries per day')
    plt.show()

    for day_num, amount_in_day in days.items():
        print(f"plotting day {day_num}")
        temp_df = df.loc[pd.to_datetime(
            df['StartTime'], format='%Y-%m-%d %H:%M').dt.day == day_num]
        print_counts(temp_df)

        plot_longitude_latitude(temp_df)

# Function to plot ship locations based on month of the year
def plot_on_month(df: pd.DataFrame, label: str) -> None:
    dt = pd.to_datetime(df[label], format='%Y-%m-%d %H:%M')
    months = dt.dt.month.value_counts().sort_index()
    months.plot(kind='bar', title='Amount of entries per month')
    for month_num, amount_in_month in months.items():
        print(f"plotting month {month_num} : {month_num_to_str(month_num)}")
        temp_df = df.loc[pd.to_datetime(
            df['StartTime'], format='%Y-%m-%d %H:%M').dt.month == month_num]
        print_counts(temp_df)

        plot_longitude_latitude(temp_df)


def plot_on_departure_month(df: pd.DataFrame) -> None:
    plot_on_month(df, 'StartTime')


def plot_on_arrival_month(df: pd.DataFrame) -> None:
    plot_on_month(df, 'EndTime')


def plot_on_transmission_month(df: pd.DataFrame) -> None:
    plot_on_month(df, 'time')


def plot_on_departure_day(df: pd.DataFrame) -> None:
    plot_on_day(df, 'StartTime')


def plot_on_arrival_day(df: pd.DataFrame) -> None:
    plot_on_day(df, 'EndTime')


def plot_on_transmission_day(df: pd.DataFrame) -> None:
    plot_on_day(df, 'time')


# def plot_on_COG(df: pd.DataFrame) -> None:
#     c = df['COG'].value_counts().sort_index(ascending=False)
#     c.plot(kind='hist', title='Amount of entries per month')
#     # types = c.index
#     # for i in range(types.shape[0]):
#     #     print(f"plotting COG {types[i]}")
#     #     temp_df = df.loc[df['COG'] == types[i]]
#     #     print_counts(temp_df)

#     #     plot_longitude_latitude(temp_df)


def plot_on_SOG(df: pd.DataFrame) -> None:
    # Create bins for grouping COG values in 10s
    group_size = 1
    bins = np.arange(0, 51, group_size)
    labels = [f"{i}-{i+(group_size-1)}" for i in range(0, 50, group_size)]

    df['SOG_group'] = pd.cut(df['SOG'], bins=bins, labels=labels)

    # Get the counts of each COG group and sort them in descending order
    c = df['SOG_group'].value_counts().sort_index(ascending=True)
    groups = c.index

    for group in groups:
        print(f"Plotting SOG group {group}")
        temp_df = df.loc[df['SOG_group'] == group]
        if temp_df.shape[0] == 0:
            continue
        print_counts(temp_df)

        plot_longitude_latitude(temp_df)

    # Drop the 'SOG_group' column after plotting
    df.drop('SOG_group', axis=1, inplace=True)

def calculate_average_speed(df: pd.DataFrame) -> pd.DataFrame:
    avg_speed = df.groupby('TripID')['SOG'].mean().reset_index()
    return avg_speed


def plot_average_speed_distribution(df: pd.DataFrame) -> None:
    plt.hist(df['SOG'])
    plt.xlabel('Average Speed (SOG)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Average Speed')
    plt.show()

def plot_average_speed_per_trip(df: pd.DataFrame) -> None:
    plt.scatter(df['TripID'], df['SOG'], s=0.01)
    plt.xlabel('TripID')
    plt.ylabel('Average Speed (SOG)')
    plt.title('Average Speed per Trip')
    plt.xticks(rotation=45)
    plt.show()