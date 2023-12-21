

# Sprint 0
Sprint 0:
    - Klevi   : As a developer, I want to be able to visualize the data, so that I can analyze it better.
    - Mohammad: As a developer, I want to analyze the meaning of the data, so that I can understand it better.
    - Artur   : As a developer, I want to clean the data, so that I can use it better.

## Task: Cleaning
Removing faulty data is important to maintain a high quality of the data. This is done by removing rows with missing values, and removing rows with faulty values.

Below is a list of things which were removed or modified to make the database more clean:
- Removed duplicated rows. Those are lines which have the same entries everywhere. This is unneeded data.
- Removed duplicated rows with the same ID. IDs should correspond to exactly one row. If they are duplicated, it means that the data is faulty. Thus in case we simply dropped it.
- Removed rows with NaN values. Rows with missing values are non valuable, and thus we dropped them. Even if the data is not relevant for our task, it is still better to drop it, than to keep it, because it might indicate a faulty transmission and other data might be wrong too.
- Filtered Trips according to their amount of entries. The goal is to remove the abnormal trips. Firstly, abnormal trips might indicate faulty data. Secondly, abnormal trips might be outliers, which might affect the results of the analysis. The criteria is somewhat arbitrary and was done by analyzing the plots. TODO: This can still be improved.
- Removed various shiptypes (IDs), which seem to be faulty. Those are:
  - 0: This is probably a NaN value caused by a transmission error.
  - 9: is only present in rtm_ham
  - 1: Doesn't have a full route.
  - 69: Very incomplete route. Nice btw :D
  - 142: very incomplete and unknown type. Seems like a tramsission error and 42 is meant. Only present in fxt_rtm
  - 139: very incomplete and unknown type. Seems like a tramsission error and 39 is meant. Only present in fxt_rtm
  - 55: very incomplete. Seems like a tramsission error. Only present in rtm_ham. Only occurs in a trip with a certain ship, where also other columns are different from the expected values.
  
## Task: Plotting

Visualizing the data helps to better understand it, spot patterns, and identify potential issues. In this task, we have created multiple plots to visualize the data in different aspects, such as spatial distribution, temporal distribution, and ship type distribution.

Below is a list of plots generated and their purposes:

- Longitude and Latitude Scatter Plot: Displays the spatial distribution of ship locations, start positions (red), and end positions (green). This plot helps to identify any anomalies in ship movement or trips.
- Amount of Entries per Hour: A bar chart showing the distribution of entries throughout the day. This plot can help identify peak and off-peak hours for ship traffic.
- Amount of Entries per Day: A bar chart showing the distribution of entries across the days of a month. This helps to identify any patterns in ship movements on a monthly basis.
- Amount of Entries per Month: A bar chart showing the distribution of entries across months. This plot provides insight into seasonal trends in ship movements.
- Amount of Entries per Day Name: A bar chart showing the distribution of entries across days of the week. This plot helps to identify any weekly patterns in ship movements.
- Course Over Ground (COG) Distribution: A scatter plot showing the distribution of ship directions (grouped into bins of 50 degrees). This plot helps to identify any patterns or anomalies in ship directions.

In addition to the above plots, the code also includes functions to filter the data based on various attributes, such as ship type, hour, day, and month. This allows for more detailed analysis and visualization of specific aspects of the data.

Here is a summary of the findings from the visualizations:

- The scatter plot of ship locations, start positions, and end positions reveals the general routes that ships take between the ports. Anomalies, such as unusual trips or outliers, can be identified through this plot.
- The distribution of entries per hour, day, day name, and month indicates the general trends in ship traffic, with certain times of the day, days of the week, and months having higher traffic than others. This information can be used to plan maintenance, resource allocation, and other activities related to ship traffic management.
- The COG distribution plot reveals the general direction ships are traveling, which can be useful for understanding ship behavior and predicting ship movements.

By visualizing the data in different ways, we can better understand the underlying patterns and trends, allowing for more informed decision-making and analysis.

## Task: Analyzing
The "Analyze" task focuses on extracting insights and understanding the data in order to gain valuable information and make informed decisions. This task involves exploring the provided data and conducting various analyses to uncover patterns, trends, and relationships.

Data Attributes
The dataset for the "Analyze" task contains information about ship trips and ship attributes. Here are the attributes included in the dataset:

- 'TripID': A unique identifier for each trip.
- 'MMSI': The Maritime Mobile Service Identity of the ship, which serves as a unique identifier.
- 'StartLatitude': The latitude of the planned starting position for the trip.
- 'StartLongitude': The longitude of the planned starting position for the trip.
- 'StartTime': The time at which the trip is planned to start.
- 'EndLatitude': The latitude of the planned ending position for the trip.
- 'EndLongitude': The longitude of the planned ending position for the trip.
- 'EndTime': The time at which the trip is planned to end.
- 'StartPort': The starting port for the trip.
- 'EndPort': The ending port for the trip.
- 'ID': The ID of the AIS message.
- 'time': The time of the AIS message.
- 'shiptype': The type of the ship.
- 'Length': The length of the ship in meters.
- 'Breadth': The breadth of the ship in meters.
- 'Draught': The draught of the ship in meters. Draught is the depth of water the ship displaces.
- 'Latitude': The latitude of the ship's current position.
- 'Longitude': The longitude of the ship's current position.
- 'SOG': The speed over ground of the ship in knots.
- 'COG': The course over ground of the ship in degrees.
- 'TH': The true heading of the ship. It represents the angle between 'true north' and the direction the ship is heading.
- 'Destination': The intended destination of the ship at the time of the AIS message.
- 'Name': The name of the ship.
- 'Callsign': The call sign of the ship.
- 'AisSource': The source of the AIS data.
- 'd': (The problem is that we don't have the defintion of this attributes to understand the menning of it. )
