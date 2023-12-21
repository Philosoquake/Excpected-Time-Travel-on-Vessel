from ..broker import ETTAgent  # type: ignore
import pandas as pd  # type: ignore
import streamlit as st  # type: ignore
from .. import plot as plot
from .. import clean as clean
import plotly.express as px  # type: ignore
import io


def determine_trip_route(df: pd.DataFrame) -> str:
    result: str = ""
    start_port = df.iloc[0]['StartPort']
    end_port = df.iloc[0]['EndPort']

    if start_port == "ROTTERDAM" and end_port == "HAMBURG":
        result = "rtm_ham"
    elif start_port == "FELIXSTOWE" and end_port == "ROTTERDAM":
        result = "fxt_rtm"
    else:
        raise ValueError("Invalid trip route")
    return result


def load_raw_data(file: io.IOBase):
    try:
        # Read the CSV file into a pandas DataFrame
        data_raw = pd.read_csv(file)
    except pd.errors.ParserError as e:
        # Handle exceptions when reading the CSV file
        print("Exception: ", e, " of type ", type(e),
              " occurred. Trying again with quotechar='''")
        file.seek(0)
        data_raw = pd.read_csv(file, quotechar="'")
    return data_raw


def init_session_state():
    # Initialize session states
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = None
    if 'file_uploaded_prev' not in st.session_state:
        st.session_state.file_uploaded_prev = None
    if 'data_cleaned' not in st.session_state:
        st.session_state.data_cleaned = None
    if 'data_raw' not in st.session_state:
        st.session_state.data_raw = None
    if 'best_agent_cached' not in st.session_state:
        st.session_state.best_agent_cached = None
    return st.session_state


def data_viz(STATE):
    # Display a select box in the sidebar to choose the plot option
    st.title("Data Visualization")
    st.write("Here your uploaded data is visualized. This helps you to identify the quality as well as relevant of the AIS data which you have uploaded.")
    viz_choice = st.radio(
        'Choose Visualization',
        [
            'Entry Location', 'Polar Plot',
        ]
    )

    data_is_cleaned = st.session_state.data_cleaned is not None
    data_used = st.session_state.data_cleaned if data_is_cleaned else STATE.data_raw

    if viz_choice == 'Entry Location':
        # Vessel Trajectory option selecteds
        st.header(
            f"Entry Location of { 'Cleaned' if data_is_cleaned else 'Raw'} Data")

        st.write("This map shows the location of entries accumulated. It helps to identify the area of entries. Thus if your desired area is not in here, you may need to have another dataset.")
        # Check if the data is cleaned and available for plotting

        # Create a scatter mapbox plot to visualize vessel trajectory
        fig = px.scatter_mapbox(data_used, lat="Latitude", lon="Longitude", color_discrete_sequence=[
                                "fuchsia"], zoom=6, height=None)
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        # Display the plot
        st.plotly_chart(fig)

    elif viz_choice == 'Vessel Movement':
        st.header("Vessel Movement")
        st.pyplot(plot.plot_vessel_movement(data_used))

    elif viz_choice == 'Polar Plot':
        st.header("Polar Plot")
        st.write("This polar plot shows the direction (COG/Course-over-Ground) of entries accumulated. It helps to identify the most common direction of entries. Thus if your desired direction is not in here, you may need to have another dataset.")
        st.pyplot(plot.polar_plot(data_used['COG']))


def get_best_agent(STATE) -> ETTAgent:
    import os
    from sklearn import metrics  # type: ignore

    ett_agents_dict: dict[str, ETTAgent] = {}
    files = os.listdir(f'./predictor/models/{determine_trip_route(st.session_state.data_raw)}')
    for file in files:
        if file.endswith('.joblib'):
            ett_agents_dict[file.replace('.joblib', '')] = ETTAgent.load(
                f'./predictor/models/{determine_trip_route(st.session_state.data_raw)}/{file}')
        else:
            print(f'File {file} is not a joblib file')

    best_agent: ETTAgent = None
    highest_r2 = 0
    for agent_name, agent in ett_agents_dict.items():
        r2 = metrics.r2_score(agent.y_test, agent.test())
        print(f"R2 for {agent_name}: {r2}")
        if r2 > highest_r2:
            highest_r2 = r2
            best_agent = agent
    return best_agent


def create_pipelines(df_data: pd.DataFrame):
    from sklearn import ensemble, linear_model, neighbors, preprocessing, pipeline  # type: ignore
    scalars = {
        '': None,
        'StandardScaler': preprocessing.StandardScaler(),
        'MinMaxScaler': preprocessing.MinMaxScaler(),
        'MaxAbsScaler': preprocessing.MaxAbsScaler(),
        'RobustScaler': preprocessing.RobustScaler(),
        # # NOTE: The scalars below are not that good.
        # 'Normalizer': preprocessing.Normalizer(),
        # 'QuantileTransformer-Normal': preprocessing.QuantileTransformer(output_distribution='normal'),
        # 'QuantileTransformer-Uniform': preprocessing.QuantileTransformer(output_distribution='uniform'),
        # 'PowerTransformer': preprocessing.PowerTransformer(),
    }

    ett_agets_dict = {
        'LinearRegression': linear_model.LinearRegression(),
        'RandomForestRegressor': ensemble.RandomForestRegressor(n_estimators=10),
        'BaggingRegressor': ensemble.BaggingRegressor(),
        'KNeighborsRegressor': neighbors.KNeighborsRegressor(),
    }
    for model_name, model in ett_agets_dict.items():
        for scalar_name, scalar in scalars.items():
            print(f"Training {model_name}-{scalar_name if scalar_name else 'NoScalar'}")
            ett_agent = ETTAgent(
                name=f"{model_name}-{scalar_name if scalar_name else 'NoScalar'}",
                estimator=pipeline.make_pipeline(scalar, model),
                data=df_data,
                custom_feature_include_list=[
                    "EndLatitude", "EndLongitude",
                    "Length",
                    "Latitude", "Longitude",
                    "SOG",
                ]
            )
            ett_agent.train()
            ett_agent.dump(
                f'./predictor/models/{determine_trip_route(df_data)}')
            print(f"Finished training {model_name}-{scalar_name if scalar_name else 'NoScalar'}")


def data_predict(STATE, data_used) -> None:
    st.title("Prediction")
    st.write("Here you can enter your current position, your desired position, the length of your vessel and the speed over ground. With this information the model will predict the estimated travel time (ETT) left for your trip. In addition, while you are selecting your current position and your desired position, you can see in the map where those dots are. We do not recommend to do it like that, but feel free.")
    longitude = st.number_input(
        label='Start Longitude', value=0.0, step=1.0, format="%.2f")
    latitude = st.number_input(
        label='Start Latitude', value=50.0, step=1.0, format="%.2f")
    end_longitude = st.number_input(
        label='Dest. Longitude', value=0.0, step=1.0, format="%.2f")
    end_latitude = st.number_input(
        label='Dest. Latitude', value=50.0, step=1.0, format="%.2f")
    speed = st.number_input(label='SOG', value=0.0, step=1.0, format="%.2f")
    length = st.number_input(label='Length', value=20, step=1)

    # Visualize the location options on a map
    fig = px.scatter_mapbox(
        pd.DataFrame({
            'lon': [longitude, end_longitude],
            'lat': [latitude, end_latitude],
            'color': ['start', 'destination']
        }),
        lat="lat",
        lon="lon",
        color='color',
        color_discrete_sequence=["red", "green"],
        zoom=6,
        height=None
    )
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig)
    import os
    if not os.path.exists(f'./predictor/models/{determine_trip_route(st.session_state.data_raw)}'):
        print("models are generated")
        message = st.empty()
        message = st.warning("Models are being generated. This only needs to be done once. After that it will be cached.")
        if st.session_state.data_cleaned is None:
            loading_message = st.empty()

            loading_message = st.warning("Since you didn't clean the data yet, it will be done now.")
            st.session_state.data_cleaned = clean.clean_up(
                st.session_state.data_raw)
            loading_message = st.success("Data cleaned")
        
        create_pipelines(st.session_state.data_cleaned)
        message = st.success("Models generated")
    else:
        print("Models already exist")
    if st.session_state.best_agent_cached is None:
        st.warning("Waiting for model to load...")
        st.session_state.best_agent_cached = get_best_agent(STATE)

    with st.form(key='my_form'):
        submit_button = st.form_submit_button('Submit')
        if submit_button:
            import datetime as dt
            from ..broker import format_timestamp  # type: ignore

            df = pd.DataFrame({
                'EndLatitude': end_latitude,
                'EndLongitude': end_longitude,
                'Length': length,
                'Latitude': latitude,
                'Longitude': longitude,
                'SOG': speed,
            }, index=[0])
            y_pred = st.session_state.best_agent_cached.test(df)
            if y_pred[0] < 0:
                st.warning("ETT prediction is negative. This means you have chosen abnormal values, like a very high speed. For convenience, we still show you the prediction, but it does not make sense.")
                ett = dt.datetime.fromtimestamp(-y_pred[0]) - \
                    dt.datetime.fromtimestamp(0)
                st.write("ETT prediction: ",
                         format_timestamp(ett), " in the past")
            else:
                ett = dt.datetime.fromtimestamp(
                    y_pred[0]) - dt.datetime.fromtimestamp(0)
                st.write("ETT prediction: ", format_timestamp(ett))


def main():
    # Set the title of the Streamlit app

    st.warning("This is only a DEMO. Do not use for navigation yet!")
    st.title("Predictor")
    st.write("Hello captain! This is a web app to predict the estimated travel time (ETT) left for your trip.")
    st.write("For this you need to upload a CSV file of AIS data trips which take your desired route. To experiment, we have kindly provided you with sample data so that you can test it out.")
    st.write("The workflow is simple. You firstly upload a CSV file of the relevant AIS data. It is further cleaned and processed. After that you can shortly explore that data visually, in the data visualization panel, to be sure that it is the desired AIS data. After that go to the prediction panel, where you can actually predict your ETT.")

    # Initialize session states
    STATE = init_session_state()

    # Upload a CSV file

    st.session_state.file_uploaded = st.file_uploader(
        "Select a CSV file", type=['csv'])
    if st.session_state.file_uploaded != st.session_state.file_uploaded_prev:
        st.session_state.data_raw = None
        st.session_state.data_cleaned = None
        st.session_state.best_agent_cached = None
    st.session_state.file_uploaded_prev = st.session_state.file_uploaded
    if st.session_state.file_uploaded is None:
        return

    if st.session_state.data_raw is None:
        loading_message = st.empty()
        loading_message.text("Loading raw data...")
        try:
            st.session_state.data_raw = load_raw_data(
                st.session_state.file_uploaded)
        except Exception as e:
            st.error("Error: " + str(e))
            return
        loading_message.text("Raw data loaded successfully!")

    if st.session_state.data_cleaned is None:
        # Display a button to clean the data
        if st.button("Clean Data"):
            # Perform data cleaning when the "Clean Data" button is clicked
            loading_message = st.empty()
            loading_message.text("Cleaning data...")

            # Call the clean_up function from the 'clean' module to clean the raw data
            st.session_state.data_cleaned = clean.clean_up(
                st.session_state.data_raw)

            loading_message.text("Data cleaned successfully!")
    else:
        st.success("Data already cleaned!")

    task = st.radio(
        "Choose the panel",
        ["None", "Data Visualization", "Prediction"]
    )

    if task == "None":
        pass
    elif task == "Data Visualization":
        data_viz(st.session_state)
    elif task == "Prediction":
        data_used = st.session_state.data_cleaned if st.session_state.data_cleaned is not None else st.session_state.data_raw
        data_predict(st.session_state, data_used)
