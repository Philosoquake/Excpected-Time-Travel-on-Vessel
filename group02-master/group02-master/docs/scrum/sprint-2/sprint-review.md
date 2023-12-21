# Klevi

## ST05: Decide on GUI tech

Had to research on which GUI tech to use. Decided to use 'PySimpleGUI' as it is easy to use and has a lot of documentation.
Had to install PySimpleGUI and test it out. Watched a tutorial on how to use it and created a simple GUI.


## ST05: Plan the GUI functionalities

Had to plan the GUI functionalities. Decided to have a main window with a upload file button, a button to plot the vessel speed and a button to plot the vessel course. The results will be shown in a new window. The results window will be saved as a temporary file and will be deleted when the program is closed.


## ST05: Plan GUI layout and work on it

Had to plan the GUI layout and work on it. Decided to use a horizontal layout for the main window and a square layout for the results window. The results window will then be shown in a new window. Im not sure if I chose the right plots for the data. I will have to ask the team for feedback.

## ST05: Create visualizations for data (plots)

Had to create visualizations for data (plots). Decided to use matplotlib and seaborn librarys to create the plots. The plots will be shown in the results window as well as in the VS-Code. Took a little bit more time than expected as I had to learn how to use the libraries.

# Artur

## ST04: Fine tune ML methods, hyperparameters
I fine tuned RandomForestRegressor and BaggingRegressor. The rule of thumb is the "the more the better", however it takes logner and longer to train them. Depending on how the application will be used in the future, it might be best to pretrain a complex version and somehow save it to disk, so that the user wouldn't have to pay the cost of waiting for the training.

The rule of thumb I'm going with is `n_estimators=20, n_jobs=-1, max_depth=50`, maybe `n_estimators=20, n_jobs=-1, max_depth=25`. This gives reasonably fast results if trains the models from scratch.
## ST04: Test ML methods based on the distance/time to target
Implemented different model + scalar combination for different distances away from the destination.

Question remains why rotterdam to hamburg is more accurate than Felixtowe to Rotterdam, even though the second is shorter and more straight forward.
## ST04: Research more about ML methods and work on data exploration
Read quite a lot about ML in scikit-learn's homepage and youtube as well as even reading part of a scientific paper talking about our topic. The goal was to get a better grasp on the .

# Mohammad
## ST05: Predictor Agent Designd
Worked on the design of the Predictor Agent. This involved determining the various components of the agent and how they would interact with each other. Focused on ensuring the agent design was efficient and capable of meeting the project requirements.
## ST05: Understand the requirement for Predictor Agent
Had to understand the detailed requirements of the Predictor Agent. This involved discussing with the team and studying the project requirements to ensure I had a comprehensive understanding. This knowledge guided my work on the agent design and API research.
## ST05: Research available APIs
Conducted research on available APIs that could be utilized in our Predictor Agent. This involved comparing the functionality, ease of use, and reliability of several APIs to determine which ones would best meet our needs.
## ST05: Evaluate the top candidate APIs
After researching the available APIs, I evaluated the top candidates based on their performance, user-friendliness, and fit with our project. This allowed us to decide on the most suitable APIs for our Predictor Agent.

