# Predicting the Expected Travel Time of a vessel

## Documentation:

Project Overview:

The goal of the project is to develop a tool that predicts the expected travel time (ETT) of a vessel based on historical AIS data. The tool aims to optimize the daily business of harbors and locks by providing accurate ETT predictions.

The project is divided into several tasks:

1. Development Infrastructure (ST01):
   - GitLab of TUHH is used as the development platform.
   - The infrastructure setup ensures that the product owner has access to all development stages.
   - At the end of each sprint, the product owner can download the resulting product, including scripts to start the application.

2. Data Preparation (ST02):
   - The raw AIS data is obtained from an AIS data provider.
   - The data must be cleaned and prepared for further stages.
   - Features/attributes are selected for machine learning, and decisions are made transparent with graphics, documentation, and explanations.

3. Machine Learning Methods for ETT and AIS Data (ST03):
   - The goal is to find the best machine learning method for predicting ETT with the lowest error.
   - At least four candidate methods are tested and compared.
   - Experiments are executable by the product owner, and an explanatory report documents the results.

4. Tools Selection for Implementation (ST04):
   - The overall architecture of the application and the implementation environment are described.
   - The installation of the final application is independent of specific operating systems, and a Docker-based implementation is preferred.

5. Creating the environment for the basic ETT setting (ST05):

   - Build the ETT application for a short tour with only one predictor agent.
   - Create a GUI for problem visualization. The GUI app should show historical data of vessel movement and speed. On request of an ETT, the actual data must also be shown.


6. Building an infrastructure for distributed decision making (ST06):

   - A broker agent can serve requests of clients that want to have an ETT prediction of a vessel.
   - The broker agent learns the weightings for all service providers and integrates the prediction of all ETT predictor agents.

7. Integration in the overall development process (ST07):

   - Integrate the testing environment with CI so that tests can be executed when new code is deployed.

   
### Project Structure:

The project is structured into several folders and files to organize the code and resources effectively. Here is an overview of the main components of the project:

1. `scrum` folder:

   - `sprint-0` folder: Contains the `sprint-backlog.md` file for sprint 0.
   - `sprint-1` folder: Contains the `sprint-backlog.md`, `sprint-review.md` files and `scrum-1.xlsx` file on working hours for sprint 1.
   - `sprint-2` folder: Contains the `sprint-backlog.md`, `sprint-review.md` files and `scrum-2.xlsx` file on working hours for sprint 2.
   - `sprint-3` folder: Contains the `sprint-review.md` file and `scrum.xlsx` file on working hours for sprint 3.

2. The `notebooks` folder contains Jupyter notebooks used for documentation, experimentation, data exploration, and testing of functions.

   - `playground-sklearn` folder: Contains the `0.ipynb` notebook for machine learning experimentation using scikit-learn.
   - `data-exploration.ipynb`: Notebook used for data exploration and analysis.
   - `learn.ipynb`: Notebook used for testing and development of functions.

3. The `predictor` folder contains modularized code for data cleaning and plotting tasks.

   - `agent` folder: Contains the code for the ETT predictor agent.
   - `clean` folder: Contains the code for data cleaning.
   - `gui` folder: Contains the code for the GUI.
   - `plot` folder: Contains the code for generating plots.
   - `__init__.py` : contains the code for the main function.


4. The `resources` folder holds the raw and cleaned CSV files for the routes (`felixstowe_rotterdam.csv` and `rotterdam_hamburg.csv`).

   - `felixstowe_rotterdam.csv`: CSV file containing raw AIS data for the Felixstowe to Rotterdam route.
   - `rotterdam_hamburg.csv`: CSV file containing raw AIS data for the Rotterdam to Hamburg route.
   - `felixstowe_rotterdam_cleaned.csv`: CSV file containing cleaned AIS data for the Felixstowe to Rotterdam route.
   - `rotterdam_hamburg_cleaned.csv`: CSV file containing cleaned AIS data for the Rotterdam to Hamburg route.


5. The `.gitignore` file specifies files and directories to be ignored by version control.

6. The `Dockerfile` contains the instructions for building the Docker image.

7. The `README.md` file provides additional information and instructions about the project.

8. The `requirements.txt` file contains the required Python packages for the project.

9. The `setup.py` file contains the instructions for installing the package.

By following this structure, the project components and resources are organized logically, allowing for efficient collaboration and development.
