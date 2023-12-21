# Artur
## ST06: Create broker agent System
I have spent a lot of time  refactoring the code and waiting for the models being fitted, to test various things. Had to create my own `ETTAgent` to test things out, and thus created code duplication with Mohammad. Later one we need to combine them to get the best of both worlds.

I couldn't implement the broker system unfortunately, because of time reasons. Though it might be ready the next weak when we hold our presentations. I also had problems integrating the future potential broker agent system with the GUI. So one has to work on that too.

## ST06: Research Broker Agent system
Researched but the information on that topic was sparse and hard to find.

## ST02: Potentially more data preparation to eliminate anomaly
I tested out various new ways to prepare the data to remove more anomalies. However I figured that the advantages of integrating it into the internal code base plus in addition testing out the various models with it, which takes always a long time to train. However, we might work on it in the future.
# Klevi

## ST05: Work on GUI layout
On this Task, the GUI layout for the code has been successfully implemented. The Streamlit app allows users to upload a CSV file, select from various plot options, and perform data cleaning with the click of a button. The plots are displayed using Streamlit's plotting methods, providing a visually appealing and interactive user experience. Overall, the GUI layout enhancements have improved the usability and effectiveness of the Vessel Visualization app.

Worked on it more than expected, had to learn how to use Streamlit and how to use it with Plotly. Had difficulties implementing the Cleaning Data function. If CSV files are uploaded with missing values, the app would crash but worked on it and fixed it. Added a new function to the app that allows users to clean the data before plotting it. The app now works as expected. It also shows the cleaned data in a table.

## ST05: Integrate GUI with other systems

There were difficulties implementing the Broker Agent and this is the reason why I couldn't finish the task. Artur spent a lot of time working on it but couldn't get it to work. I tried to help him but we couldn't figure out what was wrong. We will continue working on it in the next Week if possible.

# Mohammad

## ST04: Integrate Docker to the project
Successfully implemented Docker to the project, ensuring consistent environments from development to production. Faced challenges understanding Dockerfile syntax and setting up Docker environments. Overcame issues and established a functional containerized app.

## ST07: Integrate CI to the project

Integrated Continuous Integration into our development process, enhancing code health with each push. Challenges arose in selecting the CI platform and understanding its configuration protocols. Despite difficulties, established a robust CI pipeline.

## ST05: Integrate Agent API with other systems

Attempted to integrate an Agent API to facilitate interaction with other services. Encountered difficulties implementing the Broker Agent, and despite efforts, we couldn't resolve them in this sprint. Plan to resume and complete the task in the next sprint.