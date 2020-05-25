# Udacity_Disaster_Response_Pipeline
This is a repository to version-control my progress on the Udacity Project Disaster Response Pipeline and to share my project with Udacity Reviewers.

# Project Motivation
The motivation of this project is based on a Udacity-Nanodegree project. Task is to develop a full ML-Lifecycly project starting
with data extraction via an ETL-Pipeline, going over Model creation using a combination of a Python-based NLP/ML-Pipeline to deploying the ready an ready to use Webapplication using the WebFramework Flask.

# Installations
This project was initialized in a Jupyter Notebook using Python 3. Dependencies are implemented within the Notebook itself. 
A further Notebook-external dependency would be Flask, if you want to know more about Flask, please follow the attached url. [Flask](https://www.freecodecamp.org/news/how-to-build-a-web-application-using-flask-and-deploy-it-to-the-cloud-3551c985e492/)

# Instructions for running the Flask App:
1. Run the following commands in the project's workspace directory to set up your database and model. You can use the attached /home folder, which contains files and skripts as found in the Udacity IDE.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
        such as: 
        user:/home/workspace python data/process_data.py data/disaster_messages.csv...             
        
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        
        such as: 
        user:/home/workspace python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl...
        
2. Run the following command in the app's directory to run your web app.
    `python run.py`
     
     such as: 
     user:/home/workspace python app/run.py  

3. Go to http://0.0.0.0:3001/


# File Descriptions
ETL-Pipeline Prepartion.ipynb: Jupyter Notebook with code for the ETL-Pipeline of this project.
ML-ipeline Prepartion.ipynb: Jupyter Notebook with code for the ML-pipeline of this project.
disaster-response-pipeline-project.zip: Flask app bundle with modularized Version of Code used in both Jupyter Notebooks.

# How to interact with the repositories content
For further working on the Jupyter Notebook, you can either download it and work locally with it, or use the Python Notebook from within the online version of [Jupyter](https://jupyter.org/try).

## Author
The Author of this Notebook is the GitHub User MDK1192, usage for own interests of my files is granted. Ownership to provided datasets however resides within Figure8 and Udacity

## Acknowledgements
I would like to thank Udacity for giving an introduction into the topics needed to create this repositories content, as well as Figure8 for providing  the data accessible creating this project.


