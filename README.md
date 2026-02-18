## Email Spam Analysis System

  This project is a simple web-based system that detects whether an email or message is spam or not, built using Python and Flask. It not only classifies messages as Spam or Ham (legitimate) but also shows how confident the system is in its prediction.
  
  The goal of this project is to learn how text classification works and see how a text analysis model can be integrated into a live web application.

## What This Project Does

  -Accepts email or message text as input
  
  -Analyzes the content using a machine learning model
  
  -Determines whether the message is Spam or Ham
  
  -Displays the confidence level of the prediction
  
  -Lets users try out random messages from the dataset

## Tools and Technologies Used
  
  -Python – for core logic and data handling
  
  -Flask – to build the web interface
  
  -Scikit-learn – for building and running the classification model
  
  -Pandas – to manage and process the dataset
  
  -HTML & CSS – to create a simple, user-friendly interface

## Dataset Information
  
  -Uses the SMS Spam Collection Dataset
  
  -All messages are already labeled as Spam or Ham
  
  -If the dataset is not present locally, it is automatically downloaded

## How It Works 

  -The dataset is loaded and cleaned
  
  -Text data is converted into numerical features using TF-IDF
  
  -A Naive Bayes classifier is trained on the data
  
  -Users can input an email or message
  
  -The system predicts if it’s spam and shows the confidence score

## How to Run This Project
  # Step 1: Clone the repository
    git clone https://github.com/your-username/email-spam-analysis-system.git
    cd email-spam-analysis-system
