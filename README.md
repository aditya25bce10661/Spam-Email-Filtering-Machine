# Spam-Email-Filtering-Machine
This Project is a simple machine learning code where the machine is trained to filter an email by its subject if it is spam or not. In this code 0 has been used to train the machine that it is a spam email and 1 has been used to train the machine that it is not a spam email.


**🚀 Project Overview**

This project demonstrates a simple end-to-end pipeline for text classification:

1.	*Data Collection*: A dataset of 100+ email subjects labeled as 0 (not spam) or 1 (Spam).
   
2.	*Vectorization*: Converting text into numerical features using CountVectorizer (Bag of Words).
   
3.	*Modelling*: Training a MultinomialNB (Naive Bayes) classifier.
   
4.	*Evaluation*: Measuring accuracy and generating a classification report.
   
________________________________________

**🛠️ Requirements**

To run this project, you need Python installed along with the scikit-learn library.

  pip install scikit-learn

________________________________________

**📂 File Structure**

•	*main.py*: The core script containing the dataset, model training, and prediction logic.

•	*README.md*: Project documentation (this file).

________________________________________

**🧠 How It Works**

1. *Vectorization*
   
Computers don't understand words, they understand numbers. We use Count Vectorization to count the frequency of each word in the dataset.

2. *The Model*
   
We use the Multinomial Naive Bayes algorithm. It is highly effective for text data because it calculates the probability of a label based on the presence of specific keywords (like "Winner," "Urgent," or "Meeting").

3. *Data Alignment*
   
To prevent errors during the train_test_split, the script dynamically ensures the number of email subjects matches the number of labels:

    min_length = min(len(emails), len(labels))
    
    emails = list(emails)[:min_length]
    
    labels = labels[:min_length]

________________________________________

**📊 Performance**

The model is evaluated using:

•	*Accuracy Score*: The percentage of correct predictions.

•	*Classification Report*: Detailed metrics including Precision, Recall, and F1-Score.

________________________________________

**🔮 Usage**

To test the model with your own email subject, modify the new_email variable in main.py:

    new_email = ["Your flight confirmation is ready"]

    # Output: Predicted as not spam.


