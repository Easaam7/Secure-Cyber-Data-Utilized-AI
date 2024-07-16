##################################### Imports ######################################################################
import pandas as pd
import numpy as np
import seaborn as sns
import random
import csv
from joblib import dump,load
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
###############################################################################################
##################################### 2000 Email Extraction ##########################################################
# Load the original CSV file
input_filename = 'merged_file.csv'  # Change this to the filename of your original CSV file
df = pd.read_csv(input_filename)

# Select 2000 random rows (emails) from the DataFrame
random_emails = df.sample(n=2000, random_state=42)  # Change the random_state if you want different random samples

# Save the randomly selected emails to a new CSV file
output_filename = 'random_2000_emails.csv'
random_emails.to_csv(output_filename, index=False)

print(f"Random emails saved to {output_filename}")

# Remove the randomly selected emails from the original DataFrame
df = df.drop(random_emails.index)

# Save the modified DataFrame back to a new CSV file without the randomly selected emails
new_output_filename = 'remaining_emails.csv'
df.to_csv(new_output_filename, index=False)

print(f"Remaining emails saved to {new_output_filename}")
#########################################################################################################


##################################### Training ##########################################################
#Import the Dataset
df = pd.read_csv("remaining_emails.csv")
df.head()

# Remove Null Values
df.isna().sum()
df = df.dropna()
print(df.isna().sum())
# Count email after removing nulls
email_type_counts = df['Email Type'].value_counts()
print(email_type_counts)
# how many unique email
unique_email_types = email_type_counts.index.tolist()
# make color map for phishing Email , Safe Email
color_map = {
    'Phishing Email': 'red',
    'Safe Email': 'green',}
# unique email figure
colors = [color_map.get(email_type, 'gray') for email_type in unique_email_types]
plt.figure(figsize=(8, 6))
plt.bar(unique_email_types, email_type_counts, color=colors)
plt.xlabel('Email Type')
plt.ylabel('Count')
plt.title('Distribution of Email Types with Custom Colors')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Divide emails for test and train
Safe_Email = df[df["Email Type"]== "safe"]
Phishing_Email = df[df["Email Type"]== "phishing"]
Safe_Email = Safe_Email.sample(Phishing_Email.shape[0])
Safe_Email.shape,Phishing_Email.shape
Data= pd.concat([Safe_Email, Phishing_Email], ignore_index = True)
Data.head()
X = Data["Email Text"].values
y = Data["Email Type"].values
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.7, random_state = 0)
X_train_df = pd.DataFrame(X_train, columns=["Email Text"])
X_train_df.to_csv('X_train.csv', index=False)
X_test_df = pd.DataFrame(X_test, columns=["Email Text"])
X_test_df.to_csv('X_test.csv', index=False)
# create models
models = [
    (Pipeline([("tfidf", TfidfVectorizer()), ("classifier", RandomForestClassifier())]), {'classifier__n_estimators': [10]}),
    (Pipeline([("tfidf", TfidfVectorizer()), ("classifier", DecisionTreeClassifier())]), {'classifier__max_depth': [None, 10]}),
    (Pipeline([("tfidf", TfidfVectorizer()), ("classifier", AdaBoostClassifier())]), {'classifier__n_estimators': [50], 'classifier__learning_rate': [1.0]}),
    (Pipeline([("tfidf", TfidfVectorizer()), ("classifier", SGDClassifier())]), {'classifier__max_iter': [1000], 'classifier__tol': [1e-3]}),
    (Pipeline([("tfidf", TfidfVectorizer()), ("classifier", LogisticRegression())]), {'classifier__C': [1.0], 'classifier__max_iter': [100]})
]
accuracy_test = []
model_names = []
trained_models = []
for pipeline, params in models:
    model_name = pipeline.steps[-1][1].__class__.__name__
    print(f'###### Model => {model_name}')

    grid_search = GridSearchCV(pipeline, params, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    # Save the trained model
    model_filename = f'{model_name}.joblib'
    dump(best_model, model_filename)
    # Append the trained model to the list
    trained_models.append((model_name, model_filename))
    pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    accuracy_test.append(acc)
    model_names.append(model_name)
    print(f'Test Accuracy : {acc * 100:.5f}%')
    print('Classification Report')
    print(classification_report(y_test, pred))
    print('Confusion Matrix')
    cf_matrix = confusion_matrix(y_test, pred)
    sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
    plt.show()
    print('################### End ###################')
model_series = pd.Series(model_names, name='Model').astype(str)
accuracy_series = pd.Series(accuracy_test, name='Accuracy')
output = pd.concat([model_series, accuracy_series], axis=1)
output
plt.figure(figsize=(12, 7))
plots = sns.barplot(x='Model', y='Accuracy', data=output)
for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2%'),
                   (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')
plt.xlabel("Models", fontsize=14)
plt.ylabel("Precision", fontsize=14)
plt.title("Compare the precision of models", fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
pipelines = [
    Pipeline([("tfidf", TfidfVectorizer()), ("classifier", RandomForestClassifier(n_estimators=10))]),
    Pipeline([("tfidf", TfidfVectorizer()), ("classifier", DecisionTreeClassifier())]),
    Pipeline([("tfidf", TfidfVectorizer()), ("classifier", AdaBoostClassifier())]),
    Pipeline([("tfidf", TfidfVectorizer()), ("classifier", SGDClassifier())]),
    Pipeline([("tfidf", TfidfVectorizer()), ("classifier", LogisticRegression())])
]
accuracy_test = []
model_names = []
for pipeline in pipelines:
    model_name = pipeline.steps[-1][1].__class__.__name__
    print(f'###### Model => {model_name}')

    # Fit the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    pred = pipeline.predict(X_test)
    # test for one email
    testFile = pd.read_csv('X_train.csv')
    email = testFile.iloc[0]

    # Evaluate the model
    acc = accuracy_score(y_test, pred)
    accuracy_test.append(acc)
    model_names.append(model_name)

    print(f'Test Accuracy : {acc * 100:.5f}%')
    print('Classification Report')
    print(classification_report(y_test, pred))
    print('Confusion Matrix')
    cf_matrix = confusion_matrix(y_test, pred)
    sns.heatmap(cf_matrix / np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')
    plt.show()
    print('################### End ###################')
model_series = pd.Series(model_names, name='Model').astype(str)
accuracy_series = pd.Series(accuracy_test, name='Accuracy')
output = pd.concat([model_series, accuracy_series], axis=1)
output
plt.figure(figsize=(12, 7))
plots = sns.barplot(x='Model', y='Accuracy', data=output)

for bar in plots.patches:
    plots.annotate(format(bar.get_height(), '.2%'),
                   (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   ha='center', va='center',
                   size=15, xytext=(0, 8),
                   textcoords='offset points')
plt.xlabel("Models", fontsize=14)
plt.ylabel("Precision", fontsize=14)
plt.title("Compare the precision of models", fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
##################################### Test random email from 2000 email #######################
# Load the test email
test_email_filename = 'random_2000_emails.csv'  # Change this to the filename of your test email file
test_email_df = pd.read_csv(test_email_filename)
# safe email 10 ,  phishing email 37
f = open('op2.csv', 'w',newline="")

writer = csv.writer(f)


for i in range(2000):
 test_email_text = test_email_df.iloc[i]['Email Text']
 test_email_type = test_email_df.iloc[i]['Email Type']
 #print("Email Text:")
 #print(f"Test Email Text: {test_email_text}")
 writer.writerow([test_email_text , test_email_type])
 #print("-----------------------------------------------------")
f.close()
# List of model filenames
import pandas as pd
from joblib import load

# Load the dataset
dataset = pd.read_csv('random_2000_emails.csv')

# Extract the email texts from the dataset
email_texts = dataset['Email Text'].tolist()

# List of model filenames
model_filenames = [
    'RandomForestClassifier.joblib', 
    'DecisionTreeClassifier.joblib',
    'SGDClassifier.joblib',
    'AdaBoostClassifier.joblib',
    'LogisticRegression.joblib'
    # Add more filenames as needed
]

# Iterate over each model and make predictions for each email in the dataset
f = open('op3.csv', 'w',newline="")
writer = csv.writer(f)
for model_filename in model_filenames:
    try:
        # Load the saved model
        model = load(model_filename)
        # Iterate over each email and make a prediction
        for email_text in email_texts:
            prediction = model.predict([email_text])[0]
            # Print the prediction result
            #print("Prediction Result:")
           # print(f"Model: {model_filename}")
            #print(f"Test Email: {email_text}")
           # print(f"Predicted Email Type: {prediction}")
            writer.writerow([model_filename , email_text, prediction])
            #print("-----------------------------------------------------")
    except FileNotFoundError:
        print(f"Error: The model file {model_filename} was not found.")
    except Exception as e:
        print(f"An error occurred while predicting with {model_filename}: {e}")
f.close()        
#########################################################################################################

################################################   GUI   ###########################################
import tkinter as tk
from tkinter import messagebox
import joblib

# Load the scikit-learn model
scikit_model = joblib.load('SGDClassifier.joblib')

def predict_email_type(email_text):
    # Make prediction using the loaded scikit-learn model
    predicted_email_type = scikit_model.predict([email_text])[0]
    
    return predicted_email_type

def get_prediction():
    # Get the text from the input field
    email_text = entry.get("1.0",'end-1c')
    
    # Check if text is empty
    if not email_text:
        messagebox.showwarning("Warning", "Please enter some text.")
        return
    
    # Get prediction
    predicted_type = predict_email_type(email_text)
    
    # Show prediction in a messagebox
    messagebox.showinfo("Prediction", f"The predicted email type is: {predicted_type}")


# Create main window
root = tk.Tk()
root.title("Email Type Prediction")

# Create input field
entry = tk.Text(root, height=10, width=50)
entry.pack()

# Create predict button
predict_button = tk.Button(root, text="Predict", command=get_prediction)
predict_button.pack()

# Run the main event loop
root.mainloop()


