## About the project
We live in a world that is increasingly focused around technology which impacts all aspects of our lives - and dating is no exception. There are over thousands of different online dating apps that people can choose to use. These apps use different algorithms and themes to match people based on attributes such as similar interests, sharing the same values, and income. In our research, we are seeking to understand which attributes are most important to a successful match. This will benefit people who are seeking a partner but are unsure of what attributes to prioritize. **We utilized machine learning techniques such as decision tree classification and principal component analysis to predict to predict whether or not two people would match based on how they rated each other.** 

This project was completed as part of a requirement for DS 3001: Foundations of Data Science at Worcester Polytechnic Institute. 

## The Data
The dataset was collected by Columbia Business School from 2002 and 2004 to determine what attributes participants liked about their date at the end of a speed dating round. Participants rated their dates based on attractiveness, sincerity, intelligence, fun, ambition, shared interests. The dataset also has preliminary data on participants’ answers to a questionnaire before participating in the process. This preliminary data includes demographics, dating habits, self-perception across key attributes, beliefs on what others fine valuable in a date, lifestyle information. 

## Preprocessing the data

| ![Before image](/Images/before.JPG) |
|:---:| 
| *Screenshot showing the way data was organized before preprocessing. Note that each row represented one individual and only their partner's ID.* |

<br> <br> The original dataset contained information about how each participant felt about their dates. Each participant was given a unique ID ('iid') and their partner's unique ID ('pid'). Together, these could be used to identify pairs. In our data cleaning process, we used these two IDs to match up participants and created a processed dataset where each row represented one pair. After preprocessing, the dataset looked like this:

| ![After image](/Images/after.JPG) |
|:---:| 
| *Screenshot showing the way data was organized after preprocessing. Note that each row represents both individuals in a pair and their respective ID's.* |

<br><br>Here's an excerpt from the preprocessing code which converted the raw data (each row represents one individual) to our final data format (each row represents a pair of individuals). <br>


```markdown
## Iterate through the dataset, find pairs and create new dataframe
## with partners on the same row
for index_1, row_1 in dataset.iterrows():
    for index_2, row_2 in dataset.iterrows():
        ## Acount for rows being deleted
        if(index_1 >= dataset.shape[0] or index_2 >= dataset.shape[0]):
            break
        ## Check for pairs
        if((dataset.iloc[index_1]['iid'] == dataset.iloc[index_2]['pid']) and (dataset.iloc[index_1]['pid'] == dataset.iloc[index_2]['iid'])):
            ## If pair is found, create a new row for the post processed dataset
            new_row = pd.concat([dataset.iloc[index_1], dataset.iloc[index_2]], axis = 0)
            ## Delete rows already used (for efficiency) and reset index
            dataset.drop(index_1, inplace=True)
            dataset.drop(index_2, inplace=True)
            dataset = dataset.reset_index(drop=True)
            ## Concatenate new row to the post processed dataset
            new_df = pd.concat([new_df, new_row], axis = 1)

    ## So we can see status while code runs        
    sys.stdout.write('.'); sys.stdout.flush();

```

## Insights
### Fun vs Like

| ![Fun vs like image](/Images/fun_vs_like.png) |
|:---:| 
| *Screenshot of scattered heatmap showing how much the participant liked their date on the x axis, and how fun they thought they were on the y axis. Red points indicate a match and blue points indicate no match.* |

### Attractive vs Like

| ![Attractive vs like image](/Images/attractive_vs_like.png) |
|:--:| 
| *Screenshot of scattered heatmap showing how much the participant liked their date on the x axis, and how attractive they thought they were on the y axis. Red points indicate a match and blue points indicate no match.* |

### Ambition vs Like

| ![ambition vs like image](/Images/ambition_vs_like.png) |
|:--:| 
| *Screenshot of scattered heatmap showing how much the participant liked their date on the x axis, and how fun they thought they were on the y axis. Red points indicate a match and blue points indicate no match.* |

### Intelligence vs Like

| ![Intelligence vs like image](/Images/intelligence_vs_like.png) |
|:--:| 
| *Screenshot of scattered heatmap showing how much the participant liked their date on the x axis, and how intelligent they thought they were on the y axis. Red points indicate a match and blue points indicate no match.* |

### Sincerity vs Like

| ![Sincerity vs like image](/Images/sincerity_vs_like.png) |
|:--:| 
| *Screenshot of scattered heatmap showing how much the participant liked their date on the x axis, and how sincere they thought they were on the y axis. Red points indicate a match and blue points indicate no match.* |

### Shared Interests vs Like

| ![sharedinterests vs like image](/Images/sharedinterests_vs_like.png) |
|:--:| 
| *Screenshot of scattered heatmap showing how much the participant liked their date on the x axis, and how many shared interests on the y axis. Red points indicate a match and blue points indicate no match.* |


## Machine Learning
To design our machine learning algorithm we used random forest classification.

```markdown
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import sys

dataset = pd.read_csv('/Users/robertoesquivel/Desktop/combined_data_no_repeats.csv')
new_df = pd.concat([dataset.iloc[0], dataset.iloc[1]], axis = 0)

#new_df = new_df.T
n = dataset.shape[0]

dataset.shape

dataset.columns

print('Data Types:')
for i in dataset.columns:
    t = dataset[i].dtype
    if t != float:
        print(i, t)


dataset = dataset[['match',
       'int_corr', 'samerace', 'age_o', 'attr1_1', 'sinc1_1', 'intel1_1',
       'fun1_1', 'amb1_1', 'shar1_1', 'attr2_1', 'sinc2_1', 'intel2_1',
       'fun2_1', 'amb2_1', 'shar2_1', 'attr3_1', 'sinc3_1', 'fun3_1',
       'intel3_1', 'amb3_1', 'attr', 'sinc', 'intel', 'fun', 'amb',
       'shar', 'like', 'prob', 'int_corr.1', 'attr1_1.1',
       'sinc1_1.1', 'intel1_1.1', 'fun1_1.1', 'amb1_1.1', 'shar1_1.1',
       'attr2_1.1', 'sinc2_1.1', 'intel2_1.1', 'fun2_1.1', 'amb2_1.1',
       'shar2_1.1', 'attr3_1.1', 'sinc3_1.1', 'fun3_1.1', 'intel3_1.1',
       'amb3_1.1', 'attr.1', 'sinc.1', 'intel.1', 'fun.1', 'amb.1',
       'shar.1', 'like.1', 'prob.1']]

dataset['com_attr'] = dataset['attr'] + dataset['attr.1']
dataset['com_sinc'] = dataset['sinc'] + dataset['sinc.1']
dataset['com_intel'] = dataset['intel'] + dataset['intel.1']
dataset['com_fun'] = dataset['fun'] + dataset['fun.1']
dataset['com_amb'] = dataset['amb'] + dataset['amb.1']
dataset['com_shar'] = dataset['shar'] + dataset['shar.1']
dataset['com_like'] = dataset['like'] + dataset['like.1']

c = dataset['match'].sum() / float(len(dataset))
if c > 0.1:
    print(c)

dataset = dataset[['match', 'int_corr', 'attr1_1', 'sinc1_1',
       'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1', 'attr2_1', 'sinc2_1',
       'intel2_1', 'fun2_1', 'amb2_1', 'shar2_1', 'attr3_1', 'sinc3_1',
       'fun3_1', 'intel3_1', 'amb3_1', 'attr1_1.1', 'sinc1_1.1',
       'intel1_1.1', 'fun1_1.1', 'amb1_1.1', 'shar1_1.1', 'attr2_1.1',
       'sinc2_1.1', 'intel2_1.1', 'fun2_1.1', 'amb2_1.1', 'shar2_1.1',
       'attr3_1.1', 'sinc3_1.1', 'fun3_1.1', 'intel3_1.1', 'amb3_1.1', 'com_attr', 'com_sinc', 'com_intel', 'com_fun', 'com_amb',
       'com_shar', 'com_like']]

dataset.shape

X = dataset.iloc[:, 1:44].values
y = dataset.iloc[:, 0].values


X.shape

y.shape

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

cm

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

accuracies

accuracies.mean()

matches = dataset[dataset['match'] == 1]

652 / 3984

balanced_data = dataset.copy()

balanced_data = balanced_data[balanced_data['match'] == 0].sample(n = 652)

balanced_data = pd.concat(objs = [balanced_data, matches])

balanced_data = balanced_data.reset_index()

dataset = balanced_data

dataset.columns

dataset = dataset[['match', 'int_corr', 'attr1_1', 'sinc1_1', 'intel1_1',
       'fun1_1', 'amb1_1', 'shar1_1', 'attr2_1', 'sinc2_1', 'intel2_1',
       'fun2_1', 'amb2_1', 'shar2_1', 'attr3_1', 'sinc3_1', 'fun3_1',
       'intel3_1', 'amb3_1', 'attr1_1.1', 'sinc1_1.1', 'intel1_1.1',
       'fun1_1.1', 'amb1_1.1', 'shar1_1.1', 'attr2_1.1', 'sinc2_1.1',
       'intel2_1.1', 'fun2_1.1', 'amb2_1.1', 'shar2_1.1', 'attr3_1.1',
       'sinc3_1.1', 'fun3_1.1', 'intel3_1.1', 'amb3_1.1', 'com_attr',
       'com_sinc', 'com_intel', 'com_fun', 'com_amb', 'com_shar', 'com_like']]

dataset

X = dataset.iloc[:, 1:44].values
y = dataset.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

cm

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

accuracies

accuracies.mean()

classifier.feature_importances_.shape

index = ['int_corr', 'attr1_1', 'sinc1_1', 'intel1_1',
       'fun1_1', 'amb1_1', 'shar1_1', 'attr2_1', 'sinc2_1', 'intel2_1',
       'fun2_1', 'amb2_1', 'shar2_1', 'attr3_1', 'sinc3_1', 'fun3_1',
       'intel3_1', 'amb3_1', 'attr1_1.1', 'sinc1_1.1', 'intel1_1.1',
       'fun1_1.1', 'amb1_1.1', 'shar1_1.1', 'attr2_1.1', 'sinc2_1.1',
       'intel2_1.1', 'fun2_1.1', 'amb2_1.1', 'shar2_1.1', 'attr3_1.1',
       'sinc3_1.1', 'fun3_1.1', 'intel3_1.1', 'amb3_1.1', 'com_attr',
       'com_sinc', 'com_intel', 'com_fun', 'com_amb', 'com_shar', 'com_like']

feature_importances = pd.Series(classifier.feature_importances_, index=index)
feature_importances.sort_values(ascending=False)
ax = feature_importances.plot(kind='bar', figsize = (15,10))
ax.set(ylabel='Importance (Gini Coefficient)', title='Feature importances');

dataset = dataset[['match', 'int_corr', 'attr1_1', 'sinc1_1',
       'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1', 'attr2_1', 'sinc2_1',
       'intel2_1', 'fun2_1', 'amb2_1', 'shar2_1', 'attr3_1', 'sinc3_1',
       'fun3_1', 'intel3_1', 'amb3_1', 'attr1_1.1', 'sinc1_1.1', 'intel1_1.1',
       'fun1_1.1', 'amb1_1.1', 'shar1_1.1', 'attr2_1.1', 'sinc2_1.1',
       'intel2_1.1', 'fun2_1.1', 'amb2_1.1', 'shar2_1.1', 'attr3_1.1',
       'sinc3_1.1', 'fun3_1.1', 'intel3_1.1', 'amb3_1.1']]

dataset.shape

X = dataset.iloc[:, 1:37].values
y = dataset.iloc[:, 0].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

cm

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

accuracies

accuracies.mean()

classifier.feature_importances_

classifier.feature_importances_.sum()

index = ['int_corr', 'attr1_1', 'sinc1_1',
       'intel1_1', 'fun1_1', 'amb1_1', 'shar1_1', 'attr2_1', 'sinc2_1',
       'intel2_1', 'fun2_1', 'amb2_1', 'shar2_1', 'attr3_1', 'sinc3_1',
       'fun3_1', 'intel3_1', 'amb3_1', 'attr1_1.1', 'sinc1_1.1', 'intel1_1.1',
       'fun1_1.1', 'amb1_1.1', 'shar1_1.1', 'attr2_1.1', 'sinc2_1.1',
       'intel2_1.1', 'fun2_1.1', 'amb2_1.1', 'shar2_1.1', 'attr3_1.1',
       'sinc3_1.1', 'fun3_1.1', 'intel3_1.1', 'amb3_1.1']

feature_importances = pd.Series(classifier.feature_importances_, index=index)
feature_importances.sort_values(ascending=False)
ax = feature_importances.plot(kind='bar', figsize = (15, 10))
ax.set(ylabel='Importance (Gini Coefficient)', title='Feature importances');
```

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/hr23232323/love-at-first-swipe/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
