## About the project
We live in a world that is increasingly focused around technology which impacts all aspects of our lives - and dating is no exception. There are over thousands of different online dating apps that people can choose to use. These apps use different algorithms and themes to match people based on attributes such as similar interests, sharing the same values, and income. In our research, we are seeking to understand which attributes are most important to a successful match. This will benefit people who are seeking a partner but are unsure of what attributes to prioritize. **We utilized random forest classification to predict whether or not two people would match based on how they rated each other.** 

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
In our initial insights, we compared whether the pair had a connection ("Like") against each of the 6 attributes we were testing: Fun, Attractiveness, Ambition, Intelligence, Sincerity, and Shared Interests. We discovered that the two attributes with the strongest correlation to "Like" were Fun and Ambition, and all 6 graphs are shown below.

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
**To design our machine learning algorithm we used random forest classification.** We combined the attributes of the two individuals in a pair by adding their respective scores for each other. This enabled us to have a single combined attribute which the machine learning model could then predict based on. An issue that we originally ran into was that the dataset only contained approximately 20% "matches". This was because many more people rejected the people that they met, rather than matched with them. While this represented the real world situation very well, the machine learning model initially was overtrained to predict "no match". We tackled this issue by random sampling an even split between "matches" and "no matches" and trained the classifier on that data. This resulted in an increased prediction of true positives. 

An excerpt from our final code for training and prediction can be seen below: 

```markdown
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
```

The full code can be accessed using this [link to repository](https://github.com/hr23232323/love-at-first-swipe)


## Feature Importances

| ![Feature Importances](/Images/dating_feature_imp.png) |
|:--:| 
| *Screenshot of bar graph representing the feature importances in the random forest classifier. It can be appreciated that the factors that influence prediction the most are the degree of attractiveness that a person is rated on and how much "fun" they are.* |

## Confusion Matrix

| ![confusion matrix](/Images/confusionmatrix.png) |
|:--:| 
| *Screenshot of chart representing our true positives, true negatives, false positives, and false negatives after running the machine learning.* |

## Conclusion
In conclusion, we learned through our research that fun and attractiveness are the two attributes that people consider the most seriously when looking for a romantic partner. This was a finding that was indicated in our early insights after performing our initial data cleansing, and was supported by our classifier, as shown in feature importances. **Additionally, using 10-fold cross validation, we found a mean accuracy of 75.4% when trying to predict whether two people would match.**


