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
| *Screenshot of scattered heatmap showing how much the participant liked their date on the x axis, and how fun they thought they were on the y axis. Red points indicate a match and blue points indicate no match.* | <br>

### Attractive vs Like
| ![Attractive vs like image](/Images/attractive_vs_like.png) |
|:---:| 
| *Screenshot of scattered heatmap showing how much the participant liked their date on the x axis, and how attractive they thought they were on the y axis. Red points indicate a match and blue points indicate no match.* | <br>

### Ambition vs Like
| ![ambition vs like image](/Images/ambition_vs_like.png) |
|:---:| 
| *Screenshot of scattered heatmap showing how much the participant liked their date on the x axis, and how fun they thought they were on the y axis. Red points indicate a match and blue points indicate no match.* | <br>

### Intelligence vs Like
| ![Intelligence vs like image](/Images/intelligence_vs_like.png) |
|:---:| 
| *Screenshot of scattered heatmap showing how much the participant liked their date on the x axis, and how intelligent they thought they were on the y axis. Red points indicate a match and blue points indicate no match.* | <br>

### Sincerity vs Like
| ![Sincerity vs like image](/Images/sincerity_vs_like.png) |
|:---:| 
| *Screenshot of scattered heatmap showing how much the participant liked their date on the x axis, and how sincere they thought they were on the y axis. Red points indicate a match and blue points indicate no match.* | <br>

### Shared Interests vs Like
| ![sharedinterests vs like image](/Images/sharedinterests_vs_like.png) |
|:---:| 
| *Screenshot of scattered heatmap showing how much the participant liked their date on the x axis, and how many shared interests on the y axis. Red points indicate a match and blue points indicate no match.* | <br>

You can use the [editor on GitHub](https://github.com/hr23232323/love-at-first-swipe/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for



[Link](url) and ![Image](src)

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/hr23232323/love-at-first-swipe/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
