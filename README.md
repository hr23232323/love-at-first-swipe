## About the project
We live in a world that is increasingly focused around technology which impacts all aspects of our lives - and dating is no exception. There are over thousands of different online dating apps that people can choose to use. These apps use different algorithms and themes to match people based on attributes such as similar interests, sharing the same values, and income. In our research, we are seeking to understand which attributes are most important to a successful match. This will benefit people who are seeking a partner but are unsure of what attributes to prioritize. **We utilized machine learning techniques such as decision tree classification and principal component analysis to predict to predict whether or not two people would match based on how they rated each other.** 

This project was completed as part of a requirement for DS 3001: Foundations of Data Science at Worcester Polytechnic Institute. 

## The Data
The dataset was collected by Columbia Business School from 2002 and 2004 to determine what attributes participants liked about their date at the end of a speed dating round. Participants rated their dates based on attractiveness, sincerity, intelligence, fun, ambition, shared interests. The dataset also has preliminary data on participants’ answers to a questionnaire before participating in the process. This preliminary data includes demographics, dating habits, self-perception across key attributes, beliefs on what others fine valuable in a date, lifestyle information. 

## Preprocessing the data
The original dataset contained information about how each participant felt about their dates. Each participant was given a unique ID ('iid') and their partner's unique ID ('pid'). Together, these could be used to identify pairs. In our data cleaning process, we used these two IDs to match up participants and created a processed dataset where each row represented one pair. 

![Image](/Images/before.JPG)

## Attribute 


You can use the [editor on GitHub](https://github.com/hr23232323/love-at-first-swipe/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/hr23232323/love-at-first-swipe/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
