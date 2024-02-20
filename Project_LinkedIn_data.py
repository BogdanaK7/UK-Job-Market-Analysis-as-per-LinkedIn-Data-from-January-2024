#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("Linkedin jobs UK 21.01.2024 original dataset.csv")


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.info()


# In[7]:


LinkedIn_df = df.copy()


# In[8]:


LinkedIn_df.head()


# In[9]:


LinkedIn_df.info()


# In[10]:


LinkedIn_df = LinkedIn_df.drop('Unnamed: 14', axis=1)


# In[11]:


LinkedIn_df.head()


# In[12]:


LinkedIn_df = LinkedIn_df.drop('Unnamed: 15', axis=1)


# In[13]:


LinkedIn_df.head()


# In[14]:


LinkedIn_df = LinkedIn_df.drop('job_link', axis=1)


# In[15]:


LinkedIn_df.head()


# In[16]:


LinkedIn_df = LinkedIn_df.drop('last_processed_time', axis=1)


# In[17]:


LinkedIn_df.head()


# In[18]:


LinkedIn_df = LinkedIn_df.drop('is_being_worked', axis=1)


# In[19]:


LinkedIn_df = LinkedIn_df.drop('first_seen', axis=1)


# In[20]:


LinkedIn_df.head()


# In[21]:


# Renaming the columns for consistency
LinkedIn_df = LinkedIn_df.rename(columns={'got_summary': 'Summary','got_ner': 'Nerrative','job_title': 'Job_Title',
                                          'company': 'Company','job_location': 'Job_Location', 'search_city': 'City',
                                          'search_country': 'Country', 'search_position': 'Position',
                                          'job_level': 'Job_level','job_type': 'Job_Type'})


# In[22]:


LinkedIn_df.head()


# In[23]:


column_to_move = 'Country'

# Extracting the column
column = LinkedIn_df.pop(column_to_move)

# Inserting the column at the beginning
LinkedIn_df.insert(0, column.name, column)
LinkedIn_df.head()


# In[24]:


column_to_move2 = 'City'

# Extract the column
column = LinkedIn_df.pop(column_to_move2)

# Insert the column at the beginning
LinkedIn_df.insert(0, column.name, column)
LinkedIn_df.head()


# In[25]:


column_to_move3 = 'Job_Location'

# Extract the column
column = LinkedIn_df.pop(column_to_move3)

# Insert the column at the beginning
LinkedIn_df.insert(0, column.name, column)
LinkedIn_df.head()


# In[26]:


column_to_move4 = 'Company'

# Extract the column
column = LinkedIn_df.pop(column_to_move4)

# Insert the column at the beginning
LinkedIn_df.insert(0, column.name, column)
LinkedIn_df.head()


# In[27]:


column_to_move5 = 'Job_level'

# Extract the column
column = LinkedIn_df.pop(column_to_move5)

# Insert the column at the beginning
LinkedIn_df.insert(0, column.name, column)
LinkedIn_df.head()


# In[28]:


column_to_move6 = 'Job_Type'

# Extract the column
column = LinkedIn_df.pop(column_to_move6)

# Insert the column at the beginning
LinkedIn_df.insert(0, column.name, column)
LinkedIn_df.head()


# In[29]:


column_to_move7 = 'Job_Type'

# Extract the column
column = LinkedIn_df.pop(column_to_move7)

# Insert the column at the beginning
LinkedIn_df.insert(0, column.name, column)
LinkedIn_df.head()


# In[30]:


column_to_move8 = 'Position'

# Extract the column
column = LinkedIn_df.pop(column_to_move8)

# Insert the column at the beginning
LinkedIn_df.insert(0, column.name, column)
LinkedIn_df.head()


# In[31]:


column_to_move9 = 'Job_Title'

# Extract the column
column = LinkedIn_df.pop(column_to_move9)

# Insert the column at the beginning
LinkedIn_df.insert(0, column.name, column)
LinkedIn_df.head()


# In[32]:


#Checking for null values
null_columns = LinkedIn_df.isnull().any(axis=0)
print(null_columns)


# In[33]:


#Filtering the dataframe to show only the rows where the specified column contain null values
rows_with_null = LinkedIn_df[LinkedIn_df["Job_Location"].isnull()]
print(rows_with_null)


# In[34]:


# Count the number of null values in column 'Job_location'
column_name = 'Job_Location'  
null_count = LinkedIn_df[column_name].isnull().sum()
print(null_count)


# In[35]:


#Replace null values in column "Job_Location" with "No information"
column_name = "Job_Location"
LinkedIn_df[column_name] = LinkedIn_df[column_name].fillna("No information")


# In[36]:


#Check for null values in the DataSet to ensure that no null values are left
null_values_count = LinkedIn_df.isnull().sum()
print(null_values_count)


# In[37]:


#Replace null values in column "Company" with "No information"
column_name = "Company"
LinkedIn_df[column_name] = LinkedIn_df[column_name].fillna("No information")


# In[38]:


null_values_count = LinkedIn_df.isnull().sum()
print(null_values_count)


# In[39]:


print(LinkedIn_df.columns)


# In[40]:


# Creating a new dataset containing only the data related to the United Kingdom
uk_data = LinkedIn_df[LinkedIn_df['Country'] == 'United Kingdom']


print(uk_data)


# In[41]:


uk_data.info()


# In[42]:


# Replace 't' with True and 'False' with False in both columns "Summary" and "Nerrative"
uk_data.loc[:, 'Summary'] = uk_data['Summary'].replace({'t': True, 'False': False})
uk_data.loc[:, 'Nerrative'] = uk_data['Nerrative'].replace({'t': True, 'False': False})
print(uk_data)


# In[43]:


# Get unique job titles
unique_job_titles = uk_data['Job_Title'].unique()

# Print unique job titles
print(unique_job_titles)


# In[44]:


# Filter rows containing "data" or "Data"
data_jobs_df = uk_data[uk_data['Job_Title'].str.contains('data|Data')]

# Print DataFrame with jobs containing "data" or "Data"
print(data_jobs_df)


# In[45]:


# Count the occurrences of jobs containing "Data" or "data", "analysis", etc.
data_jobs_count7 = uk_data['Job_Title'].str.contains('Data|data|Database|database|Analyst|analyst|Analytics|analytics|\
                                Analysis|analysis|Software|software|Product|product|BI|bi\
                              Python|python|Azure|azure|programming|Programming|developer|Developer|\
                              Digital|digital|Research|research|Quantitative|quantitative|Insight|insight|Machine learning|\
                              |machine learning|AI|ai').sum()

# Count the occurrences of jobs not containing "Data" or "data"
other_jobs_count7 = len(uk_data) - data_jobs_count7

# Create a horizontal bar chart
jobs = ['Data Jobs', 'Other Jobs']
counts = [data_jobs_count7, other_jobs_count7]

plt.figure(figsize=(8, 6))
plt.barh(jobs, counts, color=['lightblue', 'lightgreen'])
plt.xlabel('Number of Jobs')
plt.title('Comparison of Data Jobs with Other Jobs in the UK (LinkedIn data 2024)')
plt.gca().invert_yaxis()  # Invert y-axis to have "Data Jobs" on top
plt.show()


# In[46]:


# Counting the occurrences of jobs containing "Data", "data", etc.
data_jobs_count1 = uk_data['Job_Title'].str.contains('Data|data|Database|database|Analyst|analyst|Analytics|analytics|\
                                Analysis|analysis|Software|software|Product|product|BI|bi\
                              Python|python|Azure|azure|programming|Programming|developer|Developer|\
                              Digital|digital|Research|research|Quantitative|quantitative|Insight|insight|Machine learning|\
                              |machine learning|AI|ai').sum()

# Counting the occurrences of jobs not containing "Data" or "data", etc.
other_jobs_count1 = len(uk_data) - data_jobs_count1

# Creating a pie chart
labels = ['Data Jobs', 'Other Jobs']
sizes = [data_jobs_count1, other_jobs_count1]
colors = ['lightblue', 'lightgreen']
explode = (0.1, 0)  # explode to highlight the 1st slice which is for Data Jobs

plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  #to ensure the pie is drawn as a cicle
plt.title('Comparison of Data Jobs with Other Jobs in the UK (LinkedIn data January 2024)')
plt.show()


# In[71]:


# Counting the occurrences of jobs containing "Data", "data", etc.
data_jobs_count1 = uk_data['Job_Title'].str.contains('Data|data|Database|database|Analyst|analyst|Analytics|analytics|\
                                Analysis|analysis|Software|software|Product|product|BI|bi\
                              Python|python|Azure|azure|programming|Programming|developer|Developer|\
                              Digital|digital|Research|research|Quantitative|quantitative|Insight|insight|Machine learning|\
                              |machine learning|AI|ai').sum()

# Counting the occurrences of jobs not containing "Data" or "data", etc.
other_jobs_count1 = len(uk_data) - data_jobs_count1

# Total jobs
total_jobs = len(uk_data)

# Custom palette

custom_palette = ["skyblue", "lightgreen"]

# Creating a bar plot
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.barplot(x=['Data Jobs', 'Other Jobs'], y=[data_jobs_count1, other_jobs_count1], palette=["lightblue", "lightgreen"])
plt.title('Comparison of of Data Jobs with Other Jobs (in %)')
plt.ylabel('Count')

# Adding percentage labels
for i, count in enumerate([data_jobs_count1, other_jobs_count1]):
    plt.text(i, count + 0.5, f'{count / total_jobs:.1%}', ha='center')

plt.show()


# In[73]:


# Count the occurrences of jobs containing "Data", "data", etc
data_jobs_count1 = uk_data['Job_Title'].str.contains('Data|data|Database|database|Analyst|analyst|Analytics|analytics|\
                                Analysis|analysis|Software|software|Product|product|BI|bi\
                              python|Azure|programming|Programming|developer|Developer|\
                              Digital|digital|Research|research|Quantitative|quantitative|Insight|insight|Machine learning|\
                              |machine learning').sum()

# Count the occurrences of jobs not containing "Data" or "data"
other_jobs_count1 = len(uk_data) - data_jobs_count1

# Create a bar plot
plt.figure(figsize=(8, 6))
sns.barplot(x=pd.Series(['Data Jobs', 'Other Jobs']), y=pd.Series([data_jobs_count1, other_jobs_count1]), palette='Set2')
plt.title('Comparison of Data Jobs with Other Jobs in the UK (LinkedIn data 2024)')
plt.ylabel('Number of Jobs')
plt.show()


# In[74]:


# Counting the occurrences of jobs containing "Data", "data", etc.
data_jobs_count1 = uk_data['Job_Title'].str.contains('Data|data|Database|database|Analyst|analyst|Analytics|analytics|\
                                Analysis|analysis|Software|software|Product|product|BI|bi\
                              Python|python|Azure|azure|programming|Programming|developer|Developer|\
                              Digital|digital|Research|research|Quantitative|quantitative|Insight|insight|Machine learning|\
                              |machine learning|AI|ai').sum()

# Counting the occurrences of jobs not containing "Data" or "data", etc.
other_jobs_count1 = len(uk_data) - data_jobs_count1

# Total jobs
total_jobs = len(uk_data)

# Creating a bar plot
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.barplot(x=['Data Jobs', 'Other Jobs'], y=[data_jobs_count1, other_jobs_count1], palette="pastel")
plt.title('Comparison of Data Jobs with Other Jobs in the UK (LinkedIn data January 2024)')
plt.ylabel('Count')

# Adding percentage labels
for i, count in enumerate([data_jobs_count1, other_jobs_count1]):
    plt.text(i, count + 0.5, f'{count / total_jobs:.1%}', ha='center')

plt.show()


# In[75]:


from bokeh.plotting import figure, show
from bokeh.io import output_notebook

# Counting the occurrences of jobs containing "Data", "data", etc
data_jobs_count1 = uk_data['Job_Title'].str.contains('Data|data|Database|database|Analyst|analyst|Analytics|analytics|\
                                Analysis|analysis|Software|software|Product|product|BI|bi\
                              Python|python|Azure|azure|programming|Programming|developer|Developer|\
                              Digital|digital|Research|research|Quantitative|quantitative|Insight|insight|Machine learning|\
                              |machine learning|AI|ai').sum()

# Counting the occurrences of jobs not containing "Data" or "data", etc.
other_jobs_count1 = len(uk_data) - data_jobs_count1

# Preparing data for plotting
categories = ['Data Jobs', 'Other Jobs']
counts = [data_jobs_count1, other_jobs_count1]

# Creating a bar plot using Bokeh
output_notebook()
p = figure(x_range=categories, height=400, title='Comparison of Data Jobs with Other Jobs in the UK (in numbers)')
p.vbar(x=categories, top=counts, width=0.7, color=['skyblue', 'lightgreen'])

# Additional visual customization
p.xgrid.grid_line_color = None
p.y_range.start = 0
p.y_range.end = max(counts) + 10
p.xaxis.axis_label = 'Job Type'
p.yaxis.axis_label = 'Number of Jobs'

show(p)


# In[76]:


# Get unique job level
unique_job_level = uk_data['Job_level'].unique()

# Print unique job level
print(unique_job_level)


# In[77]:


# Count the occurrences of each job level
job_level_counts = uk_data['Job_level'].value_counts()

# Create a horizontal bar chart
plt.figure(figsize=(8, 6))
job_level_counts.plot(kind='barh', color='pink')
plt.xlabel('Number of Jobs')
plt.ylabel('Job Level')
plt.title('Count of Different Job Levels')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest count on top
plt.show()


# In[78]:


# Get unique job types
unique_job_types = uk_data['Job_Type'].unique()

# Print unique job types
print(unique_job_types)


# In[79]:


# Count occurrences of each job type
job_type_counts = uk_data['Job_Type'].value_counts()
job_type_counts


# In[80]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Create a pivot table to get the count of occurrences for each combination of search_country and job_title
pivot_table = uk_data.pivot_table(index='City', columns='Job_level', aggfunc='size', fill_value=0)

# Plot the heatmap
plt.figure(figsize=(20, 29))
sns.heatmap(pivot_table, cmap='YlGnBu', linewidths=.5)
plt.title('Heatmap of Job levels by City')
plt.xlabel('Job_level')
plt.ylabel('City')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# In[81]:


city_job_count = uk_data.groupby('City').size().reset_index(name='Job_Count')

# Now, city_job_count DataFrame will contain the count of jobs for each city.
# You can use this DataFrame to plot the interactive UK map.

print(city_job_count.head())  # Checking the first few rows to ensure correctness


# In[82]:


merged_data = pd.merge(uk_data, city_job_count, on='City', how='left')


# In[83]:


merged_data.head()


# In[84]:


# Group the data by city and count occurrences of each job title
city_job_counts = uk_data.groupby('City')['Job_Title'].value_counts()

# Group the data by country and count occurrences of each industry
country_industry_counts = uk_data.groupby('Country')['Job_Type'].value_counts()

# Determine the most in-demand job titles or industries for each city or country
most_in_demand_city_jobs = city_job_counts.groupby(level=0).nlargest(1)
most_in_demand_country_industries = country_industry_counts.groupby(level=0).nlargest(1)

print("Most in-demand job titles in different cities:")
print(most_in_demand_city_jobs)
print("\nMost in-demand job types in different countries:")
print(most_in_demand_country_industries)


# In[85]:


# Group the data by job title and count occurrences of each company
job_company_counts = uk_data.groupby('Job_Title')['Company'].value_counts()

# Define the specific job position you're interested in
specific_job_position = "Data Scientist"   # Replace with the desired job position

# Filter the data for the specific job position
specific_job_data = job_company_counts.loc[specific_job_position]

# Get the top companies hiring for the specific job position
top_companies = specific_job_data.nlargest(5)  # Change the number as needed

print(f"Top companies hiring for '{specific_job_position}':")
print(top_companies)


# In[86]:


import re

# Define the keywords pattern
keywords_pattern = re.compile(r'Data|data|Database|database|Analyst|analyst|Analytics|analytics|\
                                Analysis|analysis|Software|software|Product|product|BI|bi\
                              Python|python|Azure|azure|programming|Programming|developer|Developer|\
                              Digital|digital|Research|research|Quantitative|quantitative|Insight|insight|Machine learning|\
                              |machine learning|AI|ai', flags=re.IGNORECASE)

# Filter the dataset for job titles containing the keywords
filtered_data = uk_data[uk_data['Job_Title'].str.contains(keywords_pattern)]

# Group the filtered data by job title and count occurrences of each company
job_company_counts = filtered_data.groupby('Job_Title')['Company'].value_counts()

# Get the top companies for each job position
top_companies_by_job = job_company_counts.groupby(level=0).nlargest(5)  # Change the number as needed

print("Top companies hiring for specific job positions:")
print(top_companies_by_job)


# In[87]:


# Define the keywords pattern
keywords_pattern = re.compile(r'Data|data|Database|database|Analyst|analyst|Analytics|analytics|\
                                Analysis|analysis|Software|software|Product|product|BI|bi\
                              Python|python|Azure|azure|programming|Programming|developer|Developer|\
                              Digital|digital|Research|research|Quantitative|quantitative|Insight|insight|Machine learning|\
                              |machine learning|AI|ai', flags=re.IGNORECASE)

# Filter the dataset for job titles containing the keywords
filtered_data = uk_data[uk_data['Job_Title'].str.contains(keywords_pattern)]

# Group the filtered data by job title and count occurrences of each company
job_company_counts = filtered_data.groupby('Company')['Job_Title'].count()

# Get the top 10 companies offering data-related jobs
top_companies = job_company_counts.nlargest(15)

# Plotting
plt.figure(figsize=(10, 6))
top_companies.plot(kind='bar', color='skyblue')
plt.title('Top 15 Companies Offering Data-Related Jobs')
plt.xlabel('Company')
plt.ylabel('Number of Data Jobs')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[88]:


keywords_pattern = re.compile(r'Data|data|Database|database|Analyst|analyst|Analytics|analytics|\
                                Analysis|analysis|Software|software|Product|product|BI|bi\
                              Python|python|Azure|azure|programming|Programming|developer|Developer|\
                              Digital|digital|Research|research|Quantitative|quantitative|Insight|insight|Machine learning|\
                              |machine learning|AI|ai', flags=re.IGNORECASE)

# Filter the dataset for job titles containing the keywords
filtered_data = uk_data[uk_data['Job_Title'].str.contains(keywords_pattern)]

# Group the filtered data by job title and count occurrences of each company
job_company_counts = filtered_data.groupby('Company')['Job_Title'].count()

# Get the total number of data jobs in the UK
total_data_jobs = job_company_counts.sum()

# Calculate the percentage of data jobs offered by each company
percentage_data_jobs = (job_company_counts / total_data_jobs) * 100

# Get the top 15 companies offering data-related jobs
top_companies = percentage_data_jobs.nlargest(15)

# Sort the top companies by percentage of data jobs in descending order
top_companies = top_companies.sort_values(ascending=False)

# Plotting
plt.figure(figsize=(10, 8))
top_companies.plot(kind='bar', color='skyblue')
plt.title('Top 15 Companies Offering Data-Related Jobs (as % of total)')
plt.xlabel('Company')
plt.ylabel('Percentage of Data Jobs')
plt.gca().yaxis.set_major_formatter('{:.1f}%'.format)
plt.tight_layout()
plt.show()


# In[89]:


# Filtering the dataset for job titles containing the keywords
filtered_data = uk_data[uk_data['Job_Title'].str.contains(keywords_pattern)]

# Grouping the filtered data by job title and count occurrences per company
job_company_counts = filtered_data.groupby('Company')['Job_Title'].count()

# Total number of data jobs
total_data_jobs = job_company_counts.sum()

# % of data jobs per company
percentage_data_jobs = (job_company_counts / total_data_jobs) * 100

# Top 15 companues offering data-related jobs
top_companies = percentage_data_jobs.nlargest(15)

# Sorting top companies (descending order)
top_companies = top_companies.sort_values(ascending=False)

# Calculating the sum of %
sum_percentage = top_companies.sum()

# Plotting
plt.figure(figsize=(10, 6))
top_companies.plot(kind='bar', color='skyblue')
plt.title('Top 15 UK Companies Offering Data-Related Jobs (as % of total)')
plt.xlabel('Company')
plt.ylabel('Percentage of Data Jobs')
plt.gca().yaxis.set_major_formatter('{:.1f}%'.format)
plt.tight_layout()
plt.show()

print("Sum of percentages of the top 15 companies:", sum_percentage)


# In[90]:


# Filtering the dataset for job titles containing the keywords
filtered_data = uk_data[uk_data['Job_Title'].str.contains(keywords_pattern)]

# Grouping the filtered data by job title and count occurrences per company
job_company_counts = filtered_data.groupby('Company')['Job_Title'].count()

# Total number of data jobs
total_data_jobs = job_company_counts.sum()

# % of data jobs per company
percentage_data_jobs = (job_company_counts / total_data_jobs) * 100

# Top 15 companies offering data-related jobs
top_companies = percentage_data_jobs.nlargest(15)

# Sorting top companies (descending order)
top_companies = top_companies.sort_values(ascending=False)

# Calculating the sum of %
sum_percentage = top_companies.sum()

# Plotting
plt.figure(figsize=(10, 6))
top_companies.plot(kind='barh', color='navy')  
plt.title('Top 15 UK Companies Offering Data-Related Jobs (as % of total)')
plt.xlabel('Percentage of Data Jobs')
plt.ylabel('Company')
plt.gca().xaxis.set_major_formatter('{:.1f}%'.format)  
plt.tight_layout()
plt.show()

print("Sum of percentages of the top 15 companies:", sum_percentage)


# In[91]:


# Plotting
plt.figure(figsize=(10, 6))
top_companies.plot(kind='barh', color='navy')  

# Annotating the sum of percentages on the graph
for i, (company, percentage) in enumerate(top_companies.items()):
    plt.text(percentage, i, f'{percentage:.1f}%', va='center', fontsize=10, color='white')

# Adding the sum of percentages to the plot
plt.text(1, len(top_companies) + 0.5, f'Sum of percentages: {sum_percentage:.1f}%', ha='right', fontsize=10, color='black')

plt.title('Top 15 UK Companies Offering Data-Related Jobs (as % of total)')
plt.xlabel('Percentage of Data Jobs')
plt.ylabel('Company')
plt.gca().xaxis.set_major_formatter('{:.1f}%'.format)  
plt.tight_layout()
plt.show()


# In[92]:


uk_data2 = uk_data.copy()

# Defining the keywords pattern for data jobs
data_job_keywords_pattern = re.compile(r'Data|data|Database|database|Analyst|analyst|Analytics|analytics|\
                                Analysis|analysis|Software|software|Product|product|BI|bi\
                              Python|python|Azure|azure|programming|Programming|developer|Developer|\
                              Digital|digital|Research|research|Quantitative|quantitative|Insight|insight|Machine learning|\
                              |machine learning|AI|ai', flags=re.IGNORECASE)

# Filtering the UK dataset based on the keywords pattern in the "Job_Title" column
filtered_data2 = uk_data2[uk_data2['Job_Title'].str.contains(data_job_keywords_pattern, na=False, regex=True)]

# Creating a new dataset containing only data jobs based on the keywords
filtered_data2.to_csv('filtered_data_jobs_dataset.csv', index=False)

# Display the first few rows of the new dataset

print(filtered_data2.head())


# In[93]:


filtered_data2.info()


# In[94]:


datajobs_uk = filtered_data2.copy()


# In[95]:


datajobs_uk.head()


# In[125]:


import pandas as pd
import matplotlib.pyplot as plt

# Loading the filtered dataset I created, that contains only data-related job titles
datajobs_uk = pd.read_csv('filtered_data_jobs_dataset.csv')

# Counting the occurrences of each position and select the top 15
top_positions = datajobs_uk['Position'].value_counts().nlargest(15)

# Create a bar plot
plt.figure(figsize=(10, 6))
top_positions.plot(kind='bar', color='skyblue')
plt.title('Top 15 Data Positions in the UK')
plt.xlabel('Position')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[97]:


# Count the occurrences of each position and select the top 20
top_positions = datajobs_uk['Position'].value_counts().nlargest(15).index

# Filter the dataset to include only the top 15 positions
datajobs_uk_top_positions = datajobs_uk[datajobs_uk['Position'].isin(top_positions)]

# Pivot the data to get frequencies of positions across cities
pivot_data = datajobs_uk_top_positions.pivot_table(index='Position', columns='City', aggfunc='size', fill_value=0)

# Create a heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(pivot_data, cmap='YlGnBu', linewidths=0.5, linecolor='grey')
plt.title('Heatmap of Top 15 Data Positions Across Cities')
plt.xlabel('City')
plt.ylabel('Position')
plt.xticks(rotation=45)


plt.tight_layout()
plt.show()


# In[126]:


# Counting the occurrences of each city
city_counts = datajobs_uk['City'].value_counts()

# Selecting the top 15 cities in the UK
top_15_cities = city_counts.nlargest(15)

# Calculating the total number of data jobs
total_data_jobs = len(datajobs_uk)

# Calculating the percentage of data jobs for each city
percentage_data_jobs = (top_15_cities / total_data_jobs) * 100

# Creating a bar plot
plt.figure(figsize=(10, 6))
percentage_data_jobs.plot(kind='bar', color='blue')
plt.title('Top 15 UK Cities with Data Jobs')
plt.xlabel('City')
plt.ylabel('Percentage of Data Jobs')
plt.xticks(rotation=45, ha='right')
plt.gca().yaxis.set_major_formatter('{:.0f}%'.format)  # Format y-axis labels as percentages
plt.tight_layout()
plt.show()


# In[127]:


# Counting the occurrences of each city
city_counts = datajobs_uk['City'].value_counts()

# Selecting the top 15 cities in the UK
top_15_cities = city_counts.nlargest(15)

# Calculating the total number of data jobs
total_data_jobs = len(datajobs_uk)

# Calculating the percentage of data jobs for each city
percentage_data_jobs = (top_15_cities / total_data_jobs) * 100

# Creating a bar plot
plt.figure(figsize=(10, 6))
percentage_data_jobs.plot(kind='bar', color='blue')
plt.title('Top 15 UK Cities with Data Jobs')
plt.xlabel('City')
plt.ylabel('Percentage of Data Jobs')
plt.xticks(rotation=45, ha='right')
plt.gca().yaxis.set_major_formatter('{:.0f}%'.format)  

# Adding text annotation for the sum of percentages
sum_percentage = percentage_data_jobs.sum()
plt.text(0.22, 0.9, f'Sum of percentages for the top 15 cities: {sum_percentage:.2f}%', transform=plt.gca().transAxes)

plt.tight_layout()
plt.show()


# In[103]:


# Define the keywords
keywords = ['intern', 'apprentice', 'apprenticeship']

# Count the total number of jobs
total_jobs = len(datajobs_uk)

# Count the occurrences of each keyword in the Job_Title column
keyword_counts = datajobs_uk['Job_Title'].str.lower().str.contains('|'.join(keywords)).sum()

# Calculate the percentage of jobs containing the keywords
percentage = (keyword_counts / total_jobs) * 100

# Create a bar plot
plt.figure(figsize=(8, 6))
plt.bar(['Intern/Apprentice/Apprenticeship'], [percentage], color='skyblue')
plt.title('Percentage of Internship/Apprenticeship Jobs in the UK (LinkedIn data 2024)')
plt.ylabel('Percentage of Jobs')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[104]:


keywords = ['Intern', 'intern', 'Intenship', 'intenship' 'Apprentice', 'apprentice', 'Apprenticeship', 'apprenticeship']

# Occurrences of each keyword in the Job_Title column
keyword_counts = uk_data['Job_Title'].str.lower().str.contains('|'.join(keywords)).sum()

print("Count of 'Intern', 'intern', 'Intenship', 'intenship' 'Apprentice', 'apprentice',\
'Apprenticeship', 'apprenticeship' in Job_Title:", keyword_counts)


# In[105]:


keywords = ['Intern', 'intern', 'Intenship', 'intenship' 'Apprentice', 'apprentice', 'Apprenticeship', 'apprenticeship']

# Occurrences of each keyword in the Job_Title column
keyword_counts = filtered_data['Job_Title'].str.contains('|'.join(keywords)).sum()

# Tital count of job titles
total_job_titles = len(filtered_data)

# & of job titles containing the keywords
percentage = (keyword_counts / total_job_titles) * 100

print("Percentage of job titles containing 'Intern', 'intern', 'Intenship', 'intenship',\
'Apprentice', 'apprentice', 'Apprenticeship','apprenticeship': {:.2f}%".format(percentage))


# In[124]:


data_jobs_percentage = 98.97
apprenticeship_internship_percentage = 1.03

# Creating a bar plot
plt.figure(figsize=(8, 7))
plt.bar(['Data Jobs', 'Apprenticeships & Internships'], [data_jobs_percentage, apprenticeship_internship_percentage],\
        color=['lightblue', 'lightgreen'])
plt.title('Comparison of Data Jobs and Apprenticeships/Internships in Data')
plt.ylabel('Percentage')
plt.ylim(0, 110)  

# Annotating bars with percentages
for x, y in enumerate([data_jobs_percentage, apprenticeship_internship_percentage]):
    plt.text(x, y + 0.4, f'{y:.2f}%', ha='center', va='bottom')

plt.show()


# In[117]:


filtered_data.info()


# In[ ]:





# In[ ]:





# In[ ]:





# In[112]:


data_jobs_percentage = (data_jobs_count1 / total_job_titles) * 100
apprenticeship_jobs_percentage = percentage

# Creating a scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=['Data Jobs', 'Apprenticeship Jobs'], y=[data_jobs_percentage, apprenticeship_jobs_percentage], color='blue')
plt.title('Comparison of Data Jobs and Apprenticeship Jobs in the UK')
plt.ylabel('Percentage')
plt.ylim(0, 100)  # Set y-axis limit to ensure percentages are between 0 and 100
plt.show()


# In[107]:


# Count the occurrences of each job level
job_level_counts = filtered_data['Job_Type'].value_counts()

# Create a horizontal bar chart
plt.figure(figsize=(8, 6))
job_level_counts.plot(kind='barh', color='pink')
plt.xlabel('Number of Jobs')
plt.ylabel('Job Level')
plt.title('Count of Different Job Levels')
plt.gca().invert_yaxis()  # Invert y-axis to have the highest count on top
plt.show()


# In[108]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Create a pivot table to get the count of occurrences for each combination of search_country and job_title
pivot_table = filtered_data.pivot_table(index='City', columns='Job_level', aggfunc='size', fill_value=0)

# Plot the heatmap
plt.figure(figsize=(20, 29))
sns.heatmap(pivot_table, cmap='YlGnBu', linewidths=.5)
plt.title('Heatmap of Job levels by City')
plt.xlabel('Job_level')
plt.ylabel('City')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# In[109]:


filtered_data.info()


# In[110]:


# Assuming "uk" is your DataFrame containing the dataset
unique_job_types = filtered_data['Job_Type'].unique()

# Print the unique job types
print(unique_job_types)


# In[ ]:





# In[ ]:




