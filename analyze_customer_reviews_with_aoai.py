#!/usr/bin/env python
# coding: utf-8

# # Analyze Customer Reviews with Azure OpenAI Service (AOAI)

# In[1]:


# Import necessary libraries.
import pandas as pd
import openai


# In[3]:


# Load csv data to Pandas dataframe and display the header.
df = pd.read_csv('./data/asos_transform_10.csv')
df.head()


# In[8]:


# Check dataframe information.
df.info()


# In[4]:


# Select the top 10 rows from the dataframe. It's part of the reviews on 14 Dec 2019.
df_10 = df.head(10)


# In[6]:


# Configure the baseline configuration of the OpenAI library.
openai.api_type = "azure"
openai.api_base = "https://turbo.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = "0ec8e07d3c8442cebeaeb1a46931e6c7"


# In[7]:


# Primary functions to interact with AOAI GPT-3 to obtain insights.
review_content_list = []

for index, headers in df_10.iterrows():
    review_content = str(headers["Contents"])
    print("Review Content: {}".format(review_content))
    # Use AOAI GPT-3 to classify the sentiment of the review content.
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt="Classify the sentiment of the following review content following categories: \
      categories: [Negative, Netural, Positive]\n\nreview content : " + review_content + "\n\nClassified sentiment:",
      max_tokens=10)
    classified_sentiment = response['choices'][0]['text'].replace(" ", "")
    # print("Classified Sentiment of Review Content: {}".format(classified_sentiment))
    print("Sentiment Classified")
    
    # Use AOAI GPT-3 to summarize the review content.
    response2 = openai.Completion.create(
      engine="text-davinci-003",
      prompt="Summarize the following review content in one sentence:" \
      + review_content + "\n\nOne Sentence:",
      max_tokens=100)
    summarized_sentence = response2['choices'][0]['text'].replace("\n","")
    # print("Summarize Sentence from the Review Content: {}".format(summarized_sentence))
    print("Summarize Sentence Generated")
    
    # Use AOAI GPT-3 to summarize 3 keyword based on the review content.
    response3 = openai.Completion.create(
      engine="text-davinci-003",
      prompt="Based on the review content, summarize in 3 keywords:" \
      + review_content + "\n\nKeywords:",
      max_tokens=20)
    summarized_keywords = response3['choices'][0]['text'].replace("\n","").replace(".","")
    # print("Summarize 3 Keywords from the Review Content: {}".format(summarized_keywords))
    print("Summarize Keywords Generated")
    
    # Use AOAI GPT-3 to craft a reply message based on the review content.
    response4 = openai.Completion.create(
      engine="text-davinci-003",
      prompt="Based on the review content, craft a reply message:" \
      + review_content + "\n\nReply Message:",
      max_tokens=100)
    reply_message = response4['choices'][0]['text'].replace("\n","")
    # print("Draft of reply message based on the Review Content: {}".format(reply_message))
    print("Draft of Reply Message Generated")
    
    # Append the insights result into a list.
    review_content_list.append([review_content, classified_sentiment, summarized_sentence, summarized_keywords, \
                                reply_message])

# Convert the list of insights into a Pandas dataframe.
review_content_df = pd.DataFrame(review_content_list, columns=['review_content', 'classified_sentiment', \
                                                               'summarized_sentence', 'summarized_keywords', \
                                                               'reply_message'])


# In[25]:


# Display the result dataframe with the insights from AOAI GPT-3.
review_content_df


# In[26]:


# Save the result dataframe into a CSV file.
review_content_df.to_csv("./data/analyzed_review_content.csv")


# In[27]:


# Concatenate all the review content into a single string.
review_content_string = review_content_df['review_content'].to_string(header=False, index=False)
print(review_content_string)


# In[28]:


# Use AOAI GPT-3 to summarize all the review content.
response5 = openai.Completion.create(
  engine="text-davinci-003",
  prompt="Summarize the following review content in 100 words" + review_content_string + "\n\nSummary:",
  max_tokens=120)

all_review_content_summary = response5['choices'][0]['text'].replace("\n","")
print(all_review_content_summary)


# In[29]:


# Visualize the classified sentiment of the review content as a pie chart.
review_content_df.groupby(['classified_sentiment']).count().plot(kind='pie',y='review_content', autopct='%1.0f%%', \
                                                                figsize =(8,8))


# In[ ]:




