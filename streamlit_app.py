import streamlit as st
import pandas as pd
import openai

# 设置 OpenAI API 凭据
openai.api_type = "azure"
openai.api_base = "https://openai-haas-01.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = "76409a14b9ed43fd9664bc1aef4cc3bc"

# 加载 OpenAI Logo 图片
logo_url = "./git-images/azure_openai_logo.png"
st.image(logo_url, width=800, caption="OpenAI GPT-3 评论分析")

# 定义分析评论的函数
def analyze_reviews(df):
        review_content_list = []

        for index, headers in df.iterrows():
                st.write("--------------------")
                review_content = str(headers["Contents"])
                st.write("评论内容: {}".format(review_content))
                # 使用 AOAI GPT-3 分类评论内容的情感。
                response = openai.Completion.create(
                    engine="Turbo-Haas-01",
                    prompt="根据以下类别分类以下评论内容的情感：\
                    类别：[Negative负面, Netural中性, Positive正面]\n\n评论内容：" + review_content + "\n\n分类情感：",
                    max_tokens=10)
                classified_sentiment = response['choices'][0]['text'].replace(" ", "")
                st.write("情感分类完成："+classified_sentiment)

                # 使用 AOAI GPT-3 总结评论内容。
                response2 = openai.Completion.create(
                    engine="Turbo-Haas-01",
                    prompt="用一句话总结以下评论内容：" \
                    + review_content + "\n\n一句话总结：",
                    max_tokens=100)
                summarized_sentence = response2['choices'][0]['text'].replace("\n","")
                st.write("总结句子生成完成："+ summarized_sentence)

                # 使用 AOAI GPT-3 根据评论内容总结 3 个关键词。
                response3 = openai.Completion.create(
                    engine="Turbo-Haas-01",
                    prompt="根据评论内容，总结 3 个关键词：" \
                    + review_content + "\n\n关键词：",
                    max_tokens=20)
                summarized_keywords = response3['choices'][0]['text'].replace("\n","").replace(".","")
                st.write("总结关键词生成完成："+summarized_keywords)

                # 使用 AOAI GPT-3 根据评论内容撰写回复消息。
                response4 = openai.Completion.create(
                    engine="Turbo-Haas-01",
                    prompt="根据评论内容，撰写回复消息：" \
                    + review_content + "\n\n回复消息：",
                    max_tokens=120)
                reply_message = response4['choices'][0]['text'].replace("\n","")
                st.write("回复消息草稿生成完成："+reply_message)

                # 将分析结果添加到列表中。
                review_content_list.append([review_content, classified_sentiment, summarized_sentence, summarized_keywords, \
                                                                        reply_message])

        # 将分析结果列表转换为 Pandas 数据框。
        review_content_df = pd.DataFrame(review_content_list, columns=['review_content', 'classified_sentiment', \
                                                                                                                                     'summarized_sentence', 'summarized_keywords', \
                                                                                                                                     'reply_message'])

        # 显示带有 AOAI GPT-3 分析结果的数据框。
        st.write(review_content_df)

        # 将结果数据框保存为 CSV 文件。
        review_content_df.to_csv("./data/analyzed_review_content.csv")

        # 将所有评论内容连接成单个字符串。
        review_content_string = review_content_df['review_content'].to_string(header=False, index=False)
       # st.write(review_content_string)



        # 将评论内容的分类情感可视化为饼图。
        st.set_option('deprecation.showPyplotGlobalUse', False)
        review_content_df.groupby(['classified_sentiment']).count().plot(kind='pie',y='review_content', autopct='%1.0f%%', \
                                                                                                                                        figsize =(8,8))
        st.pyplot()
        # 使用 AOAI GPT-3 总结所有评论内容。
        response5 = openai.Completion.create(
            engine="Turbo-Haas-01",
            prompt="总结以下评论内容，用 100 个单词：" + review_content_string + "\n\n总结：",
            max_tokens=120)

        all_review_content_summary = response5['choices'][0]['text'].replace("\n","")
        st.write(all_review_content_summary)
# 从 CSV 文件加载数据到 Pandas 数据框并显示标题。
df = pd.read_csv('./data/asos_transform_10.csv', nrows=10)

# 创建 Streamlit 应用程序
st.title('使用 Azure OpenAI 服务 (AOAI) 分析客户评论')
st.write('此Demo使用 Azure OpenAI 服务 (AOAI) 分析客户评论，评论从 CSV 文件中加载，并使用 AOAI GPT-3 ”text-davinci-003“模型进行分析。。')
st.write('点击“分析”按钮开始进行分析，结果以饼图和表格的形式显示，并生成所有评论的摘要。')
st.markdown("<p style='color:red;'>Demo受限于azure 成本，只分析前10条数据。</p>", unsafe_allow_html=True)

# 显示数据框的前 10 行
st.write('数据框的前 10 行：')
st.write(df.head(10))

# 分析评论
if st.button('分析'):
    st.write('正在分析评论...')
    analyze_reviews(df)
