# Analyze Customer Reviews with Azure OpenAI Service (AOAI)



这个演示存储库演示了如何使用 [Azure OpenAI Service (AOAI)](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/overview) 分析客户评论。我利用了来自 [Kaggle](https://www.kaggle.com/) 的 ["ASOS Customer Review"](https://www.kaggle.com/datasets/mahirahmzh/asos-customer-review-in-trustpilot) 来从客户评论内容中获得有价值的见解，例如情感、用一句话总结评论内容、用关键词总结、生成响应消息以及使用 [AOAI GPT-3 模型](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/concepts/models#gpt-3-models) 总结多个评论内容。

内容：
* [data/asos_transform.csv](https://github.com/easonlai/analyze_customer_reviews_with_aoai/blob/main/data/asos_transform.csv) <-- ASOS 的客户评论数据集。它包含 2,000 条记录。
* [data/analyzed_review_content.csv](https://github.com/easonlai/analyze_customer_reviews_with_aoai/blob/main/data/analyzed_review_content.csv) <-- 从客户评论内容中提取见解的数据框导出。
* [analyze_customer_reviews_with_aoai.ipynb](https://github.com/easonlai/analyze_customer_reviews_with_aoai/blob/main/analyze_customer_reviews_with_aoai.ipynb) <-- 执行从客户评论内容中提取见解的笔记本。

![alt text](./git-images/result.png)

享受吧！
