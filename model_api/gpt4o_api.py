import logging

import pandas as pd

from openai import AzureOpenAI
from tqdm import tqdm

prompt = """后面的“待分析文本”是一段师生对话，其中，学生话语已经剔除，只保留老师话语，请对老师的话语进行分析，具体分析方法如下所示：  
将”待分析文本“分割成”发起“、”评价“、”讲解“、“其它”四种子文本段，”发起“的分割尽可能细一点。“发起”是老师邀请、引导、鼓励学生用话语来回应的语句；“评价”是对学生回应的表扬、认可、批评等评价性话语；”讲解“是老师针对知识展开描述或对学生回应的总结；不能归属于上面三种子文本段，归属为“其它”。
按照下面“示例”输出： 
示例： 老师话语： 它分别种了什么树呢？谁来说说？于凯，你来说说看。你慢讲啊。嗯，然后呢？ 
输出： 
[{"type":"发起"，"content":"它分别种了什么树呢？谁来说说？"} 
{"type":"发起"，"content":"于凯，你来说说看。"} 
{"type":"其它"："content":"你慢讲啊。嗯，"}
{"type":"发起"："content":"然后呢？"}]
待分析文本："""

# 注意：这个部署在azure上的api接口会在2025年三月份的时候过期

client = AzureOpenAI(
    azure_endpoint="https://zonekey-gpt4o.openai.azure.com/",
    api_key="b2e709bdd54f4416a734b4a6f8f1c7a0",
    api_version="2024-02-01"
)

df = pd.read_excel("726四分类法.xlsx")
for i in tqdm(range(len(df))):
    try:
        response = client.chat.completions.create(
            model="soikit_test",  # model = "deployment_name".
            response_format={"type": "json_object"},  # 响应格式为JSON对象
            messages=[
                {"role": "system", "content": "你是一个乐于助人的小助手,并且输出是一个json，key是result"},
                {"role": "user", "content": prompt + df.loc[i, 'text']},
            ]
        )
        output = response.choices[0].message.content
        df.loc[i, 'gpt4o_predict'] = output
        logging.info(f"Row {i} processed successfully. Output: {output}")
    except Exception as e:
        df.loc[i, 'gpt4o_predict'] = "error"
        logging.error(f"Error processing row {i}: {e}")

df.to_excel("data/726四分类法.xlsx", index=False)
