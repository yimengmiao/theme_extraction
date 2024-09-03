import pandas as pd
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(
    api_key="sk-f089718a48534c4c84a0cfbc35e9fd1a",  # 替换成真实DashScope的API_KEY
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务endpoint
)
prompt = """后面的“待分析文本”是一段师生对话，其中，学生话语已经剔除，只保留老师话语，请对老师的话语进行分析，具体分析方法如下所示：  
将”待分析文本“分割成”发起“、”评价“、”讲解“、“其它”四种子文本段，”发起“的分割尽可能细一点。“发起”是老师邀请、引导、鼓励学生用话语来回应的语句；“评价”是对学生回应的表扬、认可、批评等评价性话语；”讲解“是老师针对知识展开描述或对学生回应的总结；不能归属于上面三种子文本段，归属为“其它”。
按照下面“示例”输出： 
示例： 老师话语： 它分别种了什么树呢？谁来说说？于凯，你来说说看。你慢讲啊，嗯。然后呢？ 
输出： 
[{"type":"发起"，"content":"它分别种了什么树呢？谁来说说？"} 
{"type":"发起"，"content":"于凯，你来说说看。"} 
{"type":"其它"："content":"你慢讲啊，嗯。"}
{"type":"发起"："content":"然后呢？"}]
待分析文本："""
df = pd.read_excel("data/726四分类法.xlsx")
for i in tqdm(range(len(df))):
    completion = client.chat.completions.create(
        model="qwen-long",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "你是一个乐于助人的小助手，所有的输出都是一个json列表"},
            {"role": "user", "content": prompt + df.loc[i, 'text']}
        ],
        max_tokens=1000,  # 最大生成的token数
        n=1,  # 生成的结果数量
        stop=None,  # 停止生成的标记
        temperature=0.7,  # 生成文本的多样性,
        stream=False
    )
    # content = completion["output"]['choices'][0]['message']['content']
    # print("cotent", completion)
    print(completion.choices[0].message.model_dump())
    output = completion.choices[0].message.model_dump()
    df.loc[i,'qwen_predict'] = output.get("content")

df.to_excel("726四分类法.xlsx",index=False)

