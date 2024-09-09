import pandas as pd
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(
    api_key="sk-f089718a48534c4c84a0cfbc35e9fd1a",  # 替换成真实DashScope的API_KEY
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务endpoint
)
prompt = """后面的“待分析文本”是一段师生对话，其中只保留了老师话语，学生话语已被剔除。请按照以下方法对老师话语进行分析： 

根据上下文语义的相关性，将“待分析文本”分割为“发起”、“评价”、“讲解”和“其它”四种子文本段。“发起”是老师邀请、引导、鼓励学生发言、齐读、回答问题、朗读等用话语来回应的子文本段，而不是老师让学生做动作的子文本段；“评价”是老师对学生回应的直接肯定、直接表扬、直接否定的子文本段；”讲解“是老师描述知识点、重复学生回应内容、总结学生回应的子文本段；不能归属于上面三种子文本段的，归为“其它”子文本段。
请根据“示例”中的输出格式输出分析结果，将每个子文本段以JSON格式输出，类型为type，对应的内容为content。
示例： 
老师话语： 它分别种了什么树呢？于凯，你来说说看。你慢讲啊，嗯。然后呢？ 然后种了杏树。最后呢？
输出： 
[{"type":"发起"，"content":"它分别种了什么树呢？于凯，你来说说看。"}, 
{"type":"其它"："content":"你慢讲啊，嗯。"},
{"type":"发起"："content":"然后呢？"},
{"type":"讲解"："content":"然后种了杏树。"},
{"type":"发起"："content":"最后呢？"}
]

待分析文本：


"""
df = pd.read_excel("../data/726四分类法.xlsx")
for i in tqdm(range(len(df))):
    completion = client.chat.completions.create(
        model="qwen2-72b-instruct",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "你是一个乐于助人的小助手，所有的输出都是一个json列表"},
            {"role": "user", "content": prompt + df.loc[i, 'text']}
        ],
        max_tokens=2000,  # 最大生成的token数
        n=1,  # 生成的结果数量
        stop=None,  # 停止生成的标记
        temperature=0.7,  # 生成文本的多样性,
        stream=False
    )
    # content = completion["output"]['choices'][0]['message']['content']
    # print("cotent", completion)
    print(completion.choices[0].message.model_dump())
    break
    output = completion.choices[0].message.model_dump()
    df.loc[i,'qwen2-57b_predict'] = output.get("content")

# df.to_excel("../data/726四分类法.xlsx",index=False)

