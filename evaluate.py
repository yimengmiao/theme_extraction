from tqdm import tqdm
import json
import pandas as pd
import re
from difflib import SequenceMatcher
from sklearn.metrics import classification_report

df = pd.read_excel("data/726四分类法.xlsx")
cloumn = "qwen"


def similarity(sent1, sent2):
    return SequenceMatcher(None, sent1, sent2).ratio()


def split_sentences(data):
    split_data = []
    for item in data:
        # 使用正则表达式分割句子
        sentences = re.split(r'[？。]', item['content'])
        for sentence in sentences:
            if sentence:  # 忽略空字符串
                split_data.append({'type': item['type'], 'content': sentence})
    return split_data


gpt4o_output = []
glm4_output = []
qwen_output = []
manual_label = []
for i in tqdm(range(0, len(df))):
    # print(i)
    manual = json.loads(df.loc[i, 'label'])
    split_data_manual = split_sentences(manual)
    if cloumn == 'gpt4':
        gpt4o = json.loads(df.loc[i, 'gpt4o_predict']).get("result")
        split_data_gpt4o_output = split_sentences(gpt4o)
        for item in split_data_gpt4o_output:
            for sent in split_data_manual:

                if similarity(item.get("content"), sent.get("content")) > 0.5:
                    item['label'] = sent.get("type")
                    item['manual_content'] = sent.get("content")
                    break  # 如果匹配上了，就不要再往下匹配了。

        gpt4o_output.extend(split_data_gpt4o_output)

    elif cloumn == 'glm4':
        glm4 = json.loads(df.loc[i, 'glm4_predict'])
        split_data_glm4_output = split_sentences(glm4)

        for item in split_data_glm4_output:
            for sent in split_data_manual:

                if similarity(item.get("content"), sent.get("content")) > 0.5:
                    item['label'] = sent.get("type")
                    item['manual_content'] = sent.get("content")
                    break  # 如果匹配上了，就不要再往下匹配了。

        glm4_output.extend(split_data_glm4_output)
    elif cloumn == 'qwen':
        qwen = json.loads(df.loc[i, 'qwen_predict'])
        split_data_qwen_output = split_sentences(qwen)
        for item in split_data_qwen_output:
            for sent in split_data_manual:
                if similarity(item.get("content"), sent.get("content")) > 0.5:
                    item['label'] = sent.get("type")
                    item['manual_content'] = sent.get("content")
                    break
        qwen_output.extend(split_data_qwen_output)

# 定义一个包含四个必要键值的列表
required_keys = ['type', 'content', 'label', 'manual_content']

# 遍历每个字典
# data = [
#     {'type': '讲解', 'content': '猴子呀他一开始很勤劳啊，还是种了一棵梨树', 'label': '讲解', 'manual_content': '猴子呀他一开始很勤劳啊，还是种了一棵梨树'},
#     {'type': '讲解', 'content': '可是当乌鸦来劝说它的时候，它就拔掉了梨树，改种了', 'label': '讲解', 'manual_content': '可是当乌鸦来劝说它的时候，它就拔掉了梨树，改种了'},
#     {'type': '讲解', 'content': '正当杏树成活的时候，喜鹊来劝他了', 'label': '讲解', 'manual_content': '正当杏树成活的时候，喜鹊来劝他了'}
# ]

# 遍历列表并确保每个字典都有四个键值
if cloumn == 'gpt4':
    for item in gpt4o_output:
        for key in required_keys:
            if key not in item:
                item[key] = ''  # 如果缺少键，就添加并赋值为空字符串
    # 将数据转换为DataFrame
    df = pd.DataFrame(gpt4o_output)

    # 去除df_cleaned 中任意列值为 '' 的行
    df = df[(df != '').all(axis=1)]
    df_cleaned = df.dropna(subset='label').reset_index(drop=True)

    df_cleaned['type'] = df_cleaned['type'].replace("其他", "其它")
    df_cleaned['label'] = df_cleaned['label'].replace("其他", "其它")
    df_cleaned.to_excel("data/gpt4o_T_4_result.xlsx", index=False)
    # 计算分类报告
    classification_rep = classification_report(df_cleaned['label'], df_cleaned['type'], zero_division=0)

    print(classification_rep)

if cloumn == 'glm4':
    for item in glm4_output:
        for key in required_keys:
            if key not in item:
                item[key] = ''  # 如果缺少键，就添加并赋值为空字符串
    # 将数据转换为DataFrame
    df = pd.DataFrame(glm4_output)

    # 去除df_cleaned 中任意列值为 '' 的行
    df = df[(df != '').all(axis=1)]
    df_cleaned = df.dropna(subset='label').reset_index(drop=True)

    df_cleaned['type'] = df_cleaned['type'].replace("其他", "其它")
    df_cleaned['label'] = df_cleaned['label'].replace("其他", "其它")
    df_cleaned.to_excel("data/glm4_T_4_result.xlsx", index=False)
    # 计算分类报告
    classification_rep = classification_report(df_cleaned['label'], df_cleaned['type'], zero_division=0)
    print(classification_rep)

if cloumn == 'qwen':
    for item in qwen_output:
        for key in required_keys:
            if key not in item:
                item[key] = ''
    print(qwen_output)
    df = pd.DataFrame(qwen_output)
    print(df)
    df = df[(df != '').all(axis=1)]
    print(df)
    df_cleaned = df.dropna(subset='label').reset_index(drop=True)

    df_cleaned['type'] = df_cleaned['type'].replace("其他", "其它")
    df_cleaned['label'] = df_cleaned['label'].replace("其他", "其它")
    df_cleaned.to_excel("data/qwen_T_4_result.xlsx", index=False)
    # 计算分类报告
    classification_rep = classification_report(df_cleaned['label'], df_cleaned['type'], zero_division=0)
    print(classification_rep)
