import json
import re
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm
import pandas as pd


# 去除标点符号的函数
def remove_punctuation(text):
    """
    移除输入文本中的标点符号。
    Args:
    - text: 输入的文本字符串。

    Returns:
    - 去除标点符号后的文本字符串。
    """
    return re.sub(r'[^\w\s]', '', text)


# 最长公共子文本串算法
def algrot(text1, text2):
    """
    使用动态规划算法，计算两个字符串的最长公共子文本串。
    Args:
    - text1: 第一个字符串。
    - text2: 第二个字符串。

    Returns:
    - 最长公共子文本串。
    """
    m = len(text1)
    n = len(text2)
    max_len = 0
    ending_index = m
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 动态规划填充表格，计算公共子串长度
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    ending_index = i
            else:
                dp[i][j] = 0

    # 返回最长公共子串
    return text1[ending_index - max_len: ending_index]


# 查找文本中的所有最长公共子串
def find_all_common_substrings(text1, text2):
    """
    查找文本1和文本2的所有公共子串，返回一个子串列表。
    Args:
    - text1: 第一个字符串。
    - text2: 第二个字符串。

    Returns:
    - 所有公共子串的列表。
    """
    m = len(text1)
    n = len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    substrings = []

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                substrings.append(text1[i - dp[i][j]:i])
            else:
                dp[i][j] = 0

    return list(set(substrings))  # 去重


# 计算匹配占比
def calculate_match_ratio(substring, content):
    """
    计算公共子串在目标文本中的占比。
    Args:
    - substring: 公共子串。
    - content: 目标文本。

    Returns:
    - 公共子串长度与目标文本长度的比值。
    """
    return len(substring) / len(content) if content else 0


# 查找最佳匹配
def find_best_match(predict_items, labels):
    """
    优化后的匹配算法，先进行简单的文本匹配，如果匹配成功，则直接添加结果并跳过公共子串匹配。
    如果匹配失败，则调用最长公共子串匹配。公共子串匹配只会在 start_index 后的3个 label 中进行。
    """
    start_index = 0  # 初始化从第一个 label 开始

    for predict_item in predict_items:
        best_match = None
        highest_ratio = 0
        best_match_index = -1  # 用于记录匹配的索引

        # 去除标点符号
        predict_content_no_punc = remove_punctuation(predict_item['content'])

        # 先进行快速文本匹配，逐步检查 label 的内容
        for i in range(start_index, len(labels)):
            label = labels[i]
            label_content_no_punc = remove_punctuation(label['content'])

            # 如果 predict_content_no_punc 完全匹配在 label_content_no_punc 中，直接更新并跳过公共子串匹配
            if predict_content_no_punc in label_content_no_punc:
                best_match = label
                best_match_index = i
                predict_item['best_match'] = best_match  # 添加匹配结果
                start_index = best_match_index  # 更新起始索引
                break  # 结束快速匹配，跳过公共子串匹配

        # 如果快速匹配没有找到匹配项，进入公共子串匹配阶段
        if best_match is None:
            # 公共子串匹配，范围限制在 start_index 到 start_index + 3
            for i in range(start_index, min(start_index + 3, len(labels))):
                label = labels[i]
                label_content_no_punc = remove_punctuation(label['content'])

                # 找到所有公共子串
                common_substrings = find_all_common_substrings(predict_content_no_punc, label_content_no_punc)

                # 放宽公共子串长度的限制，确保匹配的更宽松
                valid_substrings = [sub for sub in common_substrings if len(sub) / len(predict_content_no_punc) >= 0.3]

                # 如果没有符合的子串，跳过这次匹配
                if not valid_substrings:
                    continue

                # 计算公共子串与目标文本的占比，找出占比最高的
                for substring in valid_substrings:
                    ratio = calculate_match_ratio(substring, label_content_no_punc)
                    if ratio > highest_ratio:
                        highest_ratio = ratio
                        best_match = label
                        best_match_index = i  # 记录匹配成功的索引

        # 如果没有找到匹配项，标记为 "其它"
        if best_match is None:
            predict_item['best_match'] = {"content": "其它", "type": "其它"}
        else:
            predict_item['best_match'] = best_match

            # 更新 start_index 为匹配成功的索引
            if best_match_index != -1 and best_match_index > start_index:
                start_index = best_match_index  # 更新起始索引为匹配成功的索引

    return predict_items


# 处理预测与匹配
def process_predictions(predict, labels):
    """
    处理 predict 列表，找到每个 predict 项在 label 列表中的最佳匹配项。
    Args:
    - predict: 需要处理的 predict 列表。
    - labels: label 列表。

    Returns:
    - 包含匹配结果的 predict 列表。
    """
    results = []

    # 遍历 predict 列表中的所有项
    # for predict_item in predict:
    best_match = find_best_match(predict, labels)
    for item in best_match:
    # 只要找到匹配的 label 项，不管 type 是否匹配，都进行记录
        dict1 = {'content': item['content'], 'type': item['type'],"matched_content":item.get("best_match").get('content'),'matched_type':item.get("best_match").get('type')}
        # predict_item['matched_content'] = best_match.get("best_match").get('content')
        # predict_item['matched_type'] = best_match.get("best_match").get('type')

        results.append(dict1)

    return results


# 提取 filtered_result 中 matched_type 为 "发起" 和 "评价" 的数据
def filter_results(results):
    """
    从 results 中提取 matched_type 为 '发起' 和 '评价' 的项，构造 filtered_result。
    Args:
    - results: 原始匹配结果列表

    Returns:
    - filtered_result: 过滤后的列表，仅包含 '发起' 和 '评价' 类型的项。
    """
    filtered_result = [item for item in results if item['matched_type'] in ['发起', '评价']]
    return filtered_result


# 基于 filtered_result 计算 Precision, Recall, F1, 和 Accuracy
def calculate_metrics(filtered_result):
    """
    计算 '发起' 和 '评价' 两个类别的 Precision, Recall, F1 Score 和 Accuracy。
    Args:
    - filtered_result: 经过过滤的匹配结果列表

    Returns:
    - metrics: 包含 precision, recall, f1 和 accuracy 的字典
    """
    # 提取实际标签（true labels）和预测标签（predicted labels）
    y_true = [item['type'] for item in filtered_result]  # 实际标签
    y_pred = [item['matched_type'] for item in filtered_result]  # 预测标签

    # 计算 precision, recall, f1 和 accuracy
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    # 返回计算结果
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy
    }
def highlight_errors(row):
    if row['label'] == '发起' and row['predict'] != '发起':
        # 召回计算中的错误，标记为黄色,label列为发起，预测列不是发起
        return ['background-color: yellow'] * len(row)
    elif row['predict'] == '发起' and row['label'] != '发起':
        # 精确率计算中的错误，标记为深绿色，预测列是发起，人工标签列是发起
        return ['background-color: lightgreen'] * len(row)
    else:
        return [''] * len(row)

if __name__ == '__main__':
    # 定义 label 和 predict 列表
    model = "qwen_max"
    df = pd.read_excel("data/726四分类法.xlsx")
    output = []
    for i in tqdm(range(len(df))):
        label = json.loads(df.loc[i, 'label'])

        if model == "gpt4o":

            predict = json.loads(df.loc[i, 'gpt4o_predict']).get('result')
            # 执行更新后的处理
            results = process_predictions(predict, label)
            print(f"results_no_filter: {results}")
            # 提取过滤结果
            filtered_result = filter_results(results)
            output.extend(filtered_result)
            print(i)
        elif model == "glm4":

            predict = json.loads(df.loc[i, 'glm4_predict'])
            # 执行更新后的处理
            results = process_predictions(predict, label)
            print(f"results_no_filter: {results}")
            # 提取过滤结果
            filtered_result = filter_results(results)
            output.extend(filtered_result)
        elif model == "qwen_long":
            predict = json.loads(df.loc[i, 'qwen_long_predict'])
            # 执行更新后的处理
            results = process_predictions(predict, label)
            print(f"results_no_filter: {results}")
            # 提取过滤结果
            filtered_result = filter_results(results)
            output.extend(filtered_result)
        elif model == "qwen_max":
            predict = json.loads(df.loc[i, 'qwen_max_predict'])
            # 执行更新后的处理
            results = process_predictions(predict, label)
            print(f"results_no_filter: {results}")
            # 提取过滤结果
            filtered_result = filter_results(results)
            output.extend(filtered_result)

        elif model == "qwen_72B":
            predict = json.loads(df.loc[i, 'qwen_72B_predict'])
            # 执行更新后的处理
            results = process_predictions(predict, label)
            print(f"results_no_filter: {results}")
            # 提取过滤结果
            filtered_result = filter_results(results)
            output.extend(filtered_result)

        elif model == "qwen2-32b":
            predict = json.loads(df.loc[i, 'qwen1.5-32b_predict'])
            # 执行更新后的处理
            results = process_predictions(predict, label)
            print(f"results_no_filter: {results}")
            # 提取过滤结果
            filtered_result = filter_results(results)
            output.extend(filtered_result)

        elif model == "qwen2-57b":
            predict = json.loads(df.loc[i, 'qwen2-57b_predict'])
            # 执行更新后的处理
            results = process_predictions(predict, label)
            print(f"results_no_filter: {results}")
            # 提取过滤结果
            filtered_result = filter_results(results)
            output.extend(filtered_result)


    df = pd.DataFrame(output)
    df2 = pd.DataFrame()
    df2['manual_content'] = df['matched_content']
    df2['label'] = df['matched_type']
    df2['content'] = df['content']
    df2['predict'] = df['type']
    # 计算"发起的"精确率指标
    df_filter_1 = df2[df2['predict']=='发起']
    df_filter_2 = df_filter_1[df_filter_1['label']=='发起']
    print(f"发起的精确率:{len(df_filter_2)/len(df_filter_1)}")
    # 输出指标
    # 计算“发起”的召回率指标：

    # 步骤 1：筛选出 label 列等于“发起”的行
    df_filtered = df2[df2['label'] == '发起']

    # 步骤 2：按 manual_content 分组，并检查相同 manual_content 是否有任何 predict 为“发起”
    grouped = df_filtered.groupby('manual_content').apply(lambda x: (x['predict'] == '发起').any())

    # 步骤 3：计算召回
    # 成功召回的 manual_content 数量
    recall_count = grouped.sum()

    # 去重后的 manual_content 总数
    total_unique_manual_content = df_filtered['manual_content'].nunique()

    # 步骤 4：计算召回率
    recall_rate = recall_count / total_unique_manual_content
    print("“发起”的召回率",recall_rate)
    # 给预测错误的数据添加上颜色。
    df2 = df2.style.apply(highlight_errors, axis=1)
    df2.to_excel(f"data/{model}_predict.xlsx", index=False)
