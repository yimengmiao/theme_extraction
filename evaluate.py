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
    在 label 列表中为多个 predict_item 查找最佳匹配的 label 项，并将匹配结果保存到 predict_items 中。
    返回包含匹配结果的 predict_items 列表。
    """
    start_index = 0  # 初始化从第一个 label 开始

    for predict_item in predict_items:
        best_match = None
        highest_ratio = 0
        best_match_index = -1  # 用于记录匹配的索引
        print("predict_items:", predict_items)
        # 去除标点符号
        predict_content_no_punc = remove_punctuation(predict_item['content'])

        # 只在 start_index 到 start_index + 5 的范围内进行匹配
        for i in range(start_index, min(start_index + 5, len(labels))):
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

        if best_match is None:
            predict_item['best_match'] = {"content": "其它", "type": "其它"}
        else:
            predict_item['best_match'] = best_match

            # 如果匹配的索引更新了，才更新 start_index
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


if __name__ == '__main__':
    # 定义 label 和 predict 列表
    model = "qwen_72B"
    df = pd.read_excel("data/726四分类法.xlsx")
    output = []
    for i in tqdm(range(len(df))):
        label = json.loads(df.loc[i, 'label'])
        # label = [{'type': '发起', 'content': '请袁艺喊“开始上课”，声音要响亮啊。'},
        #          {'type': '其它', 'content': '同学们好，请坐。'},
        #          {'type': '评价', 'content': '真棒。'},
        #          {'type': '讲解', 'content': '昨天呀，我们一起学习了课文八。今天，小猴子呀要到我们班来做客了。'},
        #          {'type': '发起', 'content': '我们来跟他打打招呼。“猴子”。'},
        #          {'type': '讲解',
        #           'content': '昨天在写作业的时候呀，小朋友要注意哦。这个“猴”，反犬旁，旁边是一个单立人，中间有没有一个短竖啊？那么昨天在作业当中，方老师看到有人加了一竖，那就不对了，变成错别字了。明白了吗？'},
        #          {'type': '发起', 'content': '好，现在用眼睛看，用心记住这个字，“猴子”。'},
        #          {'type': '讲解', 'content': '哎，每天通过学习啊，我们知道了这个猴子啊，种了一些果树。'},
        #          {'type': '发起', 'content': '它分别种了什么树呢？谁来说说？于凯，你来说说看。'},
        #          {'type': '其它', 'content': '你慢讲啊。嗯，'},
        #          {'type': '发起', 'content': '然后呢？'},
        #          {'type': '讲解', 'content': '然后种了杏树。'},
        #          {'type': '发起', 'content': '最后呢？'},
        #          {'type': '讲解', 'content': '最后还种了桃树。'},
        #          {'type': '发起', 'content': '到最后怎么样啊？'},
        #          {'type': '发起', 'content': '好，于凯，你能不能连起来把这句话来说一说？“猴子种了”。'},
        #          {'type': '评价', 'content': '非常棒，请坐。'},
        #          {'type': '发起',
        #           'content': '我们像他一样，用一句话把这个故事，哎，就讲出来了。来，我们一起来试试看。“猴子种了”。'},
        #          {'type': '评价', 'content': '瞧，咱们一句话就能把故事的意思讲出来了。小朋友们真能干。'},
        #          {'type': '发起', 'content': '好，那我们来看看哦，他为什么什么都没种成呢？我们来读课文的第一段。来，谁来读？'},
        #          {'type': '其它', 'content': '那等一下啊。'},
        #          {'type': '发起', 'content': '好，你来读。“猴子”。'},
        #          {'type': '发起', 'content': '嗯嗯，你先不要坐下，这个“天天”说明什么呀？'}]
        #
        # predict = [{'type': '发起', 'content': '请袁艺喊“开始上课”，声音要响亮啊。'},
        #            {'type': '其它', 'content': '同学们好，请坐。'},
        #            {'type': '评价', 'content': '真棒。'},
        #            {'type': '讲解', 'content': '昨天呀，我们一起学习了课文八。今天，小猴子呀要到我们班来做客了。'},
        #            {'type': '发起', 'content': '我们来跟他打打招呼。“猴子”。'},
        #            {'type': '讲解', 'content': '昨天在写作业的时候呀，小朋友要注意哦。'},
        #            {'type': '讲解', 'content': '这个“猴”，反犬旁，旁边是一个单立人，中间有没有一个短竖啊？'},
        #            {'type': '讲解', 'content': '那么昨天在作业当中，方老师看到有人加了一竖，那就不对了，变成错别字了。'},
        #            {'type': '发起', 'content': '明白了吗？'},
        #            {'type': '发起', 'content': '好，现在用眼睛看，用心记住这个字，“猴子”。'},
        #            {'type': '讲解', 'content': '哎，每天通过学习啊，我们知道了这个猴子啊，种了一些果树。'},
        #            {'type': '发起', 'content': '它分别种了什么树呢？谁来说说？'},
        #            {'type': '发起', 'content': '于凯，你来说说看。'},
        #            {'type': '其它', 'content': '你慢讲啊。嗯，'},
        #            {'type': '发起', 'content': '然后呢？'},
        #            {'type': '讲解', 'content': '然后种了杏树。'},
        #            {'type': '发起', 'content': '最后呢？'},
        #            {'type': '讲解', 'content': '最后还种了桃树。'},
        #            {'type': '发起', 'content': '到最后怎么样啊？'},
        #            {'type': '发起', 'content': '好，于凯，你能不能连起来把这句话来说一说？“猴子种了”。'},
        #            {'type': '评价', 'content': '非常棒，请坐。'},
        #            {'type': '讲解', 'content': '我们像他一样，用一句话把这个故事，哎，就讲出来了。'},
        #            {'type': '发起', 'content': '来，我们一起来试试看。“猴子种了”。'},
        #            {'type': '讲解', 'content': '瞧，咱们一句话就能把故事的意思讲出来了。'},
        #            {'type': '评价', 'content': '小朋友们真能干。'},
        #            {'type': '发起', 'content': '好，那我们来看看哦，他为什么什么都没种成呢？我们来读课文的第一段。来，谁来读？'},
        #            {'type': '其它', 'content': '那等一下啊。'},
        #            {'type': '发起', 'content': '好，你来读。“猴子”。'},
        #            {'type': '发起', 'content': '嗯嗯，你先不要坐下，这个“天天”说明什么呀？'}]
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
            predict = json.loads(df.loc[i, 'qwen_predict'])
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
    # 计算指标
    filter_output = [item for item in output if item.get("type") in ['发起', '评价']]
    print(f"{model}预测出的发起和评价的数量: {len(filter_output)}")
    metrics = calculate_metrics(filter_output)
    # 输出指标
    print(metrics)

    df = pd.DataFrame(output)
    df2 = pd.DataFrame()
    df2['manual_content'] = df['matched_content']
    df2['label'] = df['matched_type']
    df2['content'] = df['content']
    df2['predict'] = df['type']
    df2.to_excel(f"{model}_predict.xlsx", index=False)
