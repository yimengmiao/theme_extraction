# 主题提取(theme extraction)



# 指标评估算法文档（evaluate.py)

## 需求概述

该算法的目标是处理 `predict` 列表中的内容，并与 `label` 列表中的内容进行匹配。通过找到公共子文本串来评估两者的匹配程度。新的要求是在匹配时，只要找到公共子文本串，无需考虑两者的 `type` 是否一致，直接将 `label` 中的 `type` 和 `content` 填充到匹配到的 `predict` 项中。

---

## 算法流程

### 1. 最长公共子文本串算法

我们使用动态规划算法实现最长公共子文本串算法（`algrot`），它用于找出两个文本中最长的公共子串，并作为衡量两个文本相似度的依据。

    algrot(text1, text2) 函数：
    
    功能：计算 `text1` 和 `text2` 的最长公共子文本串。
    参数：
        text1：第一个输入文本。
        text2：第二个输入文本。
    返回值：
        最长公共子文本串。

### 2. 去除标点符号

为了使文本匹配更加准确，匹配前会对 `predict` 和 `label` 列表中的文本去除标点符号。

    remove_punctuation(text) 函数：
    
    功能：移除文本中的所有标点符号。
    参数：
        text：输入的文本字符串。
    返回值：
        去除了标点符号的文本字符串。

### 3. 计算匹配占比

计算找到的最长公共子文本串在 `label` 项中的文本长度占比，以此来衡量匹配的程度。匹配程度最高的 `label` 项将被选为最佳匹配。

    calculate_match_ratio(substring, content) 函数：
    
    功能：计算公共子串在目标文本中的占比。
    参数：
        substring：最长公共子串。
        content：目标文本。
    返回值：
        公共子串长度与目标文本长度的比值。

### 4. 不再考虑 `type` 匹配

在找到最长公共子文本串后，无需再判断 `predict` 项和 `label` 项中的 `type` 是否一致。只要找到公共子文本串，即将 `label` 中的 `content` 和 `type` 填充到 `predict` 项中。

    find_best_match(predict_item, labels) 函数：
    
    功能：为每个 `predict` 项找到最佳匹配的 `label` 项。
    参数：
        predict_item：`predict` 列表中的一项。
        labels：`label` 列表。
    返回值：
        最佳匹配的 `label` 项。

### 5. 记录匹配结果

对每个 `predict` 项目，记录其与 `label` 项中的最佳匹配，包含匹配到的 `content` 和 `type`。未匹配的项目则保持原状。

    process_predictions(predict, labels) 函数：
    
    功能：处理 `predict` 列表，找到每个 `predict` 项的最佳匹配，并记录匹配结果。
    参数：
        predict：`predict` 列表。
        labels：`label` 列表。
    返回值：
        包含匹配结果的 `predict` 列表。

---

## 代码实现

```python
import re

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
def find_best_match(predict_item, labels):
    """
    在 label 列表中为 predict_item 查找最佳匹配的 label 项。
    Args:
    - predict_item: 需要匹配的 predict 项。
    - labels: label 列表。
    
    Returns:
    - 最佳匹配的 label 项。
    """
    best_match = None
    highest_ratio = 0
    
    # 先去除标点符号
    predict_content_no_punc = remove_punctuation(predict_item['content'])
    
    for label in labels:
        label_content_no_punc = remove_punctuation(label['content'])
        # 使用最长公共子串算法进行匹配
        substring = algrot(predict_content_no_punc, label_content_no_punc)
        ratio = calculate_match_ratio(substring, label_content_no_punc)
        
        # 记录匹配占比最高的 label 项
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = label
    
    return best_match

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
    for predict_item in predict:
        best_match = find_best_match(predict_item, labels)
        
        # 只要找到匹配的 label 项，不管 type 是否匹配，都进行记录
        if best_match:
            predict_item['matched_content'] = best_match['content']
            predict_item['matched_type'] = best_match['type']
        
        results.append(predict_item)
    
    return results

# 执行算法示例
# 定义 label 和 predict 列表
label = [{'type': '发起', 'content': '请袁艺喊“开始上课”，声音要响亮啊。'}, ...]

predict = [{'type': '发起', 'content': '请袁艺喊“开始上课”，声音要响亮啊。'}, ...]

# 执行预测处理
results = process_predictions(predict, label)
