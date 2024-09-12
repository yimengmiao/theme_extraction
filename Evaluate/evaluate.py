import json
import re
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
from tqdm import tqdm


class TeacherDialogueClassificationOldEvaluator:
    def __init__(self, model, df):
        """
        初始化教师对话分类旧版评估类
        :param model: 模型名称，用于选择对应的模型评估
        :param df: 要评估的数据集
        """
        self.model = model
        self.df = df

    def evaluate(self):
        """
        针对 teacher_dialogue_classification_old 任务的评估逻辑
        包含数据处理、匹配和评估指标计算
        """
        output = []
        for i in tqdm(range(len(self.df))):
            label = json.loads(self.df.loc[i, 'label'])

            predict = self._get_predict(i)

            # 执行更新后的处理
            results = self.process_predictions(predict, label)
            print(f"results_no_filter: {results}")

            # 提取过滤结果
            filtered_result = self.filter_results(results)
            output.extend(filtered_result)

        # 输出精确率和召回率等指标
        self.calculate_final_metrics(output)

    def _get_predict(self, index):
        """
        根据模型名称获取预测结果
        """
        if self.model == "gpt4o":
            return json.loads(self.df.loc[index, 'gpt4o_predict']).get('result')
        elif self.model == "glm4":
            return json.loads(self.df.loc[index, 'glm4_predict'])
        elif self.model == "qwen_long":
            return json.loads(self.df.loc[index, 'qwen_long_predict'])
        elif self.model == "qwen_max":
            return json.loads(self.df.loc[index, 'qwen_max_predict'])
        elif self.model == "qwen_72B":
            return json.loads(self.df.loc[index, 'qwen_72B_predict'])
        elif self.model == "qwen2-32b":
            return json.loads(self.df.loc[index, 'qwen-32b_predict'])
        elif self.model == "qwen2-57b":
            return json.loads(self.df.loc[index, 'qwen2-57b_predict'])
        else:
            raise ValueError(f"Unsupported model type: {self.model}")

    def process_predictions(self, predict, labels):
        """
        处理 predict 列表，找到每个 predict 项在 label 列表中的最佳匹配项。
        """
        results = []
        best_match = self.find_best_match(predict, labels)
        for item in best_match:
            if item.get("best_match") is not None:
                dict1 = {
                    'content': item['content'],
                    'type': item['type'],
                    'matched_content': item.get("best_match").get('content', '其它'),
                    'matched_type': item.get("best_match").get('type', '其它')
                }
            else:
                dict1 = {
                    'content': item['content'],
                    'type': item['type'],
                    'matched_content': '其它',
                    'matched_type': '其它'
                }
            results.append(dict1)
        return results

    def find_best_match(self, predict_items, labels):
        """
        使用匹配算法找到最佳匹配项。
        """
        start_index = 0  # 初始化从第一个 label 开始
        for predict_item in predict_items:
            best_match = None
            highest_ratio = 0
            best_match_index = -1  # 用于记录匹配的索引
            predict_content_no_punc = self.remove_punctuation(predict_item['content'])

            for i in range(start_index, len(labels)):
                label = labels[i]
                label_content_no_punc = self.remove_punctuation(label['content'])

                if predict_content_no_punc in label_content_no_punc:
                    best_match = label
                    best_match_index = i
                    predict_item['best_match'] = best_match
                    start_index = best_match_index
                    break

            if best_match is None:
                for i in range(start_index, min(start_index + 3, len(labels))):
                    label = labels[i]
                    label_content_no_punc = self.remove_punctuation(label['content'])
                    common_substrings = self.find_all_common_substrings(predict_content_no_punc, label_content_no_punc)
                    valid_substrings = [sub for sub in common_substrings if
                                        len(sub) / len(predict_content_no_punc) >= 0.3]

                    if not valid_substrings:
                        continue

                    for substring in valid_substrings:
                        ratio = self.calculate_match_ratio(substring, label_content_no_punc)
                        if ratio > highest_ratio:
                            highest_ratio = ratio
                            best_match = label
                            best_match_index = i

            if best_match is None:
                predict_item['best_match'] = {"content": "其它", "type": "其它"}
            else:
                predict_item['best_match'] = best_match
                if best_match_index != -1 and best_match_index > start_index:
                    start_index = best_match_index

        return predict_items

    def filter_results(self, results):
        """
        从 results 中提取 matched_type 为 '发起' 和 '评价' 的项
        """
        return [item for item in results if item['matched_type'] in ['发起', '评价']]

    def calculate_final_metrics(self, output):
        """
        计算和输出精确率、召回率、F1和准确率等评估指标
        """
        df = pd.DataFrame(output)
        df2 = pd.DataFrame()
        df2['manual_content'] = df['matched_content']
        df2['label'] = df['matched_type']
        df2['content'] = df['content']
        df2['predict'] = df['type']

        df_filter_1 = df2[df2['predict'] == '发起']
        df_filter_2 = df_filter_1[df_filter_1['label'] == '发起']
        print(f"发起的精确率: {len(df_filter_2) / len(df_filter_1)}")

        df_filtered = df2[df2['label'] == '发起']
        grouped = df_filtered.groupby('manual_content').apply(lambda x: (x['predict'] == '发起').any())
        recall_count = grouped.sum()
        total_unique_manual_content = df_filtered['manual_content'].nunique()
        recall_rate = recall_count / total_unique_manual_content
        print(f"“发起”的召回率: {recall_rate}")

    def remove_punctuation(self, text):
        """
        移除输入文本中的标点符号
        """
        return re.sub(r'[^\w\s]', '', text)

    def find_all_common_substrings(self, text1, text2):
        """
        查找文本1和文本2的所有公共子串，返回一个子串列表。
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

        return list(set(substrings))

    def calculate_match_ratio(self, substring, content):
        """
        计算公共子串在目标文本中的占比
        """
        return len(substring) / len(content) if content else 0


class Evaluator:
    def __init__(self, task, model, df):
        """
        初始化Evaluator类，接收task、model和数据集（df）参数。

        :param task: 任务名称，决定使用哪种评估算法
        :param model: 模型名称，用于选择对应的模型评估
        :param df: 要评估的数据集
        """
        self.task = task
        self.model = model
        self.df = df

        if self.task == "teacher_dialogue_classification_old":
            self.evaluator = TeacherDialogueClassificationOldEvaluator(self.model, self.df)
        elif self.task == "class_activity_classification":
            self.evaluator = None  # 根据实际需求实现其他任务的评估器
        else:
            raise ValueError(f"Unsupported task type: {self.task}")

    def evaluate(self):
        """
        执行评估，根据任务调用相应的评估函数
        """
        if self.evaluator:
            return self.evaluator.evaluate()
        else:
            raise NotImplementedError(f"Evaluation for task {self.task} is not implemented.")


# 使用Evaluator类进行评估示例
if __name__ == '__main__':
    model = "qwen_long"
    df = pd.read_excel("../data/726四分类法.xlsx")

    evaluator = Evaluator(task="teacher_dialogue_classification_old", model=model, df=df)
    evaluator.evaluate()
