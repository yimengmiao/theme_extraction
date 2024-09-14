import json
import re
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
from tqdm import tqdm

from test_model_ability import TeacherDialogueClassificationOldEvaluator


class TeacherDialogueClassificationEvaluator:
    pass


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

        if self.task == "test_model_ability":
            self.evaluator = TeacherDialogueClassificationOldEvaluator(self.model, self.df)
        elif self.task == "teacher_dialogue_classification":
            self.evaluator = TeacherDialogueClassificationEvaluator(self.model, self.df)
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
    model = "qwen_72B"
    task = "test_model_ability"
    df = pd.read_excel(f"../data/{task}/726四分类法.xlsx")

    evaluator = Evaluator(task=task, model=model, df=df)
    evaluator.evaluate()
