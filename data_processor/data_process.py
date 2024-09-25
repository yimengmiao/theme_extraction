import json

import pandas as pd

from .dialogue_split import DialougueProcessor
from .teacher_dialogue_classification import TeacherDialogueClassificationProcessor


class DataProcessor:
    def __init__(self, dataset, task, T=None):
        """
        初始化DataProcessor类，接收所有必要参数。

        :param dataset: 要处理的DataFrame或JSON数据
        :param task: 任务名称，决定数据处理逻辑
        :param T: 时间差阈值（由用户自定义，仅针对label为0的行）
        """
        self.task = task  # 任务类型，例如："teacher_dialogue_classification" 或 "class_activity_classification"
        self.T = T  # 分割时间差阈值，由用户传入
        self.dataset = dataset  # 处理输入的数据，支持DataFrame或JSON字符串

        # 根据任务类型初始化对应的处理逻辑
        if self.task == "teacher_dialogue_classification":
            # 选择老师四分类任务的数据处理逻辑。
            self.processor = TeacherDialogueClassificationProcessor(self.dataset, self.T)
        elif self.task == "class_activity_classification":
            self.processor = None  # 可以根据需求添加不同任务的处理类
        elif self.task == "dialogue_processing":
            self.processor = DialougueProcessor(self.dataset)  # 添加对DialougueProcessor的支持
        else:
            raise ValueError(f"Unsupported task type: {self.task}")

    def _prepare_dataset(self, dataset):
        """
        预处理输入数据，将JSON数据转换为DataFrame。

        :param dataset: 输入的数据，可能是DataFrame或JSON字符串
        :return: 处理后的DataFrame
        """
        if isinstance(dataset, str):
            return pd.read_json(dataset)
        elif isinstance(dataset, pd.DataFrame):
            return dataset
        elif isinstance(dataset, dict):
            return pd.DataFrame(dataset)
        else:
            raise ValueError("Unsupported data format. Please provide a DataFrame or JSON string.")

    def process_and_save_sub_dfs(self):
        if self.processor:
            # 根据任务类型调用相应的方法
            if self.task == "dialogue_processing":
                segments = self.processor.process_data()
                final_texts = self.processor.process_segments(segments)

                return final_texts
            elif self.task == "teacher_dialogue_classification":
                return self.processor.process()  # 假设TeacherDialogueClassificationProcessor有process方法
            # 可以继续添加其他任务类型的处理逻辑
        else:
            raise NotImplementedError("Task processing for the given task is not implemented.")


if __name__ == '__main__':
    # 示例数据
    # data = {
    #     'start_time': [27300, 35310, 40560, 45590, 47910, 50070, 52780, 53000],
    #     'end_time': [32940, 39510, 42710, 47190, 49590, 52760, 52790, 69880],
    #     'text': [
    #         "具你，为什么要我买？这是第一套。",
    #         "喂，你，吃你吃你狗，你，",
    #         "好，把语文书翻到第50页，",
    #         "然后铅笔收起来把，",
    #         "课堂练习放到左上角，",
    #         "先把语文书翻到翻到第50页，翻到这里，",
    #         "没有，50。我现在这个阳猫世，",
    #         "我看谁今天的坐姿有问题啊啊，"
    #     ],
    #     'label': [0, 1, 0, 0, 1, 0, 1, 0]
    # }
    # #
    # # df = pd.DataFrame(data)
    #
    # # 初始化DataProcessor，任务可以是"teacher_dialogue_classification"或"class_activity_classification"
    # processor = DataProcessor(
    #     dataset=data,
    #     task="teacher_dialogue_classification",  # 更改任务为"class_activity_classification"
    #     T=500
    # )
    #
    # # 处理数据
    # output = processor.process_and_save_sub_dfs()
    # print("output", output)

    data = [
        # 第一条数据
        {
            "start_time": 0,
            "end_time": 10,
            "text": "老师发起提问1，老师讲解内容1，老师发起提问2",
            "label": 0,
            "gpt4o_result": json.dumps({
                "result": [
                    {"type": "发起", "content": "老师发起提问1"},
                    {"type": "讲解", "content": "老师讲解内容1"},
                    {"type": "发起", "content": "老师发起提问2"},
                ],
                "compliance": "高"
            })
        },
        {"start_time": 10, "end_time": 20, "text": "学生回答1", "label": 1, "gpt4o_result": None},
        # 第二条数据
        {
            "start_time": 20,
            "end_time": 30,
            "text": "老师讲解内容2，老师讲解内容3，老师发起提问3，老师发起提问4。",
            "label": 0,
            "gpt4o_result": json.dumps({
                "result": [
                    {"type": "讲解", "content": "老师讲解内容2"},
                    {"type": "讲解", "content": "老师讲解内容3"},
                    {"type": "发起", "content": "老师发起提问3"},
                    {"type": "发起", "content": "老师发起提问4"}
                ],
                "compliance": "高"
            })
        },
        {"start_time": 30, "end_time": 40, "text": "学生回答2", "label": 1, "gpt4o_result": None},
        # E_after 数据
        {
            "start_time": 40,
            "end_time": 50,
            "text": "老师讲解内容4",
            "label": 0,
            "gpt4o_result": json.dumps({
                "result": [
                    {"type": "讲解", "content": "老师讲解内容4"}
                ],
                "compliance": "高"
            })
        },
        {"start_time": 50, "end_time": 60, "text": "学生回应3", "label": 1, "gpt4o_result": None},
        {
            "start_time": 60,
            "end_time": 70,
            "text": "老师讲解内容5",
            "label": 0,
            "gpt4o_result": json.dumps({
                "result": [
                    {"type": "讲解", "content": "老师讲解内容5"}
                ],
                "compliance": "高"
            })
        },
        {"start_time": 70, "end_time": 80, "text": "学生回应4", "label": 1, "gpt4o_result": None}
    ]

    processor = DataProcessor(
        dataset=data,
        task="dialogue_processing",  # 选择新的任务类型
    )

    # 处理数据
    output = processor.process_and_save_sub_dfs()
    print("output", output)
