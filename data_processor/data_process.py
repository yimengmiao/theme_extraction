import json
import pandas as pd


from teacher_dialogue_classification import TeacherDialogueClassificationProcessor


class DataProcessor:
    def __init__(self, dataset, task, T):
        """
        初始化DataProcessor类，接收所有必要参数。

        :param dataset: 要处理的DataFrame或JSON数据
        :param task: 任务名称，决定数据处理逻辑
        :param T: 时间差阈值（由用户自定义，仅针对label为0的行）
        """
        self.task = task  # 任务类型，例如："teacher_dialogue_classification" 或 "class_activity_classification"
        self.T = T  # 分割时间差阈值，由用户传入
        self.dataset = self._prepare_dataset(dataset)  # 处理输入的数据，支持DataFrame或JSON字符串

        # 根据任务类型初始化对应的处理逻辑
        if self.task == "teacher_dialogue_classification":
            # 选择老师四分类任务的数据处理逻辑。
            self.processor = TeacherDialogueClassificationProcessor(self.dataset, self.T)
        elif self.task == "class_activity_classification":
            self.processor = None  # 可以根据需求添加不同任务的处理类
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
        """
        处理整个数据集，调用对应任务的处理函数
        """
        if self.processor:
            return self.processor.process()
        else:
            raise NotImplementedError("Task processing for the given task is not implemented.")


if __name__ == '__main__':
    # 示例数据
    data = {
        'start_time': [27300, 35310, 40560, 45590, 47910, 50070, 52780, 53000],
        'end_time': [32940, 39510, 42710, 47190, 49590, 52760, 52790, 69880],
        'text': [
            "具你，为什么要我买？这是第一套。",
            "喂，你，吃你吃你狗，你，",
            "好，把语文书翻到第50页，",
            "然后铅笔收起来把，",
            "课堂练习放到左上角，",
            "先把语文书翻到翻到第50页，翻到这里，",
            "没有，50。我现在这个阳猫世，",
            "我看谁今天的坐姿有问题啊啊，"
        ],
        'label': [0, 0, 0, 0, 1, 0, 1, 0]
    }

    df = pd.DataFrame(data)

    # 初始化DataProcessor，任务可以是"teacher_dialogue_classification"或"class_activity_classification"
    processor = DataProcessor(
        dataset=df,
        task="teacher_dialogue_classification",  # 更改任务为"class_activity_classification"
        T=500
    )

    # 处理数据
    output = processor.process_and_save_sub_dfs()
    print(output)
