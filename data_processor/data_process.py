import json
import pandas as pd


class DataProcessorFixed:
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
            self.process_function = self.process_teacher_dialogue_classification
        elif self.task == "class_activity_classification":
            self.process_function = self.process_class_activity_classification
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
        else:
            raise ValueError("Unsupported data format. Please provide a DataFrame or JSON string.")

    def discard_student_before_first_teacher(self):
        """
        去掉第一个label为0的老师话语之前的所有学生话语（label为1的记录）
        """
        if (self.dataset['label'] == 0).any():
            first_teacher_idx = self.dataset[self.dataset['label'] == 0].index[0]
            self.dataset = self.dataset.loc[first_teacher_idx:].reset_index(drop=True)
        else:
            raise ValueError("No label=0 rows found in the dataset")

    def split_dataset(self):
        # 初始化
        df = self.dataset

        last_teacher_end_time = df.iloc[0]['end_time']  # 第一个教师话语的 end_time 作为初始时间
        current_start = 0  # 当前分割的起始点
        sub_datasets = []  # 存储子数据集

        for i in range(1, len(df)):
            if df.iloc[i]['label'] == 0:  # 遇到教师话语
                time_interval = df.iloc[i]['start_time'] - last_teacher_end_time  # 计算时间间隔

                # 检查是否需要分割
                if time_interval > self.T or any(df.iloc[j]['label'] == 1 for j in range(current_start, i)):
                    # 触发分割
                    sub_datasets.append(df.iloc[current_start:i])  # 将current_start到i-1的记录作为子数据集
                    current_start = i  # 更新分割起点为当前教师话语的索引

                # 更新last_teacher_end_time为当前教师话语的end_time
                last_teacher_end_time = df.iloc[i]['end_time']

        # 如果遍历结束后仍有数据未分割，处理最后一个子数据集
        if current_start < len(df):
            sub_datasets.append(df.iloc[current_start:])

        return sub_datasets

    def merge_text_by_label(self, sub_df):
        """
        根据任务类型合并文本，按相同的label进行文本合并。
        如果合并后只有label为0的行，则添加一个text为空，label为1的行。
        """
        merged_rows = []
        current_text = sub_df.iloc[0]['text']
        current_start_time = sub_df.iloc[0]['start_time']
        current_end_time = sub_df.iloc[0]['end_time']
        current_label = sub_df.iloc[0]['label']

        for i in range(1, len(sub_df)):
            row = sub_df.iloc[i]
            if row['label'] == current_label:
                current_text += row['text']
                current_end_time = row['end_time']
            else:
                merged_rows.append({
                    'start_time': current_start_time,
                    'end_time': current_end_time,
                    'text': current_text,
                    'label': current_label
                })
                current_text = row['text']
                current_start_time = row['start_time']
                current_end_time = row['end_time']
                current_label = row['label']

        merged_rows.append({
            'start_time': current_start_time,
            'end_time': current_end_time,
            'text': current_text,
            'label': current_label
        })

        if len(merged_rows) == 1 and merged_rows[0]['label'] == 0:
            merged_rows.append({
                'start_time': merged_rows[0]['start_time'],
                'end_time': merged_rows[0]['end_time'],
                'text': "",
                'label': 1
            })

        return pd.DataFrame(merged_rows)

    def process_teacher_dialogue_classification(self):
        """
        针对"teacher_dialogue_classification"任务的处理逻辑
        """
        # 丢弃第一个老师话语（label为0）之前的学生话语
        self.discard_student_before_first_teacher()

        # 分割数据
        sub_dfs = self.split_dataset()
        for item in sub_dfs:
            print(item)

        json_list = []
        for sub_df in sub_dfs:
            processed_sub_df = self.merge_text_by_label(sub_df)
            json_list.append(processed_sub_df.to_json(orient='records', force_ascii=False))
        json_list = [json.loads(item) for item in json_list]
        return json_list

    def process_class_activity_classification(self):
        """
        针对"class_activity_classification"任务的处理逻辑
        （这里可以根据实际需求添加专属逻辑）
        """
        # 示例：简单返回数据集的文本字段
        return self.dataset['text'].tolist()

    def process_and_save_sub_dfs(self):
        """
        处理整个数据集，调用对应任务的处理函数
        """
        return self.process_function()


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
    'label': [1, 1, 1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# 初始化DataProcessor，任务可以是"teacher_dialogue_classification"或"class_activity_classification"
processor = DataProcessorFixed(
    dataset=df,
    task="teacher_dialogue_classification",  # 更改任务为"class_activity_classification"
    T=500
)

# 处理数据
output = processor.process_and_save_sub_dfs()
print(output)
