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

    def split_data(self):
        """
        根据任务类型进行数据分割。针对label为0的行，按时间差T进行分割。

        :return: 分割后的子数据集列表，每个子数据集是一个DataFrame
        """
        dataset = self.dataset
        sub_dfs = []
        current_start = 0

        while current_start < len(dataset):
            ref_end_time = dataset.loc[current_start, 'end_time']
            for i in range(current_start + 1, len(dataset)):
                if dataset.loc[i, 'label'] == 0:
                    time_gap = dataset.loc[i, 'start_time'] - ref_end_time
                    if time_gap > self.T:
                        sub_dfs.append(dataset.iloc[current_start:i].copy())
                        current_start = i
                        break
                    else:
                        ref_end_time = dataset.loc[i, 'end_time']
            else:
                sub_dfs.append(dataset.iloc[current_start:].copy())
                break

        return sub_dfs

    def process_sub_dfs(self, sub_dfs):
        """
        对sub_dfs中的每个sub_df进行处理，处理label为1的行出现在label为0的行之前的情况，并进行合并和特殊处理
        """
        for idx in range(1, len(sub_dfs)):
            sub_df = sub_dfs[idx]
            list1 = []

            for i in range(len(sub_df)):
                if sub_df.iloc[i]['label'] != 0:
                    list1.append(i)
                else:
                    break

            if list1:
                sub_dfs[idx - 1] = pd.concat([sub_dfs[idx - 1], sub_df.loc[list1]]).reset_index(drop=True)
                sub_dfs[idx] = sub_df.drop(list1).reset_index(drop=True)

        return sub_dfs

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
        # 丢弃第一个label为0之前的学生话语
        self.discard_student_before_first_teacher()

        # 分割数据
        sub_dfs = self.split_data()

        # 处理label为1的行出现在label为0之前的情况
        sub_dfs = self.process_sub_dfs(sub_dfs)

        json_list = []
        for sub_df in sub_dfs:
            processed_sub_df = self.merge_text_by_label(sub_df)
            json_list.append(processed_sub_df.to_json(orient='records', force_ascii=False))

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
    'start_time': [27300, 35310, 40560, 45590, 47910, 50070, 54340, 67170],
    'end_time': [32940, 39510, 42710, 47190, 49590, 52760, 64670, 69880],
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
    'label': [0, 0, 1, 0, 0, 0, 1, 0]
}
df = pd.DataFrame(data)

# 初始化DataProcessor，任务可以是"teacher_dialogue_classification"或"class_activity_classification"
processor = DataProcessorFixed(
    dataset=df,
    task="teacher_dialogue_classification",  # 更改任务为"class_activity_classification"
    T=800
)

# 处理数据
output = processor.process_and_save_sub_dfs()
print(output)
