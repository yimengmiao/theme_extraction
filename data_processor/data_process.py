import pandas as pd
import json


class DataProcessor:
    def __init__(self, dataset, task, T):
        """
        初始化DataProcessor类，接收所有必要参数。

        :param dataset: 要处理的DataFrame或JSON数据
        :param task: 任务名称，决定数据处理逻辑
        :param T: 时间差阈值（由用户自定义，仅针对label为0的行）
        """
        self.task = task  # 任务类型，例如："teacher_dialogue_classification"
        self.T = T  # 分割时间差阈值，由用户传入
        self.dataset = self._prepare_dataset(dataset)  # 处理输入的数据，支持DataFrame或JSON字符串

    def _prepare_dataset(self, dataset):
        """
        预处理输入数据，将JSON数据转换为DataFrame。

        :param dataset: 输入的数据，可能是DataFrame或JSON字符串
        :return: 处理后的DataFrame
        """
        if isinstance(dataset, str):
            # 如果输入是JSON字符串，转换为DataFrame
            return pd.read_json(dataset)
        elif isinstance(dataset, pd.DataFrame):
            # 如果已经是DataFrame，直接返回
            return dataset
        else:
            raise ValueError("Unsupported data format. Please provide a DataFrame or JSON string.")

    def split_data(self):
        """
        根据任务类型进行数据分割。针对不同的任务，使用不同的逻辑。

        :return: 分割后的子数据集列表，每个子数据集是一个DataFrame
        """
        dataset = self.dataset
        end_positions = []  # 存储需要分割的位置
        sub_dfs = []  # 存储分割后的子DataFrame

        # 不同任务类型对应不同的分割逻辑
        if self.task == "teacher_dialogue_classification":
            # 对于师生对话分类任务，按时间差T进行分割，label为0时应用T
            for i in range(1, len(dataset)):
                time_gap = dataset.loc[i, 'start_time'] - dataset.loc[i - 1, 'end_time']
                # 针对 label 为 0 的行应用时间差阈值T进行分割
                if dataset.loc[i - 1, 'label'] == 0 and time_gap > self.T:
                    end_positions.append(i - 1)  # 记录分割点

        elif self.task == "teacher_activity_classification":
            # 对于课堂活动分类任务，可以根据其他特定条件进行分割（这里按时间差）
            for i in range(1, len(dataset)):
                time_gap = dataset.loc[i, 'start_time'] - dataset.loc[i - 1, 'end_time']
                # 在课堂活动分类任务中，也可以按自定义条件进行分割
                if time_gap > self.T:  # 这里可以是不同的逻辑
                    end_positions.append(i - 1)

        # 根据找到的分割点将 DataFrame 分割为多个子 DataFrame
        prev_idx = 0
        for end_idx in end_positions:
            sub_dfs.append(dataset.iloc[prev_idx:end_idx + 1].copy())  # 分割并复制子DataFrame
            prev_idx = end_idx + 1
        sub_dfs.append(dataset.iloc[prev_idx:].copy())  # 最后一个子 DataFrame

        return sub_dfs

    def merge_text_by_label(self, sub_df):
        """
        根据任务类型合并文本，按相同的label进行文本合并。

        :param sub_df: 子数据集 DataFrame
        :return: 合并后的子数据集 DataFrame
        """
        if self.task == "teacher_dialogue_classification":
            # 处理师生对话分类任务
            merged_rows = []
            # 初始化当前行信息
            current_text = sub_df.iloc[0]['text']
            current_start_time = sub_df.iloc[0]['start_time']
            current_end_time = sub_df.iloc[0]['end_time']
            current_label = sub_df.iloc[0]['label']

            # 遍历子数据集，合并相同label的文本
            for i in range(1, len(sub_df)):
                row = sub_df.iloc[i]
                if row['label'] == current_label:
                    # 如果label相同，合并文本
                    current_text += row['text']
                    current_end_time = row['end_time']
                else:
                    # 如果label不同，保存当前合并结果，并开始新的合并
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

            # 保存最后一行的合并结果
            merged_rows.append({
                'start_time': current_start_time,
                'end_time': current_end_time,
                'text': current_text,
                'label': current_label
            })

            return pd.DataFrame(merged_rows)

        elif self.task == "teacher_activity_classification":
            # 处理课堂活动分类任务
            merged_rows = []
            # 初始化当前行信息
            current_text = sub_df.iloc[0]['text']
            current_start_time = sub_df.iloc[0]['start_time']
            current_end_time = sub_df.iloc[0]['end_time']

            # 遍历子数据集，合并label为'activity'的行
            for i in range(1, len(sub_df)):
                row = sub_df.iloc[i]
                if row['label'] == "activity":  # 假设活动的label为 'activity'
                    current_text += row['text']
                    current_end_time = row['end_time']
                else:
                    # 保存活动相关的合并结果，并开始新的合并
                    merged_rows.append({
                        'start_time': current_start_time,
                        'end_time': current_end_time,
                        'text': current_text,
                        'label': 'activity'
                    })
                    current_text = row['text']
                    current_start_time = row['start_time']
                    current_end_time = row['end_time']

            # 保存最后一行的合并结果
            merged_rows.append({
                'start_time': current_start_time,
                'end_time': current_end_time,
                'text': current_text,
                'label': 'activity'
            })

            return pd.DataFrame(merged_rows)

    def process_and_save_sub_dfs(self):
        """
        处理整个数据集，分割并合并文本，根据任务返回处理后的JSON数据。

        :return: 包含处理结果的JSON列表
        """
        sub_dfs = self.split_data()  # 先分割数据
        json_list = []

        # 根据任务类型处理子数据集
        for sub_df in sub_dfs:
            processed_sub_df = self.merge_text_by_label(sub_df)  # 合并文本
            # 将处理后的DataFrame转换为JSON
            json_list.append(processed_sub_df.to_json(orient='records', force_ascii=False))

        return json_list


# 使用示例
if __name__ == "__main__":
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
        'label': [1, 1, 0, 0, 0, 0, 1, 0]
    }
    df = pd.DataFrame(data)

    # 初始化DataProcessor，任务为师生对话分类，时间差T由用户自行定义
    processor = DataProcessor(
        dataset=df,  # 传入数据集
        task="teacher_dialogue_classification",  # 任务名称
        T=3000  # 自定义时间差
    )

    # 处理数据
    json_output = processor.process_and_save_sub_dfs()

    # 输出结果
    for json_data in json_output:
        print(json_data)
