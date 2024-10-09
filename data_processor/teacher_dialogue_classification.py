import json

import pandas as pd

from .public_code_data_process import convert_punctuation_to_chinese
import logging

# 配置 logger，将日志记录存储到文件中
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='log/app_log.log',  # 指定日志文件路径
    filemode='a'  # 文件模式：'a' 追加模式，'w' 写入模式会覆盖之前的日志
)
logger = logging.getLogger(__name__)


# 老师四分类任务中的数据处理逻辑，具体是：将课堂音频转译文本分割为一个个师生对话文本段。
class TeacherDialogueClassificationProcessor:
    def __init__(self, dataset, T):
        """
        初始化教师对话分类处理器
        :param dataset: json格式的数据
        :param T: 时间差阈值
        """
        if not isinstance(dataset, dict):
            error_msg = "Dataset should be a dict"
            logger.error(error_msg)
            raise TypeError(error_msg)

        if not isinstance(T, (int, float)) or T < 0:
            error_msg = "T should be a non-negative number"
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            self.dataset = pd.DataFrame(dataset)
        except Exception as e:
            error_msg = f"Error converting dataset to DataFrame: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        required_columns = {'label', 'start_time', 'end_time', 'text'}
        if not required_columns.issubset(self.dataset.columns):
            missing = required_columns - set(self.dataset.columns)
            error_msg = f"Dataset must contain the following columns: {required_columns}. Missing: {missing}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        self.T = T

    def discard_student_before_first_teacher(self):
        """
        去掉第一个label为0的老师话语之前的所有学生话语（label为1的记录）
        """
        if 'label' not in self.dataset.columns:
            error_msg = "Dataset must contain a 'label' column"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if (self.dataset['label'] == 0).any():
            first_teacher_idx = self.dataset[self.dataset['label'] == 0].index[0]
            self.dataset = self.dataset.loc[first_teacher_idx:].reset_index(drop=True)
        else:
            error_msg = "No label=0 rows found in the dataset"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def split_dataset(self):
        """
        按相信老师的时间差T和老师学生标签分割数据集
        """
        df = self.dataset

        if df.empty:
            error_msg = "Dataset is empty"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if df.iloc[0]['label'] != 0:
            error_msg = "The first row must be a teacher's speech (label=0)"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not pd.api.types.is_numeric_dtype(df['end_time']) or not pd.api.types.is_numeric_dtype(df['start_time']):
            error_msg = "start_time and end_time must be numeric"
            logger.error(error_msg)
            raise TypeError(error_msg)

        last_teacher_end_time = df.iloc[0]['end_time']  # 第一个教师话语的 end_time 作为初始时间
        current_start = 0  # 当前分割的起始点
        sub_datasets = []  # 存储子数据集

        for i in range(1, len(df)):
            if df.iloc[i]['label'] == 0:  # 遇到教师话语
                time_interval = df.iloc[i]['start_time'] - last_teacher_end_time  # 计算时间间隔

                if time_interval < 0:
                    error_msg = "start_time of a speech cannot be earlier than the previous end_time"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

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

        if not sub_datasets:
            error_msg = "No valid sub-datasets created"
            logger.error(error_msg)
            raise ValueError(error_msg)

        return sub_datasets

    def merge_text_by_label(self, sub_df):
        """
        按相同的label合并文本，并处理特殊情况
        """
        if sub_df.empty:
            error_msg = "Sub-dataframe is empty"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if 'text' not in sub_df.columns:
            error_msg = "Sub-dataframe must contain a 'text' column"
            logger.error(error_msg)
            raise ValueError(error_msg)

        merged_rows = []
        try:
            current_text = convert_punctuation_to_chinese(sub_df.iloc[0]['text'])  # 转换标点符号
        except Exception as e:
            error_msg = f"Error converting punctuation: {e}"
            logger.error(error_msg)
            raise ValueError(error_msg)

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

        # 处理仅有一条老师话语且无学生话语的情况
        if len(merged_rows) == 1 and merged_rows[0]['label'] == 0:
            merged_rows.append({
                'start_time': merged_rows[0]['start_time'],
                'end_time': merged_rows[0]['end_time'],
                'text': "",
                'label': 1
            })

        return pd.DataFrame(merged_rows)

    def process(self):
        """
        处理教师对话分类任务
        """
        try:
            # 丢弃第一个老师话语（label为0）之前的学生话语
            self.discard_student_before_first_teacher()

            # 分割数据集，这就是获取 prompt1输入前的第一步数据处理工作。
            sub_dfs = self.split_dataset()
            # 下面这就是获取 prompt1输入前的第二步数据处理工作。按照 老师话语，学生话语这样的形式来作为prompt1的输入。
            json_list = []
            for sub_df in sub_dfs:
                processed_sub_df = self.merge_text_by_label(sub_df)
                json_list.append(processed_sub_df.to_json(orient='records', force_ascii=False))
            json_list = [json.loads(item) for item in json_list]
            for item in json_list:
                teacher_text = ""
                student_text = ""

                # 处理item中的每个对话
                for sub_item in item:
                    if sub_item['label'] == 0:
                        teacher_text += sub_item['text'] + " "
                    elif sub_item['label'] == 1:
                        student_text += sub_item['text'] + ""

                # 构建分析师生对话段文本
                if student_text:
                    text_to_analyze = f"""
                    “老师话语”：{teacher_text.strip()}
                    “学生话语”：{student_text}
                    """
                else:
                    text_to_analyze = f"""“老师话语”：{teacher_text.strip()}"""
                item.append({"model_input": text_to_analyze})

            return json_list

        except Exception as e:
            logger.error(f"Error during processing: {e}")
            raise

