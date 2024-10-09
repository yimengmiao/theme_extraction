import pandas as pd
import os
from business_code.teacher_classification_api_version import Teacher_four_categories
from business_code.breakpoint_and_topic_extractor import topic_extract
from config.common_config import DEFAULT_MODEL_PARAMETERS, DEFAULT_DATA_PROCESSOR


def theme_extraction(data, model_parameters=None, data_processor=None):
    """
    接口函数，输入参数为 data，依次调用 Teacher_four_categories 和 topic_extract 函数
    :param data: 输入数据，格式与 input_json['data'] 相同
    :param model_parameters: 模型参数，可选，默认为 DEFAULT_MODEL_PARAMETERS
    :param data_processor: 数据处理器参数，可选，默认为 DEFAULT_DATA_PROCESSOR
    :return: topic_extract 函数的输出结果
    """


    try:
        # 输入数据校验
        if not isinstance(data, dict):
            raise ValueError("输入的数据应为字典格式。")
        # todo: sample_data的较参中，label里的值(label值 是一个list列表)一定要有0，没有0就不让其进行下一步的逻辑处理，直接报输入数据有问题。
        required_keys = {'start_time', 'end_time', 'text', 'label'}
        if not required_keys.issubset(data.keys()):
            raise ValueError(f"输入数据缺少必要的键，必须包含 {required_keys}")

        # 使用默认的模型参数和数据处理器配置，如果有自定义的则覆盖
        model_parameters = model_parameters or DEFAULT_MODEL_PARAMETERS.copy()
        data_processor = data_processor or DEFAULT_DATA_PROCESSOR.copy()

        # 构建 input_json，用于传递给 Teacher_four_categories（老师四分类任务）
        input_json = {
            'model_parameters': model_parameters,
            'data_processor': data_processor,
            'data': data,
            'output_path': 'output.xlsx'
        }

        # 从 txt 文件中读取 Teacher_four_categories 的 prompt
        teacher_prompt_path = 'prompt/teacher_dialouge_classification_prompt1.txt'
        if not os.path.exists(teacher_prompt_path):
            raise FileNotFoundError(f"找不到指定的提示文件: {teacher_prompt_path}")

        with open(teacher_prompt_path, 'r', encoding='utf-8') as f:
            teacher_prompt = f.read()

        input_json['model_parameters']['prompt'] = teacher_prompt
        print("input_json:", input_json)

        # 调用 Teacher_four_categories 函数
        result_from_teacher = Teacher_four_categories(input_json)

        # 检查返回结果
        if not result_from_teacher:
            raise ValueError("Teacher_four_categories 返回空结果。")

        # 更新 data_processor 的任务类型
        data_processor['Task'] = 'dialogue_processing'
        data_processor['T'] = 800  # 如果需要修改，可以在调用时传入

        # 构建传递给 topic_extract 的参数
        params_for_topic_extract = {
            'model_parameters': model_parameters,
            'data_processor': data_processor,
            'data': result_from_teacher
        }

        # 从 txt 文件中读取 topic_extract 的 prompt
        topic_prompt_path = 'prompt/prompt2输出”讲解“的分割点.txt'
        if not os.path.exists(topic_prompt_path):
            raise FileNotFoundError(f"找不到指定的提示文件: {topic_prompt_path}")

        with open(topic_prompt_path, 'r', encoding='utf-8') as f:
            topic_prompt = f.read()

        params_for_topic_extract['model_parameters']['prompt'] = topic_prompt

        # 调用 topic_extract 函数
        final_output = topic_extract(params_for_topic_extract)

        # 检查返回结果
        if not final_output:
            raise ValueError("topic_extract 返回空结果。")

        # TODO: 再给每个 sub_text 加上对应的 start_time, end_time

        # 返回最终结果
        return final_output

    except Exception as e:
        # 异常处理，确保输出错误信息便于调试
        print(f"处理数据时发生错误: {e}")
        return None


# 示例调用
if __name__ == '__main__':
    # 您可以在这里替换为自己的 data
    df = pd.read_excel("data/original_data/test2.xlsx")
    sample_data = {
        'start_time': df['start_time'].to_list(),
        'end_time': df['end_time'].to_list(),
        'text': df['text'].to_list(),
        'label': df['label'].to_list()
    }

    # 调用接口函数

    result = theme_extraction(sample_data)

    # 输出结果
    print("最终输出结果:", result)
