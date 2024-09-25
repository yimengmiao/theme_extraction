import json
import pandas as pd
import re
import logging

# 引入模型 API 处理和数据处理模块
from model_api.model_api_handler import ModelAPI
from data_processor.data_process import DataProcessor
from data_processor.public_data_process import extract_json_using_patterns, remove_punctuation

# 配置 logger，用于记录日志信息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 自定义异常类，用于处理参数校验相关的错误
class InputError(Exception):
    """自定义异常，用于处理参数错误"""
    pass


def validate_input(params):
    """
    参数校验函数，确保输入参数符合预期格式
    :param params: 包含配置和数据的字典
    :return: 如果校验失败，返回错误信息，否则返回 None
    """
    try:
        # 校验 data_processor 是否存在
        if not params.get("data_processor"):
            raise InputError("data_processor is a required field.")
        # 校验 data_processor 中的 Task 是否存在
        if not params["data_processor"].get("Task"):
            raise InputError("Task in data_processor cannot be empty.")

        # 校验 data 是否存在并且非空
        if "data" not in params or not params["data"]:
            raise InputError("data is a required field and cannot be empty.")

        # 校验 model_parameters 中的各个参数是否非空
        if "model_parameters" in params:
            for key, value in params["model_parameters"].items():
                if not value:
                    raise InputError(f"{key} in model_parameters cannot be empty.")
    except InputError as e:
        # 如果捕获到 InputError，记录错误日志并返回错误信息
        logger.error("参数校验失败: %s", e)
        return {"error": str(e)}


def process_data_and_analyze(params):
    """
    处理数据并根据需求调用模型分析
    :param params: 包含模型参数、数据处理参数和数据的字典
    :return: 处理和分析后的结果
    """

    # 校验输入参数的正确性
    error = validate_input(params)
    if error:
        logger.error("参数校验失败: %s", error)
        return error  # 如果有错误，直接返回错误信息

    # 必要字段列表
    required_keys = ['start_time', 'end_time', 'text', 'label']
    data = params.get('data')

    if not data:
        logger.error("缺少 'data' 字段")
        return {"error": "Missing 'data' field in parameters."}

    # 校验数据的格式，确保必须字段存在并且为列表
    for key in required_keys:
        if key not in data:
            logger.error("缺少必要字段 '%s' 在 data 中", key)
            return {"error": f"Missing required key '{key}' in data."}
        if not isinstance(data[key], list):
            logger.error("'%s' 在 data 中必须为列表", key)
            return {"error": f"'{key}' in data must be a list."}

    # 初始化数据处理类，根据任务类型和时间阈值处理数据
    processor = DataProcessor(
        dataset=data,  # 传入数据字典
        task=params['data_processor'].get('Task', "teacher_dialogue_classification"),  # 默认任务类型
        T=params['data_processor'].get('T', 800)  # 时间差阈值，默认 800
    )

    try:
        # 处理数据并获取处理后的结果
        output = processor.process_and_save_sub_dfs()
    except Exception as e:
        # 捕获处理数据时的异常并记录日志
        logger.error("数据处理错误: %s", e)
        return {"error": f"Data processing error: {str(e)}"}

    # 如果没有传入 model_parameters，只返回数据处理结果
    if "model_parameters" not in params:
        logger.info("未传入 model_parameters, 直接返回数据处理结果")
        return output

    # 如果存在 model_parameters，调用模型进行进一步分析
    for item in output:
        model_name = params['model_parameters'].get("model_name", "glm-4-flash")  # 获取模型名称
        params['model_parameters']['text'] = item[2].get("model_input")  # 将处理后的输入文本传入模型参数
        logger.info("调用模型的入参 params: %s", params)

        # 调用模型 API 进行分析
        model_api = ModelAPI(params.get("model_parameters", ""))
        result = model_api.analyze_text()

        # 记录模型分析结果
        logger.info("模型分析结果: %s", result)

        # 将模型分析结果加入输出项中
        item.append({f"{model_name}_result": result})
        logger.debug("模型分析后的 item: %s", item)
    return output


def parse_test_result(test_result):
    """
    解析模型返回的结果，提取 JSON 数据
    :param test_result: 模型返回的字符串结果
    :return: 解析后的 JSON 数据
    """
    if test_result:
        try:
            # 尝试直接解析 JSON 数据
            result_data = json.loads(test_result)
            logger.debug("直接解析的 JSON 数据: %s", result_data)
            return result_data
        except json.JSONDecodeError:
            # 如果直接解析失败，则使用正则模式提取 JSON
            return extract_json_using_patterns(test_result)
    else:
        logger.warning("test_result 为空")
        return {}


def process_output_result(output_result, model_name):
    """
    处理模型的输出结果，生成最终的 DataFrame
    :param output_result: 模型输出结果
    :param model_name: 模型名称
    :return: 处理后的结果 DataFrame
    """
    result_list = []

    # 遍历输出结果
    for sublist in output_result:
        test_result = None
        if len(sublist) >= 4:
            third_item = sublist[3]
            test_result = next(iter(third_item.values()))
        else:
            test_result = None
        logger.debug("提取到的 test_result: %s", test_result)

        # 解析模型的返回结果
        contents = parse_test_result(test_result)
        logger.debug("解析后的内容: %s", contents)
        logger.debug("子列表: %s", sublist)

        new_sublist = []
        # 这里是以防不能从模型的输出 结果 中解析出json而做的。
        if not contents:
            for item in sublist:
                if 'start_time' in item:
                    new_item = item.copy()
                    new_item[f'{model_name}_predict'] = '{}'
                    new_sublist.append(new_item)
            result_list.append(new_sublist)
            continue

        # 匹配并将模型结果加入输出
        for item in sublist:
            if 'start_time' in item:
                new_item = item.copy()
                match_found = False
                for content in contents["result"]:
                    if content['content'] != "":
                        # 去除掉标点符号后再匹配
                        if remove_punctuation(content['content']) in remove_punctuation(item['text']):
                            logger.debug("找到匹配内容")
                            new_item[f'{model_name}_predict'] = test_result
                            match_found = True
                            break
                if not match_found:
                    new_item[f'{model_name}_predict'] = ''
                new_sublist.append(new_item)
        result_list.append(new_sublist)

    # 将所有子列表平铺成一维列表，并转换为 DataFrame
    flat_list = [item for sublist in result_list for item in sublist]
    df_result = pd.DataFrame(flat_list)
    return df_result


def main(model_parameters, data_processor, data, prompt, output_path):
    """
    主函数，处理数据并进行分析
    :param model_parameters: 包含模型相关参数的字典
    :param data_processor: 数据处理相关参数
    :param data: 输入的数据，包含 start_time, end_time, text 和 label
    :param prompt: 模型的提示语
    :param output_path: 输出文件路径
    """

    # 更新 config 中的 prompt
    model_parameters["prompt"] = prompt

    # 将所有参数打包成一个字典
    config = {
        "model_parameters": model_parameters,
        "data_processor": data_processor,
        "data": data
    }

    logger.info("配置文件: %s", config)

    # 处理和分析数据
    output_result = process_data_and_analyze(config)
    logger.info("处理和分析的结果: %s", output_result)

    # 保存结果为 JSON 文件
    Task = data_processor.get("Task")
    model_name = model_parameters.get("model_name")
    result_filename = f"不同task任务调用闭源模型生成的结果/{Task}_{model_name}_result.json"
    with open(result_filename, "w", encoding="utf-8") as f:
        json.dump(output_result, f, ensure_ascii=False)
    logger.info(f"模型结果已保存至: %s", result_filename)

    # 处理结果并保存为 Excel 文件
    df_result = process_output_result(output_result, model_name)
    df_result.to_excel(output_path, index=False)
    logger.info(f"最终结果已保存至: %s", output_path)


# 示例调用
if __name__ == '__main__':
    # 示例数据
    model_parameters = {
        'model_family': 'gpt4o',
        'api_key': 'b2e709bdd54f4416a734b4a6f8f1c7a0',
        'model_name': 'soikit_test',
        'api_version': '2024-02-01',
        'prompt': ''
    }

    data_processor = {
        'Task': 'teacher_dialogue_classification',
        'T': 500
    }

    data = {
        'start_time': [27300, 35310, 40560, 45590, 47910, 50070, 52780, 53000],
        'end_time': [32940, 39510, 42710, 47190, 49590, 52760, 52790, 69880],
        'text': [
            '具你，为什么要我买？这是第一套。',
            '喂，你，吃你吃你狗，你，',
            '好，把语文书翻到第50页，',
            '然后铅笔收起来把，',
            '课堂练习放到左上角，',
            '先把语文书翻到翻到第50页，翻到这里，',
            '没有，50。我现在这个阳猫世，',
            '我看谁今天坐姿有问题啊啊，'
        ],
        'label': [0, 1, 0, 0, 1, 0, 1, 0]
    }

    # 示例 prompt 和输出路径
    prompt = """后面的“待分析文本”是一段发生在课堂上的师生对话，其中，"老师话语”是老师说的话，“学生话语”是学生说的话。有的”待分析文本“没有采集到“学生话语”。请按照以下方法对“待分析文本”进行分析：
首先，根据”待分析文本“上下文语义的相关性，将“老师话语“分割为“发起”、“评价”、“讲解”和“其它”四种子文本段。“发起”是老师邀请、引导、鼓励学生发言、齐读、回答问题、朗读等用话语来回应的子文本段，而不是老师让学生做动作的子文本段；“评价”是老师对学生回应的直接肯定、直接表扬、直接否定的子文本段；”讲解“是老师描述知识点、重复学生回应内容、总结学生回应的子文本段；不能归属于上面三种子文本段的，归为“其它”子文本段。
然后，评估“学生话语”对应“发起”的符合度，符合度评分为：“高”、“中'、"低"、“无”。“高”表示“学生话语“与”发起”内容高度对应；“中”表示“学生话语“和”发起“内容有相关性但未完全回应所有内容；“低”表示”学生话语”与”发起”的内容基本不相关；“无”表示“老师话语”中没有被归为“发起”的子文本段或“待分析文本”中没有“学生话语”。如果”老师话语“中只有一个“发起”，直接输出符合度；如果“老师话语”中有多个“发起”，输出评分最高的符合度。
参照“示例”的输出格式进行输出。
示例：
“老师话语”：讲话的时候可以加上表情和动作，这样表演的更好，其他同学在这位同学表演的时候都在认真听讲，乌鸦，一处，你这样吧，你上来吧，好不好？哎，你上来讲好不好？

”学生话语“：一只乌鸦哇哇的对，猴子说，猴哥猴哥，你怎么种梨树呢？
输出是一个json格式：

{
"result":
[{"type":"讲解","content": "讲话的时候可以加上表情和动作，这样表演的更好，"},
{"type":"其它","content":"其他同学在这位同学表演的时候都在认真听讲，"},
{"type":"评价","content": "真棒，"},
{"type":"发起","content": "乌鸦，一处，你这样吧，你上来吧，好不好？"},
{"type":"发起" ,"content""哎，你上来讲好不好？"}],
"compliance": "高"
}

待分析文本：
"""
    output_path = "output.xlsx"

    # 调用主函数
    main(model_parameters, data_processor, data, prompt, output_path)
