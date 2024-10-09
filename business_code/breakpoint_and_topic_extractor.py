import json
import logging
import traceback

from data_processor.public_code_data_process import extract_json_using_patterns
from model_api.model_api_handler import ModelAPI
from data_processor.data_process import DataProcessor
from config.common_config import ALTERNATE_MODEL_PARAMETERS

# 配置 logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='log/主题提取.log',  # 指定日志文件路径
    filemode='a'  # 文件模式：'a' 追加模式，'w' 写入模式会覆盖之前的日志
)
logger = logging.getLogger(__name__)


def merge_texts_into_dict(list1, dict1):
    """
    根据 任务3：主题提取输出中的 d_number，与任务3的输入中对应的文本合并，并添加到 任务3 输出 中 的每个子主题的 sub_text 键中。

    参数：
        prompt3_input(list): 包含编号和文本内容的列表。
        prompt3_output(dict): 包含主题和子主题信息的字典。

    返回：
        dict: 更新后的 prompt3_output，添加了每个子主题的 sub_text。
    """
    # 创建一个字典，将 d_number 映射到对应的文本内容
    d_number_to_text = {}

    for item in list1:
        # 分割编号和文本内容
        parts = item.strip().split('\n', 1)
        if len(parts) == 2:
            try:
                number = int(parts[0])
                text = parts[1].strip()
                d_number_to_text[number] = text
            except ValueError:
                continue  # 如果编号不是整数，跳过该项

    # 根据 dict1 中的 d_number，将对应的文本合并并添加到 sub_text 键中
    for sub in dict1['split'].values():
        d_numbers = sub.get('d_number', [])
        if not isinstance(d_numbers, list):
            d_numbers = json.loads(d_numbers)
        merged_texts = []
        for num in d_numbers:
            text = d_number_to_text.get(num)
            if text:
                merged_texts.append(text)
        sub['sub_text'] = '\n\n'.join(merged_texts)

    return dict1


def topic_extract(params):
    """主题提取代码"""
    # 初始化数据处理类
    processor = DataProcessor(
        dataset=params.get("data"),
        task=params['data_processor'].get('Task', "teacher_dialogue_classification"),
        T=params['data_processor'].get('T', 800)
    )

    try:
        # 处理数据并获取输出
        output = processor.process_and_save_sub_dfs()
    except Exception as e:
        logger.error("数据处理错误: %s", e)
        logger.error("详细堆栈信息：\n%s", traceback.format_exc())

        return {"error": f"Data processing error: {str(e)}"}

    # 如果没有传入 model_parameters，直接返回数据处理的结果
    if "model_parameters" not in params:
        logger.info("未传入 model_parameters, 直接返回数据处理结果")
        return output

    # 如果有 model_parameters，则继续调用模型
    model_name = params['model_parameters'].get("model_name", "glm-4-flash")
    prompt2_input = "\n".join(output)  # 这就是Prompt2的输入
    logger.info("prompt2_input: %s", prompt2_input)
    params['model_parameters']['text'] = prompt2_input
    logger.info("prompt2调用模型的入参 params: %s", params)

    # 调用模型 API 进行分析
    try:
        model_api = ModelAPI(params.get("model_parameters", ""))
        breakpoint_json = model_api.analyze_text()
        logger.info(f"{model_name}模型分析的Prompt2的输出结果: %s", breakpoint_json)
    except Exception as e:
        logger.info("尝试调用付费模型 gpt4o")
        try:
            prompt2 = params['model_parameters'].get('prompt', "")
            params["model_parameters"] = ALTERNATE_MODEL_PARAMETERS
            params["model_parameters"]["prompt"] = prompt2
            params['model_parameters']['text'] = prompt2_input
            logger.info("调用付费模型gpt4o的入参 params: %s", params)
            model_api = ModelAPI(params.get("model_parameters", ""))
            breakpoint_json = model_api.analyze_text()
            logger.info("备用模型gpt4o 分析的Prompt2的输出结果: %s", breakpoint_json)
        except Exception as fallback_e:
            # 如果这个模型调用也失败，则返回错误信息
            logger.error("备用模型gpt4o 调用失败: %s", fallback_e)
            logger.error("详细堆栈信息：\n%s", traceback.format_exc())
            return {"error": f"Fallback model API error: {str(fallback_e)}"}

    # 处理模型返回的结果
    splitpoint = extract_json_using_patterns(breakpoint_json)
    if not splitpoint:
        logger.error("无法从模型返回的结果中提取分割点")
        return {"error": "Failed to extract split points from model output."}

    # 继续处理
    processor2 = DataProcessor(
        prompt2_input=prompt2_input,
        task="topic_extraction",  # 选择新的任务类型
        splitpoint=splitpoint
    )
    prompt2_output = processor2.process_and_save_sub_dfs()  # 这里输出的就是Prompt3的输入了

    # 读取Prompt3的内容
    with open("prompt/prompt3主题提取.txt", "r", encoding="utf-8") as f:
        prompt3 = f.read()
    params['model_parameters']['text'] = prompt2_output
    params['model_parameters']['prompt'] = prompt3

    logger.info("prompt3调用模型的入参 params: %s", params)

    # 调用模型 API 进行主题提取
    try:
        model_api = ModelAPI(params.get("model_parameters", ""))
        topic_content_raw = model_api.analyze_text()
        logger.info(f"{model_name}模型分析的Prompt3的输出结果: %s", topic_content_raw)
    except Exception as e:
        logger.error("模型 API 调用错误: %s", e)
        logger.error("详细堆栈信息：\n%s", traceback.format_exc())
        return {"error": f"Model API error during prompt3 analysis: {str(e)}"}

    # 解析模型返回的主题内容
    topic_content = extract_json_using_patterns(topic_content_raw)
    if not topic_content:
        logger.error("无法从模型返回的结果中提取主题内容")
        return {"error": "Failed to extract topic content from model output."}

    # 把相同主题的子师生对话段合并起来
    prompt3_input = prompt2_output.split('师生对话')[1:]
    logger.info("prompt3_input: %s", prompt3_input)
    logger.info("topic_content: %s", topic_content)
    updated_topic_content = merge_texts_into_dict(prompt3_input, topic_content)

    return updated_topic_content


if __name__ == '__main__':
    # 经过老师四分类任务后输出的结果。
    # data = [
    #     # 第一条数据
    #     {
    #         "start_time": 0,
    #         "end_time": 10,
    #         "text": "老师发起提问1，老师讲解内容1，老师发起提问2",
    #         "label": 0,
    #         "gpt4o_result": {
    #             "result": [
    #                 {"type": "发起", "content": "老师发起提问1"},
    #                 {"type": "讲解", "content": "老师讲解内容1"},
    #                 {"type": "发起", "content": "老师发起提问2"},
    #             ],
    #             "compliance": "高"
    #         }
    #     },
    #     {"start_time": 10, "end_time": 20, "text": "学生回答1", "label": 1, "gpt4o_result": None},
    #     # 第二条数据
    #     {
    #         "start_time": 20,
    #         "end_time": 30,
    #         "text": "老师讲解内容2，老师讲解内容3，老师发起提问3，老师发起提问4。",
    #         "label": 0,
    #         "gpt4o_result": {
    #             "result": [
    #                 {"type": "讲解", "content": "老师讲解内容2"},
    #                 {"type": "讲解", "content": "老师讲解内容3"},
    #                 {"type": "发起", "content": "老师发起提问3"},
    #                 {"type": "发起", "content": "老师发起提问4"}
    #             ],
    #             "compliance": "高"
    #         }
    #     },
    #     {"start_time": 30, "end_time": 40, "text": "学生回答2", "label": 1, "gpt4o_result": None},
    #     # E_after 数据
    #     {
    #         "start_time": 40,
    #         "end_time": 50,
    #         "text": "老师讲解内容4",
    #         "label": 0,
    #         "gpt4o_result": {
    #             "result": [
    #                 {"type": "讲解", "content": "老师讲解内容4"}
    #             ],
    #             "compliance": "高"
    #         }
    #     },
    #     {"start_time": 50, "end_time": 60, "text": "学生回应3", "label": 1, "gpt4o_result": None},
    #     {
    #         "start_time": 60,
    #         "end_time": 70,
    #         "text": "老师讲解内容5",
    #         "label": 0,
    #         "gpt4o_result": {
    #             "result": [
    #                 {"type": "讲解", "content": "老师讲解内容5"}
    #             ],
    #             "compliance": "高"
    #         }
    #     },
    #     {"start_time": 70, "end_time": 80, "text": "学生回应4", "label": 1, "gpt4o_result": None}
    # ]
    #
    # # 调用模型
    # prompt = "prompt/讲解片段判定_prompt.txt"
    # with open(prompt, "r", encoding="utf-8") as f:
    #     prompt = f.read()
    # config = "config/config.json"
    # with open(config, "r", encoding="utf-8") as f1:
    #     config = json.load(f1)
    #
    # config['data'] = data
    # config["model_parameters"]["prompt"] = prompt
    #
    # Task = config.get("data_processor").get("Task")
    # model_name = config.get("model_parameters").get("model_name")
    #
    # print("config", config)
    # output_result = process_data_and_analyze(config)
    # print("output_result",output_result)
    # 模拟从外部传入的 JSON 对象

    input_json = {
        'model_parameters': {
            'model_family': 'gpt4o',
            'api_key': 'b2e709bdd54f4416a734b4a6f8f1c7a0',
            'model_name': 'soikit_test',
            'api_version': '2024-02-01',
            'prompt': """下面的“待分析文本”是一段发生在课堂上的师生对话片段。在这段对话中，“发起”是老师引导学生用话语回应的语句；“回应”是学生对老师“发起”的回应语句；“讲解”是老师对学生回应的总结，或为下一次“发起”提供背景和引导的语句。请你按照以下方法对“待分析文本”进行分析：
根据上下文语义的相关性，请你判断“待分析文本”中相邻“发起”中间的”讲解“的归属， 如果“讲解”是对上一个学生“回应”的总结，不进行任何输出。 如果“讲解”是为下一个“发起”提供背景和引导，特别是“讲解”中提到的内容与下一个“发起”相关，可以帮助更好的理解下一个“发起”，则输出这句讲解，格式为json格式：{"breakpoint":["讲解:文本内容1"]}。如果“待分析文本”中有多个“讲解”为下一个“发起”提供背景和引导，请输出{"breakpoint":["讲解:文本内容1","讲解:文本内容2"]}。
待分析文本："""
        },
        'data_processor': {
            'Task': 'dialogue_processing',
            'T': 500
        },
        'data': [
            {
                "start_time": 0,
                "end_time": 10,
                "text": "老师发起提问1，老师讲解内容1，老师发起提问2",
                "label": 0,
                "gpt4o_result": {
                    "result": [
                        {"type": "发起", "content": "老师发起提问1"},
                        {"type": "讲解", "content": "老师讲解内容1"},
                        {"type": "发起", "content": "老师发起提问2"},
                    ],
                    "compliance": "高"
                }
            },
            {"start_time": 10, "end_time": 20, "text": "学生回答1", "label": 1, "gpt4o_result": None},
            {
                "start_time": 20,
                "end_time": 30,
                "text": "老师讲解内容2，老师讲解内容3，老师发起提问3，老师发起提问4。",
                "label": 0,
                "gpt4o_result": {
                    "result": [
                        {"type": "讲解", "content": "老师讲解内容2"},
                        {"type": "讲解", "content": "老师讲解内容3"},
                        {"type": "发起", "content": "老师发起提问3"},
                        {"type": "发起", "content": "老师发起提问4"}
                    ],
                    "compliance": "高"
                }
            },
            {"start_time": 30, "end_time": 40, "text": "学生回答2", "label": 1, "gpt4o_result": None},
            {
                "start_time": 40,
                "end_time": 50,
                "text": "老师讲解内容4",
                "label": 0,
                "gpt4o_result": {
                    "result": [
                        {"type": "讲解", "content": "老师讲解内容4"}
                    ],
                    "compliance": "高"
                }
            },
            {"start_time": 50, "end_time": 60, "text": "学生回应3", "label": 1, "gpt4o_result": None},
            {
                "start_time": 60,
                "end_time": 70,
                "text": "老师讲解内容5",
                "label": 0,
                "gpt4o_result": {
                    "result": [
                        {"type": "讲解", "content": "老师讲解内容5"}
                    ],
                    "compliance": "高"
                }
            },
            {"start_time": 70, "end_time": 80, "text": "学生回应4", "label": 1, "gpt4o_result": None}
        ]
    }

    # 调用函数处理数据并分析
    output_result = topic_extract(input_json)

    print("最终输出结果:", output_result)
