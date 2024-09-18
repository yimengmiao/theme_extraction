import json
import argparse
import pandas as pd
import re
import logging

from model_api.model_api_handler import ModelAPI
from data_processor.data_process import DataProcessor

# 配置 logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
将数据处理层和模型调用层都结合起来，写成一个函数，函数的入参是读取 config.json 中的值
 :param model_parameters: 包含模型相关参数的字典，包含：
            - model_family: 模型类别
            - api_key: 用于访问模型的 API 密钥
            - base_prompt: 基础提示语
            - model_name: 模型名称
            - api_version: API 版本（可选）
:param dataset: 要处理的 DataFrame 或 JSON 数据
:param data_processor: 包含任务相关的参数字典
            - task: 数据处理任务名称，决定数据处理逻辑
            - T: 时间差阈值（仅针对 label 为 0 的行）
"""


class InputError(Exception):
    """自定义异常，用于处理参数错误"""
    pass


def validate_input(params):
    """参数校验函数"""
    try:
        if not params.get("data_processor"):
            raise InputError("data_processor is a required field.")
        if not params["data_processor"].get("Task"):
            raise InputError("Task in data_processor cannot be empty.")

        if "data" not in params or not params["data"]:
            raise InputError("data is a required field and cannot be empty.")

        if "model_parameters" in params:
            for key, value in params["model_parameters"].items():
                if not value:
                    raise InputError(f"{key} in model_parameters cannot be empty.")
    except InputError as e:
        logger.error("参数校验失败: %s", e)
        return {"error": str(e)}


def process_data_and_analyze(params):
    """处理数据并根据需求调用模型分析"""

    # 校验参数
    error = validate_input(params)
    if error:
        logger.error("参数校验失败: %s", error)
        return error  # 如果有错误，直接返回错误信息

    # 校验 data 是否包含必要的字段，并且每个字段的值是否为列表
    required_keys = ['start_time', 'end_time', 'text', 'label']
    data = params.get('data')

    if not data:
        logger.error("缺少 'data' 字段")
        return {"error": "Missing 'data' field in parameters."}

    for key in required_keys:
        if key not in data:
            logger.error("缺少必要字段 '%s' 在 data 中", key)
            return {"error": f"Missing required key '{key}' in data."}
        if not isinstance(data[key], list):
            logger.error("'%s' 在 data 中必须为列表", key)
            return {"error": f"'{key}' in data must be a list."}

    # 初始化数据处理类
    processor = DataProcessor(
        dataset=data,  # 这里传入的是 data 字典
        task=params['data_processor'].get('Task', "teacher_dialogue_classification"),
        T=params['data_processor'].get('T', 800)
    )

    try:
        # 处理数据并获取输出
        output = processor.process_and_save_sub_dfs()
    except Exception as e:
        logger.error("数据处理错误: %s", e)
        return {"error": f"Data processing error: {str(e)}"}

    # 如果没有传入 model_parameters，直接返回数据处理的结果
    if "model_parameters" not in params:
        logger.info("未传入 model_parameters, 直接返回数据处理结果")
        return output

    # 如果有 model_parameters，则继续调用模型
    for item in output:
        model_name = params['model_parameters'].get("model_name", "glm-4-flash")
        params['model_parameters']['text'] = item[2].get("model_input")
        logger.info("调用模型的入参 params: %s", params)

        model_api = ModelAPI(params.get("model_parameters", ""))
        result = model_api.analyze_text()

        logger.info("模型分析结果: %s", result)

        item.append({f"{model_name}_result": result})
        logger.debug("模型分析后的 item: %s", item)
    return output


def extract_json_using_patterns(text):
    """使用一组正则表达式模式来提取 JSON"""
    text = text.strip()
    logger.debug("原始文本: %s", text)

    patterns = [
        r'\{[\s\S]*\}',  # 新的正则表达式模式，匹配第一个 '{' 和最后一个 '}' 之间的内容
        r'(\{[\s\S]*?\})\s*\}$',
        r'\{\s*"result"\s*:\s*\[[\s\S]*?\]\s*\}',
        r'"""json\s*(\{[\s\S]*?\})\s*"""',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            if match.lastindex:
                json_str = match.group(1)
            else:
                json_str = match.group(0)
            logger.debug("匹配到的 JSON: %s", json_str)
            try:
                result_data = json.loads(json_str)
                return result_data
            except json.JSONDecodeError as e:
                logger.error("JSON 解析失败: %s", e)
                continue

    logger.warning("未找到符合模式的 JSON 数据")
    return {}


def parse_test_result(test_result):
    """解析模型返回的结果，提取 JSON"""
    if test_result:
        try:
            result_data = json.loads(test_result)
            logger.debug("直接解析的 JSON 数据: %s", result_data)
            return result_data
        except json.JSONDecodeError:
            return extract_json_using_patterns(test_result)
    else:
        logger.warning("test_result 为空")
        return {}


def process_output_result(output_result, model_name):
    """处理模型的输出结果，生成最终的 DataFrame"""
    result_list = []

    for sublist in output_result:
        test_result = None
        if len(sublist) >= 4:
            third_item = sublist[3]
            test_result = next(iter(third_item.values()))
        else:
            test_result = None
        logger.debug("提取到的 test_result: %s", test_result)

        contents = parse_test_result(test_result)
        logger.debug("解析后的内容: %s", contents)
        logger.debug("子列表: %s", sublist)

        new_sublist = []
        if not contents:
            for item in sublist:
                if 'start_time' in item:
                    new_item = item.copy()
                    new_item[f'{model_name}_predict'] = '{}'
                    new_sublist.append(new_item)
            result_list.append(new_sublist)
            continue

        for item in sublist:
            if 'start_time' in item:
                new_item = item.copy()

                match_found = False
                for content in contents["result"]:
                    if content['content'] != "":
                        if content['content'] in item['text']:
                            logger.debug("找到匹配内容")
                            new_item[f'{model_name}_predict'] = test_result
                            match_found = True
                            break
                if not match_found:
                    new_item[f'{model_name}_predict'] = ''

                new_sublist.append(new_item)
        result_list.append(new_sublist)

    flat_list = [item for sublist in result_list for item in sublist]
    df_result = pd.DataFrame(flat_list)
    return df_result


def main():
    """主函数"""

    parser = argparse.ArgumentParser(description='Process data and analyze with model.')
    parser.add_argument('--data', required=True, help='Path to data file (xlsx format).')
    parser.add_argument('--config', required=True, help='Path to config.json.')
    parser.add_argument('--prompt', required=True, help='Path to prompt file (txt).')
    parser.add_argument('--output', required=True, help='Path to output file (xlsx format).')

    args = parser.parse_args()

    with open(args.prompt, "r", encoding="utf-8") as f:
        prompt = f.read()

    with open(args.config, "r", encoding="utf-8") as f1:
        config = json.load(f1)

    df = pd.read_excel(args.data)

    data_json = {
        "start_time": df['start_time'].to_list(),
        "end_time": df['end_time'].to_list(),
        "text": df['text'].to_list(),
        'label': df['label'].to_list()
    }

    config['data'] = data_json
    config["model_parameters"]["prompt"] = prompt

    Task = config.get("data_processor").get("Task")
    model_name = config.get("model_parameters").get("model_name")

    output_result = process_data_and_analyze(config)
    logger.info("处理和分析的结果: %s", output_result)

    result_filename = f"不同task任务调用闭源模型生成的结果/{Task}_{model_name}_result.json"
    with open(result_filename, "w", encoding="utf-8") as f:
        json.dump(output_result, f, ensure_ascii=False)
    logger.info(f"模型结果已保存至: %s", result_filename)

    df_result = process_output_result(output_result, model_name)
    df_result.to_excel(args.output, index=False)
    logger.info(f"最终结果已保存至: %s", args.output)


if __name__ == '__main__':
    main()

# 代码使用：python teacher_classification.py --data data/original_data/test.xlsx --config config/config.json --prompt prompt/teacher_dialouge_classification_prompt.txt --output output.xlsx
