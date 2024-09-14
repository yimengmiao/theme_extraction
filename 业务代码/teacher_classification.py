import json
import argparse
import pandas as pd
import re

from ..model_api.model_api_handler import ModelAPI
from ..data_processor.data_process import DataProcessor

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
        return {"error": str(e)}


def process_data_and_analyze(params):
    """处理数据并根据需求调用模型分析"""

    # 校验参数
    error = validate_input(params)
    if error:
        return error  # 如果有错误，直接返回错误信息

    # 校验 data 是否包含必要的字段，并且每个字段的值是否为列表
    required_keys = ['start_time', 'end_time', 'text', 'label']
    data = params.get('data')

    if not data:
        return {"error": "Missing 'data' field in parameters."}

    for key in required_keys:
        if key not in data:
            return {"error": f"Missing required key '{key}' in data."}
        if not isinstance(data[key], list):
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
        return {"error": f"Data processing error: {str(e)}"}

    # 如果没有传入 model_parameters，直接返回数据处理的结果
    if "model_parameters" not in params:
        return output

    # 如果有 model_parameters，则继续调用模型
    for item in output:
        # 调用模型进行分析师生对话段
        model_family = params['model_parameters'].get("model_family", "glm-4")
        api_key = params['model_parameters'].get("api_key", "default_key")
        model_name = params['model_parameters'].get("model_name", "glm-4-flash")
        api_version = params['model_parameters'].get("api_version", None)
        model = ModelAPI(model_family=model_family, api_key=api_key, api_version=api_version)
        prompt = params['model_parameters'].get("prompt", "Analyze the following conversation:")

        result = model.analyze_text(text=item, prompt=prompt, model=model_name)
        # 将模型结果添加到师生对话段（item）中

        print("result", result)

        item.append({f"{model_name}_result": result})

    return output


def parse_test_result(test_result):
    """解析模型返回的结果，提取内容"""
    if test_result:
        # 判断 test_result 是否为有效的 JSON，如果不是，尝试提取其中的 JSON 部分
        try:
            result_data = json.loads(test_result)
        except json.JSONDecodeError:
            # 提取第一个大括号后的内容到最后一个大括号
            match = re.search(r'\{.*\}', test_result, re.DOTALL)
            if match:
                json_str = match.group()
                try:
                    result_data = json.loads(json_str)
                except json.JSONDecodeError:
                    result_data = {}
            else:
                result_data = {}
        contents = [result['content'] for result in result_data.get('result', [])]
        print(contents)
        return contents
    else:
        return []


def process_output_result(output_result):
    """处理模型的输出结果，生成最终的 DataFrame"""
    result_list = []

    for sublist in output_result:
        # 提取模型的预测结果
        test_result = None
        if len(sublist) >= 3:
            third_item = sublist[2]
            # 获取第三个字典中第一个键对应的值
            test_result = next(iter(third_item.values()))
        else:
            test_result = None

        # 解析 test_result，提取内容
        contents = parse_test_result(test_result)

        # 构建新的子列表，添加 'gpt4o_predict' 键
        new_sublist = []
        for item in sublist:
            if 'start_time' in item:
                new_item = item.copy()  # 复制以避免修改原始列表
                if any(content in item['text'] for content in contents):
                    new_item['gpt4o_predict'] = test_result
                else:
                    new_item['gpt4o_predict'] = ''
                new_sublist.append(new_item)
        # 添加到结果列表
        result_list.append(new_sublist)

    # 展平列表并转换为 DataFrame
    flat_list = [item for sublist in result_list for item in sublist]
    df_result = pd.DataFrame(flat_list)
    return df_result


def main():
    """主函数"""

    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(description='Process data and analyze with model.')
    parser.add_argument('--data', required=True, help='Path to data file (xlsx format).')
    parser.add_argument('--config', required=True, help='Path to config.json.')
    parser.add_argument('--prompt', required=True, help='Path to prompt file (txt).')
    parser.add_argument('--output', required=True, help='Path to output file (xlsx format).')

    args = parser.parse_args()

    # 读取 prompt
    with open(args.prompt, "r", encoding="utf-8") as f:
        prompt = f.read()
    # 读取配置
    with open(args.config, "r", encoding="utf-8") as f1:
        config = json.load(f1)
    # 读取数据文件
    df = pd.read_excel(args.data)
    # 构造数据输入
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
    # 调用处理函数
    output_result = process_data_and_analyze(config)
    # 保存模型结果
    result_filename = f"不同task任务调用闭源模型生成的结果/{Task}_{model_name}_result.json"
    with open(result_filename, "w", encoding="utf-8") as f:
        json.dump(output_result, f, ensure_ascii=False)
    print(f"Model results saved to {result_filename}")

    # 处理输出结果，生成最终的 DataFrame
    df_result = process_output_result(output_result)
    # 保存到指定的输出文件
    df_result.to_excel(args.output, index=False)
    print(f"Final results saved to {args.output}")


if __name__ == '__main__':
    main()

# 代码使用：python teacher_classification.py --data path/to/data.xlsx --config path/to/config.json --prompt path/to/prompt.txt --output path/to/output.xlsx
