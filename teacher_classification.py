import json
import argparse
import pandas as pd
import re

from model_api.model_api_handler import ModelAPI
from data_processor.data_process import DataProcessor

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
        # model_family = params['model_parameters'].get("model_family", "glm-4")
        # api_key = params['model_parameters'].get("api_key", "08bd304ed5c588b2c9cb534405241f0e.jPN6gjmvlBe2q1ZZ")
        model_name = params['model_parameters'].get("model_name", "glm-4-flash")
        # api_version = params['model_parameters'].get("api_version", None)
        params['model_parameters']['text'] = item[2].get("model_input")
        print("入参params", params)

        model_api = ModelAPI(params.get("model_parameters", ""))
        result = model_api.analyze_text()
        # model = ModelAPI(model_family=model_family, api_key=api_key, api_version=api_version)
        # prompt = params['model_parameters'].get("prompt", "Analyze the following conversation:")

        # result = model.analyze_text(text=item[2].get("model_input"), prompt=prompt, model=model_name)
        # 将模型结果添加到师生对话段（item）中

        print("result", result)

        item.append({f"{model_name}_result": result})
        print("item", item)
    return output


def extract_json_using_patterns(text):
    """使用一组正则表达式模式来提取 JSON"""
    # 保留原始文本的换行符
    text = text.strip()
    print("text:", text)

    patterns = [
        r'\{[\s\S]*\}',  # 新的正则表达式模式，匹配第一个 '{' 和最后一个 '}' 之间的内容
        # 保留其他原有模式
        r'(\{[\s\S]*?\})\s*\}$',
        r'\{\s*"result"\s*:\s*\[[\s\S]*?\]\s*\}',
        r'"""json\s*(\{[\s\S]*?\})\s*"""',

    ]

    # 依次尝试使用每个正则表达式模式进行匹配
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            # 检查正则表达式是否使用了捕获组
            if match.lastindex:
                json_str = match.group(1)  # 获取捕获组中的内容
            else:
                json_str = match.group(0)  # 获取整个匹配的内容
            print(f"匹配到的 JSON: {json_str}")
            try:
                # 尝试将提取到的字符串解析为 JSON
                result_data = json.loads(json_str)
                return result_data
            except json.JSONDecodeError as e:
                # 如果 JSON 解析失败，继续尝试下一个模式
                print(f"JSON 解析失败: {e}")
                continue

    # 如果没有找到匹配的 JSON 数据
    print("未找到符合模式的 JSON 数据")
    return {}


def parse_test_result(test_result):
    """解析模型返回的结果，提取 JSON"""
    if test_result:
        try:
            # 尝试直接解析 JSON
            result_data = json.loads(test_result)
            print("第一轮json解析", result_data)
            return result_data
        except json.JSONDecodeError:
            # 如果解析失败，尝试使用正则表达式提取 JSON
            return extract_json_using_patterns(test_result)
    else:
        return {}


def process_output_result(output_result, model_name):
    """处理模型的输出结果，生成最终的 DataFrame"""
    result_list = []

    for sublist in output_result:
        # 提取模型的预测结果
        test_result = None
        if len(sublist) >= 4:
            third_item = sublist[3]
            # 获取第三个字典中第一个键对应的值
            test_result = next(iter(third_item.values()))
        else:
            test_result = None
        print("test_result", test_result)
        # 解析 test_result，提取内容
        contents = parse_test_result(test_result)
        print("contents", contents)
        # 如果 contents 为 {}，就跳过匹配逻辑，并将对应模型的 predict 设为 '{}'
        print("sublist", sublist)
        new_sublist = []
        if not contents:
            # contents 为空，直接设置 predict 为 '{}'
            for item in sublist:
                if 'start_time' in item:
                    new_item = item.copy()
                    new_item[f'{model_name}_predict'] = '{}'
                    new_sublist.append(new_item)
            # 添加到结果列表
            result_list.append(new_sublist)
            continue  # 跳过后续逻辑，处理下一个 sublist

        # 构建新的子列表，添加模型预测结果
        for item in sublist:
            if 'start_time' in item:
                new_item = item.copy()  # 复制以避免修改原始列表

                match_found = False
                for content in contents["result"]:
                    if content['content'] != "":
                        if content['content'] in item['text']:
                            print(1)
                            new_item[f'{model_name}_predict'] = test_result
                            match_found = True
                            break
                if not match_found:
                    new_item[f'{model_name}_predict'] = ''

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
    # 调用处理函数,数据处理和模型调用
    output_result = process_data_and_analyze(config)
    print("output_result", output_result)
    # 保存模型结果
    result_filename = f"不同task任务调用闭源模型生成的结果/{Task}_{model_name}_result.json"
    with open(result_filename, "w", encoding="utf-8") as f:
        json.dump(output_result, f, ensure_ascii=False)
    print(f"Model results saved to {result_filename}")

    # 处理输出结果，生成最终的 DataFrame
    df_result = process_output_result(output_result, model_name)
    # 保存到指定的输出文件
    df_result.to_excel(args.output, index=False)
    print(f"Final results saved to {args.output}")


if __name__ == '__main__':
    main()

# 代码使用：python teacher_classification.py --data data/original_data/test.xlsx --config config/config.json --prompt prompt/teacher_dialouge_classification_prompt.txt --output output.xlsx
