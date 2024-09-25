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


def process_data_and_analyze(params):
    # 初始化数据处理类
    processor = DataProcessor(
        dataset=data,  # 这里传入的是 data 字典
        task=params['data_processor'].get('Task', "teacher_dialogue_classification"),
        T=params['data_processor'].get('T', 800)
    )

    # try:
        # 处理数据并获取输出
    output = processor.process_and_save_sub_dfs()
    # except Exception as e:
    #     logger.error("数据处理错误: %s", e)
    #     return {"error": f"Data processing error: {str(e)}"}

    # 如果没有传入 model_parameters，直接返回数据处理的结果
    if "model_parameters" not in params:
        logger.info("未传入 model_parameters, 直接返回数据处理结果")
        return output
    print("output", output)
    # 如果有 model_parameters，则继续调用模型
    for item in output:
        model_name = params['model_parameters'].get("model_name", "glm-4-flash")
        params['model_parameters']['text'] = item
        logger.info("调用模型的入参 params: %s", params)

        model_api = ModelAPI(params.get("model_parameters", ""))
        result = model_api.analyze_text()

        logger.info("模型分析结果: %s", result)

        item.append({f"{model_name}_result": result})
        logger.debug("模型分析后的 item: %s", item)
    return output


if __name__ == '__main__':
    data = [
        # 第一条数据
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
        # 第二条数据
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
        # E_after 数据
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

    # 调用模型
    prompt = "prompt/讲解片段判定_prompt.txt"
    with open(prompt, "r", encoding="utf-8") as f:
        prompt = f.read()
    config = "config/config.json"
    with open(config, "r", encoding="utf-8") as f1:
        config = json.load(f1)

    config['data'] = data
    config["model_parameters"]["prompt"] = prompt

    Task = config.get("data_processor").get("Task")
    model_name = config.get("model_parameters").get("model_name")

    print("config", config)
    output_result = process_data_and_analyze(config)
    print("output_result",output_result)