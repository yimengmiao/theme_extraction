import json
from model_api.model_api_handler import ModelAPI
from data_processor.data_process import DataProcessor

"""
将数据处理层和模型调用层都结合起来，写成一个函数 ，函数 的入参是读取config.json中的值
 :param model_parameters: 包含模型相关参数的字典，包含：
            - model_family: 模型类别
            - api_key: 用于访问模型的API密钥
            - base_prompt: 基础提示语
            - model_name: 模型名称
            - api_version: API版本（可选）
:param  dataset: 要处理的DataFrame或JSON数据
:param  data_processor: 包含任务相关的参数字典
            - task: 数据处理任务名称，决定数据处理逻辑
            - T: 时间差阈值（仅针对label为0的行）
"""


class InputError(Exception):
    """自定义异常，用于处理参数错误"""
    pass


# 参数校验函数
def validate_input(params):
    """校验输入参数"""
    try:
        if not params.get("data_processor"):
            raise InputError("data_processor is a required field.")
        if not params["data_processor"].get("Task"):
            raise InputError("Task in data_processor cannot be empty.")

        if "data" not in params or not params["data"]:
            raise InputError("data is a required field and cannot be empty.")

        if "model_paramerters" in params:
            for key, value in params["model_paramerters"].items():
                if not value:
                    raise InputError(f"{key} in model_paramerters cannot be empty.")
    except InputError as e:
        return {"error": str(e)}


# 数据处理和模型调用函数
def process_data_and_analyze(params):
    """处理数据并根据需求调用模型分析"""

    # 校验参数
    error = validate_input(params)
    if error:
        return error  # 如果有错误，直接返回错误信息

    # 校验data是否包含必要的字段，并且每个字段的值是否为列表
    required_keys = ['start_time', 'end_time', 'text', 'label']
    data = params.get('data')

    if not data:
        return {"error": "Missing 'data' field in parameters."}

    for key in required_keys:
        if key not in data:
            return {"error": f"Missing required key '{key}' in data."}
        if not isinstance(data[key], list):
            return {"error": f"'{key}' in data must be a list."}

    # 提取数据
    start_times = data['start_time']
    end_times = data['end_time']
    texts = data['text']
    labels = data['label']

    # 初始化数据处理类
    processor = DataProcessor(
        dataset=data,  # 这里传入的是data字典
        task=params['data_processor'].get('Task', "teacher_dialogue_classification"),
        T=params['data_processor'].get('T', 800)
    )

    try:
        # 处理数据并获取输出
        output = processor.process_and_save_sub_dfs()
    except Exception as e:
        return {"error": f"Data processing error: {str(e)}"}

    # 如果没有传入 model_paramerters，直接返回数据处理的结果
    if "model_parameters" not in params:
        return output

    # 如果有 model_paramerters，则继续调用模型
    for item in output:
        teacher_text = ""
        student_text = ""

        # 处理item中的每个对话
        for sub_item in item:
            if sub_item['label'] == 0:
                teacher_text += sub_item['text'] + " "
            elif sub_item['label'] == 1:
                student_text += sub_item['text'] + " "

        # 构建分析师生对话段文本
        text_to_analyze = f"""
        老师话语：{teacher_text.strip()}
        学生话语：{student_text.strip()}
        """
        print("text_to_analyze", text_to_analyze)

        # 调用模型进行分析师生对话段
        model_family = params['model_parameters'].get("model_family", "glm-4")
        api_key = params['model_parameters'].get("api_key", "default_key")
        model_name = params['model_parameters'].get("model_name", "glm-4-flash")
        api_version = params['model_parameters'].get("api_version", None)
        model = ModelAPI(model_family=model_family, api_key=api_key, api_version=api_version)
        prompt = params['model_parameters'].get("prompt", "Analyze the following conversation:")

        result = model.analyze_text(text=text_to_analyze, prompt=prompt, model=model_name)
        # 将模型结果添加到师生对话段（item）中

        print("result", result)

        item.append({f"{model_name}_result": result})

    return output


if __name__ == '__main__':
    # 示例调用
    params = {
        "model_parameters": {
            "model_family": "gpt4o",
            "api_key": "b2e709bdd54f4416a734b4a6f8f1c7a0",
            "prompt": """后面的“待分析文本”是一段发生在课堂上的师生对话，其中，"老师话语”是老师说的话，“学生话语”是学生说的话。有的”待分析文本“没有采集到“学生话语”。请按照以下方法对“待分析文本”进行分析：
首先，根据”待分析文本“上下文语义的相关性，将“老师话语“分割为“发起”、“评价”、“讲解”和“其它”四种子文本段。“发起”是老师邀请、引导、鼓励学生发言、齐读、回答问题、朗读等用话语来回应的子文本段，而不是老师让学生做动作的子文本段；“评价”是老师对学生回应的直接肯定、直接表扬、直接否定的子文本段；”讲解“是老师描述知识点、重复学生回应内容、总结学生回应的子文本段；不能归属于上面三种子文本段的，归为“其它”子文本段。
然后，评估“学生话语”对应“发起”的符合度，符合度评分为：“高”、“中'、"低"、“无”。“高”表示“学生话语“与”发起”内容高度对应”；“中”表示“学生话语“和”发起“内容有相关性但未完全回应所有内容；“低”表示”学生话语”与”发起”的内容基本不相关；“无”表示”老师话语“中没有被归为”发起“的子文本段或“待分析文本”中没有“学生话语”。如果”老师话语“中只有一个”发起“，直接输出符合度；如果”老师话语“中有多个”发起“，输出评分最高的符合度。
参照“示例”的输出格式进行输出。
示例：
“老师话语”：讲话的时候可以加上表情和动作，这样表演的更好，其他同学在这位同学表演的时候都在认真听讲，乌鸦，一处，你这样吧，你上来吧，好不好？哎，你上来讲好不好？
”学生话语“：一只乌鸦哇哇的对，猴子说，猴哥猴哥，你怎么种梨树呢？
输出：
{
"result":
[{"type":"讲解","content": "讲话的时候可以加上表情和动作，这样表演的更好，"},
{"type":"其它","content":"其他同学在这位同学表演的时候都在认真听讲，"},
{"type":"评价","content": "真棒，"},
{"type":"发起","content": "乌鸦，一处，你这样吧，你上来吧，好不好？"},
{"type":"发起" ,"content""哎，你上来讲好不好？"}],
"compliance": "高"}

待分析文本：

    """,
            "model_name": "soikit_test",
            "api_version": "2024-02-01",
            "model_type": "closed"
        },

        "data": {
            "start_time": [27300, 35310, 40560, 45590, 47910, 50070, 52780, 53000],
            "end_time": [32940, 39510, 42710, 47190, 49590, 52760, 52790, 69880],
            "text": [
                "具你，“为什么要我买？”这是第一套。",
                "喂，你，吃你吃你狗，你，",
                "好，把语文书翻到第50页，",
                "然后铅笔收起来把，",
                "课堂练习放到左上角，",
                "先把语文书翻到翻到第50页，翻到这里，",
                "没有，50。我现在这个阳猫世，",
                "我看谁今天的坐姿有问题啊啊，"
            ],
            "label": [1, 1, 1, 0, 1, 0, 1, 0]
        },
        "data_processor": {
            "Task": "teacher_dialogue_classification",
            "T": 800
        }
    }

    # 调用函数
    output_result = process_data_and_analyze(params)
    print(output_result)
