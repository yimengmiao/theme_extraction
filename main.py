"""
将数据处理层和模型调用层都结合起来，写成一个函数 ，函数 的入参是读取config.json中的值
 :param model_parameters: 包含模型相关参数的字典，包含：
            - model_family: 模型类别
            - api_key: 用于访问模型的API密钥
            - base_prompt: 基础提示语
            - model_name: 模型名称
            - api_version: API版本（可选）
        :param dataset: 要处理的DataFrame或JSON数据
        :param Task: 包含任务相关的参数字典
            - sub_task: 子任务名称，决定数据处理逻辑
            - T: 时间差阈值（仅针对label为0的行）
"""

