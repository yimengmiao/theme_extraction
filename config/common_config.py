# common_config.py

# 公共的模型参数配置
DEFAULT_MODEL_PARAMETERS = {
    'model_family': 'qwen',
    'api_key': 'sk-f582e4fab0894a52b12b7a85c62868bc',
    'model_name': 'qwen2.5-32b-instruct',
    'api_version': '2024-02-01',
}
# 备用的模型参数配置
ALTERNATE_MODEL_PARAMETERS = {
    'model_family': 'gpt4o',
    'api_key': 'b2e709bdd54f4416a734b4a6f8f1c7a0',
    'model_name': 'soikit_test',
    'api_version': '2024-02-01',
}

# 公共的数据处理器配置
DEFAULT_DATA_PROCESSOR = {
    'Task': 'teacher_dialogue_classification',
    'T': 800
}
