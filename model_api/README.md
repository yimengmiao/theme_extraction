
# ModelAPI: Unified API Handler for Multiple Models

## Introduction
The `ModelAPI` class is designed to provide a unified interface to interact with various language models like GLM-4, GPT4o, and Qwen-Long. It simplifies the process of connecting to different models and makes the code more extensible by handling different models and their respective API calls in a structured manner.

### Key Features:
- **Support for Multiple Models**: Currently supports GLM-4, GPT4o, and Qwen-Long.
- **Extensible**: Easily extendable to support more models by adding new entries in the base URL and client handling logic.
- **Dynamic Prompt Handling**: The user can provide a dynamic `base_prompt` and `text` to be analyzed by the selected model.

## Installation

No special installation is required. Ensure you have installed the `openai` library in your environment:

```bash
pip install openai
```

## Usage

### Initialization
To initialize the `ModelAPI` class, pass in the model name, API key, and optionally a base URL. For models in the `qwen` family, the base URL will automatically default to the same endpoint.

```python
# Initialize the model
api_key = "your_api_key"
model = ModelAPI(model_name="qwen-long", api_key=api_key)
```

### Analyze Text
To analyze a text, you need to provide a base prompt and the text for analysis. The `base_prompt` will be combined with the text to form the full prompt sent to the model.

```python
# Provide the text and base prompt
text_to_analyze = "待分析文本内容"
base_prompt = "后面的“待分析文本”是一段师生对话..."

# Analyze the text
result = model.analyze_text(text=text_to_analyze, base_prompt=base_prompt)
print(result)
```

### Inputs
- `model_name` (str): The name of the model to be used (`"glm-4"`, `"gpt4o"`, `"qwen-long"`, etc.).
- `api_key` (str): The API key to authenticate with the model's API.
- `base_prompt` (str): The base instruction prompt for the model.
- `text` (str): The actual text to be analyzed by the model.

### Outputs
- The model will return the analyzed result based on the provided prompt and text. The output will be in JSON format or plain text, depending on the model's response.

## Models Supported
- **GLM-4**: Uses the `https://open.bigmodel.cn/api/paas/v4/` API endpoint.
- **GPT4o**: Uses the Azure OpenAI endpoint.
- **Qwen-Long**: Automatically uses the `https://dashscope.aliyuncs.com/compatible-mode/v1` endpoint for all `qwen` family models.

## Extending the Class
To extend this class to support new models, you can add entries to the `_get_base_url` and `_get_client` methods with the corresponding API logic.

## Example
```python
api_key = "your_api_key"
text_to_analyze = "待分析文本内容"
base_prompt = "后面的“待分析文本”是一段师生对话..."

# Initialize for GLM-4
glm4_model = ModelAPI(model_name="glm-4", api_key=api_key)
glm4_result = glm4_model.analyze_text(text=text_to_analyze, base_prompt=base_prompt)

# Initialize for GPT4o
gpt4o_model = ModelAPI(model_name="gpt4o", api_key=api_key)
gpt4o_result = gpt4o_model.analyze_text(text=text_to_analyze, base_prompt=base_prompt)

# Initialize for Qwen-Long
qwen_long_model = ModelAPI(model_name="qwen-long", api_key=api_key)
qwen_long_result = qwen_long_model.analyze_text(text=text_to_analyze, base_prompt=base_prompt)
```
