

# 使用指南

该框架旨在为不同的业务需求提供一个可扩展的解决方案，包含模型层、数据处理层和业务层。通过该框架，您可以方便地集成新的业务逻辑，复用模型层和数据处理层，实现快速开发。

## 框架结构

- **模型层（Model Layer）**：提供统一的接口与多种语言模型进行交互，例如 GLM-4、GPT4o、Qwen-Long 等。该层设计为通用模块，可以在不同的业务中复用，并支持未来添加新的模型。
- **数据处理层（Data Processing Layer）**：根据不同的任务类型，处理和预处理数据，支持多种输入格式。您可以继承 `DataProcessor` 类，为新的业务需求实现特定的数据处理逻辑。
- **业务层（Business Layer）**：将模型层和数据处理层结合，实现具体的业务逻辑。业务层利用模型层和数据处理层提供的通用接口，简化了业务逻辑的实现过程。

## 使用步骤

### 1. 安装依赖

确保您的 Python 环境满足以下要求：

- **Python 版本**：Python 3.6 及以上
- **必要的 Python 库**：
  - `pandas`
  - `openai`
  - 以及框架中包含的自定义模块

安装必要的库：

```bash
pip install pandas openai
```

### 2. 准备数据

根据您的业务需求，准备输入数据。框架支持多种数据格式，包括：

- **Pandas DataFrame**
- **JSON 字符串**
- **字典**

数据应包含业务所需的字段，例如：

- `start_time`
- `end_time`
- `text`
- `label`

### 3. 数据处理

根据您的任务类型，使用相应的数据处理器。如果现有的处理器不满足需求，您可以继承 `DataProcessor` 类，创建新的数据处理器。

**示例**：

```python
from data_processor import DataProcessor

# 初始化数据处理器
processor = DataProcessor(dataset=your_dataset, task="your_task", T=your_threshold)

# 处理数据
processed_data = processor.process_and_save_sub_dfs()
```

- `dataset`：您的输入数据
- `task`：任务名称，例如 `"teacher_dialogue_classification"`
- `T`：时间差阈值，根据任务需求设定

### 4. 模型调用

使用 `ModelAPI` 类与指定的模型进行交互。您可以在模型层中添加新的模型支持。

**示例**：

```python
from model_api_handler import ModelAPI

# 初始化模型 API
api_key = "your_api_key"
model = ModelAPI(model_name="your_model_name", api_key=api_key)

# 分析文本
result = model.analyze_text(text="待分析的文本", base_prompt="您的提示语")
```

- `model_name`：模型名称，例如 `"glm-4"`
- `api_key`：模型 API 密钥
- `text`：待分析的文本
- `base_prompt`：模型提示语

### 5. 业务集成

在业务层，将数据处理和模型调用结合，实现完整的业务流程。

**示例**：

```python
# 数据处理
processor = DataProcessor(dataset=your_dataset, task="your_task", T=your_threshold)
processed_data = processor.process_and_save_sub_dfs()

# 模型调用
api_key = "your_api_key"
model = ModelAPI(model_name="your_model_name", api_key=api_key)

results = []
for data_segment in processed_data:
    text_to_analyze = data_segment['text']
    base_prompt = "您的提示语"
    result = model.analyze_text(text=text_to_analyze, base_prompt=base_prompt)
    results.append({
        'text': text_to_analyze,
        'model_output': result
    })

# 保存结果
import pandas as pd

df = pd.DataFrame(results)
df.to_excel("output.xlsx", index=False)
```

### 6. 扩展新业务

当有新的业务需求时，您可以按照以下步骤进行扩展：

#### 扩展数据处理层

- 创建新的数据处理器类，继承自 `DataProcessor`。
- 实现特定任务的数据处理逻辑。

**示例**：

```python
from data_processor import DataProcessor

class NewTaskProcessor(DataProcessor):
    def __init__(self, dataset, T):
        super().__init__(dataset, task="new_task", T=T)
    
    def process(self):
        # 实现新的数据处理逻辑
        pass
```

#### 扩展模型层

- 在 `ModelAPI` 类中添加对新模型的支持。
- 实现新模型的 API 调用逻辑。

**示例**：

```python
class ModelAPI:
    # 现有代码...

    def _get_base_url(self):
        # 添加新模型的 API 端点
        base_urls = {
            "glm-4": "https://api.glm-4.com/v1/",
            "new-model": "https://api.new-model.com/v1/"
        }
        return base_urls.get(self.model_name)
    
    def _get_client(self):
        # 实现新模型的客户端逻辑
        if self.model_name == "new-model":
            # 新模型的 API 调用方式
            pass
```

#### 更新业务层

- 在业务层中，使用新的数据处理器和模型，实现新的业务逻辑。

**示例**：

```python
# 数据处理
processor = NewTaskProcessor(dataset=your_dataset, T=your_threshold)
processed_data = processor.process()

# 模型调用
model = ModelAPI(model_name="new-model", api_key="your_api_key")

# 业务逻辑
for data_segment in processed_data:
    result = model.analyze_text(text=data_segment['text'], base_prompt="新的提示语")
    # 处理结果
```

### 7. 运行脚本

您可以通过命令行参数来运行脚本，实现不同的业务需求。

**示例**：

```bash
python your_script.py --data path/to/data.xlsx --config path/to/config.json --prompt path/to/prompt.txt --output path/to/output.xlsx
```

- `--data`：数据文件路径
- `--config`：配置文件路径，包含模型和数据处理参数
- `--prompt`：模型提示语文件路径
- `--output`：输出文件路径

## 注意事项

- **模块复用**：充分利用模型层和数据处理层的可复用性，避免重复开发。
- **代码规范**：遵循良好的代码规范和注释习惯，提升代码的可读性和可维护性。
- **错误处理**：在实际应用中，添加必要的错误处理和异常捕获，确保程序的稳定运行。
- **性能优化**：针对大规模数据处理，考虑性能优化方案，例如批量处理、异步调用等。

## 示例项目

您可以参考提供的老师四分类业务脚本，了解如何使用该框架实现具体的业务需求。该示例展示了如何：

- 使用 `TeacherDialogueClassificationProcessor` 处理数据
- 利用 `ModelAPI` 调用模型进行分析
- 将结果保存为 Excel 文件，方便查看和分析

---

希望这个使用指南能帮助您快速上手该框架，并利用其灵活的扩展性满足不同的业务需求。
