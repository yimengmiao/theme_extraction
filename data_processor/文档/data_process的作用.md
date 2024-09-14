
# DataProcessor 类

## 概述
`DataProcessor` 是一个用于根据不同任务处理数据集的 Python 类。该类支持多种输入格式，并根据用户指定的任务类型应用相应的数据处理逻辑。

## 功能
- **灵活的输入格式**：支持 pandas DataFrame、JSON 字符串或字典作为输入。
- **任务驱动的处理**：根据用户提供的任务类型执行不同的数据处理逻辑。
- **自定义时间差阈值**：允许用户定义特定任务下的时间差阈值（T），特别是针对 label 为 `0` 的数据行。

## 用法

### 初始化

```python
processor = DataProcessor(dataset, task, T)
```

- `dataset`: 要处理的数据集，支持 pandas DataFrame、JSON 字符串或字典格式。
- `task`: 任务名称，用于决定使用哪种处理逻辑。目前支持的任务包括：
  - `teacher_dialogue_classification`: 基于教师对话分类任务的数据处理逻辑。
  - `class_activity_classification`: 课堂活动分类逻辑（目前仅为占位符，尚未实现）。
- `T`: 针对特定任务的自定义时间差阈值，适用于 label 为 `0` 的行。

### 方法

#### `_prepare_dataset(self, dataset)`
- 将 JSON 或字典格式的数据转换为 pandas DataFrame 格式。
- 确保输入数据格式统一，便于进一步处理。

#### `process_and_save_sub_dfs(self)`
- 根据提供的任务执行相应的处理函数。
- 如果处理器已实现，则调用对应任务的处理逻辑。

## 示例

```python
# 使用 DataProcessor 类的示例
data_processor = DataProcessor(dataset=my_data, task="teacher_dialogue_classification", T=10)
processed_data = data_processor.process_and_save_sub_dfs()
```

## 自定义
你可以通过在 `__init__` 方法中添加更多的任务处理器来扩展该类，以支持其他任务。

## 未来工作
- 实现对 `class_activity_classification` 任务的支持或其他任务需求。
- 增加更多任务特定的数据处理功能。

