# 就是为了构造prompt2输入的数据处理步骤

## 需求分析：
数据处理的对象：前一个老师四分类任务（从课堂文本中分割师生文本段）得出的结果，记作 output1.xlsx。
1. 从 output1.xlsx 分离出 IRE,EIR,IR,EIRE片段。最后再拼接在一起。 

## 方案设计：
### 名词解释：
- **gpt4o_result** 列：output1.xlsx 中的一列，表示调用 gpt4o 模型输出的教师四分类的子文本段。
- **filtered_gpt4o_result** 列：经过删除操作后，生成的的一个新列。
- **label** 列：output1.xlsx 中的一列，表示说话人的角色。0 表示老师，1 表示学生。
- **I（发起）**：模型在 gpt4o_result列中type为“发起”的文本。
- **R（回答）**：学生对“发起”的回应，即 output1.xlsx 中 label 列为 0 的行。
- **E  (讲解)**：模型在 gpt4o_result列中type为“讲解”的文本。

### 一、提前的数据处理
1. 解析 gpt4o_result 列
gpt4o_result 列中的数据以字符串形式存储，需要将其解析为字典列表。
- 解析逻辑：每个单元格包含一个由 result 列表组成的字符串，每个列表项是一个字典，字典中包含两个字段：type: 文本片段的类型，可能值为 "发起"、"讲解" 等。
    - content: 对应类型的文本内容。
    - 解析过程需将字符串转换为实际的字典列表。
- 解析操作：使用合适的解析方法，将 gpt4o_result 列从字符串转换为字典列表，方便后续处理。

2. 过滤字典
为了简化后续的片段提取操作，我们需要对 gpt4o_result 列进行预处理，保留符合条件的内容，并合并相邻相同类型的字典。
- 过滤逻辑：
    - 解析 gpt4o_predict 列的 result 列表：检查每行的 result 列表，保留 包含“发起”或“讲解” 的字典。
        - 如果 result 列表中只有一个字典：如果 type 既不是“发起”也不是“讲解”，则删除该行及其后紧跟的 label=1 的行。
        - 如果 type 为“发起”或“讲解”，则保留该行。
- 生成 filtered_gpt4o_result 列：
    - 通过上述过滤和合并，生成新的列 filtered_gpt4o_result，该列将用于后续的片段提取。

### 二、片段提取逻辑
在数据预处理完成后，根据 label 列和 filtered_gpt4o_result 列进行片段提取。每一个片段（IRE、EIR、EIRE、IR）都基于交替出现的 label=0,1 对进行处理。
#### 1. label 列定义
- label=0 代表教师发言。
- label=1 代表学生发言。
每个 label=0 后面会跟随一个 label=1，即教师发言与学生发言交替出现。

#### 片段的提取逻辑分为两种主要情况，基于教师发言中第一个字典的 type 是否为 "发起" 或 "讲解"。以下是两种情况的详细逻辑。
##### 条件一：发起开头（IRE、IR）
如果在 label=0 的行中，filtered_gpt4o_result 列中的第一个字典的 type 为 "发起"。
1. **I（发起）**：
    - 提取 filtered_gpt4o_result 列中的所有 content 作为 I 的内容。具体操作为：如果 type="发起" 是第一个字典，则将整个 result 列表中所有字典的 content 合并，作为 I。
2. **R（回应）**：
    - 查找接下来的 label=1 行，并将 text 列的内容作为 R（回应）。如果该行的 text 列为空，则认为学生回应缺失，标记为 "学生文本可能缺失"。
3. **E（讲解）**：
    - 查找下一个label=0行，如果该行的result 列表长度不为1，或者（result列表长度为1但type="发起" 的字典），终止 E 的寻找。直接输出 IR。
    - 查找下一个 label=0 行，如果该行的 filtered_gpt4o_result 列中 result 列表的长度为1，且字典类型为 "讲解"，则将 content 作为 E。
    - 继续向下查找，直到遇到：result 列表长度不为1，或者（result列表长度为1但type="发起" 的字典），终止 E 的寻找。
    - 在未终止E的寻找之前，碰到result列表中的content内容都要整合到E中。
4. **片段存储**：
    - 如果找到 I 和 R，将其与 E 一起组合形成 IRE 片段。
    - 如果没有找到 E，则只存储 IR 片段。
    - 如果找到多个 E，则形成 IREEEE 片段。(后面多个E的内容会放在一个叫E_after列表中)。

##### 条件二：讲解开头（EI、EIR、EIRE）
如果在 label=0 的行中，filtered_gpt4o_result 列中的第一个字典的 type 为 "讲解"，则进入条件二处理逻辑：
5. **E（讲解）**：
    - 将 filtered_gpt4o_result 列中的 content 合并，作为 E。
    - 继续查找后续的 label=0 行，如果 result 列表长度为1，且 type="讲解"，则将内容继续合并到 E，直到遇到下一个 result 列表长度不为1。
6. **I（发起）**：
    - 查找下一个 label=0 行，判断其 filtered_gpt4o_result 列是否包含 type="发起" 的字典。如果 type="发起" 的字典出现在 result 列表中的任意索引位置，从这个索引位置开始，将之后的内容作为 I。
    - 如果 I 前面的字典类型为 "讲解"，将这些内容继续放在一个叫E_before列表 中。
    - 举例 ：如果 type="发起" 并不是result列表中的第一个字典（假设在第三个位置），则将 result 列表前两个字典的 content 放到 E_before 中，之后的字典中的content合并起来，作为 I。

7. 有了EI后再从条件一中的第2 步开始处理，继续寻找 R 和 E。

### 三、跳转条件
- **重新开始条件**：当一个片段（IR、IRE、EIR、EIRE）提取完毕后，处理逻辑将跳到下一个 label=0,1 对，继续处理新的片段，直到遍历完所有数据。

## 四. 输入，输出

### 输入格式
```python
data = [
    {
        "start_time": 0,
        "end_time": 10,
        "text": "老师发言1",
        "label": 0,
        "gpt4o_result": '{"result": [{"type": "发起", "content": "老师发起提问1"}, {"type": "讲解", "content": "老师讲解内容1"}], "compliance": "高"}'
    },
    {"start_time": 10, "end_time": 20, "text": "学生回答1", "label": 1, "gpt4o_result": None},
    # ...（更多数据）
]
```

### 输出格式
最终结果中列表里面每个内容都是分割好的IRE片段。

示例：
```python
最终结果 = [
    '''发起：老师发起提问1老师讲解内容1
        回应：学生回答1''',
    # ...
]
```

**Todo**: 把这个最终结果放到prompt2中去，让其判断输出归属于上一个片段的“讲解”的分割点。