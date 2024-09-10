class ModelAPI:
    def __init__(self, model_family, api_key, base_url=None, api_version=None):
        self.model_family = model_family.lower()  # 更新为model_family
        self.api_key = api_key
        self.base_url = base_url or self._get_base_url()
        self.api_version = api_version  # 针对GPT4o，添加api_version参数
        self.client = self._get_client()

    def _get_base_url(self):
        # 处理不同模型家族的base_url
        if self.model_family == "glm-4":
            return "https://open.bigmodel.cn/api/paas/v4/"
        elif self.model_family == "gpt4o":
            return "https://zonekey-gpt4o.openai.azure.com/"  # Azure API端点
        elif self.model_family.startswith("qwen"):  # 检查是否是qwen家族模型
            return "https://dashscope.aliyuncs.com/compatible-mode/v1"
        else:
            raise ValueError(f"Unsupported model family: {self.model_family}")

    def _get_client(self):
        # 根据不同的模型家族返回不同的客户端实例
        if self.model_family == "glm-4":
            from openai import OpenAI
            return OpenAI(api_key=self.api_key, base_url=self.base_url)
        elif self.model_family == "gpt4o":
            from openai import AzureOpenAI
            # 针对GPT4o模型，传入api_version参数
            return AzureOpenAI(api_key=self.api_key, azure_endpoint=self.base_url, api_version=self.api_version)
        elif self.model_family.startswith("qwen"):  # 统一处理qwen家族模型
            from openai import OpenAI
            return OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            raise ValueError(f"Unsupported model family: {self.model_family}")

    def analyze_text(self, text, base_prompt, model):
        # 调用模型接口生成结果
        prompt = base_prompt + text
        response = self.client.chat.completions.create(
            model=model,  # 这里使用传入的模型名称
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": "你是一个乐于助人的小助手,并且每次输出的结果要是json格式"},
                      {"role": "user", "content": prompt}],
            max_tokens=1000,
            n=1,
            temperature=0.7,
        )
        return response.choices[0].message.content


if __name__ == '__main__':
    # 示例调用
    api_key = "your_api_key"
    text_to_analyze = "它分别种了什么树呢？谁来说说？于凯，你来说说看。你慢讲啊。嗯，然后呢？"
    base_prompt = """后面的“待分析文本”是一段师生对话，其中，学生话语已经剔除，只保留老师话语，请对老师的话语进行分析，具体分析方法如下所示：  
将”待分析文本“分割成”发起“、”评价“、”讲解“、“其它”四种子文本段，”发起“的分割尽可能细一点。“发起”是老师邀请、引导、鼓励学生用话语来回应的语句；“评价”是对学生回应的表扬、认可、批评等评价性话语；”讲解“是老师针对知识展开描述或对学生回应的总结；不能归属于上面三种子文本段，归属为“其它”。
按照下面“示例”输出： """
    # 调用GPT4o模型

    # 调用GPT4o模型，传入deployment_name而不是gpt4o
    # gpt4o_model = ModelAPI(model_family="gpt4o", api_key="b2e709bdd54f4416a734b4a6f8f1c7a0",
    #                        api_version="2024-02-01")
    # gpt4o_result = gpt4o_model.analyze_text(text=text_to_analyze, base_prompt=base_prompt, model="soikit_test")
    # print("gpt4o_result:", gpt4o_result)
    #
    # # 调用GLM-4模型
    # glm4_model = ModelAPI(model_family="glm-4", api_key="08bd304ed5c588b2c9cb534405241f0e.jPN6gjmvlBe2q1ZZ")
    # glm4_result = glm4_model.analyze_text(text=text_to_analyze, base_prompt=base_prompt,model='glm-4')
    # print("glm4_result:", glm4_result)

    # # 调用Qwen-Long模型
    qwen_long_model = ModelAPI(model_family="qwen", api_key="sk-f582e4fab0894a52b12b7a85c62868bc")
    qwen_long_result = qwen_long_model.analyze_text(text=text_to_analyze, base_prompt=base_prompt,model="qwen2-72b-instruct")
    print("qwen_long_result:", qwen_long_result)
