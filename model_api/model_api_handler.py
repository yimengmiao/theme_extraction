class ModelAPI:
    def __init__(self, params):
        self.model_family = params.get('model_family', '').lower()
        if not self.model_family:
            raise ValueError("Parameter 'model_family' is required.")

        self.api_key = params.get('api_key')
        if not self.api_key:
            raise ValueError("Parameter 'api_key' is required.")

        self.base_url = params.get('base_url') or self._get_base_url()
        self.api_version = params.get('api_version')
        self.text = params.get('text', '')
        self.prompt = params.get('prompt', '')
        self.model = params.get('model_name')
        if not self.model:
            raise ValueError("Parameter 'model' is required.")

        self.max_tokens = params.get('max_tokens', 1000)
        self.n = params.get('n', 1)
        self.temperature = params.get('temperature', 0.7)
        self.client = self._get_client()

    def _get_base_url(self):
        if self.model_family == "glm-4":
            return "https://open.bigmodel.cn/api/paas/v4/"
        elif self.model_family == "gpt4o":
            return "https://zonekey-gpt4o.openai.azure.com/"
        elif self.model_family.startswith("qwen"):
            return "https://dashscope.aliyuncs.com/compatible-mode/v1"
        else:
            raise ValueError(f"Unsupported model family: {self.model_family}")

    def _get_client(self):
        if self.model_family == "glm-4":
            from openai import OpenAI
            return OpenAI(api_key=self.api_key, base_url=self.base_url)
        elif self.model_family == "gpt4o":
            from openai import AzureOpenAI
            return AzureOpenAI(api_key=self.api_key, azure_endpoint=self.base_url, api_version=self.api_version)
        elif self.model_family.startswith("qwen"):
            from openai import OpenAI
            return OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            raise ValueError(f"Unsupported model family: {self.model_family}")

    def analyze_text(self):
        user_input = self.prompt + self.text
        response = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "你是一个乐于助人的小助手"},
                {"role": "user", "content": user_input}
            ],
            max_tokens=self.max_tokens,
            n=self.n,
            temperature=self.temperature,
        )
        return response.choices[0].message.content


if __name__ == '__main__':
    # 定义参数
    params = {
        "model_family": "glm-4",
        "api_key": "08bd304ed5c588b2c9cb534405241f0e.jPN6gjmvlBe2q1ZZ",
        "text": "它分别种了什么树呢？谁来说说？于凯，你来说说看。你慢讲啊。嗯，然后呢？",
        "prompt": """后面的'待分析文本'是一段师生对话，其中，学生话语已经剔除，只保留老师话语，请对老师的话语进行分析，具体分析方法如下所示：
    将'待分析文本'分割成'发起'、'评价'、'讲解'、'其它'四种子文本段，'发起'的分割尽可能细一点。'发起'是老师邀请、引导、鼓励学生用话语来回应的语句；'评价'是对学生回应的表扬、认可、批评等评价性话语；'讲解'是老师针对知识展开描述或对学生回应的总结；不能归属于上面三种子文本段，归属为'其它'。
    按照下面'示例'输出：
    {"result":
    [{"type":"发起","content":"它分别种了什么树呢？于凯，你来说说看。"}, 
    {"type":"其它","content":"你慢讲啊，嗯。"},
    {"type":"发起","content":"然后呢？"},
    {"type":"讲解","content":"然后种了杏树。"},
    {"type":"发起","content":"最后呢？"}
    ]}""",
        "model_name": "glm-4-flash",
        "max_tokens": 1000,
        "n": 1,
        "temperature": 0.7
    }
    # 调用GPT4o模型

    # 调用GPT4o模型，传入deployment_name而不是gpt4o
    # gpt4o_model = ModelAPI(model_family="gpt4o", api_key="b2e709bdd54f4416a734b4a6f8f1c7a0",
    #                        api_version="2024-02-01")
    # gpt4o_result = gpt4o_model.analyze_text(text=text_to_analyze, base_prompt=base_prompt, model="soikit_test")
    # print("type(gpt4o_result)=", type(gpt4o_result))
    # print("gpt4o_result:", gpt4o_result)

    # # 调用GLM-4模型
    # 创建 ModelAPI 实例并调用方法
    model_api = ModelAPI(params)
    glm4_result = model_api.analyze_text()
    print("Result:", glm4_result)
    print("type(glm4_result)=", type(glm4_result))
    print("glm4_result:", glm4_result)

    # # 调用Qwen-Long模型
    # qwen_long_model = ModelAPI(model_family="qwen", api_key=api_key)
    # qwen_long_result = qwen_long_model.analyze_text(text=text_to_analyze, base_prompt=base_prompt,
    #                                                 model="qwen2-72b-instruct")
    # print("qwen_long_result:", qwen_long_result)
