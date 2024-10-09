import re
import json
import logging
import string

# 配置 logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_json_using_patterns(text):
    """使用一组正则表达式模式来提取 JSON"""
    text = text.strip()
    logger.debug("原始文本: %s", text)

    patterns = [
        r'\{[\s\S]*\}',  # 新的正则表达式模式，匹配第一个 '{' 和最后一个 '}' 之间的内容
        r'(\{[\s\S]*?\})\s*\}$',
        r'\{\s*"result"\s*:\s*\[[\s\S]*?\]\s*\}',
        r'"""json\s*(\{[\s\S]*?\})\s*"""',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            if match.lastindex:
                json_str = match.group(1)
            else:
                json_str = match.group(0)
            logger.debug("匹配到的 JSON: %s", json_str)
            try:
                result_data = json.loads(json_str)
                return result_data
            except json.JSONDecodeError as e:
                logger.error("JSON 解析失败: %s", e)
                continue

    logger.warning("未找到符合模式的 JSON 数据")
    return {}


def convert_punctuation_to_chinese(text):
    """
    将英文标点符号转换为中文标点符号
    """
    punctuations = {
        ',': '，',
        '.': '。',
        '?': '？',
        '!': '！',
        ':': '：',
        ';': '；',
        '"': '“',
        '\'': '‘',
        '(': '（',
        ')': '）',
        '[': '【',
        ']': '】'
    }

    for eng_punc, zh_punc in punctuations.items():
        text = text.replace(eng_punc, zh_punc)

    return text


# 去除标点符号

def remove_punctuation(text):
    # 定义中英文标点符号
    punctuation = string.punctuation + '，。；：“”‘’？！《》、'
    # 使用正则表达式去除标点
    text = re.sub(f"[{re.escape(punctuation)}]", '', text)
    return text

if __name__ == '__main__':
    text = """
    
    ```json
{
    "result": [
        {"type": "发起", "content": "它分别种了什么树呢？谁来说说？"},
        {"type": "发起", "content": "于凯，你来说说看。"},
        {"type": "其它", "content": "你慢讲啊。"},
        {"type": "其它", "content": "嗯。"},
        {"type": "发起", "content": "然后呢？"}
    ]
}
```"""
    print(type(extract_json_using_patterns(text)))