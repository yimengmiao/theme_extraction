import re
import json
import logging

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
    # 中英文标点符号的正则表达式
    punctuation_pattern = r'[，。！？、；：“”‘’（）《》【】…—]+' + r'[.,\/#!$%\^&\*;:{}=\-_`~()]'
    # 使用正则表达式替换标点符号为空字符串
    no_punctuation_text = re.sub(punctuation_pattern, '', text)
    return no_punctuation_text
