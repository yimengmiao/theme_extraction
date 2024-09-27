import traceback

import pandas as pd
import json
from typing import Any, Dict, List, Optional, Tuple
import logging

# 配置 logger，将日志记录存储到文件中
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='log/app_log.log',  # 指定日志文件路径
    filemode='a'  # 文件模式：'a' 追加模式，'w' 写入模式会覆盖之前的日志
)
logger = logging.getLogger(__name__)


class DialougueProcessor:
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
        self.df = None

    def validate_input_data(self) -> bool:
        required_keys = {'start_time', 'end_time', 'text', 'label', 'gpt4o_result'}
        for idx, item in enumerate(self.data):
            if not isinstance(item, dict):
                raise ValueError(f"第 {idx + 1} 个元素不是字典类型")
            missing_keys = required_keys - item.keys()
            if missing_keys:
                raise ValueError(f"第 {idx + 1} 个元素缺少必要的键: {missing_keys}")
        return True

    def parse_gpt4o_result(self, gpt4o_result: Optional[str]) -> Optional[Dict[str, Any]]:
        if pd.isnull(gpt4o_result):
            return None
        try:
            if isinstance(gpt4o_result, str):
                s = gpt4o_result.replace('\n', '').replace('\\n', '').replace('\\\\', '\\')
                return json.loads(s)
            else:
                return gpt4o_result
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")

            logger.info(f"原始字符串: {gpt4o_result}")
            logger.error("详细堆栈信息：\n%s", traceback.format_exc())  # 添加详细堆栈跟踪信息

            return None

    def filter_and_merge(self, parsed_gpt4o_result: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], bool]:
        if parsed_gpt4o_result is None:
            return None, False
        result = parsed_gpt4o_result.get('result', [])
        filtered_result = [d for d in result if d.get('type') in ['发起', '讲解']]
        if len(result) == 1 and result[0].get('type') not in ['发起', '讲解']:
            return None, True
        if not filtered_result:
            return None, False
        return {'result': filtered_result, 'compliance': parsed_gpt4o_result.get('compliance')}, False

    def should_delete_row(self, delete_flag: bool, idx: int) -> bool:
        return delete_flag

    def get_student_response(self, i: int) -> Tuple[str, int]:
        if i < len(self.df) and self.df.iloc[i]['label'] == 1:
            R_content = self.df.iloc[i]['text'] or '[空白]'
            i += 1
        else:
            R_content = '[空白]'
        return R_content, i

    def get_E_after(self, i: int) -> Tuple[List[str], int]:
        E_after = []
        while i < len(self.df):
            if i + 1 >= len(self.df):
                break
            next_row = self.df.iloc[i]
            if next_row['label'] == 0:
                next_filtered_result = next_row['filtered_gpt4o_result']
                if next_filtered_result is not None:
                    next_result_list = next_filtered_result.get('result', [])
                    if len(next_result_list) == 1 and next_result_list[0].get('type') == '讲解':
                        E_after.append(next_result_list[0].get('content', ''))
                        i += 1
                        next_student_row = self.df.iloc[i]
                        if next_student_row['label'] == 1:
                            i += 1
                            continue
                        else:
                            break
                    else:
                        break
                else:
                    break
            else:
                break
        return E_after, i

    def process_teacher_row(self, i: int) -> Tuple[Optional[Dict[str, Any]], int]:
        row = self.df.iloc[i]
        filtered_gpt4o_result = row['filtered_gpt4o_result']
        if filtered_gpt4o_result is not None:
            result_list = filtered_gpt4o_result.get('result', [])
            E_before = []
            I_content = ''
            first_initiate_idx = next((idx for idx, item in enumerate(result_list) if item.get('type') == '发起'), None)
            if first_initiate_idx is not None:
                for idx in range(0, first_initiate_idx):
                    if result_list[idx].get('type') == '讲解':
                        E_before.append(result_list[idx].get('content', ''))
                I_content = ''.join(item.get('content', '') for item in result_list[first_initiate_idx:])
            else:
                i += 1
                return None, i
            i += 1
            R_content, i = self.get_student_response(i)
            E_after_list, i = self.get_E_after(i)
            segment = {'I': I_content, 'R': R_content}
            if E_before:
                segment['E_before'] = E_before
            if E_after_list:
                segment['E_after'] = E_after_list
            segment['type'] = self.determine_segment_type(segment)
            if segment['type'] is None:
                return None, i
            return segment, i
        else:
            i += 1
            return None, i

    def determine_segment_type(self, segment: Dict[str, Any]) -> Optional[str]:
        has_I = 'I' in segment and segment['I']
        has_E_before = 'E_before' in segment and segment['E_before']
        has_E_after = 'E_after' in segment and segment['E_after']
        if not has_I:
            return None
        if has_E_before and has_E_after:
            return 'EIRE'
        elif has_E_before:
            return 'EIR'
        elif has_E_after:
            return 'IRE'
        else:
            return 'IR'

    def extract_segments(self) -> List[Dict[str, Any]]:
        segments = []
        i = 0
        while i < len(self.df):
            row = self.df.iloc[i]
            if row['label'] == 0:
                segment, i = self.process_teacher_row(i)
                if segment:
                    segments.append(segment)
                else:
                    continue
            else:
                i += 1
        return segments

    def process_data(self) -> List[Dict[str, Any]]:
        self.validate_input_data()
        self.df = pd.DataFrame(self.data)
        self.df['parsed_gpt4o_result'] = self.df['gpt4o_result'].apply(self.parse_gpt4o_result)

        filtered_results = []
        delete_flags = []

        for idx, row in self.df.iterrows():
            parsed_gpt4o_result = row['parsed_gpt4o_result']
            filtered_gpt4o_result, delete_flag = self.filter_and_merge(parsed_gpt4o_result)
            filtered_results.append(filtered_gpt4o_result)
            delete_flags.append(delete_flag)

        self.df['filtered_gpt4o_result'] = filtered_results
        self.df['delete_flag'] = delete_flags
        self.df['to_delete'] = self.df.apply(lambda x: self.should_delete_row(x['delete_flag'], x.name), axis=1)
        self.df = self.df[self.df['to_delete'] == False].reset_index(drop=True)

        segments = self.extract_segments()
        return segments

    def process_segments(self, segments: List[Dict[str, Any]]) -> List[str]:
        final_texts = []
        for segment in segments:
            segment_type = segment.get('type')
            I_text = f"发起：{segment.get('I', '')}"
            # 当学生文本为空时，赋值为[空白]
            R_content = segment.get('R', '[空白]') or '[空白]'
            R_text = f"回应：{R_content}"

            if segment_type == "IR":
                final_text = f"{I_text}\n{R_text}"
            elif segment_type == "IRE":
                E_after_list = segment.get('E_after', [])
                E_after_text = "\n".join([f"讲解：{content}" for content in E_after_list])
                final_text = f"{I_text}\n{R_text}\n{E_after_text}"
            elif segment_type == "EIRE":
                E_before_list = segment.get('E_before', [])
                E_before_text = "\n".join([f"讲解：{content}" for content in E_before_list])
                E_after_list = segment.get('E_after', [])
                E_after_text = "\n".join([f"讲解：{content}" for content in E_after_list])
                final_text = f"{E_before_text}\n{I_text}\n{R_text}\n{E_after_text}"
            elif segment_type == "EIR":
                E_before_list = segment.get('E_before', [])
                E_before_text = "\n".join([f"讲解：{content}" for content in E_before_list])
                final_text = f"{E_before_text}\n{I_text}\n{R_text}"
            else:
                continue
            final_texts.append(final_text)
        return final_texts


# 示例使用
if __name__ == "__main__":

    data = [
        # 第一条数据
        {
            "start_time": 0,
            "end_time": 10,
            "text": "老师发起提问1，老师讲解内容1，老师发起提问2",
            "label": 0,
            "gpt4o_result": json.dumps({
                "result": [
                    {"type": "发起", "content": "老师发起提问1"},
                    {"type": "讲解", "content": "老师讲解内容1"},
                    {"type": "发起", "content": "老师发起提问2"},
                ],
                "compliance": "高"
            })
        },
        {"start_time": 10, "end_time": 20, "text": "学生回答1", "label": 1, "gpt4o_result": None},
        # 第二条数据
        {
            "start_time": 20,
            "end_time": 30,
            "text": "老师讲解内容2，老师讲解内容3，老师发起提问3，老师发起提问4。",
            "label": 0,
            "gpt4o_result": json.dumps({
                "result": [
                    {"type": "讲解", "content": "老师讲解内容2"},
                    {"type": "讲解", "content": "老师讲解内容3"},
                    {"type": "发起", "content": "老师发起提问3"},
                    {"type": "发起", "content": "老师发起提问4"}
                ],
                "compliance": "高"
            })
        },
        {"start_time": 30, "end_time": 40, "text": "", "label": 1, "gpt4o_result": None},
        # E_after 数据
        {
            "start_time": 40,
            "end_time": 50,
            "text": "老师讲解内容4",
            "label": 0,
            "gpt4o_result": json.dumps({
                "result": [
                    {"type": "讲解", "content": "老师讲解内容4"}
                ],
                "compliance": "高"
            })
        },
        {"start_time": 50, "end_time": 60, "text": "学生回应3", "label": 1, "gpt4o_result": None},
        {
            "start_time": 60,
            "end_time": 70,
            "text": "老师讲解内容5",
            "label": 0,
            "gpt4o_result": json.dumps({
                "result": [
                    {"type": "讲解", "content": "老师讲解内容5"}
                ],
                "compliance": "高"
            })
        },
        {"start_time": 70, "end_time": 80, "text": "学生回应4", "label": 1, "gpt4o_result": None}
    ]

    processor = DialougueProcessor(data)
    segments = processor.process_data()
    print("segments", segments)
    final_texts = processor.process_segments(segments)

    for text in final_texts:
        print(text)
