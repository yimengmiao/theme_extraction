import re
from data_processor.public_code_data_process import remove_punctuation


class Prompt3inputProcessor:
    """处理文本的类，包含去重、过滤索引、合并段落等操作"""

    def __init__(self, prompt2_input, splitpoint):
        """
        初始化函数，接受输入文本和分割点字典
        :param prompt2_input: 需要处理的文本
        :param splitpoint: 分割点，用于根据指定内容进行文本分割
        """
        self.prompt2_input = prompt2_input
        self.splitpoint = splitpoint
        self.input1_list = self.prompt2_input.split('\n')
        self.breakpoint = self.splitpoint.get("breakpoint", [])

    @staticmethod
    def method_dict(lst):
        """对列表进行去重"""
        return list(dict.fromkeys(lst))

    @staticmethod
    def filter_consecutive(lst):
        """
        处理相邻索引，保留较小的索引生成新的
        :param lst: 索引列表
        :return: 过滤后的索引列表
        """
        if not lst:
            return []
        result = [lst[0]]
        for i in range(1, len(lst)):
            if lst[i] != lst[i - 1] + 1:
                result.append(lst[i])
        return result

    @staticmethod
    def merge_explanations(piece):
        """
        合并相邻的讲解内容
        :param piece: 段落列表
        :return: 合并后的段落列表
        """
        new_piece = []
        merged = ""
        for item in piece:
            if item.startswith("讲解："):
                merged += item.replace("讲解：", "") + ","
            else:
                if merged:
                    new_piece.append("讲解：" + merged[:-1])  # 去掉最后一个逗号
                    merged = ""
                new_piece.append(item)
        if merged:
            new_piece.append("讲解：" + merged[:-1])
        return new_piece

    @staticmethod
    def split_by_initiate(list_all):
        """
        分割合并后的子列表
        根据'发起'分割段落
        :param list_all: 全部文本列表
        :return: 根据发起分割后的文本段落
        """
        segments = []
        current_segment = []
        for item in list_all:
            if item.startswith('发起：'):
                if current_segment:  # 保存之前的段落
                    segments.append(current_segment)
                    current_segment = []  # 重置当前段落
            current_segment.append(item)
        if current_segment:
            segments.append(current_segment)
        if segments[0][0].startswith('讲解：'):
            segments[0].extend(segments[1])
            segments.pop(1)
        return segments

    def process(self):
        """
        主函数：Construct the input of prompt3
        处理原始文本，将其分割并重组为指定格式的输入文本
        :return: 构造好的 prompt3 输入文本
        """
        list1 = []

        # 处理索引
        for i, item in enumerate(self.input1_list):
            for item2 in self.breakpoint:
                if remove_punctuation(item.strip()) == remove_punctuation(item2.strip()):
                    list1.append(i)

        list1 = self.method_dict(list1)  # 去重
        list2 = self.filter_consecutive(list1)  # 过滤连续索引

        # 根据 list2 切分 input1_list
        pieces = []
        if list2:
            pieces.append(self.input1_list[0:list2[0]])
            for i in range(len(list2) - 1):
                pieces.append(self.input1_list[list2[i]:list2[i + 1]])
            pieces.append(self.input1_list[list2[-1]:])
        else:
            pieces.append(self.input1_list)

        # 合并相邻讲解内容
        merged_pieces = [self.merge_explanations(piece) for piece in pieces if piece]

        list_all = []

        # 检查 breakpoint 是否为空
        if not self.breakpoint:
            list_all.extend(self.split_by_initiate(self.input1_list))
        else:
            for merged_piece in merged_pieces:
                list_all.extend(self.split_by_initiate(merged_piece))

        # 生成 prompt3 的输入文本
        prompt3_input = ""
        for idx, segment in enumerate(list_all, start=1):
            prompt3_input += f"师生对话{idx}\n" + "\n".join(segment) + "\n\n"

        return prompt3_input.strip()


# 示例调用
if __name__ == '__main__':
    prompt2_input = """讲解：瞧那么一句话，就能把故事的意思讲出来了，\n发起：好，那我们来看看哦，他为什么什么树都没种成呢？\n回应：[空白]\n发起：我们来读课文的第一段，来，谁来读？\n回应：[空白]\n发起：嗯，好，你来读\n回应：猴子，种了一棵梨树苗天，天浇水施肥等着将来吃梨子，嗯，\n发起：你不要坐下，是不天天说明什么呀？\n回应：就是每天\n讲解：每天对呀，就说明这个猴子怎么样\n讲解：他每天都去浇，天天都去浇水，施肥\n发起：说明他怎么样\n回应：勤劳淡实，\n讲解：英语非常勤劳，\n讲解：猴子是不是很好，\n发起：你们看，\n回应：[空白]\n讲解：这个猴子种树啊，它有一个动作的过程，\n发起：你找到了动作的吗？它有哪些动作？\n回应：浇水\n发起：天气就直接浇水了吗？\n回应：[空白]\n讲解：所以先说重好的，\n发起：然后呢，\n回应：胶水\n讲解：嗯浇，嗯，施肥，"""
    splitpoint = {
  "breakpoint": [

  ]
}

    # 创建类的实例
    processor = Prompt3inputProcessor(prompt2_input, splitpoint)
    # 调用处理方法
    result = processor.process()
    print(result)
