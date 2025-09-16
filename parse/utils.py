import re

# 创建一个映射字典，将单词数字映射到整数
number_words = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000
}

# 定义一个函数，将单词数字转换为整数
def word_to_int(word):
    if word in number_words:
        return number_words[word]
    else:
        return None

# 解析字符串并提取单词数字
def parser_number(an):
    words = an.split()
    numbers = []
    for word in words:
        num = word_to_int(word.lower())
        if num is not None:
            numbers.append(num)
    if numbers:
        # 将提取到的数字合并成一个整数
        return sum(numbers)
    else:
        return None