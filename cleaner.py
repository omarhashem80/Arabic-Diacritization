import re


class TextCleaner:
    @staticmethod
    def replace_regex(text, pattern, replacement=''):
        return pattern.sub(replacement, text)

    @staticmethod
    def clean_lines(lines):
        for i in range(len(lines)):
            
            reg = r'\(\s*(\d+)\s*\)|\(\s*(\d+)\s*\/\s*(\d+)\s*\)|\d+'
            lines[i] = TextCleaner.replace_regex(lines[i], re.compile(reg))
            
            reg_brackets = r'[\[\{\(\]\}\)]'
            lines[i] = re.compile(reg_brackets).sub('', lines[i])
            
            reg = r'[/\/\\\-]'
            lines[i] = TextCleaner.replace_regex(lines[i], re.compile(reg))
            
            reg = r'[,»–\';«*\u200f\u200d\u200b\u200c\u200e"\\~`%…_]'
            lines[i] = TextCleaner.replace_regex(lines[i], re.compile(reg))
            
            reg = r'[a-zA-Z]'
            lines[i] = TextCleaner.replace_regex(lines[i], re.compile(reg))
            
            reg = r'[\u00BC-\u00BE\u2150-\u215E]'
            lines[i] = TextCleaner.replace_regex(lines[i], re.compile(reg))
            
            reg = r'[\U0001F000-\U0001FFFF]'
            lines[i] = TextCleaner.replace_regex(lines[i], re.compile(reg))
            
            reg = r'[\u2600-\u26FF\u2700-\u27BF]'
            lines[i] = TextCleaner.replace_regex(lines[i], re.compile(reg))
            
            reg = r'\s+'
            lines[i] = TextCleaner.replace_regex(lines[i], re.compile(reg), ' ')
        return lines

    @staticmethod
    def remove_diacritics(lines):
        diacritics_pattern = r'[\u064B-\u065F\u0670\uFE70-\uFE7F]'
        for i in range(len(lines)):
            lines[i] = TextCleaner.replace_regex(lines[i], re.compile(diacritics_pattern))
        return lines

    @staticmethod
    def count_spaces(text):
        return len(re.findall(r'\s', text))