import re


class TextCleaner:
    @staticmethod
    def replace_regex(text, pattern, replacement=''):
        return pattern.sub(replacement, text)

    @staticmethod
    def clean_lines(lines):
        for i in range(len(lines)):
            # Remove brackets with only numbers inside and all standalone numbers
            number_pattern = r'\(\s*(\d+)\s*\)|\(\s*(\d+)\s*\/\s*(\d+)\s*\)|\d+'
            lines[i] = TextCleaner.replace_regex(lines[i], re.compile(number_pattern))

            # Remove all brackets
            brackets_pattern = r'[\[\{\(\]\}\)]'
            lines[i] = re.compile(brackets_pattern).sub('', lines[i])

            # Remove unwanted characters
            unwanted_chars = r'[/\/\\\-]'
            lines[i] = TextCleaner.replace_regex(lines[i], re.compile(unwanted_chars))

            punctuation_chars = r'[,»–\';«*\u200f"\\~`]'
            lines[i] = TextCleaner.replace_regex(lines[i], re.compile(punctuation_chars))

            # Normalize spaces
            space_pattern = r'\s+'
            lines[i] = TextCleaner.replace_regex(lines[i], re.compile(space_pattern), ' ')

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