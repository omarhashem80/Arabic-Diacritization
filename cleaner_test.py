import re


class TextCleaner:
    # Pre-compiled regex patterns (faster + cleaner)
    PATTERN_NUM_BRACKETS = re.compile(r'\(\s*\d+\s*\)|\(\s*\d+\s*/\s*\d+\s*\)|\d+')
    PATTERN_BRACKETS = re.compile(r'[\[\{\(\]\}\)]')
    PATTERN_UNWANTED_1 = re.compile(r'[/\\/\\-]')
    PATTERN_UNWANTED_2 = re.compile(r'[,»–\'\;«*\u200f\u200d\u200b\u200c\u200e"\\~`%…_]')
    PATTERN_ENGLISH = re.compile(r'[a-zA-Z]')
    PATTERN_FRACTIONS = re.compile(r'[\u00BC-\u00BE\u2150-\u215E]')
    PATTERN_EMOJIS = re.compile(r'[\U0001F000-\U0001FFFF]')
    PATTERN_MISC = re.compile(r'[\u2600-\u26FF\u2700-\u27BF]')
    PATTERN_SPACES = re.compile(r'\s+')
    PATTERN_DIACRITICS = re.compile(r'[\u064B-\u065F\u0670\uFE70-\uFE7F]')

    @staticmethod
    def replace(text, pattern, repl=''):
        return pattern.sub(repl, text)

    @staticmethod
    def clean_lines(lines):
        for i, line in enumerate(lines):
            line = TextCleaner.replace(line, TextCleaner.PATTERN_NUM_BRACKETS)
            line = TextCleaner.replace(line, TextCleaner.PATTERN_BRACKETS)
            line = TextCleaner.replace(line, TextCleaner.PATTERN_UNWANTED_1)
            line = TextCleaner.replace(line, TextCleaner.PATTERN_UNWANTED_2)
            line = TextCleaner.replace(line, TextCleaner.PATTERN_ENGLISH)
            line = TextCleaner.replace(line, TextCleaner.PATTERN_FRACTIONS)
            line = TextCleaner.replace(line, TextCleaner.PATTERN_EMOJIS)
            line = TextCleaner.replace(line, TextCleaner.PATTERN_MISC)
            line = TextCleaner.replace(line, TextCleaner.PATTERN_SPACES, ' ')
            lines[i] = line.strip()
        return lines

    @staticmethod
    def remove_diacritics(lines):
        for i in range(len(lines)):
            lines[i] = TextCleaner.replace(lines[i], TextCleaner.PATTERN_DIACRITICS)
        return lines

    @staticmethod
    def count_spaces(text):
        return len(TextCleaner.PATTERN_SPACES.findall(text))
