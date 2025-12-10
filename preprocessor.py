import re
import textwrap


class TextPreprocessor:
    def __init__(self, cleaner, input_path='.', output_path='.', max_length=600, with_labels=True):
        self.cleaner = cleaner
        self.input_path = input_path
        self.output_path = output_path
        self.max_length = max_length
        self.with_labels = with_labels

    def clean_lines(self, lines, data_type):
        lines = self.cleaner.clean_lines(lines)

        if not lines:
            return lines

        
        if self.with_labels:
            with open(f'{self.output_path}cleaned_{data_type}_with_diacritics.txt', 'a+', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')

        
        lines_no_diacritics = self.cleaner.remove_diacritics(lines)

        
        with open(f'{self.output_path}cleaned_{data_type}_without_diacritics.txt', 'a+', encoding='utf-8') as f:
            f.write('\n'.join(lines_no_diacritics) + '\n')

        return lines_no_diacritics

    def preprocess_file(self, data_type, limit=None):
        
        open(f'{self.output_path}cleaned_{data_type}_with_diacritics.txt', 'w', encoding='utf-8').close()
        open(f'{self.output_path}cleaned_{data_type}_without_diacritics.txt', 'w', encoding='utf-8').close()

        
        with open(f'{self.input_path}{data_type}.txt', 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]

        if limit is not None:
            lines = lines[:limit]

        return self.clean_lines(lines, data_type)

    def tokenize_file(self, data_type):
        tokenized_no_diacritics = []
        space_counts = []

        
        with open(f'{self.output_path}cleaned_{data_type}_without_diacritics.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = re.sub(r'[\n\r\t]', '', line)
                line = re.sub(r'\s+', ' ', line).strip()
                sentences = textwrap.wrap(line, self.max_length)
                for sentence in sentences:
                    tokenized_no_diacritics.append(sentence)
                    space_counts.append(self.cleaner.count_spaces(sentence))

        tokenized_with_diacritics = []

        if self.with_labels:
            space_index = 0
            with open(f'{self.output_path}cleaned_{data_type}_with_diacritics.txt', 'r', encoding='utf-8') as f:
                for line in f:
                    line = re.sub(r'[\n\r\t]', '', line)
                    line = re.sub(r'\s+', ' ', line).strip()
                    remaining_text = line
                    while remaining_text:
                        spaces_to_include = space_counts[space_index]
                        space_index += 1
                        words = remaining_text.split()
                        if len(words) <= spaces_to_include + 1:
                            tokenized_with_diacritics.append(remaining_text.strip())
                            break
                        sentence = ' '.join(words[:spaces_to_include + 1])
                        tokenized_with_diacritics.append(sentence.strip())
                        remaining_text = ' '.join(words[spaces_to_include + 1:]).strip()

        return tokenized_no_diacritics, tokenized_with_diacritics
