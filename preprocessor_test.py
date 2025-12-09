import re
import textwrap


class TextPreprocessor:
    CLEAN_NEWLINES = re.compile(r'[\n\r\t]')
    CLEAN_SPACES = re.compile(r'\s+')

    def __init__(self, cleaner, input_path='.', output_path='.', max_length=600, with_labels=True):
        self.cleaner = cleaner
        self.input_path = input_path
        self.output_path = output_path
        self.max_length = max_length
        self.with_labels = with_labels

    def _write(self, filename, lines, mode='a+'):
        with open(filename, mode, encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')

    def clean_lines(self, lines, data_type):
        lines = self.cleaner.clean_lines(lines)
        if not lines:
            return lines

        # Save lines with diacritics
        if self.with_labels:
            self._write(f'{self.output_path}cleaned_{data_type}_with_diacritics.txt', lines)

        # Remove diacritics
        lines_no_diacritics = self.cleaner.remove_diacritics(lines)

        # Save lines without diacritics
        self._write(f'{self.output_path}cleaned_{data_type}_without_diacritics.txt', lines_no_diacritics)

        return lines_no_diacritics

    def preprocess_file(self, data_type, limit=None):
        # Reset outputs
        self._write(f'{self.output_path}cleaned_{data_type}_with_diacritics.txt', [], mode='w')
        self._write(f'{self.output_path}cleaned_{data_type}_without_diacritics.txt', [], mode='w')

        # Read input file
        with open(f'{self.input_path}{data_type}.txt', 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f]

        if limit is not None:
            lines = lines[:limit]

        return self.clean_lines(lines, data_type)

    def _clean_line(self, line):
        line = self.CLEAN_NEWLINES.sub('', line)
        line = self.CLEAN_SPACES.sub(' ', line)
        return line.strip()

    def tokenize_file(self, data_type):
        tokenized_no_diacritics = []
        space_counts = []

        # Read no-diacritics cleaned file
        with open(f'{self.output_path}cleaned_{data_type}_without_diacritics.txt', 'r', encoding='utf-8') as f:
            for raw_line in f:
                line = self._clean_line(raw_line)
                sentences = textwrap.wrap(line, self.max_length)
                for sentence in sentences:
                    tokenized_no_diacritics.append(sentence)
                    space_counts.append(self.cleaner.count_spaces(sentence))

        tokenized_with_diacritics = []

        if self.with_labels:
            space_index = 0

            # Read with-diacritics cleaned file
            with open(f'{self.output_path}cleaned_{data_type}_with_diacritics.txt', 'r', encoding='utf-8') as f:
                for raw_line in f:
                    line = self._clean_line(raw_line)
                    remaining = line

                    while remaining:
                        spaces_to_take = space_counts[space_index]
                        space_index += 1

                        words = remaining.split()

                        if len(words) <= spaces_to_take + 1:
                            tokenized_with_diacritics.append(remaining)
                            break

                        sentence = ' '.join(words[:spaces_to_take + 1])
                        tokenized_with_diacritics.append(sentence)

                        remaining = ' '.join(words[spaces_to_take + 1:]).strip()

        return tokenized_no_diacritics, tokenized_with_diacritics
