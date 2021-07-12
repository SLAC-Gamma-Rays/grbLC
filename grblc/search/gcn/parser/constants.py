import re

# Regex
dash_line = re.compile(r"\-{4,}|\={4,}")
table_entry_check = re.compile(r"([\w｜\.｜\+｜\-)]+)((?<!:)([^\S\n]{3,}|\t)+)|\-{4,}|\={4,}")
sentence_check = re.compile(r"(((?<!(z|\\|\))))(\=|(?:of))\s*((\d+(\.\d+)(?!(a|\:|G|d|\%|\"))))|(?:(\d+\s*\berg\b)|(\d+\s*\bmJy\b)))")
header_check = re.compile(r"(?<!\d)(\t+([()a-zA-Z']{1,})){2,}")

# new ones
date_check = re.compile(r"\d{2}\/\d{2}\/\d{2}|\d{4}-\d{2}-\d{2}")

# labels
labels = []