import re

# fmt: off
# Regex
dash_line = re.compile(r"\-{4,}|\={4,}")
table_entry_check = re.compile(r"([\w｜\.｜\+｜\-)]+)((?<!:)([^\S\n]{3,}|\t)+)|\-{4,}|\={4,}") # noqa
sentence_check = re.compile(r"(((?<!(z|\\|\))))(\=|(?:of))\s*((\d+(\.\d+)(?!(a|\:|G|d|\%|\"))))|(?:(\d+\s*\berg\b)|(\d+\s*\bmJy\b)))") # noqa E501
header_check = re.compile(r"(?<!\d)(\t+([()a-zA-Z']{1,})){2,}")


# new ones
date_check = re.compile(r"\d{2}\/\d{2}\/\d{2}|\d{4}-\d{2}-\d{2}")
mag_check = re.compile(r"((([a-z]?[a-z])\W?\n?\s?band \n?\s?magnitude \n?\s?was|([a-z]?[a-z])\W?\n?\s?band\n?\s?magnitude\n?\s? of|([a-z]?[a-z])\n?\s?\=?)\n?\s*(\d\d\.\d+)\s?\n?(\D\D\D)\s?\n?(\d?\.\d+))",  # noqa: E501
                       flags=re.IGNORECASE
                       )
flux_check = re.compile(r"", flags=re.IGNORECASE)
# labels
labels = []
