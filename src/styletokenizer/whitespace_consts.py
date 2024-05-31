# TODO: list not yet complete
tab = "\u0009"
vertical_tab = "\u000B"
form_feed = "\u000C"
carriage_return = "\u000D"
newline = "\u000A"
space = "\u0020"
next_line = "\u0085"
non_breaking_space = "\u00A0"
ogham_space_mark = "\u1680"
mongolian_vowel_separator = "\u180E"
en_quad = "\u2000"
em_quad = "\u2001"
en_space = "\u2002"
em_space = "\u2003"
three_per_em_space = "\u2004"
four_per_em_space = "\u2005"
six_per_em_space = "\u2006"
figure_space = "\u2007"
punctuation_space = "\u2008"
thin_space = "\u2009"
hair_space = "\u200A"
line_separator = "\u2028"
paragraph_separator = "\u2029"
narrow_no_break_space = "\u202F"
medium_mathematical_space = "\u205F"
ideographic_space = "\u3000"
UNICODE_WHITESPACE = [tab, vertical_tab, form_feed, carriage_return, newline, space, next_line, non_breaking_space,
                      ogham_space_mark, mongolian_vowel_separator, en_quad, em_quad, en_space, em_space,
                      three_per_em_space, four_per_em_space, six_per_em_space, figure_space, punctuation_space,
                      thin_space, hair_space, line_separator, paragraph_separator, narrow_no_break_space,
                      medium_mathematical_space, ideographic_space]
COMMON_WHITESPACE = [tab, newline, space, non_breaking_space, ideographic_space]
WHITESPACE_PATTERN = f"[{''.join(UNICODE_WHITESPACE)}]"

# zero width
zero_width_no_break_space = "\uFEFF"
zero_width_space = "\u200B"
zero_width_non_joiner = "\u200C"
zero_width_joiner = "\u200D"

# apostrophe encodings
apostrophe = "\u0027"  # general purpose
right_single_quotation_mark = "\u2019"  # typographic apostrophe in typeset text, preferred in word processors (e.g., Word)
modifier_letter_apostrophe = "\u02BC"
left_single_quotation_mark = "\u2018"
prime = "\u2032"  # denotes feet and arcminutes
reversed_prime = "\u2035"  # alternate for prime
modifier_letter_turned_comma = "\u02BB"  # glottal stops in Hawaiian and other Polynesian languages
right_half_ring = "\u02BE"
modifier_letter_prime = "\u02B9"  # used in transliteration systems for Semitic languages
modifier_letter_reverse_prime = "\u02BD"  # used in phonetic transcription
APOSTROPHES = [apostrophe, right_single_quotation_mark, modifier_letter_apostrophe, left_single_quotation_mark, prime,
               right_half_ring, modifier_letter_turned_comma, reversed_prime, modifier_letter_prime,
               modifier_letter_reverse_prime]
COMMON_APOSTROPHE = [apostrophe, right_single_quotation_mark, left_single_quotation_mark]
APOSTROPHE_PATTERN = f"[{''.join(APOSTROPHES)}]"

# get python ws: https://stackoverflow.com/questions/37903317/is-there-a-python-constant-for-unicode-whitespace
import re
import sys

s = ''.join(chr(c) for c in range(sys.maxunicode + 1))
ws = ''.join(re.findall(r'\s', s))


def get_ws_types(text):
    """
        for a given text, return the count of the common whitespace characters: \t, \n, \r, " ", \u00a0 and \u3000
    :param text:
    :return:
    """
    ws_counts = {char: text.count(char) for char in COMMON_WHITESPACE}
    ws_vec = [ws_counts[char] for char in COMMON_WHITESPACE]
    return ws_vec, COMMON_WHITESPACE


def common_ws_tokenize(text):
    """
        given a text, only return the common whitespace characters: \t, \n, \r, " ", \u00a0 and \u3000 in a list
    :param text:
    :return:
    """
    return [char for char in text if char in COMMON_WHITESPACE]


def common_apostrophe_tokenize(text):
    """
        given a text, only return the common apostrophe characters: ', `, ´ in a list
    :param text:
    :return:
    """
    return [char for char in text if char in COMMON_APOSTROPHE]
