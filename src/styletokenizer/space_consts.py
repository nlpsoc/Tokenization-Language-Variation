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

# zero width
zero_width_no_break_space = "\uFEFF"
zero_width_space = "\u200B"
zero_width_non_joiner = "\u200C"
zero_width_joiner = "\u200D"


# get python ws: https://stackoverflow.com/questions/37903317/is-there-a-python-constant-for-unicode-whitespace
import re
import sys
s = ''.join(chr(c) for c in range(sys.maxunicode+1))
ws = ''.join(re.findall(r'\s', s))
