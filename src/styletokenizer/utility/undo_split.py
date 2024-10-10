import re

def fix_sentence(sentence):
    # Remove spaces before punctuation
    sentence = re.sub(r'\s+([.,!?;:%])', r'\1', sentence)

    # Remove spaces before currency symbols
    sentence = re.sub(r'([$€£¥])\s+', r'\1', sentence)

    # Fix contractions and possessives
    sentence = re.sub(r"\b( 't| 's| 'm| 're| 've| 'd| 'll)\b", lambda x: x.group().replace(' ', ''), sentence)


    # Process quotes one by one, keeping track of whether they are opening or closing
    result = []
    inside_quote = False  # Flag to track if we are inside a quote pair

    i = 0
    while i < len(sentence):
        char = sentence[i]

        if char == '"':
            if inside_quote:
                # Closing quote: remove any space before it
                if result and result[-1] == ' ':
                    result.pop()
                result.append('"')
                inside_quote = False
            else:
                # Opening quote: remove any space after it
                result.append('"')
                inside_quote = True
                i += 1
                while i < len(sentence) and sentence[i] == ' ':
                    i += 1
                continue  # Skip incrementing i at the end to avoid doubling it
        else:
            result.append(char)

        i += 1

    return ''.join(result).strip()

# List of example sentences to be corrected
sentences = [
    "He said the foodservice pie business doesn 't fit the company 's long-term growth strategy .",
    'His wife said he was " 100 percent behind George Bush " and looked forward to using his years of training in the war .',
    'The dollar was at 116.78 yen JPY = , virtually flat on the session , and at 1.2871 against the Swiss franc CHF = , down 0.1 percent .',
    "This integrates with Rational PurifyPlus and allows developers to work in supported versions of Java , Visual C # and Visual Basic .NET.",
    "The top rate will go to 4.45 percent for all residents with taxable incomes above $ 500,000 ."
]

# Applying the function to each sentence
fixed_sentences = [fix_sentence(sentence) for sentence in sentences]

# Display corrected sentences
for i, fixed_sentence in enumerate(fixed_sentences, 1):
    print(f"Original: {sentences[i-1]}")
    print(f"Fixed: {fixed_sentence}")
    print()