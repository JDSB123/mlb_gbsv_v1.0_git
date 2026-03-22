import re

with open('pyproject.toml', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('strict = true', 'strict = false\ndisallow_any_generics = false\ndisallow_untyped_defs = false')

with open('pyproject.toml', 'w', encoding='utf-8') as f:
    f.write(text)
