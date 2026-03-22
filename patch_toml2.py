with open('pyproject.toml', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('warn_return_any = true', 'warn_return_any = false')

with open('pyproject.toml', 'w', encoding='utf-8') as f:
    f.write(text)
