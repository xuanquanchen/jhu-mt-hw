# Data Directory

Place your training, validation, and test data files here.

## File Format

Each line should be: `word punctuation`

Example (`train`, `valid`, `test` files):
```
你好 O
世界 ，
今天 O
天气 O
很好 。
```

## Required Files

- `train` - Training data
- `valid` - Validation data  
- `test` - Test data (for evaluation)

## Supported Punctuation

### Standard Set (V1, V2):
- `O` - No punctuation
- `，` - Comma (COMMA)
- `。` - Period (PERIOD)
- `？` - Question mark (QUESTION)
- `！` - Exclamation mark (EXCLAMATION)
- `；` - Semicolon (SEMICOLON)
- `、` - Enumeration comma (ENUMERATION)

### Expanded Set (V3):
All standard punctuation marks plus:
- `《` - Left book title mark (LEFT_BOOK_TITLE)
- `》` - Right book title mark (RIGHT_BOOK_TITLE)

**Note:** V3 supports 9 classes total (O + 8 punctuation marks)

## Data Source

You can use data from:
- IWSLT Chinese dataset
- Chinese Wikipedia
- Your own Chinese text corpus

Make sure to format it as shown above (one word + punctuation per line).

