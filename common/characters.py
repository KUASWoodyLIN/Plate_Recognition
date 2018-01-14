import string

ALL_LABELS = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9'.split()
characters = string.digits + string.ascii_uppercase
id2name = {i: name for i, name in enumerate(characters)}
name2id = {name: i for i, name in id2name.items()}


## 翻译字符到整型，整型到字符的两个函数
## 因为keras的输出只接受整型

# Translation of characters to unique integer values
def text_to_labels(text):
    ret = []
    for char in text:
        ret.append(characters.find(char))
    return ret


# Reverse translation of numerical classes back to characters
def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(characters):  # CTC Blank
            ret.append("")
        else:
            ret.append(characters[c])
    return "".join(ret)


if __name__ == '__main__':
  labels = text_to_labels('ASV1234')
  print(labels)
  text = labels_to_text(labels)
  print(text)
  print('Exit')
