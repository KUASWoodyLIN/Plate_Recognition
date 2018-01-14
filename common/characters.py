import string

ALL_LABELS = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z 0 1 2 3 4 5 6 7 8 9'.split()
characters = string.digits + string.ascii_uppercase
id2name = {i: name for i, name in enumerate(characters)}
name2id = {name: i for i, name in id2name.items()}
