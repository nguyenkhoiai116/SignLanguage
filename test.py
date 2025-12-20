# tạo folder từ tên là a -> z vd data/a \\ data/b ...
import os
for char in range(ord('a'), ord('z') + 1):
    folder_name = f"data/{chr(char)}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")