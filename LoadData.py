def read_file(filepath, mode = 'r'): # rb for raw binary
    f = open(filepath, mode)# encoding = encoding)
    file_content = ""
    try:
        file_content = f.read()
    finally:
        f.close()
    return file_content