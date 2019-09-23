import module as md
import os.path as op

i = 0
for filename in md.FileNames :
    while True:
        dirname = input(">>「" + md.ClassNames[i] + "」の画像のあるディレクトリ：")
        if op.isdir(dirname):
            break
        print(">> そのディレクトリは存在しません")

    md.PreProcess(dirname, filename, var_amount=1)
    i += 1