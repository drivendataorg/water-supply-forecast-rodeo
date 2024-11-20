import os
import glob

files = glob.glob("runs/**/*p50.bin", recursive=True)
removed_files = [x for x in files if "reg" not in x]
len(removed_files)
for file in removed_files:
    os.remove(file)
    print(f"{file} is removed!")

files = glob.glob("runs/main/*/*/preds/**.csv", recursive=True)
for file in files:
    os.remove(file)

files = glob.glob("runs/main/*/*/evals/**.csv", recursive=True)
for file in files:
    os.remove(file)
