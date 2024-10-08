import os
from os.path import isfile, join

mypath = "C:/Users/for_a/Downloads/spotdl/lyrics"
files = [f for f in os.listdir(mypath) if isfile(join(mypath, f))]
print(files)

counter = 0

for file in files:
    old_file = os.path.join(mypath, file)
    split = file.split(" - ")
    new_file = os.path.join(mypath + "new", split[1].replace(".lrc", "") + " - " + split[0] + ".lrc")
    os.rename(old_file, new_file)
    counter+=1
    print(counter)