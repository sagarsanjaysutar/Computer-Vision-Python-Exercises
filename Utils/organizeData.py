import os
import shutil
import random
import glob

# This scripts makes a sub-set of dataset obtaining from cats-vs-dogs from kaggle.

os.chdir("../Datasets/")
if os.path.isdir("Training/dog") or os.path.isdir("Training/cat") is False:

    os.makedirs("Training/dog")
    os.makedirs("Training/cat")
    os.makedirs("Valid/dog")
    os.makedirs("Valid/cat")
    os.makedirs("Testing/dog")
    os.makedirs("Testing/cat")

    for i in random.sample(glob.glob('train/cat*'), 500):
        shutil.move(i, 'Training/cat')      
    for i in random.sample(glob.glob('train/dog*'), 500):
        shutil.move(i, 'Training/dog')
    for i in random.sample(glob.glob('train/cat*'), 100):
        shutil.move(i, 'Valid/cat')        
    for i in random.sample(glob.glob('train/dog*'), 100):
        shutil.move(i, 'Valid/dog')
    for i in random.sample(glob.glob('train/cat*'), 50):
        shutil.move(i, 'Testing/cat')      
    for i in random.sample(glob.glob('train/dog*'), 50):
        shutil.move(i, 'Testing/dog')
    print("Done")
    