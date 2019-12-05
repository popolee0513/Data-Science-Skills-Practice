
# coding: utf-8

# In[1]:


import os

def get_words_in_file(filename):
    """ Returns a list of all words in the file at filename. """
    with open(filename, 'r',encoding="utf-8") as f:
        # read() reads in a string from a file pointer, and split() splits a
        # string into words based on whitespace
        words = f.read().split()
    return words

def get_files_in_folder(folder):
    """ Returns a list of files in folder (including the path to the file) """
    filenames = os.listdir(folder)
    # os.path.join combines paths while dealing with /s and \s appropriately
    full_filenames = [os.path.join(folder, filename) for filename in filenames]
    return full_filenames

