# The classes the model uses

import re

# Classes array, used when mapping an index to a class
classes = [
    "other", # Overflow class - model puts unidentifiable images in this class
    "form", # Classes
    "record"
]
class SiameseClassDictionary():

    def __getClassFromIndex__(index):
        i = int(index)
        if(i < 0 or i >= len(classes)):
            return classes[0]
        else:
            return classes[i]

    def __getIndexFromClass__(className):
        for i in classes:
            if(classes[i].__eq__(className)):
                return i
            else:
                return 0

    def len():
        return len(classes)
        