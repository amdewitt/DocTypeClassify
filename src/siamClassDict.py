# The classes the model uses

# Imports
import re

# Classes dictionary, mapping values to keys, used for lookup generation
classes = {
    "other" : [] # overflow class
    ,"form" : ["form"]
    ,"medicalRecord" : ["medical record", "record"]
}

# (Helper Method) Generates the dictionary in reverse, to map values to the appropriate class
def __generateLookup()
    mappings = {}
    nameList = []

    classArray = list(classes.keys())
    for key, values in classes.items():
        index = classArray.index(key)
        nameList.extend(values)
        for name in values:
               mappings[name] = index
    return mappings

lookup = __generateLookup()

# Retuns the list of classes
def __getClasses__():
    return list(classes.keys())

# Returns the index with the given class name
def __findClassFromValue__(className):
    pass

    # Returns the length of the class list
def __len__():
    return len(classes.keys())