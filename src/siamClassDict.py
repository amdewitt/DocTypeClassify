# The classes the model uses

# Classes dictionary, mapping values to keys, used for lookup generation
classes = {
    "other" : [""] # overflow class
    ,"form" : ["form"]
    ,"medicalRecord" : ["medical record", "medicalrecord"]
}

# (Helper Method) Generates the dictionary in reverse, to map values to the appropriate class
def __generateLookup():
    mappings = {}
    classArray = list(classes.keys())
    for key, values in classes.items():
        index = classArray.index(key)
        keyAtIndex = classArray[index]
        for name in values:
            mappings[name] = keyAtIndex
        mappings[index] = keyAtIndex
    return mappings

lookup = __generateLookup()

# Returns the length of the class list
def __len__():
    return len(classes.keys())

# Returns the index with the given class name
def __findClass__(value):
    for i in lookup.keys():
        if str(i).lower() == str(value).lower():
            return lookup[value]
    else:
        return lookup[""]

def __classFromIndex__(index):
    if int(index) < 0 or index >= __len__():
        return 0
    else:
        classArray = list(classes.keys())
        return classArray[index]

# Retuns the list of classes
def __printLookup__():
    print(lookup)