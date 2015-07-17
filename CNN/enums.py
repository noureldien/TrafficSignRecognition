import enum


class SuperclassType(enum.Enum):
    _00_All = 0
    _01_Prohibitory = 1
    _02_Warning = 2
    _03_Mandatory = 3


class ClassifierType(enum.Enum):
    logit = 1
    svm = 2
