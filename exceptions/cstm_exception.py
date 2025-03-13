"""
this file containing the defination of custome exceptions
"""
class InvalidDataFrame(Exception):
    """this exception is basically raised when dataset not being loaded from path"""

    def __init__(self, msg:str="Unable to load DataFrame from Given Path"): 
        super().__init__(msg)