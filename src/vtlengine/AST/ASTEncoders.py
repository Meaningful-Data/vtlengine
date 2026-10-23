import json

from vtlengine import AST
from vtlengine.Exceptions import InputValidationException
from vtlengine.Model import Dataset


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "toJSON"):
            return obj.toJSON()
        # Makes a circular reference error if we do not check for this
        elif isinstance(obj, Dataset):
            return "dataset"
        else:
            return json.__dict__


class ComplexDecoder(json.JSONDecoder):
    @staticmethod
    def object_hook(dictionary):
        if "class_name" in dictionary:
            # A JSON AST is an input: what it names must exist in the AST (issue #1192)
            name = dictionary["class_name"]
            if not hasattr(AST, name):
                raise InputValidationException(code="0-5-1-11", name=name)

            ast_class = getattr(AST, name)
            del dictionary["class_name"]
            try:
                return ast_class(**dictionary)
            except TypeError as error:
                raise InputValidationException(code="0-5-1-12", name=name, error=error) from error
        else:
            return dictionary
