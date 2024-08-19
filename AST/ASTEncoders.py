import json

import AST


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "toJSON"):
            return obj.toJSON()
        else:
            return json.__dict__


class ComplexDecoder(json.JSONDecoder):
    @staticmethod
    def object_hook(dictionary):
        if "class_name" in dictionary:
            if not hasattr(AST, dictionary["class_name"]):
                raise ValueError(f"Class {dictionary['class_name']} not found in AST")

            ast_class = getattr(AST, dictionary["class_name"])
            del dictionary["class_name"]
            try:
                return ast_class(**dictionary)
            except TypeError as e:
                raise e
        else:
            return dictionary
