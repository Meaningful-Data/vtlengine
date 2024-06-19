import json

from AST import Start, Assignment, PersistentAssignment, VarID, UnaryOp, BinOp, MulOp, ParamOp, JoinOp, \
    Constant, ParamConstant, Identifier, Optional, ID, Role, Collection, Analytic, OrderBy, Windowing, \
    RegularAggregation, Aggregation, TimeAggregation, If, Validation, Operator, DefIdentifier, DPRIdentifier, Types, \
    Argument, HRBinOp, HRUnOp, HRule, DPRule, HRuleset, DPRuleset, EvalOp

AST_Classes = {

    'Start': Start,
    'Assignment': Assignment,
    'PersistentAssignment': PersistentAssignment,
    'VarID': VarID,
    'UnaryOp': UnaryOp,
    'BinOp': BinOp,
    'MulOp': MulOp,
    'ParamOp': ParamOp,
    'JoinOp': JoinOp,
    'Constant': Constant,
    'ParamConstant': ParamConstant,
    'Identifier': Identifier,
    'Optional': Optional,
    'ID': ID,
    'Role': Role,
    'Collection': Collection,
    'Analytic': Analytic,
    'OrderBy': OrderBy,
    'Windowing': Windowing,
    'RegularAggregation': RegularAggregation,
    'Aggregation': Aggregation,
    'TimeAggregation': TimeAggregation,
    'If': If,
    'Validation': Validation,
    'Operator': Operator,
    'DefIdentifier': DefIdentifier,
    'DPRIdentifier': DPRIdentifier,
    'Types': Types,
    'Argument': Argument,
    'HRBinOp': HRBinOp,
    'HRUnOp': HRUnOp,
    'HRule': HRule,
    'DPRule': DPRule,
    'HRuleset': HRuleset,
    'DPRuleset': DPRuleset,
    'EvalOp': EvalOp
}


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'toJSON'):
            return obj.toJSON()
        else:
            return json.__dict__


class ComplexDecoder(json.JSONDecoder):
    @staticmethod
    def object_hook(dictionary):
        if "class_name" in dictionary:
            name = dictionary["class_name"]
            del dictionary["class_name"]
            try:
                return AST_Classes[name](**dictionary)
            except TypeError as e:
                print(name)
                raise e
        else:
            return dictionary