"""
AST.AST.py
==========

Description
-----------
Basic AST nodes.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class AST:
    """
    AST: (children)
    """

    @classmethod
    def __all_annotations(cls) -> Dict[str, Any]:
        class_attributes = {}
        for c in cls.__mro__:
            if "__annotations__" in c.__dict__:
                class_attributes.update(c.__annotations__)
        return dict(reversed(list(class_attributes.items())))

    def __str__(self) -> str:
        """Returns a human-friendly description."""
        out = []
        name = self.__class__.__name__
        for k in self.__all_annotations().keys():
            v = self.__getattribute__(k)
            if v:
                out.append(f"{k}={str(v)}")
        return f"<{name}({', '.join(out)})>"

    def toJSON(self):
        base = {
            'class_name': self.__class__.__name__
        }
        for k in self.__all_annotations().keys():
            v = self.__getattribute__(k)
            base[k] = v
        return base

    __repr__ = __str__


@dataclass
class Start(AST):
    """
    Start: (children)
    """

    children: List[AST]


@dataclass
class Assignment(AST):
    """
    Assignment: (left, op, right)
    """

    left: AST
    op: str
    right: AST


@dataclass
class PersistentAssignment(AST):
    """
    PersistentAssignment: (left, op, right)
    """

    left: AST
    op: str
    right: AST


@dataclass
class VarID(AST):
    """
    VarID: (value)
    The Var node is constructed out of ID token.
    Could be: DATASET or a COMPONENT.
    """
    value: Any


@dataclass
class UnaryOp(AST):
    """
    UnaryOp: (op, operand)
    op types: "+", "-", CEIL, FLOOR, ABS, EXP, LN, SQRT, LEN, TRIM, LTRIM, RTRIM, UPCASE, LCASE,
              ISNULL, FLOW_TO_STOCK, STOCK_TO_FLOW, SUM, AVG, COUNT, MEDIAN, MAX,
              RANK, STDDEV_POP, STDDEV_SAMP , VAR_POP, VAR_SAMP
    """

    op: str
    operand: AST


@dataclass
class BinOp(AST):
    """
    BinOp: (left, op, right)
    op types: "+", "-", "*", "/",MOD, MEMBERSHIP, PIVOT, UNPIVOT, LOG, POWER, CHARSET_MATCH, NVL, MOD
    """

    left: AST
    op: str
    right: AST


@dataclass
class MulOp(AST):
    """
    MulOp: (op, children)
    op types: BETWEEN, GROUP BY, GROUP EXCEPT, GROUP ALL
    """

    op: str
    children: List[AST]


@dataclass
class ParamOp(AST):
    """
    ParamOp: (op, children, params)

    op types: ROUND, TRUNC, SUBSTR, INSTR, REPLACE
    """

    op: str
    children: List[AST]
    params: List[AST]


@dataclass
class JoinOp(AST):
    """
    JoinOp: (op, clauses, using)

    op types: INNER_JOIN, LEFT_JOIN, FULL_JOIN, CROSS_JOIN.
    clauses types:
    body types:
    """

    op: str
    clauses: List[AST]
    using: Optional[List[AST]]
    isLast: bool = False


@dataclass
class Constant(AST):
    """
    Constant: (type, value)

    types: INTEGER_CONSTANT, FLOAT_CONSTANT, BOOLEAN_CONSTANT,
           STRING_CONSTANT, NULL_CONSTANT
    """

    type_: str
    value: Optional[Union[str, int, float, bool]]


@dataclass
class ParamConstant(Constant):
    """
    Constant: (type, value)

    types: ALL
    """

    type_: str
    value: str


@dataclass
class Identifier(AST):
    """
    Identifier: (value)
    """

    value: str
    kind: str


@dataclass
class ID(AST):
    """
    ID: (type, value)

    for a general purpose Terminal nodes Node
    """

    type_: str
    value: str


@dataclass
class Role(AST):
    """
    Role: (role)

    roles: MEASURE, COMPONENT, DIMENSION, ATTRIBUTE, VIRAL ATTRIBUTE
    """

    role: str


@dataclass
class Collection(AST):
    """
    Collection: (name, type, children)

    types: Sets and ValueDomains
    """

    name: str
    type: str
    children: List[AST]
    kind: str = 'Set'


@dataclass
class Analytic(AST):
    """
    Analytic: (op, operand, partition_by, order_by, params)

    op: SUM, AVG, COUNT, MEDIAN, MIN, MAX, STDDEV_POP, STDDEV_SAMP, VAR_POP, VAR_SAMP, FIRST_VALUE, LAST_VALUE, LAG,
        LEAD, RATIO_TO_REPORT

    partition_by: List of components.
    order_by: List of components + mode (ASC, DESC).
    params: Windowing clause (no need to validate them) or Scalar Item in LAG/LEAD.
    """
    op: str
    operand: AST
    partition_by: List[AST]
    order_by: List[AST]
    params: List[AST]


@dataclass
class OrderBy(AST):
    component: str
    order: Optional[str]


@dataclass
class Windowing(AST):
    """
    Windowing: (type, first, second, first_mode, second_mode)

    type: RANGE, ROWS, GROUPS
    first: int
    second: int
    first_mode: int
    second_mode: int
    """

    type_: str
    start: int
    start_mode: int
    stop: int
    stop_mode: int


@dataclass
class RegularAggregation(AST):
    """
    RegularAggregation: (dataset, op, children)

    op types: FILTER, CALC, KEEP, DROP, RENAME, SUBSPACE
    """

    op: str
    children: List[AST]
    dataset: Optional[AST] = None
    isLast: bool = False


@dataclass
class RenameNode(AST):
    """
    RenameNode: (old_name, new_name)
    """

    old_name: str
    new_name: str


@dataclass
class Aggregation(AST):
    """
    Aggregation: (op, operand, grouping_op, grouping)

    op types: AGGREGATE, SUM, AVG , COUNT, MEDIAN, MIN, MAX, STDDEV_POP, STDDEV_SAMP,
              VAR_POP, VAR_SAMP

    grouping types: 'group by', 'group except', 'group all'.
    """
    op: str
    operand: Optional[AST] = None
    grouping_op: Optional[str] = None
    grouping: Optional[List[AST]] = None
    having_clause: Optional[AST] = None
    param: Optional[str] = None


@dataclass
class TimeAggregation(AST):
    """
    TimeAggregation: (op, operand, params, conf)

    op types: TIME_AGG
    """

    op: str
    operand: Optional[AST]
    params: List[AST]
    conf: Optional[str]


@dataclass
class If(AST):
    """
    If: (condition, thenOp, elseOp)
    """
    condition: AST
    thenOp: AST
    elseOp: AST


@dataclass
class Validation(AST):
    """
    Validation: (op, validation, params, inbalance, invalid)
    """

    op: str
    validation: str
    params: List[AST]
    inbalance: Optional[AST]
    invalid: Optional[AST]


@dataclass
class Operator(AST):
    """
    Operator: (operator, parameters, outputType, expression)
    """

    op: str
    parameters: list
    outputType: str
    expression: AST


# TODO: Is this class necessary?
@dataclass
class DefIdentifier(AST):
    """
    DefIdentifier: (value, kind)
    """
    value: str
    kind: str


@dataclass
class DPRIdentifier(AST):
    """
    DefIdentifier: (value, kind, alias)
    """
    value: str
    kind: str
    alias: Optional[str]


@dataclass
class Types(AST):
    """
    Types: (name, kind, type_, constraints, nullable)

    kind:
            - basicScalarType
                - STRING, INTEGER, NUMBER, BOOLEAN, DATE, TIME_PERIOD, DURATION, SCALAR, TIME.
            -
    """

    name: Optional[str]
    kind: str
    type_: str
    constraints: Optional[Dict[str, Any]]
    nullable: bool


@dataclass
class Argument(AST):
    """
    Argument: (name, type_, default)
    """
    name: str
    type_: str
    default: Optional[AST]


# TODO: Are HRBinOp and HRUnOp necessary?
@dataclass
class HRBinOp(AST):
    """
    HRBinOp: (left, op, right)
    op types: '+','-', '=', '>', '<', '>=', '<='.
    """
    left: DefIdentifier
    op: str
    right: DefIdentifier


@dataclass
class HRUnOp(AST):
    """
    HRUnOp: (op, operand)
    op types: '+','-'.
    """

    op: str
    operand: DefIdentifier


# TODO: Unify HRule and DPRule?
class HRule(AST):
    """
    HRule: (name, rule, erCode, erLevel)
    """

    name: Optional[str]
    rule: HRBinOp
    erCode: Optional[Constant]
    erLevel: Optional[Constant]


@dataclass
class DPRule(AST):
    """
    DPRule: (name, rule, erCode, erLevel)
    """

    name: Optional[str]
    rule: HRBinOp
    erCode: Optional[Constant]
    erLevel: Optional[Constant]


# TODO: Unify HRuleset and DPRuleset?
@dataclass
class HRuleset(AST):
    """
    HRuleset: (name, element, rules)
    """

    name: str
    element: DefIdentifier
    rules: List[HRule]


@dataclass
class DPRuleset(AST):
    """
    DPRuleset: (name, element, rules)
    """

    name: str
    element: Union[DefIdentifier, list]
    rules: List[DPRule]


@dataclass
class EvalOp(AST):
    """
    EvalOp: (name, children, output, language)

    op types:
    """

    name: str
    children: List[AST]
    output: Optional[str]
    language: Optional[str]

    def __init__(self, name, children, output, language):
        super().__init__()
        self.name = name
        self.children = children
        self.output = output
        self.language = language


@dataclass
class NoOp(AST):
    """
    NoOp: ()
    """
    pass
