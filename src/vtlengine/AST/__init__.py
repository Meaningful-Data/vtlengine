"""
AST.AST.py
==========

Description
-----------
Basic AST nodes.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union

from vtlengine.DataTypes import ScalarType
from vtlengine.Model import Dataset, Role


@dataclass
class AST:
    """
    AST: (children)
    """

    line_start: int
    column_start: int
    line_stop: int
    column_stop: int

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
        for k in self.__all_annotations():
            v = self.__getattribute__(k)
            if v:
                out.append(f"{k}={str(v)}")
        return f"<{name}({', '.join(out)})>"

    def toJSON(self):
        base = {"class_name": self.__class__.__name__}
        for k in self.__all_annotations():
            v = self.__getattribute__(k)
            base[k] = v
        return base

    __repr__ = __str__

    def ast_equality(self, other):
        if not isinstance(other, self.__class__):
            return False
        for k in self.__all_annotations():
            if (
                getattr(self, k) != getattr(other, k)
                and k not in AST.__annotations__
                and k != "children"  # We do not want to compare the children order here
            ):
                return False
        return True

    __eq__ = ast_equality


@dataclass
class Comment(AST):
    """
    Comment: (value)
    """

    value: str
    __eq__ = AST.ast_equality


@dataclass
class Start(AST):
    """
    Start: (children)
    """

    children: List[AST]

    __eq__ = AST.ast_equality


@dataclass
class Assignment(AST):
    """
    Assignment: (left, op, right)
    """

    left: AST
    op: str
    right: AST

    __eq__ = AST.ast_equality


@dataclass
class PersistentAssignment(Assignment):
    """
    PersistentAssignment: (left, op, right)
    """

    pass


@dataclass
class VarID(AST):
    """
    VarID: (value)
    The Var node is constructed out of ID token.
    Could be: DATASET or a COMPONENT.
    """

    value: str

    __eq__ = AST.ast_equality


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

    __eq__ = AST.ast_equality


@dataclass
class BinOp(AST):
    """
    BinOp: (left, op, right)
    op types: "+", "-", "*", "/",MOD, MEMBERSHIP, PIVOT, UNPIVOT, LOG,
    POWER, CHARSET_MATCH, NVL, MOD
    """

    left: AST
    op: str
    right: AST

    __eq__ = AST.ast_equality


@dataclass
class MulOp(AST):
    """
    MulOp: (op, children)
    op types: BETWEEN, GROUP BY, GROUP EXCEPT, GROUP ALL
    """

    op: str
    children: List[AST]

    __eq__ = AST.ast_equality


@dataclass
class ParamOp(AST):
    """
    ParamOp: (op, children, params)

    op types: ROUND, TRUNC, SUBSTR, INSTR, REPLACE
    """

    op: str
    children: List[AST]
    params: List[AST]

    __eq__ = AST.ast_equality


@dataclass
class UDOCall(AST):
    op: str
    params: List[AST]

    __eq__ = AST.ast_equality


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
    using: Optional[List[str]]
    isLast: bool = False

    __eq__ = AST.ast_equality


@dataclass
class Constant(AST):
    """
    Constant: (type, value)

    types: INTEGER_CONSTANT, FLOAT_CONSTANT, BOOLEAN_CONSTANT,
           STRING_CONSTANT, NULL_CONSTANT
    """

    type_: str
    value: Optional[Union[str, int, float, bool]]

    __eq__ = AST.ast_equality


@dataclass
class ParamConstant(Constant):
    """
    Constant: (type, value)

    types: ALL
    """

    type_: str
    value: str

    __eq__ = AST.ast_equality


@dataclass
class Identifier(AST):
    """
    Identifier: (value)
    """

    value: str
    kind: str

    __eq__ = AST.ast_equality


@dataclass
class ID(AST):
    """
    ID: (type, value)

    for a general purpose Terminal nodes Node
    """

    type_: str
    value: str

    __eq__ = AST.ast_equality


@dataclass
class Collection(AST):
    """
    Collection: (name, type, children)

    types: Sets and ValueDomains
    """

    name: str
    type: str
    children: List[AST]
    kind: str = "Set"

    __eq__ = AST.ast_equality


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
    start: Union[int, str]
    start_mode: str
    stop: Union[int, str]
    stop_mode: str

    __eq__ = AST.ast_equality


@dataclass
class OrderBy(AST):
    component: str
    order: str

    def __post_init__(self):
        if self.order not in ["asc", "desc"]:
            raise ValueError(f"Invalid order: {self.order}")

    __eq__ = AST.ast_equality


@dataclass
class Analytic(AST):
    """
    Analytic: (op, operand, partition_by, order_by, params)

    op: SUM, AVG, COUNT, MEDIAN, MIN, MAX, STDDEV_POP, STDDEV_SAMP, VAR_POP, VAR_SAMP,
        FIRST_VALUE, LAST_VALUE, LAG, LEAD, RATIO_TO_REPORT

    partition_by: List of components.
    order_by: List of components + mode (ASC, DESC).
    params: Windowing clause (no need to validate them) or Scalar Item in LAG/LEAD.
    """

    op: str
    operand: Optional[AST]
    window: Optional[Windowing] = None
    params: Optional[List[int]] = None
    partition_by: Optional[List[str]] = None
    order_by: Optional[List[OrderBy]] = None

    def __post_init__(self):
        if self.partition_by is None and self.order_by is None:
            raise ValueError("Partition by or order by must be provided on Analytic.")

    __eq__ = AST.ast_equality


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

    __eq__ = AST.ast_equality


@dataclass
class RenameNode(AST):
    """
    RenameNode: (old_name, new_name)
    """

    old_name: str
    new_name: str

    __eq__ = AST.ast_equality


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

    __eq__ = AST.ast_equality


@dataclass
class TimeAggregation(AST):
    """
    TimeAggregation: (op, operand, params, conf)

    op types: TIME_AGG
    """

    op: str
    period_to: str
    period_from: Optional[str] = None
    operand: Optional[AST] = None
    conf: Optional[str] = None

    __eq__ = AST.ast_equality


@dataclass
class If(AST):
    """
    If: (condition, thenOp, elseOp)
    """

    condition: AST
    thenOp: AST
    elseOp: AST

    __eq__ = AST.ast_equality


@dataclass
class CaseObj(AST):
    condition: AST
    thenOp: AST


@dataclass
class Case(AST):
    """
    Case: (condition, thenOp, elseOp)
    """

    cases: List[CaseObj]
    elseOp: AST

    __eq__ = AST.ast_equality


@dataclass
class Validation(AST):
    """
    Validation: (op, validation, error_code, error_level, imbalance, invalid)
    """

    op: str
    validation: str
    error_code: Optional[str]
    error_level: Optional[int]
    imbalance: Optional[AST]
    invalid: bool

    __eq__ = AST.ast_equality


@dataclass
class ComponentType(AST):
    """
    ComponentType: (data_type, role)
    """

    name: str
    data_type: Optional[Type[ScalarType]] = None
    role: Optional[Role] = None

    __eq__ = AST.ast_equality


@dataclass
class ASTScalarType(AST):
    data_type: Type[ScalarType]


@dataclass
class DatasetType(AST):
    """
    DatasetType: (name, components)
    """

    components: List[ComponentType]

    __eq__ = AST.ast_equality


@dataclass
class Types(AST):
    """
    Types: (name, kind, type_, constraints, nullable)

    kind:
            - basicScalarType
                - STRING, INTEGER, NUMBER, BOOLEAN, DATE, TIME_PERIOD, DURATION, SCALAR, TIME.
            -
    """

    kind: str
    type_: str
    constraints: List[AST]
    nullable: Optional[bool]
    name: Optional[str] = None

    __eq__ = AST.ast_equality


@dataclass
class Argument(AST):
    """
    Argument: (name, type_, default)
    """

    name: str
    type_: Type[ScalarType]
    default: Optional[AST]

    __eq__ = AST.ast_equality


@dataclass
class Operator(AST):
    """
    Operator: (operator, parameters, outputType, expression)
    """

    op: str
    parameters: List[Argument]
    output_type: str
    expression: AST

    __eq__ = AST.ast_equality


# TODO: Is this class necessary?
@dataclass
class DefIdentifier(AST):
    """
    DefIdentifier: (value, kind)
    """

    value: str
    kind: str

    __eq__ = AST.ast_equality


@dataclass
class DPRIdentifier(AST):
    """
    DefIdentifier: (value, kind, alias)
    """

    value: str
    kind: str
    alias: Optional[str] = None

    __eq__ = AST.ast_equality


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

    __eq__ = AST.ast_equality


@dataclass
class HRUnOp(AST):
    """
    HRUnOp: (op, operand)
    op types: '+','-'.
    """

    op: str
    operand: DefIdentifier

    __eq__ = AST.ast_equality


# TODO: Unify HRule and DPRule?
@dataclass
class HRule(AST):
    """
    HRule: (name, rule, erCode, erLevel)
    """

    name: Optional[str]
    rule: HRBinOp
    erCode: Optional[str]
    erLevel: Optional[int]

    __eq__ = AST.ast_equality


@dataclass
class DPRule(AST):
    """
    DPRule: (name, rule, erCode, erLevel)
    """

    name: Optional[str]
    rule: HRBinOp
    erCode: Optional[str]
    erLevel: Optional[int]

    __eq__ = AST.ast_equality


# TODO: Unify HRuleset and DPRuleset?
@dataclass
class HRuleset(AST):
    """
    HRuleset: (name, element, rules)
    """

    name: str
    signature_type: str
    element: DefIdentifier
    rules: List[HRule]

    __eq__ = AST.ast_equality


@dataclass
class DPRuleset(AST):
    """
    DPRuleset: (name, element, rules)
    """

    name: str
    signature_type: str
    params: Union[DefIdentifier, list]
    rules: List[DPRule]

    __eq__ = AST.ast_equality


@dataclass
class EvalOp(AST):
    """
    EvalOp: (name, children, output, language)

    op types:
    """

    name: str
    operands: List[AST]
    output: Optional[Dataset]
    language: Optional[str]

    __eq__ = AST.ast_equality


@dataclass
class NoOp(AST):
    """
    NoOp: ()
    """

    pass


@dataclass
class ParFunction(AST):
    """
    ParFunction: (operand)
    """

    operand: AST

    __eq__ = AST.ast_equality
