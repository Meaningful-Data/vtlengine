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


class ParamConstant(Constant):
    """
    Constant: (type, value)

    types: ALL
    """

    def __init__(self, type_, value):
        super().__init__(type_=type_, value=value)

    def __str__(self):
        return "<AST(name='{name}',type='{type}', value='{value}')>".format(
            name=self.__class__.__name__,
            type=self.type_,
            value=self.value
        )

    def __eq__(self, other):
        return super().__eq__(other)

    def toJSON(self):
        return super().toJSON()

    __repr__ = __str__


class Identifier(AST):
    """
    Identifier: (value)
    """

    def __init__(self, value, kind=None):
        super().__init__()
        self.value = value
        self.kind = kind

    def __str__(self):
        return "<AST(name='{name}', value='{value}')>".format(
            name=self.__class__.__name__,
            value=self.value
        )

    def __eq__(self, other):
        if isinstance(other, Identifier):
            return self.value == other.value and self.kind == other.kind
        return False

    def toJSON(self):
        return {
            'class_name': self.__class__.__name__,
            'value': self.value,
            'kind': self.kind
        }

    __repr__ = __str__


class ID(AST):
    """
    ID: (type, value)

    for a general purpose Terminal nodes Node
    """

    def __init__(self, type_, value):
        super().__init__()
        self.type = type_
        self.value = value

    def __str__(self):
        return "<AST(name='{name}',type='{type}', value='{value}')>".format(
            name=self.__class__.__name__,
            type=self.type,
            value=self.value
        )

    def __eq__(self, other):
        if isinstance(other, ID):
            return self.type == other.type and self.value == other.value
        return False

    def toJSON(self):
        return {
            'class_name': self.__class__.__name__,
            'value': self.value,
            'type_': self.type
        }

    __repr__ = __str__


class Role(AST):
    """
    Role: (role)

    roles: MEASURE, COMPONENT, DIMENSION, ATTRIBUTE, VIRAL ATTRIBUTE
    """

    def __init__(self, role):
        super().__init__()
        self.role = role

    def __str__(self):
        return "<AST(name='{name}', role='{role}')>".format(
            name=self.__class__.__name__,
            role=self.role
        )

    def __eq__(self, other):
        if isinstance(other, Role):
            return self.role == other.role
        return False

    def toJSON(self):
        return {
            'class_name': self.__class__.__name__,
            'role': self.role
        }

    __repr__ = __str__


class Collection(AST):
    """
    Collection: (name, type, children)

    types: Sets and ValueDomains
    """

    def __init__(self, name, type_, children, kind='Set'):
        super().__init__()
        self.name = name
        self.type = type_
        self.children = children
        self.kind = kind

    def __str__(self):
        return "<AST(name='{name}', collectionname={collectionname}, type='{type}', children={children}, kind={kind})>".format(
            name=self.__class__.__name__,
            collectionname=self.name,
            type=self.type,
            children=self.children,
            kind=self.kind
        )

    def __eq__(self, other):
        if isinstance(other, Collection):
            return (self.name == other.name and self.type == other.type and
                    self.children == other.children and self.kind == other.kind)
        return False

    def toJSON(self):
        return {
            'class_name': self.__class__.__name__,
            'name': self.name,
            'type_': self.type,
            'children': self.children,
            'kind': self.kind
        }

    __repr__ = __str__


class Analytic(AST):
    """
    Analytic: (op, operand, partition_by, order_by, params)

    op: SUM, AVG, COUNT, MEDIAN, MIN, MAX, STDDEV_POP, STDDEV_SAMP, VAR_POP, VAR_SAMP, FIRST_VALUE, LAST_VALUE, LAG,
        LEAD, RATIO_TO_REPORT

    partition_by: List of components.
    order_by: List of components + mode (ASC, DESC).
    params: Windowing clause (no need to validate them) or Scalar Item in LAG/LEAD.
    """

    def __init__(self, op, operand, partition_by, order_by, params):
        super().__init__()
        self.op = op
        self.operand = operand
        self.partition_by = partition_by
        self.order_by = order_by
        self.params = params

    def __str__(self):
        return f"<AST(op='{self.op}', operand='{self.operand}', partition_by='{self.partition_by}', order_by='{self.order_by}', params='{self.params})>"

    def __eq__(self, other):
        if isinstance(other, Analytic):
            return (self.op == other.op and self.operand == other.operand and
                    self.partition_by == other.partition_by and self.order_by == other.order_by and
                    self.params == other.params)
        return False

    def toJSON(self):
        return {
            'class_name': self.__class__.__name__,
            'op': self.op,
            'operand': self.operand,
            'partition_by': self.partition_by,
            'order_by': self.order_by,
            'params': self.params
        }

    __repr__ = __str__


class OrderBy(AST):
    def __init__(self, component, order=None):
        super().__init__()
        self.component = component
        self.order = order

    def __eq__(self, other):
        if isinstance(other, OrderBy):
            return self.component == other.component and self.order == other.order
        return False

    def toJSON(self):
        return {
            'class_name': self.__class__.__name__,
            'component': self.component,
            'order': self.order
        }


class Windowing(AST):
    def __init__(self, type_, first: int, second: int, first_mode: int, second_mode: int):
        super().__init__()
        self.type_ = type_
        self.start = first
        self.start_mode = first_mode
        self.stop = second
        self.stop_mode = second_mode

    def __eq__(self, other):
        if isinstance(other, Windowing):
            return (
                    self.type_ == other.type_ and self.start == other.start and self.start_mode == other.start_mode and
                    self.stop == other.stop and self.stop_mode == other.stop_mode)
        return False

    def toJSON(self):
        return {
            'class_name': self.__class__.__name__,
            'type_': self.type_,
            'first': self.start,
            'first_mode': self.start_mode,
            'second': self.stop,
            'second_mode': self.stop_mode
        }


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


class Aggregation(AST):
    """
    Aggregation: (op, operand, grouping_op, grouping)

    op types: AGGREGATE, SUM, AVG , COUNT, MEDIAN, MIN, MAX, STDDEV_POP, STDDEV_SAMP,
              VAR_POP, VAR_SAMP

    grouping types: 'group by', 'group except', 'group all'.
    """

    def __init__(self, op: str, operand: AST, grouping_op: Optional[str],
                 grouping: Optional[List[AST]], role: Optional[str], comp_mode: bool,
                 having_clause: Optional[ParamOp],
                 having_expression: Optional[str]):
        super().__init__()
        self.op: str = op
        self.operand: AST = operand
        self.grouping_op: Optional[str] = grouping_op
        self.grouping: Optional[List[AST]] = grouping
        self.role = role
        self.comp_mode = comp_mode
        self.having_clause: ParamOp = having_clause
        self.having_expression = having_expression

    def __str__(self):
        return "<AST(name='{name}',op='{op}', operand={operand}, role={role}, grouping_op={grouping_op}, grouping={grouping}, having_clause={having_clause}, having_expression={having_expression})>".format(
            name=self.__class__.__name__,
            op=self.op,
            operand=self.operand,
            role=self.role,
            grouping_op=self.grouping_op,
            grouping=self.grouping,
            having_clause=self.having_clause,
            having_expression=self.having_expression
        )

    def __eq__(self, other):
        if isinstance(other, Aggregation):
            return (
                    self.op == other.op and self.operand == other.operand and self.grouping_op == other.grouping_op and
                    self.grouping == other.grouping and self.role == other.role and self.comp_mode == other.comp_mode
                    and self.having_clause == other.having_clause and self.having_expression == self.having_expression)
        return False

    def toJSON(self):
        return {
            'class_name': self.__class__.__name__,
            'op': self.op,
            'operand': self.operand,
            'grouping_op': self.grouping_op,
            'grouping': self.grouping,
            'role': self.role,
            'comp_mode': self.comp_mode,
            'having_clause': self.having_clause,
            'having_expression': self.having_expression
        }

    __repr__ = __str__


class AggregationComp(AST):
    """
    AggregationComp: (op, operand)

    op type: SUM, AVG , COUNT, MEDIAN, MIN, MAX, STDDEV_POP, STDDEV_SAMP,
              VAR_POP, VAR_SAMP
    """

    def __init__(self, op: str, operand: Optional[AST]):
        super().__init__()
        self.op: str = op
        self.operand: Optional[AST] = operand

    def __str__(self):
        return "<AST(name='{name}', op='{op}', operand={operand})>"

    def __eq__(self, other):
        if isinstance(other, Aggregation):
            return self.op == other.op and self.operand == other.operand
        return False

    def toJSON(self):
        return {
            'class_name': self.__class__.__name__,
            'op': self.op,
            'operand': self.operand
        }

    __repr__ = __str__


class TimeAggregation(AST):
    """
    TimeAggregation: (op, operand, params, conf)

    op types: TIME_AGG
    """

    def __init__(self, op: str, operand: Optional[AST], params: List[AST], conf: Optional[str]):
        super().__init__()
        self.op: str = op
        self.operand: Optional[AST] = operand
        self.params: List[AST] = params
        self.conf: Optional[list] = conf

    def __str__(self):
        return "<AST(name='{name}',op='{op}', operand={operand}, params={params}, conf={conf})>".format(
            name=self.__class__.__name__,
            op=self.op,
            operand=self.operand,
            params=self.params,
            conf=self.conf
        )

    def __eq__(self, other):
        if isinstance(other, TimeAggregation):
            return (
                    self.op == other.op and self.operand == other.operand and self.params == other.params and
                    self.conf == other.conf)
        return False

    def toJSON(self):
        return {
            'class_name': self.__class__.__name__,
            'op': self.op,
            'operand': self.operand,
            'params': self.params,
            'conf': self.conf
        }

    __repr__ = __str__


class If(AST):
    """
    If: (condition, thenOp, elseOp)
    """

    def __init__(self, condition, thenOp, elseOp):
        super().__init__()
        self.condition = condition
        self.thenOp = thenOp
        self.elseOp = elseOp

    def __str__(self):
        return "<AST(name='{name}', condition={condition}, thenOp={thenOp}, elseOp={elseOp})>".format(
            name=self.__class__.__name__,
            condition=self.condition,
            thenOp=self.thenOp,
            elseOp=self.elseOp
        )

    def __eq__(self, other):
        if isinstance(other, If):
            return self.condition == other.condition and self.thenOp == other.thenOp and self.elseOp == other.elseOp
        return False

    def toJSON(self):
        return {
            'class_name': self.__class__.__name__,
            'condition': self.condition,
            'thenOp': self.thenOp,
            'elseOp': self.elseOp
        }

    __repr__ = __str__


class Validation(AST):
    """
    Validation: (op, validation, params, inbalance, invalid)
    """

    def __init__(self, op, validation, params, inbalance, invalid):
        super().__init__()
        self.op = op
        self.validation = validation
        self.params = params
        self.inbalance = inbalance
        self.invalid = invalid

    def __str__(self):
        return "<AST(name='{name}', op={op}, validation={validation}, params={params}, inbalance={inbalance}, invalid={invalid})>".format(
            name=self.__class__.__name__,
            op=self.op,
            validation=self.validation,
            params=self.params,
            inbalance=self.inbalance,
            invalid=self.invalid
        )

    def __eq__(self, other):
        if isinstance(other, Validation):
            return (
                    self.op == other.op and self.validation == other.validation and self.params == other.params and
                    self.inbalance == other.inbalance and self.invalid == other.invalid)
        return False

    def toJSON(self):
        return {
            'class_name': self.__class__.__name__,
            'op': self.op,
            'validation': self.validation,
            'params': self.params,
            'inbalance': self.inbalance,
            'invalid': self.invalid
        }

    __repr__ = __str__


class Operator(AST):
    """
    Operator: (operator, parameters, outputType, expression)
    """

    def __init__(self, operator, parameters, outputType, expresion):
        super().__init__()
        self.operator: str = operator
        self.parameters: list = parameters
        self.outputType = outputType
        self.expression = expresion

    def __str__(self):
        return "<AST(name='{name}', operator={operator}, parameters={parameters}, output={output}, expresion={expresion}))>".format(
            name=self.__class__.__name__,
            operator=self.operator,
            parameters=self.parameters,
            output=self.outputType,
            expresion=self.expression
        )

    def __eq__(self, other):
        if isinstance(other, Operator):
            return (self.operator == other.operator and self.parameters == other.parameters and
                    self.outputType == other.outputType and self.expression == other.expression)
        return False

    def toJSON(self):
        return {
            'class_name': self.__class__.__name__,
            'operator': self.operator,
            'parameters': self.parameters,
            'outputType': self.outputType,
            'expression': self.expression
        }

    __repr__ = __str__


class DefIdentifier(AST):
    """
    DefIdentifier: (value, kind)
    """

    def __init__(self, value, kind):
        super().__init__()
        self.value = value
        self.kind = kind

    def __str__(self):
        return "<AST(name='{name}', kind='{kind}', value='{value}')>".format(
            name=self.__class__.__name__,
            kind=self.kind,
            value=self.value
        )

    def __eq__(self, other):
        if isinstance(other, DefIdentifier):
            return self.value == other.value and self.kind == other.kind
        return False

    def toJSON(self):
        return {
            'class_name': self.__class__.__name__,
            'value': self.value,
            'kind': self.kind
        }

    __repr__ = __str__


class DPRIdentifier(AST):
    """
    DefIdentifier: (value, kind, alias)
    """

    def __init__(self, value, kind, alias):
        super().__init__()
        self.value = value
        self.kind = kind
        self.alias = alias

    def __str__(self):
        return "<AST(name='{name}', kind='{kind}', value='{value}', alias={alias})>".format(
            name=self.__class__.__name__,
            kind=self.kind,
            value=self.value,
            alias=self.alias
        )

    def __eq__(self, other):
        if isinstance(other, DPRIdentifier):
            return self.value == other.value and self.kind == other.kind and self.alias == other.alias
        return False

    def toJSON(self):
        return {
            'class_name': self.__class__.__name__,
            'value': self.value,
            'kind': self.kind,
            'alias': self.alias
        }

    __repr__ = __str__


class Types(AST):
    """
    Types: (name, kind, type_, constraints, nullable)

    kind:
            - basicScalarType
                - STRING, INTEGER, NUMBER, BOOLEAN, DATE, TIME_PERIOD, DURATION, SCALAR, TIME.
            -
    """

    def __init__(self, kind, type_, constraints, nullable, name=None):
        super().__init__()
        self.name = name
        self.kind = kind
        self.type_ = type_
        self.constraints = constraints
        self.nullable = nullable

    def __str__(self):
        return "<AST(name='{name}', kind='{kind}', type='{type_}', constraints='{constraints}', nullable='{nullable}')>".format(
            name=self.__class__.__name__,
            kind=self.kind,
            type_=self.type_,
            constraints=self.constraints,
            nullable=self.nullable
        )

    def __eq__(self, other):
        if isinstance(other, Types):
            return (
                    self.name == other.name and self.kind == other.kind and self.type_ == other.type_ and
                    self.constraints == other.constraints and self.nullable == other.nullable)
        return False

    def toJSON(self):
        return {
            'class_name': self.__class__.__name__,
            'name': self.name,
            'kind': self.kind,
            'type_': self.type_,
            'constraints': self.constraints,
            'nullable': self.nullable
        }

    __repr__ = __str__


class Argument(AST):
    """
    Argument: (name, type_, default)
    """

    def __init__(self, name, type, default=None):
        super().__init__()
        self.name = name
        self.type_ = type
        self.default = default

    def __str__(self):
        return "<AST(name='{name}', vname='{vname}', type='{type_}', default='{default}')>".format(
            name=self.__class__.__name__,
            vname=self.name,
            type_=self.type_,
            default=self.default
        )

    def __eq__(self, other):
        if isinstance(other, Argument):
            return self.name == other.name and self.type_ == other.type_ and self.default == other.default
        return False

    def toJSON(self):
        return {
            'class_name': self.__class__.__name__,
            'name': self.name,
            'type': self.type_,
            'default': self.default
        }

    __repr__ = __str__


class HRBinOp(AST):
    """
    HRBinOp: (left, op, right)
    op types: '+','-', '=', '>', '<', '>=', '<='.
    """

    def __init__(self, left: DefIdentifier or None, op: str, right: DefIdentifier or None):
        super().__init__()
        self.left: DefIdentifier = left
        self.op: str = op
        self.right: DefIdentifier = right

    def __str__(self):
        return "<AST(name='{name}', op='{op}', left={left}, right={right})>".format(
            name=self.__class__.__name__,
            op=self.op,
            left=self.left,
            right=self.right
        )

    def __eq__(self, other):
        if isinstance(other, HRBinOp):
            return self.left == other.left and self.op == other.op and self.right == other.right
        return False

    def toJSON(self):
        return {
            'class_name': self.__class__.__name__,
            'op': self.op,
            'left': self.left,
            'right': self.right
        }

    __repr__ = __str__


class HRUnOp(AST):
    """
    HRUnOp: (op, operand)
    op types: '+','-'.
    """

    def __init__(self, op: str, operand: DefIdentifier):
        super().__init__()
        self.op: str = op
        self.operand: DefIdentifier = operand

    def __str__(self):
        return "<AST(name='{name}', op='{op}', operand={operand})>".format(
            name=self.__class__.__name__,
            op=self.op,
            operand=self.operand
        )

    def __eq__(self, other):
        if isinstance(other, HRUnOp):
            return self.op == other.op and self.operand == other.operand
        return False

    def toJSON(self):
        return {
            'class_name': self.__class__.__name__,
            'op': self.op,
            'operand': self.operand,
        }

    __repr__ = __str__


class HRule(AST):
    """
    HRule: (name, rule, erCode, erLevel)
    """

    def __init__(self, name: Optional[str], rule: HRBinOp, erCode: Optional[Constant],
                 erLevel: Optional[Constant]):
        super().__init__()
        self.name: Optional[str] = name
        self.rule: HRBinOp = rule
        self.erCode: Optional[Constant] = erCode
        self.erLevel: Optional[Constant] = erLevel

    def __str__(self):
        return "<AST(name='{name}', rulename='{rulename}', rule='{rule}', erCode='{erCode}', erLevel='{erLevel}')>".format(
            name=self.__class__.__name__,
            rulename=self.name,
            rule=self.rule,
            erCode=self.erCode,
            erLevel=self.erLevel
        )

    def __eq__(self, other):
        if isinstance(other, HRule):
            return (
                    self.name == other.name and self.rule == other.rule and self.erCode == other.erCode and
                    self.erLevel == other.erLevel)
        return False

    def toJSON(self):
        return {
            'class_name': self.__class__.__name__,
            'name': self.name,
            'rule': self.rule,
            'erCode': self.erCode,
            'erLevel': self.erLevel
        }

    __repr__ = __str__


class DPRule(AST):
    """
    DPRule: (name, rule, erCode, erLevel)
    """

    def __init__(self, name: Optional[str], rule: HRBinOp, erCode: Optional[Constant],
                 erLevel: Optional[Constant]):
        super().__init__()
        self.name: Optional[str] = name
        self.rule: HRBinOp = rule
        self.erCode: Optional[Constant] = erCode
        self.erLevel: Optional[Constant] = erLevel

    def __str__(self):
        return "<AST(name='{name}', rulename='{rulename}', rule='{rule}', erCode='{erCode}', erLevel='{erLevel}')>".format(
            name=self.__class__.__name__,
            rulename=self.name,
            rule=self.rule,
            erCode=self.erCode,
            erLevel=self.erLevel
        )

    def __eq__(self, other):
        if isinstance(other, DPRule):
            return (
                    self.name == other.name and self.rule == other.rule and self.erCode == other.erCode and
                    self.erLevel == other.erLevel)
        return False

    def toJSON(self):
        return {
            'class_name': self.__class__.__name__,
            'name': self.name,
            'rule': self.rule,
            'erCode': self.erCode,
            'erLevel': self.erLevel
        }

    __repr__ = __str__


class HRuleset(AST):
    """
    HRuleset: (name, element, rules)
    """

    def __init__(self, name: str, element: [DefIdentifier, list], rules: List[HRule]):
        super().__init__()
        self.name: str = name
        self.element: DefIdentifier = element
        self.rules: List[HRule] = rules

    def __str__(self):
        return "<AST(name='{name}', rule={rule}, element={element}, rules={rules})>".format(
            name=self.__class__.__name__,
            rule=self.name,
            element=self.element,
            rules=self.rules
        )

    def __eq__(self, other):
        if isinstance(other, HRuleset):
            return self.name == other.name and self.element == other.element and self.rules == other.rules
        return False

    def toJSON(self):
        return {
            'class_name': self.__class__.__name__,
            'name': self.name,
            'element': self.element,
            'rules': self.rules
        }

    __repr__ = __str__


class DPRuleset(AST):
    """
    DPRuleset: (name, element, rules)
    """

    def __init__(self, name: str, element: [DefIdentifier, list], rules: List[DPRule]):
        super().__init__()
        self.name: str = name
        self.element: list = element
        self.rules: List[DPRule] = rules

    def __str__(self):
        return "<AST(name='{name}', rule={rule}, element={element}, rules={rules})>".format(
            name=self.__class__.__name__,
            rule=self.name,
            element=self.element,
            rules=self.rules
        )

    def __eq__(self, other):
        if isinstance(other, DPRuleset):
            return self.name == other.name and self.element == other.element and self.rules == other.rules
        return False

    def toJSON(self):
        return {
            'class_name': self.__class__.__name__,
            'name': self.name,
            'element': self.element,
            'rules': self.rules
        }

    __repr__ = __str__


class EvalOp(AST):
    """
    EvalOp: (name, children, output, language)

    op types:
    """

    def __init__(self, name, children, output, language):
        super().__init__()
        self.name = name
        self.children = children
        self.output = output
        self.language = language

    def __str__(self):
        return "<AST(name='{name}', routine='{routine}', children={children}, output={output}, language={language})>".format(
            name=self.__class__.__name__,
            routine=self.name,
            children=self.children,
            output=self.output,
            language=self.language
        )

    def __eq__(self, other):
        if isinstance(other, EvalOp):
            return (
                    self.name == other.name and self.children == other.children and self.output == other.output and
                    self.language == other.language)
        return False

    def toJSON(self):
        return {
            'class_name': self.__class__.__name__,
            'name': self.name,
            'children': self.children,
            'output': self.output,
            'language': self.language
        }

    __repr__ = __str__


class NoOp(AST):
    """
    NoOp: ()
    """

    def __str__(self):
        return "<AST(name='{name}')>".format(
            name=self.__class__.__name__
        )

    __repr__ = __str__
