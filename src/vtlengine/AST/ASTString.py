import copy
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union

from vtlengine import AST
from vtlengine.AST import (
    Assignment,
    PersistentAssignment,
    TimeAggregation,
)
from vtlengine.AST.ASTTemplate import ASTTemplate
from vtlengine.AST.Grammar._cpp_parser import LITERAL_NAMES
from vtlengine.AST.Grammar.tokens import (
    AGGREGATE,
    APPLY,
    ATTRIBUTE,
    CALC,
    CAST,
    CHARSET_MATCH,
    CHECK_DATAPOINT,
    DATE_ADD,
    DATEDIFF,
    DROP,
    FILL_TIME_SERIES,
    FILTER,
    HAVING,
    IDENTIFIER,
    INSTR,
    INTERSECT,
    KEEP,
    LOG,
    MAX,
    MEASURE,
    MEMBERSHIP,
    MIN,
    MINUS,
    MOD,
    NOT,
    NVL,
    PLUS,
    POWER,
    RANDOM,
    RENAME,
    REPLACE,
    ROUND,
    SETDIFF,
    STRING_DISTANCE,
    SUBSTR,
    SYMDIFF,
    TIMESHIFT,
    TRUNC,
    UNION,
    VIRAL_ATTRIBUTE,
)
from vtlengine.DataTypes import SCALAR_TYPES_CLASS_REVERSE
from vtlengine.Model import Component, Dataset, Scalar

nl = "\n"
tab = "\t"
RESERVED_WORDS = {x.replace("'", ""): x for x in LITERAL_NAMES}
# A name the lexer reads as an IDENTIFIER without quotes (VtlTokens.g4)
UNQUOTED_NAME = re.compile(
    r"(?:[0-9][A-Za-z0-9_.]*)?[A-Za-z][A-Za-z0-9_.]*"
    r"|[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_.]*"
    r"(?:\((?:[0-9]+(?:\.[0-9]+)*(?:\.[_+*~])?|[_+*~])\))?(?::[.A-Za-z0-9_]+)?"
)
# String literals, quoted names and comments, which pretty printing must not change
QUOTED_TEXT = re.compile(r"(\"[^\"]*\"|'[^']*'|//[^\n]*|/\*.*?\*/)", re.DOTALL)


def _format_number(value: float) -> str:
    """A float written as a number literal that reads back to the same Number.

    The grammar takes no exponent and needs digits on both sides of the point, so
    the shortest digits that round-trip are written out in full. A whole value keeps
    its ".0": written as 2 it would read back as the Integer 2.
    """
    text = format(Decimal(str(value)), "f")
    return text if "." in text else f"{text}.0"


def _handle_literal(value: Union[str, int, float, bool]):
    if isinstance(value, str):
        if '"' in value:
            return value
        return f'"{value}"'
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, float):
        return _format_number(value)
    return str(value)


def _format_reserved_word(value: str):
    if value in RESERVED_WORDS:
        return RESERVED_WORDS[value]
    return value


def _format_name(value: str) -> str:
    """A name written so that it reads back as the same name.

    The parser drops the quotes around a name, so they go back on reserved words and
    on any name the lexer would not read as one identifier, such as 'Me 1'.
    """
    if value not in RESERVED_WORDS and not UNQUOTED_NAME.fullmatch(value):
        return f"'{value}'"
    return _format_reserved_word(value)


def _format_constraint(constraint: Optional[AST.TypeConstraint]) -> str:
    """The constraints of a scalar type, written as declared."""
    if constraint is None:
        return ""
    text = ""
    if constraint.condition is not None:
        text += f" [{ASTString().render(constraint.condition)}]"
    if constraint.values is not None:
        text += f" {{{', '.join(ASTString().render(value) for value in constraint.values)}}}"
    if constraint.not_null:
        text += " not null"
    return text


def _format_component_type(
    component: Component, constraint: Optional[AST.TypeConstraint] = None
) -> str:
    role = "component" if component.role is None else component.role.value.lower()
    # A component declared without a type is parsed with a String instance, not the class
    if isinstance(component.data_type, type):
        return f"{role}<{_format_type(component.data_type)}{_format_constraint(constraint)}>"
    return role


def _format_type(type_: Any, constraints: Optional[AST.SignatureConstraints] = None) -> str:
    """The type of an operator parameter or result, written as declared."""
    if isinstance(type_, Dataset):
        named = constraints.components if constraints is not None else {}
        unnamed = constraints.unnamed if constraints is not None else []
        components = [
            f"{_format_component_type(component, named.get(name))} {_format_name(name)}"
            for name, component in type_.components.items()
        ] + [
            f"{_format_component_type(entry.component, entry.constraint)} _{entry.multiplicity}"
            for entry in unnamed
        ]
        return f"dataset {{{', '.join(components)}}}" if components else "dataset"
    constraint = constraints.value if constraints is not None else None
    if isinstance(type_, Component):
        return _format_component_type(type_, constraint)
    if isinstance(type_, Scalar):
        return f"scalar{_format_constraint(constraint)}"
    return f"{SCALAR_TYPES_CLASS_REVERSE[type_].lower()}{_format_constraint(constraint)}"


def _format_dataset_eval(dataset: Dataset) -> str:
    """The components of the dataset an eval returns, one per line."""
    components = f",{nl}{tab * 3}".join(
        [
            f"{_format_component_type(component)} {_format_name(component.name)}"
            for component in dataset.components.values()
        ]
    )
    return f"dataset {{{nl}{tab * 3}{components}{nl}{tab * 2}}}"


def _replace_outside_quotes(text: str, old: str, new: str) -> str:
    """text.replace(old, new), leaving string literals and quoted names as they are."""
    parts = QUOTED_TEXT.split(text)
    return "".join(part if i % 2 else part.replace(old, new) for i, part in enumerate(parts))


def _strip_line_ends(text: str) -> str:
    """The spaces and tabs that end each line removed, outside string literals."""
    parts = [
        part if i % 2 else re.sub(r"[ \t]+(?=\n)", "", part)
        for i, part in enumerate(QUOTED_TEXT.split(text))
    ]
    parts[-1] = parts[-1].rstrip(" \t")
    return "".join(parts)


@dataclass
class ASTString(ASTTemplate):
    vtl_script: str = ""
    pretty: bool = False
    is_first_assignment: bool = False
    is_from_agg: bool = False  # Handler to write grouping at aggr level
    # Comments written inside the ruleset being rendered, which go before their rule
    ruleset_comments: List[AST.Comment] = field(default_factory=list)

    def render(self, ast: AST.AST) -> str:
        self.vtl_script = ""
        result = self.visit(ast)
        if result:
            self.vtl_script += result
        return self.vtl_script

    def visit_Start(self, node: AST.Start) -> Any:
        inner_comments = self._inner_comments(node.children) if self.pretty else {}
        moved = {id(comment) for comments in inner_comments.values() for comment in comments}
        for child in node.children:
            if id(child) in moved:
                continue
            comments = inner_comments.get(id(child), [])
            if isinstance(child, (AST.DPRuleset, AST.HRuleset)):
                self.ruleset_comments = comments
            else:
                # Any other statement is written whole, after the comments inside it
                for comment in comments:
                    self.vtl_script += f"{self._comment_text(comment)}{nl}"
            if isinstance(child, (Assignment, PersistentAssignment)):
                self.is_first_assignment = True
            start = len(self.vtl_script)
            self.visit(child)
            if self.pretty:
                self.vtl_script = self.vtl_script[:start] + _strip_line_ends(
                    self.vtl_script[start:]
                )
            self.vtl_script += "\n"

    @staticmethod
    def _inner_comments(children: List[AST.AST]) -> Dict[int, List[AST.Comment]]:
        """The comments written inside each statement, by the id of the statement.

        The children come sorted by position, so a statement is followed by its comments.
        """
        inner: Dict[int, List[AST.Comment]] = {}
        statement: Optional[AST.AST] = None
        for child in children:
            if not isinstance(child, AST.Comment):
                statement = child
            elif statement is not None and (child.line_start, child.column_start) < (
                statement.line_stop,
                statement.column_stop,
            ):
                inner.setdefault(id(statement), []).append(child)
        return inner

    def _comment_text(self, node: AST.Comment) -> str:
        value = node.value[:-1] if node.value[-1] == "\n" else node.value
        if self.pretty:
            value = nl.join([line.rstrip() for line in value.split(nl)])
        return value

    def _comments_before(self, rule: Optional[AST.AST], indent: str) -> str:
        """The ruleset comments written before a rule (all those left, without a rule)."""
        lines = ""
        while self.ruleset_comments and (
            rule is None
            or (self.ruleset_comments[0].line_start, self.ruleset_comments[0].column_start)
            < (rule.line_start, rule.column_start)
        ):
            lines += f"{indent}{self._comment_text(self.ruleset_comments.pop(0))}{nl}"
        return lines

    # ---------------------- Rulesets ----------------------
    def visit_HRuleset(self, node: AST.HRuleset) -> None:
        if isinstance(node.element, list):
            sep = nl + tab if self.pretty else " "
            conditons = ", ".join([self.visit(e) for e in node.element[:-1]])
            signature = (
                f"{sep if self.pretty else ''}{node.signature_type}{sep}"
                f"condition {conditons}{sep}rule {node.element[-1].value}"
            )
        else:
            signature = f"{node.signature_type} rule {node.element.value}"

        # create_ast sorts the rules by dependency, so they go back in the order written
        written_rules = sorted(node.rules, key=lambda rule: (rule.line_start, rule.column_start))
        rules_strs = []
        if self.pretty:
            self.vtl_script += f"define hierarchical ruleset {node.name}({signature}) is{nl}"
            for _i, rule in enumerate(written_rules):
                rule_str = self.visit(rule)
                # An unnamed when rule comes indented already
                if rule.name is not None or rule.rule.op != "when":
                    rule_str = f"{tab}{rule_str}"
                if rule.erCode is not None:
                    rule_str += f"{nl}{tab}errorcode {_handle_literal(rule.erCode)}"
                if rule.erLevel is not None:
                    rule_str += f"{nl}{tab}errorlevel {_handle_literal(rule.erLevel)}"
                indent = rule_str[: len(rule_str) - len(rule_str.lstrip(tab))]
                rules_strs.append(self._comments_before(rule, indent) + rule_str)
            rules_sep = f";{nl * 2}" if len(rules_strs) > 1 else ""
            rules = rules_sep.join(rules_strs)
            self.vtl_script += rules + nl + self._comments_before(None, tab)
            self.vtl_script += f"end hierarchical ruleset;{nl}"
        else:
            for rule in written_rules:
                rule_str = self.visit(rule)
                if rule.erCode is not None:
                    rule_str += f" errorcode {_handle_literal(rule.erCode)}"
                if rule.erLevel is not None:
                    rule_str += f" errorlevel {_handle_literal(rule.erLevel)}"
                rules_strs.append(rule_str)
            rules_sep = "; " if len(rules_strs) > 1 else ""
            rules = rules_sep.join(rules_strs)
            self.vtl_script += (
                f"define hierarchical ruleset {node.name} ({signature}) is {rules} "
                f"end hierarchical ruleset;"
            )

    def visit_HRule(self, node: AST.HRule) -> str:
        vtl_script = ""
        if node.name is not None:
            # In pretty mode a when rule starts on the line after its name
            vtl_script += (
                f"{node.name}:{nl}"
                if self.pretty and node.rule.op == "when"
                else (f"{node.name}: ")
            )
        vtl_script += f"{self.visit(node.rule)}"
        return vtl_script

    def visit_HRBinOp(self, node: AST.HRBinOp) -> str:
        if node.op == "when":
            if self.pretty:
                return (
                    f"{tab * 3}when{nl}"
                    f"{tab * 4}{self.visit(node.left)}{nl}"
                    f"{tab * 3}then{nl}"
                    f"{tab * 4}{self.visit(node.right)}"
                )
            else:
                return f"{node.op} {self.visit(node.left)} then {self.visit(node.right)}"
        return f"{self.visit(node.left)} {node.op} {self.visit(node.right)}"

    def visit_HRUnOp(self, node: AST.HRUnOp) -> str:
        return f"{node.op} {self.visit(node.operand)}"

    def visit_DefIdentifier(self, node: AST.DefIdentifier) -> str:
        if node.was_quoted:
            return f"'{node.value}'"
        return _format_reserved_word(node.value)

    def visit_DPRule(self, node: AST.DPRule) -> str:
        if self.pretty:
            lines = []
            if node.name is not None:
                lines.append(f"{tab * 2}{node.name}:")
            rule = self.visit(node.rule)
            # A when rule comes indented already, and any other rule goes at its depth
            is_when = isinstance(node.rule, AST.HRBinOp) and node.rule.op == "when"
            lines.append(rule if is_when else f"{tab * 3}{rule}")
            if node.erCode is not None:
                lines.append(f"{tab * 3}errorcode {_handle_literal(node.erCode)}")
            if node.erLevel is not None:
                lines.append(f"{tab * 3}errorlevel {_handle_literal(node.erLevel)}")
            return nl.join(lines)
        else:
            vtl_script = ""
            if node.name is not None:
                vtl_script += f"{node.name}: "
            vtl_script += f"{self.visit(node.rule)}"
            if node.erCode is not None:
                vtl_script += f" errorcode {_handle_literal(node.erCode)}"
            if node.erLevel is not None:
                vtl_script += f" errorlevel {_handle_literal(node.erLevel)}"
            return vtl_script

    def visit_DPRIdentifier(self, node: AST.DPRIdentifier) -> str:
        vtl_script = _format_name(node.value)
        if node.alias is not None:
            vtl_script += f" as {_format_name(node.alias)}"
        return vtl_script

    def visit_DPRuleset(self, node: AST.DPRuleset) -> None:
        rules_sep = "; " if len(node.rules) > 1 else ""
        signature_sep = ", " if len(node.params) > 1 else ""
        params = signature_sep.join([self.visit(x) for x in node.params])
        signature = f"{node.signature_type} {params}" if params else node.signature_type

        if self.pretty:
            self.vtl_script += f"define datapoint ruleset {node.name}({signature}) is{nl}"
            rules = ""
            for i, rule in enumerate(node.rules):
                rule_str = self.visit(rule)
                indent = rule_str[: len(rule_str) - len(rule_str.lstrip(tab))]
                rules += self._comments_before(rule, indent) + rule_str
                if i != len(node.rules) - 1:
                    rules += f";{nl * 2}"
                else:
                    rules += f"{nl}"
            self.vtl_script += rules + self._comments_before(None, tab)
            self.vtl_script += f"end datapoint ruleset;{nl}"
        else:
            rules = rules_sep.join([self.visit(x) for x in node.rules])
            self.vtl_script += (
                f"define datapoint ruleset {node.name} "
                f"({signature}) is {rules} end datapoint ruleset;"
            )

    # ---------------------- Viral Propagation ----------------------
    @staticmethod
    def _vp_constant(value: Any) -> str:
        """A viral propagation constant written the way it was read.

        ``null`` reaches the rule as None and is a keyword rather than a value, so
        quoting it turned it into the string "None". Numbers and booleans
        are not quoted either.
        """
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            return _format_number(value)
        if isinstance(value, int):
            return str(value)
        return f'"{value}"'

    def visit_ViralPropagationDef(self, node: AST.ViralPropagationDef) -> None:
        clauses_strs: list[str] = []
        for clause in node.enumerated_clauses:
            clause_str = ""
            if clause.name is not None:
                clause_str += f"{clause.name} : "
            values_str = " and ".join([self._vp_constant(v) for v in clause.values])
            clause_str += f"when {values_str} then {self._vp_constant(clause.result)}"
            clauses_strs.append(clause_str)
        if node.aggregate_clause is not None:
            clauses_strs.append(f"aggregate {node.aggregate_clause.function}")
        if node.has_default:
            clauses_strs.append(f"else {self._vp_constant(node.default_value)}")

        name = _format_name(node.name)
        target = _format_name(node.target)
        if self.pretty:
            self.vtl_script += (
                f"define viral propagation {name}({node.signature_type} {target}) is{nl}"
            )
            for i, c in enumerate(clauses_strs):
                self.vtl_script += f"{tab}{c}"
                if i != len(clauses_strs) - 1:
                    self.vtl_script += f";{nl}"
                else:
                    self.vtl_script += nl
            self.vtl_script += f"end viral propagation;{nl}"
        else:
            clauses_joined = "; ".join(clauses_strs)
            self.vtl_script += (
                f"define viral propagation {name} "
                f"({node.signature_type} {target}) is "
                f"{clauses_joined} "
                f"end viral propagation;"
            )

    # ---------------------- User Defined Operators ----------------------

    def visit_Argument(self, node: AST.Argument) -> str:
        default = f" default {self.visit(node.default)}" if node.default is not None else ""

        name = _format_name(node.name)

        return f"{name} {_format_type(node.type_, node.constraints)}{default}"

    def visit_Operator(self, node: AST.Operator) -> None:
        signature_sep = ", " if len(node.parameters) > 1 else ""
        signature = signature_sep.join([self.visit(x) for x in node.parameters])
        returns = (
            ""
            if node.output_type is None
            else f"returns {node.output_type.lower()} "
            if node.return_type is None
            else f"returns {_format_type(node.return_type, node.return_constraints)} "
        )
        if self.pretty:
            self.vtl_script += f"define operator {node.op}({signature}){nl}"
            self.vtl_script += f"\t{returns}is{nl}"
            expression = self.visit(node.expression)
            if "(" in expression:
                expression = _replace_outside_quotes(expression, "(", f"({nl}{tab * 2}")
                expression = _replace_outside_quotes(expression, ")", f"{nl}{tab * 2})")

            self.vtl_script += f"{tab * 2}{expression}{nl}"
            self.vtl_script += f"end operator;{nl}"
        else:
            body = f"{returns}is {self.visit(node.expression)}"
            self.vtl_script += f"define operator {node.op}({signature}) {body} end operator;"

    # ---------------------- Basic Operators ----------------------
    def visit_Assignment(self, node: AST.Assignment) -> Optional[str]:
        return_element = not copy.deepcopy(self.is_first_assignment)
        is_first = self.is_first_assignment
        if is_first:
            self.is_first_assignment = False
        left = self.visit(node.left)
        role = getattr(node.left, "role", None)
        if role is not None:
            left = f"{role.value.lower()} {left}"
        if self.pretty and is_first:
            expression = f"{left} {node.op}{nl}{tab}{self.visit(node.right)}"
        else:
            expression = f"{left} {node.op} {self.visit(node.right)}"
        if return_element:
            return expression
        self.vtl_script += f"{expression};"

    def visit_PersistentAssignment(self, node: AST.PersistentAssignment) -> Optional[str]:
        return self.visit_Assignment(node)

    def visit_BinOp(self, node: AST.BinOp) -> str:
        if node.op in [NVL, LOG, MOD, POWER, RANDOM, TIMESHIFT, DATEDIFF, CHARSET_MATCH]:
            return f"{node.op}({self.visit(node.left)}, {self.visit(node.right)})"
        elif node.op == MEMBERSHIP:
            return f"{self.visit(node.left)}{node.op}{self.visit(node.right)}"
        if self.pretty:
            return f"{self.visit(node.left)} {node.op} {self.visit(node.right)}"

        return f"{self.visit(node.left)} {node.op} {self.visit(node.right)}"

    def visit_UnaryOp(self, node: AST.UnaryOp) -> str:
        if node.op in [PLUS, MINUS]:
            return f"{node.op}{self.visit(node.operand)}"
        elif node.op in [IDENTIFIER, ATTRIBUTE, VIRAL_ATTRIBUTE, NOT]:
            return f"{node.op} {self.visit(node.operand)}"
        elif node.op == MEASURE:
            if getattr(node, "is_implicit_role", False):
                return self.visit(node.operand)
            return f"{node.op} {self.visit(node.operand)}"

        return f"{node.op}({self.visit(node.operand)})"

    def visit_MulOp(self, node: AST.MulOp) -> str:
        sep = ", " if len(node.children) > 1 else ""
        body = sep.join([self.visit(x) for x in node.children])
        if self.pretty:
            return f"{node.op}({body})"
        return f"{node.op}({body})"

    def visit_ParamOp(self, node: AST.ParamOp) -> str:  # noqa: C901
        if node.op == HAVING:
            return f"{node.op} {self.visit(node.params)}"
        elif node.op == STRING_DISTANCE:
            method = self.visit(node.params[0])
            s1 = self.visit(node.children[0])
            s2 = self.visit(node.children[1])
            return f"{node.op}({method}, {s1}, {s2})"
        elif node.op in [SUBSTR, INSTR, REPLACE, ROUND, TRUNC, UNION, SETDIFF, SYMDIFF, INTERSECT]:
            params_sep = ", " if len(node.params) > 1 else ""
            param_values = [self.visit(x) for x in node.params]
            if not param_values:
                return f"{node.op}({self.visit(node.children[0])})"
            return f"{node.op}({self.visit(node.children[0])}, {params_sep.join(param_values)})"

        elif node.op == CAST:
            operand = self.visit(node.children[0])
            data_type = SCALAR_TYPES_CLASS_REVERSE[node.children[1]].lower()
            mask = ""
            if len(node.params) == 1:
                mask = f", {_handle_literal(self.visit(node.params[0]))}"
            if self.pretty:
                return f"{node.op}({operand}, {data_type}{mask})"
            else:
                return f"{node.op}({operand}, {data_type}{mask})"
        elif node.op == FILL_TIME_SERIES:
            operand = self.visit(node.children[0])
            param = node.params[0].value if node.params else "all"
            if self.pretty:
                return f"{node.op}({operand},{param})"
            else:
                return f"{node.op}({operand}, {param})"
        elif node.op == DATE_ADD:
            operand = self.visit(node.children[0])
            shift_number = self.visit(node.params[0])
            period_indicator = self.visit(node.params[1])
            if self.pretty:
                return (
                    f"{node.op}({nl}{tab * 2}{operand},{nl}{tab * 2}{shift_number},"
                    f"{nl}{tab * 2}"
                    f"{period_indicator})"
                )
            else:
                return f"{node.op}({operand}, {shift_number}, {period_indicator})"
        return ""

    def visit_HROperation(self, node: AST.HROperation) -> str:
        operand = self.visit(node.dataset)
        clause_sep = f"{nl}{tab * 2}" if self.pretty else " "

        # The condition and rule clauses are optional, independently of each other
        clauses = [node.ruleset_name]
        if node.conditions:
            conditions = ", ".join([self.visit(condition) for condition in node.conditions])
            clauses.append(f"condition {conditions}")
        if node.rule_component is not None:
            clauses.append(f"rule {self.visit(node.rule_component)}")

        # Modes are written whenever the script wrote them, defaults included
        modes = "".join(
            f" {mode.value}"
            for mode in (node.validation_mode, node.input_mode, node.output)
            if mode is not None
        )
        body = f"{clause_sep.join(clauses)}{modes}"

        if self.pretty:
            closing = "" if node.rule_component is not None else nl
            return f"{node.op}({nl}{tab * 2}{operand},{nl}{tab * 2}{body}{closing})"
        return f"{node.op}({operand}, {body})"

    def visit_DPValidation(self, node: AST.DPValidation) -> str:
        operand = self.visit(node.dataset)
        rule_name = node.ruleset_name

        # Build components string if present
        components_str = ""
        if node.components:
            components_str = " components " + ", ".join(
                [_format_name(component) for component in node.components]
            )

        # Output mode (written whenever the script wrote it, default "invalid" included)
        output_str = ""
        if node.output is not None:
            output_str = f" {node.output.value}"

        if self.pretty:
            # In pretty mode the output mode goes on its own line, if the script wrote it
            output_line = f"{nl}{tab}{node.output.value}" if node.output is not None else ""
            return (
                f"{CHECK_DATAPOINT}({nl}{tab}{operand},{nl}{tab}{rule_name}"
                f"{components_str}{output_line}{nl})"
            )
        else:
            return f"{CHECK_DATAPOINT}({operand}, {rule_name}{components_str}{output_str})"

    # ---------------------- Individual operators ----------------------

    def _handle_grouping_having(self, node: AST) -> Tuple[str, str]:
        if self.is_from_agg:
            return "", ""
        grouping = ""
        if node.grouping is not None:
            names = []
            time_aggregations = []
            for grouping_value in node.grouping:
                if isinstance(grouping_value, TimeAggregation):
                    time_aggregations.append(self.visit(grouping_value))
                else:
                    names.append(_format_name(grouping_value.value))
            # time_agg follows the components without a comma: group by Id_1 time_agg("A")
            grouping_values = [", ".join(names)] if names else []
            grouping = " " + " ".join([node.grouping_op] + grouping_values + time_aggregations)
        having = f" {self.visit(node.having_clause)}" if node.having_clause is not None else ""
        return grouping, having

    def visit_Aggregation(self, node: AST.Aggregation) -> str:
        grouping, having = self._handle_grouping_having(node)
        operand = "" if node.operand is None else f"{self.visit(node.operand)}"
        if self.pretty and node.op not in (MAX, MIN):
            return f"{node.op}({nl}{tab * 2}{operand}{grouping}{having}{nl}{tab * 2})"
        return f"{node.op}({operand}{grouping}{having})"

    def visit_Analytic(self, node: AST.Analytic) -> str:
        operand = "" if node.operand is None else self.visit(node.operand)
        partition = ""
        if node.partition_op == "except all":
            partition = "partition except all"
        elif node.partition_by:
            keyword = "except" if node.partition_op == "except" else "by"
            partition_values = [_format_name(x) for x in node.partition_by]
            partition = f"partition {keyword} {', '.join(partition_values)}"
        order = ""
        if node.order_by:
            order_sep = ", " if len(node.order_by) > 1 else ""
            order = f"order by {order_sep.join([self.visit(x) for x in node.order_by])}"
        window = self.visit(node.window) if node.window is not None else ""
        params = ""
        if node.params:
            rendered_params = [
                self.visit(p) if isinstance(p, AST.AST) else str(p) for p in node.params
            ]
            params = ", " + ", ".join(rendered_params)
        over = " ".join([clause for clause in (partition, order, window) if clause])
        # An analytic function without operand, such as rank, starts with over
        arguments = f"{operand}{params} " if operand or params else ""
        if self.pretty:
            result = f"{node.op}({nl}{tab * 3}{arguments}over({over}){nl}{tab * 2})"
        else:
            result = f"{node.op}({arguments}over ({over}))"

        return result

    def visit_Case(self, node: AST.Case) -> str:
        if self.pretty:
            else_str = f"{nl}{tab * 2}else{nl}{tab * 3}{self.visit(node.elseOp)}"
            body_sep = " " if len(node.cases) > 1 else ""
            body = body_sep.join([self.visit(x) for x in node.cases])
            return f"case {body} {else_str}"
        else:
            else_str = f"else {self.visit(node.elseOp)}"
            body_sep = " " if len(node.cases) > 1 else ""
            body = body_sep.join([self.visit(x) for x in node.cases])
            return f"case {body} {else_str}"

    def visit_CaseObj(self, node: AST.CaseObj) -> str:
        if self.pretty:
            return (
                f"{nl}{tab * 2}when{nl}{tab * 3}{self.visit(node.condition)}{nl}{tab * 2}then"
                f"{nl}{tab * 3}{self.visit(node.thenOp)}"
            )
        else:
            return f"when {self.visit(node.condition)} then {self.visit(node.thenOp)}"

    def visit_EvalOp(self, node: AST.EvalOp) -> str:
        operand_sep = ", " if len(node.operands) > 1 else ""
        if self.pretty:
            operands = operand_sep.join([self.visit(x) for x in node.operands])
            ext_routine = f"{nl}{tab * 2}{node.name}({operands})"
            language = f"{nl}{tab * 2}language {_handle_literal(node.language)}"
            output = f"{nl}{tab * 2}returns {_format_dataset_eval(node.output)}"
            return f"eval({ext_routine}{language}{output})"
        else:
            operands = operand_sep.join([self.visit(x) for x in node.operands])
            ext_routine = f"{node.name}({operands})"
            language = f"language {_handle_literal(node.language)}"
            output = f"returns {_format_type(node.output)}"
            return f"eval({ext_routine} {language} {output})"

    def visit_If(self, node: AST.If) -> str:
        if self.pretty:
            else_str = (
                f"else{nl}{tab * 5}{self.visit(node.elseOp)}" if node.elseOp is not None else ""
            )
            return (
                f"{nl}{tab * 4}if {nl}{tab * 5}{self.visit(node.condition)} "
                f"{nl}{tab * 4}then {nl}{tab * 5}{self.visit(node.thenOp)}{nl}{tab * 4}{else_str}"
            )
        else:
            else_str = f"else {self.visit(node.elseOp)}" if node.elseOp is not None else ""
            return f"if {self.visit(node.condition)} then {self.visit(node.thenOp)} {else_str}"

    def visit_JoinOp(self, node: AST.JoinOp) -> str:
        nvl_str = ""
        if node.nvl:
            nvl_parts = [
                f"nvl({_format_name(p.component)}, {self.visit(p.default)})" for p in node.nvl
            ]
            nvl_str = ", " + ", ".join(nvl_parts)
        if self.pretty:
            sep = f",{nl}{tab * 2}" if len(node.clauses) > 1 else ""
            clauses = sep.join([self.visit(x) for x in node.clauses])
            using = ""
            if node.using is not None:
                using_sep = ", " if len(node.using) > 1 else ""
                using_values = [_format_name(x) for x in node.using]
                using = f"using {using_sep.join(using_values)}{nvl_str}"
            return f"{node.op}({nl}{tab * 2}{clauses}{nl}{tab * 2}{using})"
        else:
            sep = ", " if len(node.clauses) > 1 else ""
            clauses = sep.join([self.visit(x) for x in node.clauses])
            using = ""
            if node.using is not None:
                using_sep = ", " if len(node.using) > 1 else ""
                using_values = [_format_name(x) for x in node.using]
                using = f" using {using_sep.join(using_values)}{nvl_str}"
            return f"{node.op}({clauses}{using})"

    def visit_ParFunction(self, node: AST.ParFunction) -> str:
        return f"({self.visit(node.operand)})"

    def _in_join_body(self, node: AST.RegularAggregation) -> bool:
        dataset = node.dataset
        if isinstance(dataset, AST.JoinOp):
            return not dataset.isLast
        if isinstance(dataset, AST.RegularAggregation) and not dataset.isLast:
            return self._in_join_body(dataset)
        return False

    def visit_RegularAggregation(self, node: AST.RegularAggregation) -> str:
        child_sep = ", " if len(node.children) > 1 else ""
        if node.op == AGGREGATE:
            self.is_from_agg = True
            body = child_sep.join([self.visit(x) for x in node.children])
            self.is_from_agg = False
            grouping, having = self._handle_grouping_having(node.children[0].right)
            if self.pretty:
                body = f"{nl}{tab * 3}{body}{nl}{tab * 3}{grouping}{having}{nl}{tab * 2}"
            else:
                body = f"{body}{grouping}{having}"
        elif node.op == DROP and self.pretty:
            drop_sep = f",{nl}{tab * 3}" if len(node.children) > 1 else ""
            body = f"{drop_sep.join([self.visit(x) for x in node.children])}{nl}{tab * 2}"
        elif node.op == FILTER and self.pretty:
            condition = self.visit(node.children[0])
            if " and " in condition or " or " in condition:
                for op in (" and ", " or "):
                    condition = _replace_outside_quotes(condition, op, f"{op}{nl}{tab * 5}")
            body = f"{nl}{tab * 4}{condition}{nl}{tab * 2}"
        else:
            body = child_sep.join([self.visit(x) for x in node.children])
        if self._in_join_body(node) and node.op in [
            AGGREGATE,
            CALC,
            DROP,
            FILTER,
            KEEP,
            RENAME,
            APPLY,
        ]:
            dataset = self.visit(node.dataset)
            if self.pretty:
                return f"{dataset[:-1]} {(node.op)} {body}{nl}{tab})"
            else:
                return f"{dataset[:-1]} {node.op} {body})"
        else:
            dataset = self.visit(node.dataset)
            if self.pretty:
                return f"{dataset}{nl}{tab * 2}[{node.op} {body}]"
            else:
                return f"{dataset}[{node.op} {body}]"

    def visit_RenameNode(self, node: AST.RenameNode) -> str:
        return f"{_format_name(node.old_name)} to {_format_name(node.new_name)}"

    def visit_TimeAggregation(self, node: AST.TimeAggregation) -> str:
        if node.period_to_ref is not None:
            period_to = self.visit(node.period_to_ref)
        else:
            period_to = _handle_literal(node.period_to)
        operand = "" if node.operand is None else f", {self.visit(node.operand)}"

        if node.period_from is not None:
            period_from = f", {_handle_literal(node.period_from)}"
        elif node.period_from_optional:
            period_from = ", _"
        else:
            period_from = ""
        config = "" if node.conf is None else f", {node.conf}"
        return f"{node.op}({period_to}{period_from}{operand}{config})"

    def visit_UDOCall(self, node: AST.UDOCall) -> str:
        params_sep = ", " if len(node.params) > 1 else ""
        params = params_sep.join([self.visit(x) for x in node.params])
        return f"{node.op}({params})"

    def visit_Validation(self, node: AST.Validation) -> str:
        operand = self.visit(node.validation)
        imbalance = f" imbalance {self.visit(node.imbalance)}" if node.imbalance is not None else ""
        error_code = (
            f" errorcode {_handle_literal(node.error_code)}" if node.error_code is not None else ""
        )
        error_level = (
            f" errorlevel {_handle_literal(node.error_level)}"
            if node.error_level is not None
            else ""
        )
        invalid = " invalid" if node.invalid else " all"
        if self.pretty:
            error_code = f"{nl}{tab * 2}{error_code.strip()}" if error_code else ""
            error_level = f"{nl}{tab * 2}{error_level.strip()}" if error_level else ""
            imbalance = f"{nl}{tab * 2}{imbalance.strip()}" if imbalance else ""
            invalid = f"{nl}{tab * 2}{invalid.strip()}" if invalid else ""
            return (
                f"{node.op}({nl}{tab * 2}{operand}{error_code}{error_level}"
                f"{imbalance}{invalid}{nl}{tab})"
            )
        return f"{node.op}({operand}{error_code}{error_level}{imbalance}{invalid})"

    # ---------------------- Constants and IDs ----------------------

    def visit_VarID(self, node: AST.VarID) -> str:
        return _format_name(node.value)

    def visit_Identifier(self, node: AST.Identifier) -> Any:
        return _format_name(node.value)

    def visit_Constant(self, node: AST.Constant) -> str:
        if node.value is None:
            return "null"
        return _handle_literal(node.value)

    def visit_ParamConstant(self, node: AST.ParamConstant) -> Any:
        if node.value is None:
            return "null"
        return node.value

    def visit_Collection(self, node: AST.Collection) -> str:
        if node.kind == "ValueDomain":
            return node.name
        sep = ", " if len(node.children) > 1 else ""
        return f"{{{sep.join([self.visit(x) for x in node.children])}}}"

    def visit_Windowing(self, node: AST.Windowing) -> str:
        if (
            node.type_ == "data"
            and node.start == -1
            and node.start_mode == "preceding"
            and node.stop == 0
            and node.stop_mode == "current"
        ):
            return ""
        if node.start == -1:
            start = f"unbounded {node.start_mode}"
        elif node.start_mode == "current":
            start = "current data point"
        elif isinstance(node.start, AST.AST):
            start = f"{self.visit(node.start)} {node.start_mode}"
        else:
            start = f"{node.start if node.start != 'current row' else 0} {node.start_mode}"
        if isinstance(node.stop, AST.AST):
            stop = f"{self.visit(node.stop)} {node.stop_mode}"
        else:
            stop = f"{node.stop if node.stop != 'current row' else 0} {node.stop_mode}"
        if node.stop_mode == "current":
            stop = "current data point"
        mode = "data points" if node.type_ == "data" else "range"
        return f"{mode} between {start} and {stop}"

    def visit_OrderBy(self, node: AST.OrderBy) -> str:
        if node.order == "asc":
            return f"{_format_name(node.component)}"
        return f"{_format_name(node.component)} {node.order}"

    def visit_Comment(self, node: AST.Comment) -> None:
        self.vtl_script += self._comment_text(node)

    def visit_ID(self, node: AST.ID) -> AST.AST:
        return node.value
