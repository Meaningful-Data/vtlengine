import copy
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

from vtlengine import AST
from vtlengine.AST import Comment, DPRuleset, HRuleset, Operator, TimeAggregation
from vtlengine.AST.ASTTemplate import ASTTemplate
from vtlengine.AST.Grammar.lexer import Lexer
from vtlengine.AST.Grammar.tokens import (
    AGGREGATE,
    ATTRIBUTE,
    CAST,
    CHARSET_MATCH,
    CHECK_DATAPOINT,
    CHECK_HIERARCHY,
    DATE_ADD,
    DATEDIFF,
    DROP,
    FILL_TIME_SERIES,
    FILTER,
    HAVING,
    HIERARCHY,
    IDENTIFIER,
    INSTR,
    INTERSECT,
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
    REPLACE,
    ROUND,
    SETDIFF,
    SUBSTR,
    SYMDIFF,
    TIMESHIFT,
    TRUNC,
    UNION,
    VIRAL_ATTRIBUTE,
)
from vtlengine.DataTypes import SCALAR_TYPES_CLASS_REVERSE
from vtlengine.Model import Component, Dataset

nl = "\n"
tab = "\t"


def _handle_literal(value: Union[str, int, float, bool]):
    if isinstance(value, str):
        if '"' in value:
            return value
        return f'"{value}"'
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, float):
        decimal = str(value).split(".")[1]
        if len(decimal) > 4:
            return f"{value:f}".rstrip("0")
        else:
            return f"{value:g}"
    return str(value)


def _format_dataset_eval(dataset: Dataset) -> str:
    def __format_component(component: Component) -> str:
        return (
            f"\n\t\t\t{component.role.value.lower()}"
            f"<{SCALAR_TYPES_CLASS_REVERSE[component.data_type].lower()}> "
            f"{component.name}"
        )

    return f"{{ {', '.join([__format_component(x) for x in dataset.components.values()])} \n\t\t}}"


def _format_reserved_word(value: str):
    reserved_words = {x.replace("'", ""): x for x in Lexer.literalNames}
    if value in reserved_words:
        return reserved_words[value]
    elif value[0] == "_":
        return f"'{value}'"
    return value


@dataclass
class ASTString(ASTTemplate):
    vtl_script: str = ""
    pretty: bool = False
    is_first_assignment: bool = False
    is_from_agg: bool = False  # Handler to write grouping at aggr level

    def render(self, ast: AST.AST) -> str:
        self.vtl_script = ""
        result = self.visit(ast)
        if result:
            self.vtl_script += result
        return self.vtl_script

    def visit_Start(self, node: AST.Start) -> Any:
        transformations = [
            x for x in node.children if not isinstance(x, (HRuleset, DPRuleset, Operator, Comment))
        ]
        for child in node.children:
            if child in transformations:
                self.is_first_assignment = True
            self.visit(child)
            self.vtl_script += "\n"

    # ---------------------- Rulesets ----------------------
    def visit_HRuleset(self, node: AST.HRuleset) -> None:
        signature = f"{node.signature_type} rule {node.element.value}"
        if self.pretty:
            self.vtl_script += f"define hierarchical ruleset {node.name}({signature}) is{nl}"
            for i, rule in enumerate(node.rules):
                self.vtl_script += f"{tab}{self.visit(rule)}{nl}"
                if rule.erCode:
                    self.vtl_script += f"{tab}errorcode {_handle_literal(rule.erCode)}{nl}"
                if rule.erLevel:
                    self.vtl_script += f"{tab}errorlevel {rule.erLevel}"
                    if i != len(node.rules) - 1:
                        self.vtl_script += f";{nl}"
                    self.vtl_script += nl
            self.vtl_script += f"end hierarchical ruleset;{nl}"
        else:
            rules_strs = []
            for rule in node.rules:
                rule_str = self.visit(rule)
                if rule.erCode:
                    rule_str += f" errorcode {_handle_literal(rule.erCode)}"
                if rule.erLevel:
                    rule_str += f" errorlevel {rule.erLevel}"
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
            vtl_script += f"{node.name}: "
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
        return _format_reserved_word(node.value)

    def visit_DPRule(self, node: AST.DPRule) -> str:
        if self.pretty:
            lines = []
            if node.name is not None:
                lines.append(f"{tab}{node.name}: ")
            lines.append(self.visit(node.rule))
            if node.erCode is not None:
                lines.append(f"{tab * 3}errorcode  {_handle_literal(node.erCode)}")
            if node.erLevel is not None:
                lines.append(f"{tab * 3}errorlevel {node.erLevel}")
            return nl.join(lines)
        else:
            vtl_script = ""
            if node.name is not None:
                vtl_script += f"{node.name}: "
            vtl_script += f"{self.visit(node.rule)}"
            if node.erCode is not None:
                vtl_script += f" errorcode {_handle_literal(node.erCode)}"
            if node.erLevel is not None:
                vtl_script += f" errorlevel {node.erLevel}"
            return vtl_script

    def visit_DPRIdentifier(self, node: AST.DPRIdentifier) -> str:
        vtl_script = f"{node.value}"
        if node.alias is not None:
            vtl_script += f" as {node.alias}"
        return vtl_script

    def visit_DPRuleset(self, node: AST.DPRuleset) -> None:
        rules_sep = "; " if len(node.rules) > 1 else ""
        signature_sep = ", " if len(node.params) > 1 else ""
        signature = (
            f"{node.signature_type} {signature_sep.join([self.visit(x) for x in node.params])}"
        )

        if self.pretty:
            self.vtl_script += f"define datapoint ruleset {node.name}({signature}) is {nl}"
            rules = ""
            for i, rule in enumerate(node.rules):
                rules += f"\t{self.visit(rule)}"
                if i != len(node.rules) - 1:
                    rules += f";{nl * 2}"
                else:
                    rules += f"{nl}"
            self.vtl_script += rules
            self.vtl_script += f"end datapoint ruleset;{nl}"
        else:
            rules = rules_sep.join([self.visit(x) for x in node.rules])
            self.vtl_script += (
                f"define datapoint ruleset {node.name} "
                f"({signature}) is {rules} end datapoint ruleset;"
            )

    # ---------------------- User Defined Operators ----------------------

    def visit_Argument(self, node: AST.Argument) -> str:
        default = f" default {self.visit(node.default)}" if node.default is not None else ""

        if isinstance(node.type_, Dataset):
            argument_type = "dataset"
        elif isinstance(node.type_, Component):
            argument_type = "component"
        else:
            argument_type = node.type_.__name__.lower()

        name = _format_reserved_word(node.name)

        return f"{name} {argument_type}{default}"

    def visit_Operator(self, node: AST.Operator) -> None:
        signature_sep = ", " if len(node.parameters) > 1 else ""
        signature = signature_sep.join([self.visit(x) for x in node.parameters])
        if self.pretty:
            self.vtl_script += f"define operator {node.op}({signature}){nl}"
            self.vtl_script += f"\treturns {node.output_type.lower()} is{nl}"
            expression = self.visit(node.expression)
            if "(" in expression:
                expression = expression.replace("(", f"({nl}{tab * 2}")
                expression = expression.replace(")", f"{nl}{tab * 2})")

            self.vtl_script += f"{tab * 2}{expression}{nl}"
            self.vtl_script += f"end operator;{nl}"
        else:
            body = f"returns {node.output_type.lower()} is {self.visit(node.expression)}"
            self.vtl_script += f"define operator {node.op}({signature}) {body} end operator;"

    # ---------------------- Basic Operators ----------------------
    def visit_Assignment(self, node: AST.Assignment) -> Optional[str]:
        return_element = not copy.deepcopy(self.is_first_assignment)
        is_first = self.is_first_assignment
        if is_first:
            self.is_first_assignment = False
        if self.pretty:
            if is_first:
                expression = f"{self.visit(node.left)} {node.op}{nl}{tab}{self.visit(node.right)}"
            else:
                expression = f"{self.visit(node.left)} {node.op} {self.visit(node.right)}"
        else:
            expression = f"{self.visit(node.left)} {node.op} {self.visit(node.right)}"
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
            return self.visit(node.operand)

        return f"{node.op}({self.visit(node.operand)})"

    def visit_MulOp(self, node: AST.MulOp) -> str:
        sep = ", " if len(node.children) > 1 else ""
        body = sep.join([self.visit(x) for x in node.children])
        if self.pretty:
            return f"{node.op}({body})"
        return f"{node.op}({body})"

    def visit_ParamOp(self, node: AST.ParamOp) -> str:
        if node.op == HAVING:
            return f"{node.op} {self.visit(node.params)}"
        elif node.op in [SUBSTR, INSTR, REPLACE, ROUND, TRUNC, UNION, SETDIFF, SYMDIFF, INTERSECT]:
            params_sep = ", " if len(node.params) > 1 else ""
            return (
                f"{node.op}({self.visit(node.children[0])}, "
                f"{params_sep.join([self.visit(x) for x in node.params])})"
            )
        elif node.op in (CHECK_HIERARCHY, HIERARCHY):
            operand = self.visit(node.children[0])
            component_name = self.visit(node.children[1])
            rule_name = self.visit(node.children[2])
            param_mode_value = node.params[0].value
            param_input_value = node.params[1].value
            param_output_value = node.params[2].value

            default_value_input = "dataset" if node.op == CHECK_HIERARCHY else "rule"
            default_value_output = "invalid" if node.op == CHECK_HIERARCHY else "computed"

            param_mode = f" {param_mode_value}" if param_mode_value != "non_null" else ""
            param_input = (
                f" {param_input_value}" if param_input_value != default_value_input else ""
            )
            param_output = (
                f" {param_output_value}" if param_output_value != default_value_output else ""
            )
            if self.pretty:
                return (
                    f"{node.op}({nl}{tab * 2}{operand},{nl}{tab * 2}{rule_name}{nl}{tab * 2}rule "
                    f"{component_name}"
                    f"{param_mode}{param_input}{param_output})"
                )
            else:
                return (
                    f"{node.op}({operand}, {rule_name} rule {component_name}"
                    f"{param_mode}{param_input}{param_output})"
                )

        elif node.op == CHECK_DATAPOINT:
            operand = self.visit(node.children[0])
            rule_name = node.children[1]
            output = ""
            if len(node.params) == 1 and node.params[0] != "invalid":
                output = f"{node.params[0]}"
            if self.pretty:
                return f"{node.op}({nl}{tab}{operand},{nl}{tab}{rule_name}{nl}{tab}{output}{nl})"
            else:
                return f"{node.op}({operand}, {rule_name}{output})"
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

    # ---------------------- Individual operators ----------------------

    def _handle_grouping_having(self, node: AST) -> Tuple[str, str]:
        if self.is_from_agg:
            return "", ""
        grouping = ""
        if node.grouping is not None:
            grouping_sep = ", " if len(node.grouping) > 1 else ""
            grouping_values = []
            for grouping_value in node.grouping:
                if isinstance(grouping_value, TimeAggregation):
                    grouping_values.append(self.visit(grouping_value))
                else:
                    grouping_values.append(_format_reserved_word(grouping_value.value))
            grouping = f" {node.grouping_op} {grouping_sep.join(grouping_values)}"
        having = f" {self.visit(node.having_clause)}" if node.having_clause is not None else ""
        return grouping, having

    def visit_Aggregation(self, node: AST.Aggregation) -> str:
        grouping, having = self._handle_grouping_having(node)
        if self.pretty and node.op not in (MAX, MIN):
            operand = self.visit(node.operand)
            return f"{node.op}({nl}{tab * 2}{operand}{grouping}{having}{nl}{tab * 2})"
        return f"{node.op}({self.visit(node.operand)}{grouping}{having})"

    def visit_Analytic(self, node: AST.Analytic) -> str:
        operand = "" if node.operand is None else self.visit(node.operand)
        partition = ""
        if node.partition_by:
            partition_sep = ", " if len(node.partition_by) > 1 else ""
            partition_values = [_format_reserved_word(x) for x in node.partition_by]
            partition = f"partition by {partition_sep.join(partition_values)}"
        order = ""
        if node.order_by:
            order_sep = ", " if len(node.order_by) > 1 else ""
            order = f" order by {order_sep.join([self.visit(x) for x in node.order_by])}"
        window = f" {self.visit(node.window)}" if node.window is not None else ""
        params = ""
        if node.params:
            params = "" if len(node.params) == 0 else f", {int(node.params[0])}"
        if self.pretty:
            result = (
                f"{node.op}({nl}{tab * 3}{operand}{params} over({partition}{order} {window})"
                f"{nl}{tab * 2})"
            )
        else:
            result = f"{node.op}({operand}{params} over ({partition}{order}{window}))"

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
            language = f"{nl}{tab * 2}language {_handle_literal(node.language)}{nl}"
            output = f"{tab * 2}returns dataset {_format_dataset_eval(node.output)}"
            return f"eval({ext_routine} {language} {output})"
        else:
            operands = operand_sep.join([self.visit(x) for x in node.operands])
            ext_routine = f"{node.name}({operands})"
            language = f"language {_handle_literal(node.language)}"
            output = f"returns dataset {_format_dataset_eval(node.output)}"
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
        if self.pretty:
            sep = f",{nl}{tab * 2}" if len(node.clauses) > 1 else ""
            clauses = sep.join([self.visit(x) for x in node.clauses])
            using = ""
            if node.using is not None:
                using_sep = ", " if len(node.using) > 1 else ""
                using_values = [_format_reserved_word(x) for x in node.using]
                using = f"using {using_sep.join(using_values)}"
            return f"{node.op}({nl}{tab * 2}{clauses}{nl}{tab * 2}{using})"
        else:
            sep = ", " if len(node.clauses) > 1 else ""
            clauses = sep.join([self.visit(x) for x in node.clauses])
            using = ""
            if node.using is not None:
                using_sep = ", " if len(node.using) > 1 else ""
                using_values = [_format_reserved_word(x) for x in node.using]
                using = f" using {using_sep.join(using_values)}"
            return f"{node.op}({clauses}{using})"

    def visit_ParFunction(self, node: AST.ParFunction) -> str:
        return f"({self.visit(node.operand)})"

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
                    condition = condition.replace(op, f"{op}{nl}{tab * 5}")
            body = f"{nl}{tab * 4}{condition}{nl}{tab * 2}"
        else:
            body = child_sep.join([self.visit(x) for x in node.children])
        if isinstance(node.dataset, AST.JoinOp):
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
        return f"{node.old_name} to {node.new_name}"

    def visit_TimeAggregation(self, node: AST.TimeAggregation) -> str:
        operand = self.visit(node.operand)
        period_from = "_" if node.period_from is None else _handle_literal(node.period_from)
        period_to = _handle_literal(node.period_to)
        if self.pretty:
            return f"{node.op}({period_to}, {period_from}, {operand})"
        else:
            return f"{node.op}({period_to}, {period_from}, {operand})"

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
        error_level = f" errorlevel {node.error_level}" if node.error_level is not None else ""
        invalid = " invalid" if node.invalid else " all"
        return f"{node.op}({operand}{error_code}{error_level}{imbalance}{invalid})"

    # ---------------------- Constants and IDs ----------------------

    def visit_VarID(self, node: AST.VarID) -> str:
        return _format_reserved_word(node.value)

    def visit_Identifier(self, node: AST.Identifier) -> Any:
        return _format_reserved_word(node.value)

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
        else:
            start = f"{node.start} {node.start_mode}"
        stop = f"{node.stop} {node.stop_mode}"
        if node.stop_mode == "current":
            stop = "current data point"
        mode = "data points" if node.type_ == "data" else "range"
        return f"{mode} between {start} and {stop}"

    def visit_OrderBy(self, node: AST.OrderBy) -> str:
        if node.order == "asc":
            return f"{_format_reserved_word(node.component)}"
        return f"{_format_reserved_word(node.component)} {node.order}"

    def visit_Comment(self, node: AST.Comment) -> None:
        value = copy.copy(node.value)
        value = value[:-1] if value[-1] == "\n" else value
        self.vtl_script += value
