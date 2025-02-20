import copy
from dataclasses import dataclass
from typing import Any, Optional, Union

import vtlengine.AST.Grammar.tokens
from vtlengine import AST
from vtlengine.AST import DPRuleset, HRuleset, Operator, Start
from vtlengine.AST.ASTTemplate import ASTTemplate
from vtlengine.AST.Grammar.tokens import (
    CAST,
    CHECK_DATAPOINT,
    CHECK_HIERARCHY,
    HAVING,
    HIERARCHY,
    INSTR,
    LOG,
    MEMBERSHIP,
    MINUS,
    MOD,
    NVL,
    PLUS,
    POWER,
    RANDOM,
    REPLACE,
    ROUND,
    SUBSTR,
    TRUNC,
)
from vtlengine.DataTypes import SCALAR_TYPES_CLASS_REVERSE
from vtlengine.Model import Component, Dataset


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
            f"{component.role.value.lower()}"
            f"<{SCALAR_TYPES_CLASS_REVERSE[component.data_type].lower()}> "
            f"{component.name}"
        )

    return f"{{ {', '.join([__format_component(x) for x in dataset.components.values()])} }}"


def _format_reserved_word(value: str):
    tokens_dict = vtlengine.AST.Grammar.tokens.__dict__
    tokens_dict = {k: v for k, v in tokens_dict.items() if not k.startswith("__")}
    if value in tokens_dict.values():
        return f"'{value}'"
    return value


@dataclass
class ASTString(ASTTemplate):
    vtl_script: str = ""
    pretty: bool = False
    is_first_assignment: bool = False

    def render(self, ast: Start):
        self.visit(ast)
        return self.vtl_script

    def visit_Start(self, node: AST.Start) -> Any:
        hierarchies = [x for x in node.children if isinstance(x, HRuleset)]
        datapoints = [x for x in node.children if isinstance(x, DPRuleset)]
        udos = [x for x in node.children if isinstance(x, Operator)]
        definitions = datapoints + hierarchies + udos
        transformations = [
            x for x in node.children if not isinstance(x, (HRuleset, DPRuleset, Operator))
        ]
        for child in definitions:
            self.visit(child)
            self.vtl_script += "\n"
        for child in transformations:
            self.is_first_assignment = True
            self.visit(child)
            self.vtl_script += "\n"

    # ---------------------- Rulesets ----------------------

    def visit_HRuleset(self, node: AST.HRuleset) -> None:
        rules_sep = "; " if len(node.rules) > 1 else ""
        signature = f"{node.signature_type} rule {node.element.value}"
        rules = rules_sep.join([self.visit(x) for x in node.rules])
        self.vtl_script += (
            f"define hierarchical ruleset {node.name} ({signature}) is {rules} "
            f"end hierarchical ruleset;"
        )

    def visit_HRule(self, node: AST.HRule) -> str:
        vtl_script = ""
        if node.name is not None:
            vtl_script += f"{node.name}: "
        vtl_script += f"{self.visit(node.rule)}"
        if node.erCode is not None:
            vtl_script += f" errorcode {_handle_literal(node.erCode)}"
        if node.erLevel is not None:
            vtl_script += f" errorlevel {node.erLevel}"
        return vtl_script

    def visit_HRBinOp(self, node: AST.HRBinOp) -> str:
        if node.op == "when":
            return f"{node.op} {self.visit(node.left)} then {self.visit(node.right)}"
        return f"{self.visit(node.left)} {node.op} {self.visit(node.right)}"

    def visit_HRUnOp(self, node: AST.HRUnOp) -> str:
        return f"{node.op} {self.visit(node.operand)}"

    def visit_DefIdentifier(self, node: AST.DefIdentifier) -> str:
        return node.value

    def visit_DPRule(self, node: AST.DPRule) -> str:
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
        rules = rules_sep.join([self.visit(x) for x in node.rules])
        self.vtl_script += (
            f"define datapoint ruleset {node.name} ({signature}) is {rules} end datapoint ruleset;"
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
        body = f"returns {node.output_type.lower()} is {self.visit(node.expression)}"
        self.vtl_script += f"define operator {node.op}({signature}) {body} end operator;"

    # ---------------------- Basic Operators ----------------------
    def visit_Assignment(self, node: AST.Assignment) -> Optional[str]:
        return_element = not copy.deepcopy(self.is_first_assignment)
        if self.is_first_assignment:
            self.is_first_assignment = False
        expression = f"{self.visit(node.left)} {node.op} {self.visit(node.right)}"
        if return_element:
            return expression
        self.vtl_script += f"{expression};"

    def visit_PersistentAssignment(self, node: AST.PersistentAssignment) -> Optional[str]:
        return self.visit_Assignment(node)

    def visit_BinOp(self, node: AST.BinOp) -> str:
        if node.op in [NVL, LOG, MOD, POWER, RANDOM]:
            return f"{node.op}({self.visit(node.left)}, {self.visit(node.right)})"
        elif node.op == MEMBERSHIP:
            return f"{self.visit(node.left)}{node.op}{self.visit(node.right)}"
        return f"{self.visit(node.left)} {node.op} {self.visit(node.right)}"

    def visit_UnaryOp(self, node: AST.UnaryOp) -> str:
        if node.op in [PLUS, MINUS]:
            return f"{node.op}{self.visit(node.operand)}"
        return f"{node.op}({self.visit(node.operand)})"

    def visit_MulOp(self, node: AST.MulOp) -> str:
        sep = ", " if len(node.children) > 1 else ""
        body = sep.join([self.visit(x) for x in node.children])
        return f"{node.op}({body})"

    def visit_ParamOp(self, node: AST.ParamOp) -> str:
        if node.op == HAVING:
            return f"{node.op} {self.visit(node.params)}"
        elif node.op in [
            SUBSTR,
            INSTR,
            REPLACE,
            ROUND,
            TRUNC,
        ]:
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
            return (
                f"{node.op}({operand}, {rule_name} rule {component_name}"
                f"{param_mode}{param_input}{param_output})"
            )

        elif node.op == CHECK_DATAPOINT:
            operand = self.visit(node.children[0])
            rule_name = node.children[1]
            output = ""
            if len(node.params) == 1 and node.params[0] != "invalid":
                output = f" {node.params[0]}"
            return f"{node.op}({operand}, {rule_name}{output})"
        elif node.op == CAST:
            operand = self.visit(node.children[0])
            data_type = SCALAR_TYPES_CLASS_REVERSE[node.children[1]].lower()
            mask = ""
            if len(node.params) == 1:
                mask = f", {self.visit(node.params[0])}"
            return f"{node.op}({operand}, {data_type}{mask})"

    # ---------------------- Individual operators ----------------------

    def visit_Aggregation(self, node: AST.Aggregation) -> str:
        grouping = ""
        if node.grouping is not None:
            grouping_sep = ", " if len(node.grouping) > 1 else ""
            grouping = f" {node.grouping_op} {grouping_sep.join([x.value for x in node.grouping])}"
        having = f" {self.visit(node.having_clause)}" if node.having_clause is not None else ""
        return f"{node.op}({self.visit(node.operand)}{grouping}{having})"

    def visit_Analytic(self, node: AST.Analytic) -> str:
        return "CHECK_ANALYTIC"

    def visit_Case(self, node: AST.Case) -> str:
        else_str = f"else {self.visit(node.elseOp)}"
        body_sep = " " if len(node.cases) > 1 else ""
        body = body_sep.join([self.visit(x) for x in node.cases])
        return f"case {body} {else_str}"

    def visit_CaseObj(self, node: AST.CaseObj) -> str:
        return f"when {self.visit(node.condition)} then {self.visit(node.thenOp)}"

    def visit_EvalOp(self, node: AST.EvalOp) -> str:
        operand_sep = ", " if len(node.operands) > 1 else ""
        operands = operand_sep.join([self.visit(x) for x in node.operands])
        ext_routine = f"{node.name}({operands})"
        language = f"language {_handle_literal(node.language)}"
        output = f"returns dataset {_format_dataset_eval(node.output)}"
        return f"eval({ext_routine} {language} {output})"

    def visit_RegularAggregation(self, node: AST.RegularAggregation) -> str:
        child_sep = ", " if len(node.children) > 1 else ""
        body = child_sep.join([self.visit(x) for x in node.children])
        dataset = self.visit(node.dataset)
        return f"{dataset}[{node.op} {body}]"

    def visit_If(self, node: AST.If) -> str:
        else_str = f" else {self.visit(node.elseOp)}" if node.elseOp is not None else ""
        return f"if {self.visit(node.condition)} then {self.visit(node.thenOp)}{else_str}"

    def visit_JoinOp(self, node: AST.JoinOp) -> str:
        return "CHECK_JOIN"

    def visit_UDOCall(self, node: AST.UDOCall) -> str:
        params_sep = ", " if len(node.params) > 1 else ""
        params = params_sep.join([self.visit(x) for x in node.params])
        return f"{node.op}({params})"

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
        return _handle_literal(node.value)

    def visit_Collection(self, node: AST.Collection) -> str:
        return f"[{', '.join([self.visit(x) for x in node.children])}]"
