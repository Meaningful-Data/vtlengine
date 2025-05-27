from copy import copy, deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import pandas as pd

import vtlengine.AST as AST
import vtlengine.Exceptions
import vtlengine.Operators as Operators
from vtlengine.AST import VarID
from vtlengine.AST.ASTTemplate import ASTTemplate
from vtlengine.AST.DAG import HRDAGAnalyzer
from vtlengine.AST.DAG._words import DELETE, GLOBAL, INSERT, PERSISTENT
from vtlengine.AST.Grammar.tokens import (
    AGGREGATE,
    ALL,
    APPLY,
    AS,
    BETWEEN,
    CALC,
    CAST,
    CHECK_DATAPOINT,
    CHECK_HIERARCHY,
    COUNT,
    CURRENT_DATE,
    DATE_ADD,
    DROP,
    EQ,
    EXISTS_IN,
    EXTERNAL,
    FILL_TIME_SERIES,
    FILTER,
    HAVING,
    HIERARCHY,
    INSTR,
    KEEP,
    MEMBERSHIP,
    REPLACE,
    ROUND,
    SUBSTR,
    TRUNC,
    WHEN,
)
from vtlengine.DataTypes import (
    BASIC_TYPES,
    SCALAR_TYPES_CLASS_REVERSE,
    Boolean,
    ScalarType,
    check_unary_implicit_promotion,
)
from vtlengine.Exceptions import SemanticError
from vtlengine.files.output import save_datapoints
from vtlengine.files.output._time_period_representation import TimePeriodRepresentation
from vtlengine.files.parser import _fill_dataset_empty_data, load_datapoints
from vtlengine.Model import (
    Component,
    DataComponent,
    Dataset,
    ExternalRoutine,
    Role,
    Scalar,
    ScalarSet,
    ValueDomain,
)
from vtlengine.Operators.Aggregation import extract_grouping_identifiers
from vtlengine.Operators.Assignment import Assignment
from vtlengine.Operators.CastOperator import Cast
from vtlengine.Operators.Comparison import Between, ExistIn
from vtlengine.Operators.Conditional import Case, If
from vtlengine.Operators.General import Eval
from vtlengine.Operators.HROperators import (
    HAAssignment,
    Hierarchy,
    get_measure_from_dataset,
)
from vtlengine.Operators.Numeric import Round, Trunc
from vtlengine.Operators.String import Instr, Replace, Substr
from vtlengine.Operators.Time import (
    Current_Date,
    Date_Add,
    Fill_time_series,
    Time_Aggregation,
)
from vtlengine.Operators.Validation import Check, Check_Datapoint, Check_Hierarchy
from vtlengine.Utils import (
    AGGREGATION_MAPPING,
    ANALYTIC_MAPPING,
    BINARY_MAPPING,
    HR_COMP_MAPPING,
    HR_NUM_BINARY_MAPPING,
    HR_UNARY_MAPPING,
    JOIN_MAPPING,
    REGULAR_AGGREGATION_MAPPING,
    ROLE_SETTER_MAPPING,
    SET_MAPPING,
    THEN_ELSE,
    UNARY_MAPPING,
)


# noinspection PyTypeChecker
@dataclass
class InterpreterAnalyzer(ASTTemplate):
    # Model elements
    datasets: Dict[str, Dataset]
    value_domains: Optional[Dict[str, ValueDomain]] = None
    external_routines: Optional[Dict[str, ExternalRoutine]] = None
    # Analysis mode
    only_semantic: bool = False
    # Memory efficient
    ds_analysis: Optional[Dict[str, Any]] = None
    datapoints_paths: Optional[Dict[str, Path]] = None
    output_path: Optional[Union[str, Path]] = None
    # Time Period Representation
    time_period_representation: Optional[TimePeriodRepresentation] = None
    # Return only persistent
    return_only_persistent: bool = True
    # Flags to change behavior
    nested_condition: Union[str, bool] = False
    is_from_assignment: bool = False
    is_from_component_assignment: bool = False
    is_from_regular_aggregation: bool = False
    is_from_grouping: bool = False
    is_from_having: bool = False
    is_from_if: bool = False
    is_from_rule: bool = False
    is_from_join: bool = False
    is_from_condition: bool = False
    is_from_hr_val: bool = False
    is_from_hr_agg: bool = False
    condition_stack: Optional[List[str]] = None
    # Handlers for simplicity
    regular_aggregation_dataset: Optional[Dataset] = None
    aggregation_grouping: Optional[List[str]] = None
    aggregation_dataset: Optional[Dataset] = None
    then_condition_dataset: Optional[List[Any]] = None
    else_condition_dataset: Optional[List[Any]] = None
    ruleset_dataset: Optional[Dataset] = None
    rule_data: Optional[pd.DataFrame] = None
    ruleset_signature: Optional[Dict[str, str]] = None
    udo_params: Optional[List[Dict[str, Any]]] = None
    hr_agg_rules_computed: Optional[Dict[str, pd.DataFrame]] = None
    ruleset_mode: Optional[str] = None
    hr_input: Optional[str] = None
    hr_partial_is_valid: Optional[List[bool]] = None
    hr_condition: Optional[Dict[str, str]] = None
    # DL
    dprs: Optional[Dict[str, Optional[Dict[str, Any]]]] = None
    udos: Optional[Dict[str, Optional[Dict[str, Any]]]] = None
    hrs: Optional[Dict[str, Optional[Dict[str, Any]]]] = None

    # **********************************
    # *                                *
    # *          Memory efficient      *
    # *                                *
    # **********************************
    def _load_datapoints_efficient(self, statement_num: int) -> None:
        if self.datapoints_paths is None:
            return
        if self.ds_analysis is None:
            return
        if statement_num not in self.ds_analysis[INSERT]:
            return
        for ds_name in self.ds_analysis[INSERT][statement_num]:
            if ds_name in self.datapoints_paths:
                self.datasets[ds_name].data = load_datapoints(
                    self.datasets[ds_name].components,
                    ds_name,
                    self.datapoints_paths[ds_name],
                )
            elif ds_name in self.datasets and self.datasets[ds_name].data is None:
                _fill_dataset_empty_data(self.datasets[ds_name])

    def _save_datapoints_efficient(self, statement_num: int) -> None:
        if self.output_path is None:
            # Keeping the data in memory if no output path is provided
            return
        if self.ds_analysis is None:
            return
        if statement_num not in self.ds_analysis[DELETE]:
            return
        for ds_name in self.ds_analysis[DELETE][statement_num]:
            if (
                ds_name not in self.datasets
                or not isinstance(self.datasets[ds_name], Dataset)
                or self.datasets[ds_name].data is None
            ):
                continue
            if ds_name in self.ds_analysis[GLOBAL]:
                # We do not save global input datasets, only results of transformations
                self.datasets[ds_name].data = None
                continue
            if self.return_only_persistent and ds_name not in self.ds_analysis[PERSISTENT]:
                self.datasets[ds_name].data = None
                continue
            # Saving only datasets, no scalars
            save_datapoints(
                self.time_period_representation,
                self.datasets[ds_name],
                self.output_path,
            )
            self.datasets[ds_name].data = None

    # **********************************
    # *                                *
    # *          AST Visitors          *
    # *                                *
    # **********************************

    def visit_Start(self, node: AST.Start) -> Any:
        statement_num = 1
        if self.only_semantic:
            Operators.only_semantic = True
        else:
            Operators.only_semantic = False
        results = {}
        for child in node.children:
            if isinstance(child, (AST.Assignment, AST.PersistentAssignment)):
                vtlengine.Exceptions.dataset_output = child.left.value  # type: ignore[attr-defined]
                self._load_datapoints_efficient(statement_num)
            if not isinstance(
                child, (AST.HRuleset, AST.DPRuleset, AST.Operator)
            ) and not isinstance(child, (AST.Assignment, AST.PersistentAssignment)):
                raise SemanticError("1-3-17")
            result = self.visit(child)

            # Reset some handlers (joins and if)
            self.is_from_join = False
            self.condition_stack = None
            self.then_condition_dataset = None
            self.else_condition_dataset = None
            self.nested_condition = False

            if result is None:
                continue

            # Removing output dataset
            vtlengine.Exceptions.dataset_output = None
            # Save results
            self.datasets[result.name] = copy(result)
            results[result.name] = result
            self._save_datapoints_efficient(statement_num)
            statement_num += 1

        return results

    # Definition Language

    def visit_Operator(self, node: AST.Operator) -> None:
        if self.udos is None:
            self.udos = {}
        elif node.op in self.udos:
            raise ValueError(f"User Defined Operator {node.op} already exists")

        param_info: List[Dict[str, Union[str, Type[ScalarType], AST.AST]]] = []
        for param in node.parameters:
            if param.name in [x["name"] for x in param_info]:
                raise ValueError(f"Duplicated Parameter {param.name} in UDO {node.op}")
            # We use a string for model types, but the data type class for basic types
            # (Integer, Number, String, Boolean, ...)
            if isinstance(param.type_, (Dataset, Component, Scalar)):
                type_ = param.type_.__class__.__name__
            else:
                type_ = param.type_
            param_info.append({"name": param.name, "type": type_})
            if param.default is not None:
                param_info[-1]["default"] = param.default
            if len(param_info) > 1:
                previous_default = param_info[0]
                for i in [1, len(param_info) - 1]:
                    if previous_default and not param_info[i]:
                        raise SemanticError("1-3-12")
                    previous_default = param_info[i]

        self.udos[node.op] = {
            "params": param_info,
            "expression": node.expression,
            "output": node.output_type,
        }

    def visit_DPRuleset(self, node: AST.DPRuleset) -> None:
        # Rule names are optional, if not provided, they are generated.
        # If provided, all must be provided
        rule_names = [rule.name for rule in node.rules if rule.name is not None]
        if len(rule_names) != 0 and len(node.rules) != len(rule_names):
            raise SemanticError("1-4-1-7", type="Datapoint Ruleset", name=node.name)
        if len(rule_names) == 0:
            for i, rule in enumerate(node.rules):
                rule.name = (i + 1).__str__()

        if len(rule_names) != len(set(rule_names)):
            not_unique = [name for name in rule_names if rule_names.count(name) > 1]
            raise SemanticError(
                "1-4-1-5",
                type="Datapoint Ruleset",
                names=", ".join(not_unique),
                ruleset_name=node.name,
            )

        # Signature has the actual parameters names or aliases if provided
        signature_actual_names = {}
        if not isinstance(node.params, AST.DefIdentifier):
            for param in node.params:
                if param.alias is not None:
                    signature_actual_names[param.alias] = param.value
                else:
                    signature_actual_names[param.value] = param.value

        ruleset_data = {
            "rules": node.rules,
            "signature": signature_actual_names,
            "params": (
                [x.value for x in node.params]
                if not isinstance(node.params, AST.DefIdentifier)
                else []
            ),
            "signature_type": node.signature_type,
        }

        # Adding the ruleset to the dprs dictionary
        if self.dprs is None:
            self.dprs = {}
        elif node.name in self.dprs:
            raise ValueError(f"Datapoint Ruleset {node.name} already exists")

        self.dprs[node.name] = ruleset_data

    def visit_HRuleset(self, node: AST.HRuleset) -> None:
        if self.hrs is None:
            self.hrs = {}

        if node.name in self.hrs:
            raise ValueError(f"Hierarchical Ruleset {node.name} already exists")

        rule_names = [rule.name for rule in node.rules if rule.name is not None]
        if len(rule_names) != 0 and len(node.rules) != len(rule_names):
            raise ValueError("All rules must have a name, or none of them")
        if len(rule_names) == 0:
            for i, rule in enumerate(node.rules):
                rule.name = (i + 1).__str__()

        cond_comp: List[Any] = []
        if isinstance(node.element, list):
            cond_comp = [x.value for x in node.element[:-1]]
            node.element = node.element[-1]

        signature_actual_name = node.element.value

        ruleset_data = {
            "rules": node.rules,
            "signature": signature_actual_name,
            "condition": cond_comp,
            "node": node,
        }

        self.hrs[node.name] = ruleset_data

    # Execution Language
    def visit_Assignment(self, node: AST.Assignment) -> Any:
        if (
            self.is_from_join
            and isinstance(node.left, AST.Identifier)
            and node.left.kind == "ComponentID"
        ):
            self.is_from_component_assignment = True
        self.is_from_assignment = True
        left_operand: str = self.visit(node.left)
        self.is_from_assignment = False
        right_operand: Union[Dataset, DataComponent] = self.visit(node.right)
        self.is_from_component_assignment = False
        return Assignment.analyze(left_operand, right_operand)

    def visit_PersistentAssignment(self, node: AST.PersistentAssignment) -> Any:
        return self.visit_Assignment(node)

    def visit_ParFunction(self, node: AST.ParFunction) -> Any:
        return self.visit(node.operand)

    def visit_BinOp(self, node: AST.BinOp) -> Any:
        is_from_if = False
        if (
            not self.is_from_condition
            and node.op != MEMBERSHIP
            and self.condition_stack is not None
            and len(self.condition_stack) > 0
        ):
            is_from_if = self.is_from_if
            self.is_from_if = False

        if (
            self.is_from_join
            and node.op in [MEMBERSHIP, AGGREGATE]
            and hasattr(node.left, "value")
            and hasattr(node.right, "value")
        ):
            if self.udo_params is not None and node.right.value in self.udo_params[-1]:
                comp_name = f"{node.left.value}#{self.udo_params[-1][node.right.value]}"
            else:
                comp_name = f"{node.left.value}#{node.right.value}"
            ast_var_id = AST.VarID(
                value=comp_name,
                line_start=node.right.line_start,
                line_stop=node.right.line_stop,
                column_start=node.right.column_start,
                column_stop=node.right.column_stop,
            )
            return self.visit(ast_var_id)
        left_operand = self.visit(node.left)
        right_operand = self.visit(node.right)
        if is_from_if:
            left_operand, right_operand = self.merge_then_else_datasets(left_operand, right_operand)
        if node.op == MEMBERSHIP:
            if right_operand not in left_operand.components and "#" in right_operand:
                right_operand = right_operand.split("#")[1]
            if self.is_from_component_assignment:
                return BINARY_MAPPING[node.op].analyze(
                    left_operand, right_operand, self.is_from_component_assignment
                )
            elif self.is_from_regular_aggregation:
                raise SemanticError("1-1-6-6", dataset_name=left_operand, comp_name=right_operand)
            elif len(left_operand.get_identifiers()) == 0:
                raise SemanticError("1-3-27", op=node.op)
        return BINARY_MAPPING[node.op].analyze(left_operand, right_operand)

    def visit_UnaryOp(self, node: AST.UnaryOp) -> None:
        operand = self.visit(node.operand)
        if node.op not in UNARY_MAPPING and node.op not in ROLE_SETTER_MAPPING:
            raise NotImplementedError
        if (
            self.is_from_regular_aggregation
            and self.regular_aggregation_dataset is not None
            and node.op in ROLE_SETTER_MAPPING
        ):
            if self.regular_aggregation_dataset.data is None:
                data_size = 0
            else:
                data_size = len(self.regular_aggregation_dataset.data)
            return ROLE_SETTER_MAPPING[node.op].analyze(operand, data_size)
        return UNARY_MAPPING[node.op].analyze(operand)

    def visit_Aggregation(self, node: AST.Aggregation) -> None:
        # Having takes precedence as it is lower in the AST
        if self.is_from_having:
            if node.operand is not None:
                self.visit(node.operand)
            operand = self.aggregation_dataset
        elif self.is_from_regular_aggregation and self.regular_aggregation_dataset is not None:
            operand = self.regular_aggregation_dataset
            if node.operand is not None and operand is not None:
                op_comp: DataComponent = self.visit(node.operand)
                comps_to_keep = {}
                for (
                    comp_name,
                    comp,
                ) in self.regular_aggregation_dataset.components.items():
                    if comp.role == Role.IDENTIFIER:
                        comps_to_keep[comp_name] = copy(comp)
                comps_to_keep[op_comp.name] = Component(
                    name=op_comp.name,
                    data_type=op_comp.data_type,
                    role=op_comp.role,
                    nullable=op_comp.nullable,
                )
                if operand.data is not None:
                    data_to_keep = operand.data[operand.get_identifiers_names()]
                    data_to_keep[op_comp.name] = op_comp.data
                else:
                    data_to_keep = None
                operand = Dataset(name=operand.name, components=comps_to_keep, data=data_to_keep)
        else:
            operand = self.visit(node.operand)

        if not isinstance(operand, Dataset):
            raise SemanticError("2-3-4", op=node.op, comp="dataset")

        for comp in operand.components.values():
            if isinstance(comp.data_type, ScalarType):
                raise SemanticError("2-1-12-1", op=node.op)

        if node.having_clause is not None and node.grouping is None:
            raise SemanticError("1-3-33")

        groupings: Any = []
        having = None
        grouping_op = node.grouping_op
        if node.grouping is not None:
            if grouping_op == "group all":
                data = None if self.only_semantic else copy(operand.data)
                self.aggregation_dataset = Dataset(
                    name=operand.name, components=operand.components, data=data
                )
            # For Component handling in operators like time_agg
            self.is_from_grouping = True
            for x in node.grouping:
                groupings.append(self.visit(x))
            self.is_from_grouping = False
            if grouping_op == "group all":
                comp_grouped = groupings[0]
                if (
                    operand.data is not None
                    and comp_grouped.data is not None
                    and len(comp_grouped.data) > 0
                ):
                    operand.data[comp_grouped.name] = comp_grouped.data
                groupings = [comp_grouped.name]
                self.aggregation_dataset = None
            if node.having_clause is not None:
                self.aggregation_dataset = Dataset(
                    name=operand.name,
                    components=deepcopy(operand.components),
                    data=pd.DataFrame(columns=operand.get_components_names()),
                )
                self.aggregation_grouping = extract_grouping_identifiers(
                    operand.get_identifiers_names(), node.grouping_op, groupings
                )
                self.is_from_having = True
                # Empty data analysis on having - we do not care about the result
                self.visit(node.having_clause)
                # Reset to default values
                self.is_from_having = False
                self.aggregation_grouping = None
                self.aggregation_dataset = None
                having = getattr(node.having_clause, "expr", "")
                having = self._format_having_expression_udo(having)

        elif self.is_from_having:
            groupings = self.aggregation_grouping
            # Setting here group by as we have already selected the identifiers we need
            grouping_op = "group by"

        return AGGREGATION_MAPPING[node.op].analyze(operand, grouping_op, groupings, having)

    def _format_having_expression_udo(self, having: str) -> str:
        if self.udo_params is None:
            return having
        for k, v in self.udo_params[-1].items():
            old_param = None
            if f"{k} " in having:
                old_param = f"{k} "
            elif f" {k}" in having:
                old_param = f" {k}"
            if old_param is not None:
                if isinstance(v, str):
                    new_param = f" {v}"
                elif isinstance(v, (Dataset, Scalar)):
                    new_param = f" {v.name}"
                else:
                    new_param = f" {v.value}"
                having = having.replace(old_param, new_param)
        return having

    def visit_Analytic(self, node: AST.Analytic) -> Any:  # noqa: C901
        component_name = None
        if self.is_from_regular_aggregation:
            if self.regular_aggregation_dataset is None:
                raise SemanticError("1-1-6-10")
            if node.operand is None:
                operand = self.regular_aggregation_dataset
            else:
                operand_comp = self.visit(node.operand)
                component_name = operand_comp.name
                measure_names = self.regular_aggregation_dataset.get_measures_names()
                dataset_components = self.regular_aggregation_dataset.components.copy()
                for name in measure_names:
                    if name != operand_comp.name:
                        dataset_components.pop(name)

                if self.only_semantic or self.regular_aggregation_dataset.data is None:
                    data = None
                else:
                    data = self.regular_aggregation_dataset.data[dataset_components.keys()]

                operand = Dataset(
                    name=self.regular_aggregation_dataset.name,
                    components=dataset_components,
                    data=data,
                )

        else:
            operand = self.visit(node.operand)
        partitioning: Any = []
        ordering = []
        if self.udo_params is not None:
            if node.partition_by is not None:
                for comp_name in node.partition_by:
                    if comp_name in self.udo_params[-1]:
                        partitioning.append(self.udo_params[-1][comp_name])
                    elif comp_name in operand.get_identifiers_names():
                        partitioning.append(comp_name)
                    else:
                        raise SemanticError(
                            "2-3-9",
                            comp_type="Component",
                            comp_name=comp_name,
                            param="UDO parameters",
                        )
            if node.order_by is not None:
                for o in node.order_by:
                    if o.component in self.udo_params[-1]:
                        o.component = self.udo_params[-1][o.component]
                    elif o.component not in operand.get_identifiers_names():
                        raise SemanticError(
                            "2-3-9",
                            comp_type="Component",
                            comp_name=o.component,
                            param="UDO parameters",
                        )
                ordering = node.order_by

        else:
            partitioning = node.partition_by
            ordering = node.order_by if node.order_by is not None else []
        if not isinstance(operand, Dataset):
            raise SemanticError("2-3-4", op=node.op, comp="dataset")
        if node.partition_by is None:
            order_components = (
                [x.component for x in node.order_by] if node.order_by is not None else []
            )
            partitioning = [x for x in operand.get_identifiers_names() if x not in order_components]

        params = []
        if node.params is not None:
            for param in node.params:
                if isinstance(param, AST.Constant):
                    params.append(param.value)
                else:
                    params.append(param)

        result = ANALYTIC_MAPPING[node.op].analyze(
            operand=operand,
            partitioning=partitioning,
            ordering=ordering,
            window=node.window,
            params=params,
            component_name=component_name,
        )
        if not self.is_from_regular_aggregation:
            return result

        # Extracting the components we need (only identifiers)
        id_columns = (
            self.regular_aggregation_dataset.get_identifiers_names()
            if (self.regular_aggregation_dataset is not None)
            else None
        )

        # # Extracting the component we need (only measure)
        if component_name is None or node.op == COUNT:
            measure_name = result.get_measures_names()[0]
        else:
            measure_name = component_name
        # Joining the result with the original dataset
        if self.only_semantic:
            data = None
        else:
            if (
                self.regular_aggregation_dataset is not None
                and self.regular_aggregation_dataset.data is not None
            ):
                joined_result = pd.merge(
                    self.regular_aggregation_dataset.data[id_columns],
                    result.data,
                    on=id_columns,
                    how="inner",
                )
                data = joined_result[measure_name]
            else:
                data = None

        return DataComponent(
            name=measure_name,
            data=data,
            data_type=result.components[measure_name].data_type,
            role=result.components[measure_name].role,
            nullable=result.components[measure_name].nullable,
        )

    def visit_MulOp(self, node: AST.MulOp) -> None:
        """
        MulOp: (op, children)

        op: BETWEEN : 'between'.

        Basic usage:

            for child in node.children:
                self.visit(child)
        """
        # Comparison Operators
        if node.op == BETWEEN:
            operand_element = self.visit(node.children[0])
            from_element = self.visit(node.children[1])
            to_element = self.visit(node.children[2])

            return Between.analyze(operand_element, from_element, to_element)

        # Comparison Operators
        elif node.op == EXISTS_IN:
            dataset_1 = self.visit(node.children[0])
            if not isinstance(dataset_1, Dataset):
                raise SemanticError("2-3-11", pos="First")
            dataset_2 = self.visit(node.children[1])
            if not isinstance(dataset_2, Dataset):
                raise SemanticError("2-3-11", pos="Second")

            retain_element = None
            if len(node.children) == 3:
                retain_element = self.visit(node.children[2])
                if isinstance(retain_element, Scalar):
                    retain_element = retain_element.value
                if retain_element == ALL:
                    retain_element = None

            return ExistIn.analyze(dataset_1, dataset_2, retain_element)

        # Set Operators.
        elif node.op in SET_MAPPING:
            datasets = []
            for child in node.children:
                datasets.append(self.visit(child))

            for ds in datasets:
                if not isinstance(ds, Dataset):
                    raise ValueError(f"Expected dataset, got {type(ds).__name__}")

            return SET_MAPPING[node.op].analyze(datasets)

        elif node.op == CURRENT_DATE:
            return Current_Date.analyze()

        else:
            raise SemanticError("1-3-5", op_type="MulOp", node_op=node.op)

    def visit_VarID(self, node: AST.VarID) -> Any:  # noqa: C901
        if self.is_from_assignment:
            return node.value
        # Having takes precedence as it is lower in the AST
        if self.udo_params is not None and node.value in self.udo_params[-1]:
            udo_element = self.udo_params[-1][node.value]
            if isinstance(udo_element, (Scalar, Dataset, DataComponent)):
                return udo_element
            # If it is only the component or dataset name, we rename the node.value
            node.value = udo_element
        if self.aggregation_dataset is not None and (self.is_from_having or self.is_from_grouping):
            if node.value not in self.aggregation_dataset.components:
                raise SemanticError(
                    "1-1-1-10",
                    op=None,
                    comp_name=node.value,
                    dataset_name=self.aggregation_dataset.name,
                )
            if self.aggregation_dataset.data is None:
                data = None
            else:
                data = self.aggregation_dataset.data[node.value]
            return DataComponent(
                name=node.value,
                data=data,
                data_type=self.aggregation_dataset.components[node.value].data_type,
                role=self.aggregation_dataset.components[node.value].role,
                nullable=self.aggregation_dataset.components[node.value].nullable,
            )
        if self.is_from_regular_aggregation:
            if self.is_from_join and node.value in self.datasets:
                return self.datasets[node.value]
            if self.regular_aggregation_dataset is not None:
                if node.value in self.datasets and isinstance(self.datasets[node.value], Scalar):
                    if node.value in self.regular_aggregation_dataset.components:
                        raise SemanticError("1-1-6-11", comp_name=node.value)
                    return self.datasets[node.value]
                if self.regular_aggregation_dataset.data is not None:
                    if (
                        self.is_from_join
                        and node.value
                        not in self.regular_aggregation_dataset.get_components_names()
                    ):
                        is_partial_present = 0
                        found_comp = None
                        for comp_name in self.regular_aggregation_dataset.get_components_names():
                            if (
                                "#" in comp_name
                                and comp_name.split("#")[1] == node.value
                                or "#" in node.value
                                and node.value.split("#")[1] == comp_name
                            ):
                                is_partial_present += 1
                                found_comp = comp_name
                        if is_partial_present == 0:
                            raise SemanticError(
                                "1-1-1-10",
                                comp_name=node.value,
                                dataset_name=self.regular_aggregation_dataset.name,
                            )
                        elif is_partial_present == 2:
                            raise SemanticError("1-1-13-9", comp_name=node.value)
                        node.value = found_comp  # type:ignore[assignment]
                    if node.value not in self.regular_aggregation_dataset.components:
                        raise SemanticError(
                            "1-1-1-10",
                            comp_name=node.value,
                            dataset_name=self.regular_aggregation_dataset.name,
                        )
                    data = self.regular_aggregation_dataset.data[node.value]
                else:
                    data = None
                return DataComponent(
                    name=node.value,
                    data=data,
                    data_type=self.regular_aggregation_dataset.components[node.value].data_type,
                    role=self.regular_aggregation_dataset.components[node.value].role,
                    nullable=self.regular_aggregation_dataset.components[node.value].nullable,
                )
        if (
            self.is_from_rule
            and self.ruleset_dataset is not None
            and self.ruleset_signature is not None
        ):
            if node.value not in self.ruleset_signature:
                raise SemanticError("1-1-10-7", comp_name=node.value)
            comp_name = self.ruleset_signature[node.value]
            if comp_name not in self.ruleset_dataset.components:
                raise SemanticError(
                    "1-1-1-10",
                    comp_name=node.value,
                    dataset_name=self.ruleset_dataset.name,
                )
            data = None if self.rule_data is None else self.rule_data[comp_name]
            return DataComponent(
                name=comp_name,
                data=data,
                data_type=self.ruleset_dataset.components[comp_name].data_type,
                role=self.ruleset_dataset.components[comp_name].role,
                nullable=self.ruleset_dataset.components[comp_name].nullable,
            )
        if node.value not in self.datasets:
            raise SemanticError("2-3-6", dataset_name=node.value)
        return self.datasets[node.value]

    def visit_Collection(self, node: AST.Collection) -> Any:
        if node.kind == "Set":
            elements = []
            duplicates = []
            for child in node.children:
                ref_element = child.children[1] if isinstance(child, AST.ParamOp) else child
                if ref_element in elements:
                    duplicates.append(ref_element)
                elements.append(self.visit(child).value)
            if len(duplicates) > 0:
                raise SemanticError("1-3-9", duplicates=duplicates)
            for element in elements:
                if type(element) is not type(elements[0]):
                    raise Exception("All elements in a set must be of the same type")
            if len(elements) == 0:
                raise Exception("A set must contain at least one element")
            if len(elements) != len(set(elements)):
                raise Exception("A set must not contain duplicates")
            return ScalarSet(data_type=BASIC_TYPES[type(elements[0])], values=elements)
        elif node.kind == "ValueDomain":
            if self.value_domains is None:
                raise SemanticError("2-3-10", comp_type="Value Domains")
            if node.name not in self.value_domains:
                raise SemanticError("1-3-23", name=node.name)
            vd = self.value_domains[node.name]
            return ScalarSet(data_type=vd.type, values=vd.setlist)
        else:
            raise SemanticError("1-3-26", name=node.name)

    def visit_RegularAggregation(self, node: AST.RegularAggregation) -> None:  # noqa: C901
        operands = []
        dataset = self.visit(node.dataset)
        if isinstance(dataset, Scalar):
            raise SemanticError("1-1-1-20", op=node.op)
        self.regular_aggregation_dataset = dataset
        if node.op == APPLY:
            op_map = BINARY_MAPPING
            return REGULAR_AGGREGATION_MAPPING[node.op].analyze(dataset, node.children, op_map)
        for child in node.children:
            self.is_from_regular_aggregation = True
            operands.append(self.visit(child))
            self.is_from_regular_aggregation = False
        if node.op == CALC and any(isinstance(operand, Dataset) for operand in operands):
            raise SemanticError("1-3-35", op=node.op)
        if node.op == AGGREGATE:
            # Extracting the role encoded inside the children assignments
            role_info = {
                child.left.value: child.left.role
                for child in node.children
                if hasattr(child, "left")
            }
            dataset = copy(operands[0])
            if self.regular_aggregation_dataset is not None:
                dataset.name = self.regular_aggregation_dataset.name
            dataset.components = {
                comp_name: comp
                for comp_name, comp in dataset.components.items()
                if comp.role != Role.MEASURE
            }
            if dataset.data is not None:
                dataset.data = dataset.data[dataset.get_identifiers_names()]
            aux_operands = []
            for operand in operands:
                measure = operand.get_component(operand.get_measures_names()[0])
                data = operand.data[measure.name] if operand.data is not None else None
                # Getting role from encoded information
                # (handling also UDO params as it is present in the value of the mapping)
                if self.udo_params is not None and operand.name in self.udo_params[-1].values():
                    role = None
                    for k, v in self.udo_params[-1].items():
                        if isinstance(v, str) and v == operand.name:
                            role_key = k
                            role = role_info[role_key]
                else:
                    role = role_info[operand.name]
                aux_operands.append(
                    DataComponent(
                        name=operand.name,
                        data=data,
                        data_type=measure.data_type,
                        role=role if role is not None else measure.role,
                        nullable=measure.nullable,
                    )
                )
            operands = aux_operands
        self.regular_aggregation_dataset = None
        if node.op == FILTER:
            if not isinstance(operands[0], DataComponent) and hasattr(child, "left"):
                measure = child.left.value
                operands[0] = DataComponent(
                    name=measure,
                    data=operands[0].data[measure],
                    data_type=operands[0].components[measure].data_type,
                    role=operands[0].components[measure].role,
                    nullable=operands[0].components[measure].nullable,
                )
            return REGULAR_AGGREGATION_MAPPING[node.op].analyze(operands[0], dataset)
        if self.is_from_join:
            if node.op in [DROP, KEEP]:
                operands = [
                    (
                        operand.get_measures_names()
                        if isinstance(operand, Dataset)
                        else (
                            operand.name
                            if isinstance(operand, DataComponent)
                            and operand.role is not Role.IDENTIFIER
                            else operand
                        )
                    )
                    for operand in operands
                ]
                operands = list(
                    set(
                        [
                            item
                            for sublist in operands
                            for item in (sublist if isinstance(sublist, list) else [sublist])
                        ]
                    )
                )
            result = REGULAR_AGGREGATION_MAPPING[node.op].analyze(operands, dataset)
            if node.isLast:
                if result.data is not None:
                    result.data.rename(
                        columns={col: col[col.find("#") + 1 :] for col in result.data.columns},
                        inplace=True,
                    )
                result.components = {
                    comp_name[comp_name.find("#") + 1 :]: comp
                    for comp_name, comp in result.components.items()
                }
                for comp in result.components.values():
                    comp.name = comp.name[comp.name.find("#") + 1 :]
                if result.data is not None:
                    result.data.reset_index(drop=True, inplace=True)
                self.is_from_join = False
            return result
        return REGULAR_AGGREGATION_MAPPING[node.op].analyze(operands, dataset)

    def visit_If(self, node: AST.If) -> Dataset:
        self.is_from_condition = True
        condition = self.visit(node.condition)
        self.is_from_condition = False

        if isinstance(condition, Scalar):
            thenValue = self.visit(node.thenOp)
            elseValue = self.visit(node.elseOp)
            if not isinstance(thenValue, Scalar) or not isinstance(elseValue, Scalar):
                raise SemanticError(
                    "1-1-9-3",
                    op="If_op",
                    then_name=thenValue.name,
                    else_name=elseValue.name,
                )
            if condition.value:
                return self.visit(node.thenOp)
            else:
                return self.visit(node.elseOp)

        # Analysis for data component and dataset
        else:
            if self.condition_stack is None:
                self.condition_stack = []
            if self.then_condition_dataset is None:
                self.then_condition_dataset = []
            if self.else_condition_dataset is None:
                self.else_condition_dataset = []
            self.generate_then_else_datasets(copy(condition))

        self.condition_stack.append(THEN_ELSE["then"])
        self.is_from_if = True
        self.nested_condition = "T" if isinstance(node.thenOp, AST.If) else False
        thenOp = self.visit(node.thenOp)
        if isinstance(thenOp, Scalar) or not isinstance(node.thenOp, AST.BinOp):
            self.then_condition_dataset.pop()
            self.condition_stack.pop()

        self.condition_stack.append(THEN_ELSE["else"])
        self.is_from_if = True
        self.nested_condition = "E" if isinstance(node.elseOp, AST.If) else False
        elseOp = self.visit(node.elseOp)
        if isinstance(elseOp, Scalar) or (
            not isinstance(node.elseOp, AST.BinOp) and not isinstance(node.elseOp, AST.If)
        ):
            if len(self.else_condition_dataset) > 0:
                self.else_condition_dataset.pop()
            if len(self.condition_stack) > 0:
                self.condition_stack.pop()

        return If.analyze(condition, thenOp, elseOp)

    def visit_Case(self, node: AST.Case) -> Any:
        conditions: List[Any] = []
        thenOps: List[Any] = []

        if self.condition_stack is None:
            self.condition_stack = []

        while node.cases:
            case = node.cases.pop(0)
            self.is_from_condition = True
            conditions.append(self.visit(case.condition))
            self.is_from_condition = False
            thenOps.append(self.visit(case.thenOp))

        return Case.analyze(conditions, thenOps, self.visit(node.elseOp))

    def visit_RenameNode(self, node: AST.RenameNode) -> Any:
        if self.udo_params is not None:
            if "#" in node.old_name:
                if node.old_name.split("#")[1] in self.udo_params[-1]:
                    comp_name = self.udo_params[-1][node.old_name.split("#")[1]]
                    node.old_name = f"{node.old_name.split('#')[0]}#{comp_name}"
            else:
                if node.old_name in self.udo_params[-1]:
                    node.old_name = self.udo_params[-1][node.old_name]

        if (
            self.is_from_join
            and self.regular_aggregation_dataset is not None
            and node.old_name not in self.regular_aggregation_dataset.components
        ):
            node.old_name = node.old_name.split("#")[1]

        return node

    def visit_Constant(self, node: AST.Constant) -> Any:
        return Scalar(
            name=str(node.value),
            value=node.value,
            data_type=BASIC_TYPES[type(node.value)],
        )

    def visit_JoinOp(self, node: AST.JoinOp) -> None:
        clause_elements = []
        for clause in node.clauses:
            clause_elements.append(self.visit(clause))
            if hasattr(clause, "op") and clause.op == AS:
                # TODO: We need to delete somewhere the join datasets with alias that are added here
                self.datasets[clause_elements[-1].name] = clause_elements[-1]

        # No need to check using, regular aggregation is executed afterwards
        self.is_from_join = True
        return JOIN_MAPPING[node.op].analyze(clause_elements, node.using)

    def visit_ParamConstant(self, node: AST.ParamConstant) -> str:
        return node.value

    def visit_ParamOp(self, node: AST.ParamOp) -> None:  # noqa: C901
        if node.op == ROUND:
            op_element = self.visit(node.children[0])
            param_element = self.visit(node.params[0]) if len(node.params) != 0 else None
            return Round.analyze(op_element, param_element)

        # Numeric Operator
        elif node.op == TRUNC:
            op_element = self.visit(node.children[0])
            param_element = None
            if len(node.params) != 0:
                param_element = self.visit(node.params[0])

            return Trunc.analyze(op_element, param_element)

        elif node.op == SUBSTR or node.op == REPLACE or node.op == INSTR:
            params = [None, None, None]
            op_element = self.visit(node.children[0])
            for i, node_param in enumerate(node.params):
                params[i] = self.visit(node_param)
            param1, param2, param3 = tuple(params)
            if node.op == SUBSTR:
                return Substr.analyze(op_element, param1, param2)
            elif node.op == REPLACE:
                return Replace.analyze(op_element, param1, param2)
            elif node.op == INSTR:
                return Instr.analyze(op_element, param1, param2, param3)
            else:
                raise NotImplementedError
        elif node.op == HAVING:
            if self.aggregation_dataset is not None and self.aggregation_grouping is not None:
                for id_name in self.aggregation_grouping:
                    if id_name not in self.aggregation_dataset.components:
                        raise SemanticError("1-1-2-4", op=node.op, id_name=id_name)
                if len(self.aggregation_dataset.get_measures()) != 1:
                    raise ValueError("Only one measure is allowed")
                # Deepcopy is necessary for components to avoid changing the original dataset
                self.aggregation_dataset.components = {
                    comp_name: deepcopy(comp)
                    for comp_name, comp in self.aggregation_dataset.components.items()
                    if comp_name in self.aggregation_grouping or comp.role == Role.MEASURE
                }

                self.aggregation_dataset.data = (
                    self.aggregation_dataset.data[
                        self.aggregation_dataset.get_identifiers_names()
                        + self.aggregation_dataset.get_measures_names()
                    ]
                    if (self.aggregation_dataset.data is not None)
                    else None
                )
            result = self.visit(node.params)
            measure = result.get_measures()[0]
            if measure.data_type != Boolean:
                raise SemanticError("1-1-2-3", type=SCALAR_TYPES_CLASS_REVERSE[Boolean])
            return None
        elif node.op == FILL_TIME_SERIES:
            mode = self.visit(node.params[0]) if len(node.params) == 1 else "all"
            return Fill_time_series.analyze(self.visit(node.children[0]), mode)
        elif node.op == DATE_ADD:
            params = [self.visit(node.params[0]), self.visit(node.params[1])]
            return Date_Add.analyze(self.visit(node.children[0]), params)
        elif node.op == CAST:
            operand = self.visit(node.children[0])
            scalar_type = node.children[1]
            mask = None
            if len(node.params) > 0:
                mask = self.visit(node.params[0])
            return Cast.analyze(operand, scalar_type, mask)

        elif node.op == CHECK_DATAPOINT:
            if self.dprs is None:
                raise SemanticError("1-3-19", node_type="Datapoint Rulesets", node_value="")
            # Checking if ruleset exists
            dpr_name: Any = node.children[1]
            if dpr_name not in self.dprs:
                raise SemanticError("1-3-19", node_type="Datapoint Ruleset", node_value=dpr_name)
            dpr_info = self.dprs[dpr_name]

            # Extracting dataset
            dataset_element = self.visit(node.children[0])
            if not isinstance(dataset_element, Dataset):
                raise SemanticError("1-1-1-20", op=node.op)
            # Checking if list of components supplied is valid
            if len(node.children) > 2:
                for comp_name in node.children[2:]:
                    if comp_name.__str__() not in dataset_element.components:
                        raise SemanticError(
                            "1-1-1-10",
                            comp_name=comp_name,
                            dataset_name=dataset_element.name,
                        )
                if dpr_info is not None and dpr_info["signature_type"] == "variable":
                    for i, comp_name in enumerate(node.children[2:]):
                        if comp_name != dpr_info["params"][i]:
                            raise SemanticError(
                                "1-1-10-3",
                                op=node.op,
                                expected=dpr_info["params"][i],
                                found=comp_name,
                            )

            output: Any = node.params[0]  # invalid, all_measures, all
            if dpr_info is None:
                dpr_info = {}

            rule_output_values = {}
            self.ruleset_dataset = dataset_element
            self.ruleset_signature = dpr_info["signature"]
            self.ruleset_mode = output
            # Gather rule data, adding the ruleset dataset to the interpreter
            if dpr_info is not None:
                for rule in dpr_info["rules"]:
                    rule_output_values[rule.name] = {
                        "errorcode": rule.erCode,
                        "errorlevel": rule.erLevel,
                        "output": self.visit(rule),
                    }
            self.ruleset_mode = None
            self.ruleset_signature = None
            self.ruleset_dataset = None

            # Datapoint Ruleset final evaluation
            return Check_Datapoint.analyze(
                dataset_element=dataset_element,
                rule_info=rule_output_values,
                output=output,
            )
        elif node.op in (CHECK_HIERARCHY, HIERARCHY):
            if len(node.children) == 3:
                dataset, component, hr_name = (self.visit(x) for x in node.children)
                cond_components = []
            else:
                children = [self.visit(x) for x in node.children]
                dataset = children[0]
                component = children[1]
                hr_name = children[2]
                cond_components = children[3:]

            # Input is always dataset
            mode, input_, output = (self.visit(param) for param in node.params)

            # Sanitise the hierarchical ruleset and the call

            if self.hrs is None:
                raise SemanticError("1-3-19", node_type="Hierarchical Rulesets", node_value="")
            else:
                if hr_name not in self.hrs:
                    raise SemanticError(
                        "1-3-19", node_type="Hierarchical Ruleset", node_value=hr_name
                    )

                if not isinstance(dataset, Dataset):
                    raise SemanticError("1-1-1-20", op=node.op)

                hr_info = self.hrs[hr_name]
            if hr_info is not None:
                if len(cond_components) != len(hr_info["condition"]):
                    raise SemanticError("1-1-10-2", op=node.op)

                if (
                    hr_info["node"].signature_type == "variable"
                    and hr_info["signature"] != component
                ):
                    raise SemanticError(
                        "1-1-10-3",
                        op=node.op,
                        found=component,
                        expected=hr_info["signature"],
                    )
                elif hr_info["node"].signature_type == "valuedomain" and component is None:
                    raise SemanticError("1-1-10-4", op=node.op)

                cond_info = {}
                for i, cond_comp in enumerate(hr_info["condition"]):
                    if (
                        hr_info["node"].signature_type == "variable"
                        and cond_components[i] != cond_comp
                    ):
                        raise SemanticError(
                            "1-1-10-6",
                            op=node.op,
                            expected=cond_comp,
                            found=cond_components[i],
                        )
                    cond_info[cond_comp] = cond_components[i]

                if node.op == HIERARCHY:
                    aux = []
                    for rule in hr_info["rules"]:
                        if rule.rule.op == EQ or rule.rule.op == WHEN and rule.rule.right.op == EQ:
                            aux.append(rule)
                    # Filter only the rules with HRBinOP as =,
                    # as they are the ones that will be computed
                    if len(aux) == 0:
                        raise SemanticError("1-1-10-5")
                    hr_info["rules"] = aux

                    hierarchy_ast = AST.HRuleset(
                        name=hr_name,
                        signature_type=hr_info["node"].signature_type,
                        element=hr_info["node"].element,
                        rules=aux,
                        line_start=node.line_start,
                        line_stop=node.line_stop,
                        column_start=node.column_start,
                        column_stop=node.column_stop,
                    )
                    HRDAGAnalyzer().visit(hierarchy_ast)

                Check_Hierarchy.validate_hr_dataset(dataset, component)

                # Gather rule data, adding the necessary elements to the interpreter
                # for simplicity
                self.ruleset_dataset = dataset
                self.ruleset_signature = {**{"RULE_COMPONENT": component}, **cond_info}
                self.ruleset_mode = mode
                self.hr_input = input_
                rule_output_values = {}
                if node.op == HIERARCHY:
                    self.is_from_hr_agg = True
                    self.hr_agg_rules_computed = {}
                    for rule in hr_info["rules"]:
                        self.visit(rule)
                    self.is_from_hr_agg = False
                else:
                    self.is_from_hr_val = True
                    for rule in hr_info["rules"]:
                        rule_output_values[rule.name] = {
                            "errorcode": rule.erCode,
                            "errorlevel": rule.erLevel,
                            "output": self.visit(rule),
                        }
                    self.is_from_hr_val = False
                self.ruleset_signature = None
                self.ruleset_dataset = None
                self.ruleset_mode = None
                self.hr_input = None

                # Final evaluation
                if node.op == CHECK_HIERARCHY:
                    result = Check_Hierarchy.analyze(
                        dataset_element=dataset,
                        rule_info=rule_output_values,
                        output=output,
                    )
                    del rule_output_values
                else:
                    result = Hierarchy.analyze(dataset, self.hr_agg_rules_computed, output)
                    self.hr_agg_rules_computed = None
                return result

        raise SemanticError("1-3-5", op_type="ParamOp", node_op=node.op)

    def visit_DPRule(self, node: AST.DPRule) -> None:
        self.is_from_rule = True
        if self.ruleset_dataset is not None:
            if self.ruleset_dataset.data is None:
                self.rule_data = None
            else:
                self.rule_data = self.ruleset_dataset.data.copy()
        validation_data = self.visit(node.rule)
        if isinstance(validation_data, DataComponent):
            if self.rule_data is not None and self.ruleset_dataset is not None:
                aux = self.rule_data.loc[:, self.ruleset_dataset.get_components_names()]
                aux["bool_var"] = validation_data.data
                validation_data = aux
            else:
                validation_data = None
        if self.ruleset_mode == "invalid" and validation_data is not None:
            validation_data = validation_data[validation_data["bool_var"] == False]
        self.rule_data = None
        self.is_from_rule = False
        return validation_data

    def visit_HRule(self, node: AST.HRule) -> None:
        self.is_from_rule = True
        if self.ruleset_dataset is not None:
            self.rule_data = (
                None if self.ruleset_dataset.data is None else self.ruleset_dataset.data.copy()
            )
        rule_result = self.visit(node.rule)
        if rule_result is None:
            self.is_from_rule = False
            return None
        if self.is_from_hr_agg:
            measure_name = rule_result.get_measures_names()[0]
            if (
                self.hr_agg_rules_computed is not None
                and rule_result.data is not None
                and len(rule_result.data[measure_name]) > 0
            ):
                self.hr_agg_rules_computed[rule_result.name] = rule_result.data
        else:
            rule_result = rule_result.data
        self.rule_data = None
        self.is_from_rule = False
        return rule_result

    def visit_HRBinOp(self, node: AST.HRBinOp) -> Any:
        if node.op == WHEN:
            filter_comp = self.visit(node.left)
            if self.rule_data is None:
                return None
            filtering_indexes = list(filter_comp.data[filter_comp.data == True].index)
            nan_indexes = list(filter_comp.data[filter_comp.data.isnull()].index)
            # If no filtering indexes, then all datapoints are valid on DPR and HR
            if len(filtering_indexes) == 0 and not (self.is_from_hr_agg or self.is_from_hr_val):
                self.rule_data["bool_var"] = True
                self.rule_data.loc[nan_indexes, "bool_var"] = None
                return self.rule_data
            non_filtering_indexes = list(set(filter_comp.data.index) - set(filtering_indexes))

            original_data = self.rule_data.copy()
            self.rule_data = self.rule_data.iloc[filtering_indexes].reset_index(drop=True)
            result_validation = self.visit(node.right)
            if self.is_from_hr_agg or self.is_from_hr_val:
                # We only need to filter rule_data on DPR
                return result_validation
            self.rule_data["bool_var"] = result_validation.data
            original_data = original_data.merge(
                self.rule_data, how="left", on=original_data.columns.tolist()
            )
            original_data.loc[non_filtering_indexes, "bool_var"] = True
            original_data.loc[nan_indexes, "bool_var"] = None
            return original_data
        elif node.op in HR_COMP_MAPPING:
            self.is_from_assignment = True
            if self.ruleset_mode in ("partial_null", "partial_zero"):
                self.hr_partial_is_valid = []
            left_operand = self.visit(node.left)
            self.is_from_assignment = False
            right_operand = self.visit(node.right)
            if isinstance(right_operand, Dataset):
                right_operand = get_measure_from_dataset(right_operand, node.right.value)

            if self.ruleset_mode in ("partial_null", "partial_zero"):
                # Check all values were present in the dataset
                if self.hr_partial_is_valid and not any(self.hr_partial_is_valid):
                    right_operand.data = right_operand.data.map(lambda x: "REMOVE_VALUE")
                self.hr_partial_is_valid = []

            if self.is_from_hr_agg:
                return HAAssignment.analyze(left_operand, right_operand, self.ruleset_mode)
            else:
                result = HR_COMP_MAPPING[node.op].analyze(
                    left_operand, right_operand, self.ruleset_mode
                )
                left_measure = left_operand.get_measures()[0]
                if left_operand.data is None:
                    result.data = None
                else:
                    left_original_measure_data = left_operand.data[left_measure.name]
                    result.data[left_measure.name] = left_original_measure_data
                result.components[left_measure.name] = left_measure
                return result
        else:
            left_operand = self.visit(node.left)
            right_operand = self.visit(node.right)
            if (
                isinstance(left_operand, Dataset)
                and isinstance(right_operand, Dataset)
                and self.ruleset_mode in ("partial_null", "partial_zero")
                and not self.only_semantic
            ):
                measure_name = left_operand.get_measures_names()[0]
                if left_operand.data is None:
                    left_operand.data = pd.DataFrame({measure_name: []})
                if right_operand.data is None:
                    right_operand.data = pd.DataFrame({measure_name: []})
                left_null_indexes = set(
                    left_operand.data[left_operand.data[measure_name].isnull()].index
                )
                right_null_indexes = set(
                    right_operand.data[right_operand.data[measure_name].isnull()].index
                )
                # If no indexes are in common, then one datapoint is not null
                invalid_indexes = list(left_null_indexes.intersection(right_null_indexes))
                if len(invalid_indexes) > 0:
                    left_operand.data.loc[invalid_indexes, measure_name] = "REMOVE_VALUE"
            if isinstance(left_operand, Dataset):
                left_operand = get_measure_from_dataset(left_operand, node.left.value)
            if isinstance(right_operand, Dataset):
                right_operand = get_measure_from_dataset(right_operand, node.right.value)
            return HR_NUM_BINARY_MAPPING[node.op].analyze(left_operand, right_operand)

    def visit_HRUnOp(self, node: AST.HRUnOp) -> None:
        operand = self.visit(node.operand)
        return HR_UNARY_MAPPING[node.op].analyze(operand)

    def visit_Validation(self, node: AST.Validation) -> Dataset:
        validation_element = self.visit(node.validation)
        if not isinstance(validation_element, Dataset):
            raise ValueError(f"Expected dataset, got {type(validation_element).__name__}")

        imbalance_element = None
        if node.imbalance is not None:
            imbalance_element = self.visit(node.imbalance)
            if not isinstance(imbalance_element, Dataset):
                raise ValueError(f"Expected dataset, got {type(validation_element).__name__}")

        return Check.analyze(
            validation_element=validation_element,
            imbalance_element=imbalance_element,
            error_code=node.error_code,
            error_level=node.error_level,
            invalid=node.invalid,
        )

    def visit_EvalOp(self, node: AST.EvalOp) -> Dataset:
        """
        EvalOp: (name, children, output, language)

        Basic usage:

            for child in node.children:
                self.visit(child)
            if node.output != None:
                self.visit(node.output)

        """
        if node.language not in EXTERNAL:
            raise Exception(f"Language {node.language} not supported on Eval")

        if self.external_routines is None:
            raise SemanticError("2-3-10", comp_type="External Routines")

        if node.name not in self.external_routines:
            raise SemanticError("1-3-5", op_type="External Routine", node_op=node.name)
        external_routine = self.external_routines[node.name]
        operands = {}
        for operand in node.operands:
            element = self.visit(operand)
            if not isinstance(element, Dataset):
                raise ValueError(f"Expected dataset, got {type(element).__name__} as Eval Operand")
            operands[element.name.split(".")[1] if "." in element.name else element.name] = element
        output_to_check = node.output
        return Eval.analyze(operands, external_routine, output_to_check)

    def generate_then_else_datasets(self, condition: Union[Dataset, DataComponent]) -> None:
        components = {}
        if self.then_condition_dataset is None:
            self.then_condition_dataset = []
        if self.else_condition_dataset is None:
            self.else_condition_dataset = []
        if isinstance(condition, Dataset):
            if (
                len(condition.get_measures()) != 1
                or condition.get_measures()[0].data_type != BASIC_TYPES[bool]
            ):
                raise ValueError("Only one boolean measure is allowed on condition dataset")
            name = condition.get_measures_names()[0]
            if condition.data is None or condition.data.empty:
                data = None
            else:
                data = condition.data[name]
                components = {comp.name: comp for comp in condition.get_identifiers()}

        else:
            if condition.data_type != BASIC_TYPES[bool]:
                raise ValueError("Only boolean scalars are allowed on data component condition")
            name = condition.name
            data = None if condition.data is None else condition.data

        if data is not None:
            if self.nested_condition and self.condition_stack is not None:
                merge_df = (
                    self.then_condition_dataset[-1]
                    if self.condition_stack[-1] == THEN_ELSE["then"]
                    else self.else_condition_dataset[-1]
                )
                indexes = merge_df.data[merge_df.data.columns[-1]]
            else:
                indexes = data[data.notnull()].index

            if isinstance(condition, Dataset):
                filtered_data = data.iloc[indexes]
                then_data: Any = (
                    condition.data[condition.data[name] == True]
                    if (condition.data is not None)
                    else []
                )
                then_indexes: Any = list(filtered_data[filtered_data == True].index)
                if len(then_data) > len(then_indexes):
                    then_data = then_data.iloc[then_indexes]
                then_data[name] = then_indexes
                else_data: Any = (
                    condition.data[condition.data[name] != True]
                    if (condition.data is not None)
                    else []
                )
                else_indexes: Any = list(set(indexes) - set(then_indexes))
                if len(else_data) > len(else_indexes):
                    else_data = else_data.iloc[else_indexes]
                else_data[name] = else_indexes
            else:
                filtered_data = data.iloc[indexes]
                then_indexes = list(filtered_data[filtered_data == True].index)
                else_indexes = list(set(indexes) - set(then_indexes))
                then_data = pd.DataFrame({name: then_indexes})
                else_data = pd.DataFrame({name: else_indexes})
        else:
            then_data = pd.DataFrame({name: []})
            else_data = pd.DataFrame({name: []})
        components.update(
            {
                name: Component(
                    name=name,
                    data_type=BASIC_TYPES[int],
                    role=Role.MEASURE,
                    nullable=True,
                )
            }
        )

        if self.condition_stack and len(self.condition_stack) > 0:
            last_condition_dataset = (
                self.then_condition_dataset[-1]
                if self.condition_stack[-1] == THEN_ELSE["then"]
                else (self.else_condition_dataset[-1])
            )
            measure_name = last_condition_dataset.get_measures_names()[0]
            then_data = then_data[then_data[name].isin(last_condition_dataset.data[measure_name])]
            else_data = else_data[else_data[name].isin(last_condition_dataset.data[measure_name])]
        then_dataset = Dataset(name=name, components=components, data=then_data)
        else_dataset = Dataset(name=name, components=components, data=else_data)

        self.then_condition_dataset.append(then_dataset)
        self.else_condition_dataset.append(else_dataset)

    def merge_then_else_datasets(self, left_operand: Any, right_operand: Any) -> Any:
        if (
            self.then_condition_dataset is None
            or self.else_condition_dataset is None
            or self.condition_stack is None
        ):
            return left_operand, right_operand

        merge_dataset = (
            self.then_condition_dataset.pop()
            if self.condition_stack.pop() == THEN_ELSE["then"]
            else (self.else_condition_dataset.pop())
        )

        merge_index = merge_dataset.data[merge_dataset.get_measures_names()[0]].to_list()
        ids = merge_dataset.get_identifiers_names()
        if isinstance(left_operand, (Dataset, DataComponent)):
            if left_operand.data is None:
                return left_operand, right_operand
            if isinstance(left_operand, Dataset):
                dataset_index = left_operand.data.index[
                    left_operand.data[ids]
                    .apply(tuple, 1)
                    .isin(merge_dataset.data[ids].apply(tuple, 1))
                ]
                left = left_operand.data[left_operand.get_measures_names()[0]]
                left_operand.data[left_operand.get_measures_names()[0]] = left.reindex(
                    dataset_index, fill_value=None
                )
            else:
                left = left_operand.data
                left_operand.data = left.reindex(merge_index, fill_value=None)
        if isinstance(right_operand, (Dataset, DataComponent)):
            if right_operand.data is None:
                return left_operand, right_operand
            if isinstance(right_operand, Dataset):
                dataset_index = right_operand.data.index[
                    right_operand.data[ids]
                    .apply(tuple, 1)
                    .isin(merge_dataset.data[ids].apply(tuple, 1))
                ]
                right = right_operand.data[right_operand.get_measures_names()[0]]
                right_operand.data[right_operand.get_measures_names()[0]] = right.reindex(
                    dataset_index, fill_value=None
                )
            else:
                right = right_operand.data
                right_operand.data = right.reindex(merge_index, fill_value=None)
        return left_operand, right_operand

    def visit_Identifier(self, node: AST.Identifier) -> Union[AST.AST, Dataset, str]:
        """
        Identifier: (value)

        Basic usage:

            return node.value
        """

        if self.udo_params is not None and node.value in self.udo_params[-1]:
            return self.udo_params[-1][node.value]

        if node.value in self.datasets:
            if self.is_from_assignment:
                return self.datasets[node.value].name
            return self.datasets[node.value]
        return node.value

    def visit_DefIdentifier(self, node: AST.DefIdentifier) -> Any:
        """
        DefIdentifier: (value, kind)

        Basic usage:

            return node.value
        """
        partial_is_valid = True
        # Only for Hierarchical Rulesets
        if not (self.is_from_rule and node.kind == "CodeItemID"):
            return node.value

        # Getting Dataset elements
        result_components = {
            comp_name: copy(comp)
            for comp_name, comp in self.ruleset_dataset.components.items()  # type: ignore[union-attr]
        }
        if self.ruleset_signature is not None:
            hr_component = self.ruleset_signature["RULE_COMPONENT"]
        name = node.value

        if self.rule_data is None:
            return Dataset(name=name, components=result_components, data=None)

        condition = None
        if hasattr(node, "_right_condition"):
            condition: DataComponent = self.visit(node._right_condition)  # type: ignore[no-redef]
            if condition is not None:
                condition = condition.data[condition.data == True].index

        if (
            self.hr_agg_rules_computed is not None
            and self.hr_input == "rule"
            and node.value in self.hr_agg_rules_computed
        ):
            df = self.hr_agg_rules_computed[node.value].copy()
            return Dataset(name=name, components=result_components, data=df)

        df = self.rule_data.copy()
        if condition is not None:
            df = df.loc[condition].reset_index(drop=True)

        measure_name = self.ruleset_dataset.get_measures_names()[0]  # type: ignore[union-attr]
        if node.value in df[hr_component].values:
            rest_identifiers = [
                comp.name
                for comp in result_components.values()
                if comp.role == Role.IDENTIFIER and comp.name != hr_component
            ]
            code_data = df[df[hr_component] == node.value].reset_index(drop=True)
            code_data = code_data.merge(df[rest_identifiers], how="right", on=rest_identifiers)
            code_data = code_data.drop_duplicates().reset_index(drop=True)

            # If the value is in the dataset, we create a new row
            # based on the hierarchy mode
            # (Missing data points are considered,
            # lines 6483-6510 of the reference manual)
            if self.ruleset_mode in ("partial_null", "partial_zero"):
                # We do not care about the presence of the leftCodeItem in Hierarchy Roll-up
                if self.is_from_hr_agg and self.is_from_assignment:
                    pass
                elif code_data[hr_component].isnull().any():
                    partial_is_valid = False

            if self.ruleset_mode in ("non_zero", "partial_zero", "always_zero"):
                fill_indexes = code_data[code_data[hr_component].isnull()].index
                code_data.loc[fill_indexes, measure_name] = 0
            code_data[hr_component] = node.value
            df = code_data
        else:
            # If the value is not in the dataset, we create a new row
            # based on the hierarchy mode
            # (Missing data points are considered,
            # lines 6483-6510 of the reference manual)
            if self.ruleset_mode in ("partial_null", "partial_zero"):
                # We do not care about the presence of the leftCodeItem in Hierarchy Roll-up
                if self.is_from_hr_agg and self.is_from_assignment:
                    pass
                elif self.ruleset_mode == "partial_null":
                    partial_is_valid = False
            df = df.head(1)
            df[hr_component] = node.value
            if self.ruleset_mode in ("non_zero", "partial_zero", "always_zero"):
                df[measure_name] = 0
            else:  # For non_null, partial_null and always_null
                df[measure_name] = None
        if self.hr_partial_is_valid is not None and self.ruleset_mode in (
            "partial_null",
            "partial_zero",
        ):
            self.hr_partial_is_valid.append(partial_is_valid)
        return Dataset(name=name, components=result_components, data=df)

    def visit_UDOCall(self, node: AST.UDOCall) -> None:  # noqa: C901
        if self.udos is None:
            raise SemanticError("2-3-10", comp_type="User Defined Operators")
        elif node.op not in self.udos:
            raise SemanticError("1-3-5", node_op=node.op, op_type="User Defined Operator")

        operator = self.udos[node.op]
        signature_values = {}

        if operator is None:
            raise SemanticError("1-3-5", node_op=node.op, op_type="User Defined Operator")
        if operator["output"] == "Component" and not (
            self.is_from_regular_aggregation or self.is_from_rule
        ):
            raise SemanticError("1-3-29", op=node.op)

        for i, param in enumerate(operator["params"]):
            if i >= len(node.params):
                if "default" in param:
                    value = self.visit(param["default"]).value
                    signature_values[param["name"]] = Scalar(
                        name=str(value), value=value, data_type=BASIC_TYPES[type(value)]
                    )
                else:
                    raise SemanticError(
                        "1-3-28",
                        op=node.op,
                        received=len(node.params),
                        expected=len(operator["params"]),
                    )
            else:
                if isinstance(param["type"], str):  # Scalar, Dataset, Component
                    if param["type"] == "Scalar":
                        signature_values[param["name"]] = self.visit(node.params[i])
                    elif param["type"] in ["Dataset", "Component"]:
                        if isinstance(node.params[i], AST.VarID):
                            signature_values[param["name"]] = node.params[i].value  # type: ignore[attr-defined]
                        else:
                            param_element = self.visit(node.params[i])
                            if isinstance(param_element, Dataset):
                                if param["type"] == "Component":
                                    raise SemanticError(
                                        "1-4-1-1",
                                        op=node.op,
                                        option=param["name"],
                                        type_1=param["type"],
                                        type_2="Dataset",
                                    )
                            elif isinstance(param_element, Scalar) and param["type"] in [
                                "Dataset",
                                "Component",
                            ]:
                                raise SemanticError(
                                    "1-4-1-1",
                                    op=node.op,
                                    option=param["name"],
                                    type_1=param["type"],
                                    type_2="Scalar",
                                )
                            signature_values[param["name"]] = param_element

                    else:
                        raise NotImplementedError
                elif issubclass(param["type"], ScalarType):  # Basic types
                    # For basic Scalar types (Integer, Float, String, Boolean)
                    # We validate the type is correct and cast the value
                    param_element = self.visit(node.params[i])
                    if isinstance(param_element, (Dataset, DataComponent)):
                        type_2 = "Dataset" if isinstance(param_element, Dataset) else "Component"
                        raise SemanticError(
                            "1-4-1-1",
                            op=node.op,
                            option=param["name"],
                            type_1=param["type"],
                            type_2=type_2,
                        )
                    scalar_type = param["type"]
                    if not check_unary_implicit_promotion(param_element.data_type, scalar_type):
                        raise SemanticError(
                            "2-3-5",
                            param_type=scalar_type,
                            type_name=param_element.data_type,
                            op=node.op,
                            param_name=param["name"],
                        )
                    signature_values[param["name"]] = Scalar(
                        name=param_element.name,
                        value=scalar_type.cast(param_element.value),
                        data_type=scalar_type,
                    )
                else:
                    raise NotImplementedError

        # We set it here to a list to start the stack of UDO params
        if self.udo_params is None:
            self.udo_params = []

        # Adding parameters to the stack
        self.udo_params.append(signature_values)

        # Calling the UDO AST, we use deepcopy to avoid changing the original UDO AST
        if operator is not None:
            result = self.visit(deepcopy(operator["expression"]))

        if self.is_from_regular_aggregation or self.is_from_rule:
            result_type = "Component" if isinstance(result, DataComponent) else "Scalar"
        else:
            result_type = "Scalar" if isinstance(result, Scalar) else "Dataset"

        if result_type != operator["output"]:
            raise SemanticError(
                "1-4-1-1",
                op=node.op,
                option="output",
                type_1=operator["output"],
                type_2=result_type,
            )

        # We pop the last element of the stack (current UDO params)
        # to avoid using them in the next UDO call
        self.udo_params.pop()

        # We set to None if empty to ensure we do not use these params anymore
        if len(self.udo_params) == 0:
            self.udo_params = None
        return result

    def visit_TimeAggregation(self, node: AST.TimeAggregation) -> None:
        if node.operand is not None:
            operand = self.visit(node.operand)
        else:
            if self.aggregation_dataset is None:
                raise SemanticError("1-1-19-11")
            component_name = Time_Aggregation._get_time_id(self.aggregation_dataset)
            ast_operand = VarID(
                value=component_name,
                line_start=node.line_start,
                line_stop=node.line_stop,
                column_start=node.column_start,
                column_stop=node.column_stop,
            )
            operand = self.visit(ast_operand)
        return Time_Aggregation.analyze(
            operand=operand,
            period_from=node.period_from,
            period_to=node.period_to,
            conf=node.conf,
        )
