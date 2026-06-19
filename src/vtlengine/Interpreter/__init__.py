from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union

import vtlengine.AST as AST
import vtlengine.Exceptions
from vtlengine.AST.ASTTemplate import ASTTemplate
from vtlengine.AST.DAG import HRDAGAnalyzer
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
    STRING_DISTANCE,
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
from vtlengine.Model import (
    CaseInsensitiveDict,
    Component,
    DataComponent,
    Dataset,
    ExternalRoutine,
    Role,
    Scalar,
    ScalarSet,
    ValueDomain,
    names_equal,
    normalize_name,
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
from vtlengine.Operators.String import DISTANCE_DISPATCH, Instr, Replace, Substr
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
    UNARY_MAPPING,
)
from vtlengine.Utils.__Virtual_Assets import VirtualCounter
from vtlengine.ViralPropagation import (
    ViralPropagationRegistry,
    ViralPropagationRule,
    get_current_registry,
    set_current_registry,
)


# noinspection PyTypeChecker
@dataclass
class InterpreterAnalyzer(ASTTemplate):
    # Model elements
    datasets: Dict[str, Dataset]
    scalars: Optional[Dict[str, Scalar]] = None
    value_domains: Optional[Dict[str, ValueDomain]] = None
    external_routines: Optional[Dict[str, ExternalRoutine]] = None
    # Flags to change behavior
    is_from_assignment: bool = False
    is_from_component_assignment: bool = False
    is_from_regular_aggregation: bool = False
    is_from_grouping: bool = False
    is_from_having: bool = False
    is_from_rule: bool = False
    is_from_join: bool = False
    is_from_hr_agg: bool = False
    # Handlers for simplicity
    regular_aggregation_dataset: Optional[Dataset] = None
    aggregation_grouping: Optional[List[str]] = None
    aggregation_dataset: Optional[Dataset] = None
    ruleset_dataset: Optional[Dataset] = None
    ruleset_signature: Optional[Dict[str, str]] = None
    udo_params: Optional[List[Dict[str, Any]]] = None
    ruleset_mode: Optional[str] = None
    # DL
    dprs: Optional[Dict[str, Optional[Dict[str, Any]]]] = None
    udos: Optional[Dict[str, Optional[Dict[str, Any]]]] = None
    hrs: Optional[Dict[str, Optional[Dict[str, Any]]]] = None
    signature_values: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        # VTL regular names are case-insensitive: keep all symbol tables in
        # CaseInsensitiveDict so lookups match regardless of the written casing.
        self.datasets = CaseInsensitiveDict(self.datasets)
        if self.scalars is not None:
            self.scalars = CaseInsensitiveDict(self.scalars)
        if self.value_domains is not None:
            self.value_domains = CaseInsensitiveDict(self.value_domains)
        if self.external_routines is not None:
            self.external_routines = CaseInsensitiveDict(self.external_routines)
        self.datasets_inputs = {normalize_name(k) for k in self.datasets}
        self.scalars_inputs = {normalize_name(k) for k in self.scalars} if self.scalars else set()

    # **********************************
    # *                                *
    # *          AST Visitors          *
    # *                                *
    # **********************************

    def visit_Start(self, node: AST.Start) -> Any:
        set_current_registry(ViralPropagationRegistry())

        results = {}
        invalid_dataset_outputs = []
        invalid_scalar_outputs = []
        for child in node.children:
            if isinstance(child, (AST.Assignment, AST.PersistentAssignment)):
                vtlengine.Exceptions.dataset_output = child.left.value  # type: ignore[attr-defined]
            if not isinstance(
                child,
                (AST.HRuleset, AST.DPRuleset, AST.Operator, AST.ViralPropagationDef),
            ) and not isinstance(child, (AST.Assignment, AST.PersistentAssignment)):
                raise SemanticError("1-2-5")
            result = self.visit(child)
            if isinstance(result, Dataset) and normalize_name(result.name) in self.datasets_inputs:
                invalid_dataset_outputs.append(result.name)
            if isinstance(result, Scalar) and normalize_name(result.name) in self.scalars_inputs:
                invalid_scalar_outputs.append(result.name)

            self.is_from_join = False
            VirtualCounter.reset()

            if result is None:
                continue

            vtlengine.Exceptions.dataset_output = None
            self.datasets[result.name] = copy(result)
            results[result.name] = result
            if isinstance(result, Scalar):
                if self.scalars is None:
                    self.scalars = CaseInsensitiveDict()
                self.scalars[result.name] = copy(result)
        if invalid_dataset_outputs:
            raise SemanticError("0-1-2-8", names=", ".join(invalid_dataset_outputs))
        if invalid_scalar_outputs:
            raise SemanticError("0-1-2-8", names=", ".join(invalid_scalar_outputs))

        return results

    # Definition Language

    def visit_Operator(self, node: AST.Operator) -> None:
        if self.udos is None:
            self.udos = CaseInsensitiveDict()
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
            raise SemanticError("1-3-1-7", type="Datapoint Ruleset", name=node.name)
        if len(rule_names) == 0:
            for i, rule in enumerate(node.rules):
                rule.name = (i + 1).__str__()

        if len(rule_names) != len(set(rule_names)):
            not_unique = [name for name in rule_names if rule_names.count(name) > 1]
            raise SemanticError(
                "1-3-1-5",
                type="Datapoint Ruleset",
                names=", ".join(not_unique),
                ruleset_name=node.name,
            )

        # Signature has the actual parameters names or aliases if provided
        signature_actual_names: CaseInsensitiveDict[str] = CaseInsensitiveDict()
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
            self.dprs = CaseInsensitiveDict()
        elif node.name in self.dprs:
            raise ValueError(f"Datapoint Ruleset {node.name} already exists")

        self.dprs[node.name] = ruleset_data

    def visit_HRuleset(self, node: AST.HRuleset) -> None:
        if self.hrs is None:
            self.hrs = CaseInsensitiveDict()

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

    def visit_ViralPropagationDef(self, node: AST.ViralPropagationDef) -> None:
        """Validate and store the viral propagation definition in the registry."""
        registry = get_current_registry()

        # Validate: cannot mix enumerated and aggregate clauses
        if node.enumerated_clauses and node.aggregate_clause:
            raise SemanticError("1-3-3-3", name=node.name)

        # Validate: no duplicate enumeration combinations
        seen_values: Set[frozenset[str]] = set()
        for clause in node.enumerated_clauses:
            key = frozenset(clause.values)
            if key in seen_values:
                raise SemanticError("1-3-3-4", values=clause.values, name=node.name)
            seen_values.add(key)

        # Validate: no duplicate rules for the same target (variable or value domain)
        existing = registry.get_existing(node.signature_type, node.target)
        if existing is not None:
            code = "1-3-3-1" if node.signature_type == "variable" else "1-3-3-2"
            raise SemanticError(code, name=node.target)

        enumerated_clauses = [
            {"values": clause.values, "result": clause.result} for clause in node.enumerated_clauses
        ]
        aggregate_function = node.aggregate_clause.function if node.aggregate_clause else None

        rule = ViralPropagationRule(
            name=node.name,
            signature_type=node.signature_type,
            target=node.target,
            enumerated_clauses=enumerated_clauses,
            aggregate_function=aggregate_function,
            default_value=node.default_value,
        )
        registry.register(rule)

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
        result = Assignment.validate(left_operand, right_operand)
        if isinstance(result, (Dataset, Scalar)):
            result.persistent = isinstance(node, AST.PersistentAssignment)
        return result

    def visit_PersistentAssignment(self, node: AST.PersistentAssignment) -> Any:
        return self.visit_Assignment(node)

    def visit_ParFunction(self, node: AST.ParFunction) -> Any:
        return self.visit(node.operand)

    def visit_BinOp(self, node: AST.BinOp) -> Any:
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

        if node.op == MEMBERSHIP:
            if right_operand not in left_operand.components and "#" in right_operand:
                right_operand = right_operand.split("#")[1]
            if not self.is_from_component_assignment:
                if self.is_from_regular_aggregation:
                    raise SemanticError(
                        "1-1-6-6", dataset_name=left_operand, comp_name=right_operand
                    )
                if len(left_operand.get_identifiers()) == 0:
                    raise SemanticError("1-2-10", op=node.op)
        return BINARY_MAPPING[node.op].validate(left_operand, right_operand)

    def visit_UnaryOp(self, node: AST.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        if node.op not in UNARY_MAPPING and node.op not in ROLE_SETTER_MAPPING:
            raise NotImplementedError
        if (
            self.is_from_regular_aggregation
            and self.regular_aggregation_dataset is not None
            and node.op in ROLE_SETTER_MAPPING
        ):
            return ROLE_SETTER_MAPPING[node.op].validate(operand)
        return UNARY_MAPPING[node.op].validate(operand)

    @staticmethod
    def _apply_time_agg_grouping(
        groupings: List[Any],
        grouping_op: Optional[str],
    ) -> List[Any]:
        """Extract TimeAggregation DataComponent name from groupings."""
        time_comp = None
        regular_groupings: List[Any] = []
        for g in groupings:
            if isinstance(g, DataComponent):
                time_comp = g
            else:
                regular_groupings.append(g)
        if time_comp is not None and grouping_op != "group except":
            regular_groupings.append(time_comp.name)
        return regular_groupings

    def _resolve_aggregation_operand(self, node: AST.Aggregation) -> Any:
        """Resolve the operand for an aggregation node."""
        if self.is_from_having:
            if node.operand is not None:
                self.visit(node.operand)
            return self.aggregation_dataset
        if self.is_from_regular_aggregation and self.regular_aggregation_dataset is not None:
            operand = self.regular_aggregation_dataset
            if node.operand is not None and operand is not None:
                op_comp: DataComponent = self.visit(node.operand)
                comps_to_keep = {
                    comp_name: copy(comp)
                    for comp_name, comp in self.regular_aggregation_dataset.components.items()
                    if comp.role == Role.IDENTIFIER
                }
                comps_to_keep[op_comp.name] = Component(
                    name=op_comp.name,
                    data_type=op_comp.data_type,
                    role=op_comp.role,
                    nullable=op_comp.nullable,
                )
                return Dataset(name=operand.name, components=comps_to_keep, data=None)
            return operand
        return self.visit(node.operand)

    def visit_Aggregation(self, node: AST.Aggregation) -> Any:
        operand = self._resolve_aggregation_operand(node)

        if not isinstance(operand, Dataset):
            raise SemanticError("2-3-4", op=node.op, comp="dataset")

        for comp in operand.components.values():
            if isinstance(comp.data_type, ScalarType):
                raise SemanticError("2-1-12-1", op=node.op)

        if node.having_clause is not None and node.grouping is None:
            raise SemanticError("1-2-13")

        groupings: Any = []
        grouping_op = node.grouping_op
        if node.grouping is not None:
            has_time_agg = any(isinstance(x, AST.TimeAggregation) for x in node.grouping)
            if grouping_op == "group all" or has_time_agg:
                self.aggregation_dataset = Dataset(
                    name=operand.name, components=operand.components, data=None
                )
            self.is_from_grouping = True
            for x in node.grouping:
                groupings.append(self.visit(x))
            self.is_from_grouping = False
            if grouping_op == "group all" or has_time_agg:
                groupings = self._apply_time_agg_grouping(groupings, grouping_op)
                self.aggregation_dataset = None
            if node.having_clause is not None:
                self.aggregation_dataset = Dataset(
                    name=operand.name,
                    components=deepcopy(operand.components),
                    data=None,
                )
                self.aggregation_grouping = extract_grouping_identifiers(
                    operand.get_identifiers_names(), node.grouping_op, groupings
                )
                self.is_from_having = True
                self.visit(node.having_clause)
                self.is_from_having = False
                self.aggregation_grouping = None
                self.aggregation_dataset = None

        elif self.is_from_having:
            groupings = self.aggregation_grouping
            grouping_op = "group by"

        result = AGGREGATION_MAPPING[node.op].validate(operand, grouping_op, groupings)
        if not self.is_from_regular_aggregation:
            result.name = VirtualCounter._new_ds_name()
        return result

    def visit_Analytic(self, node: AST.Analytic) -> Any:  # noqa: C901
        component_name = None
        analytic_component_name: Optional[str] = None
        operand_id_collision = False
        if self.is_from_regular_aggregation:
            if self.regular_aggregation_dataset is None:
                raise SemanticError("1-1-6-10")
            if node.operand is None:
                operand = self.regular_aggregation_dataset
            else:
                operand_comp = self.visit(node.operand)
                component_name = operand_comp.name
                id_names = self.regular_aggregation_dataset.get_identifiers_names()
                measure_names = self.regular_aggregation_dataset.get_measures_names()
                attribute_names = self.regular_aggregation_dataset.get_attributes_names()
                dataset_components = self.regular_aggregation_dataset.components.copy()
                for name in measure_names + attribute_names:
                    dataset_components.pop(name)

                operand_id_collision = (
                    operand_comp.role == Role.IDENTIFIER and operand_comp.name in id_names
                )
                analytic_component_name = (
                    f"__vtl_op_{operand_comp.name}" if operand_id_collision else operand_comp.name
                )
                analytic_role = Role.MEASURE if operand_id_collision else operand_comp.role

                dataset_components[analytic_component_name] = Component(
                    name=analytic_component_name,
                    data_type=operand_comp.data_type,
                    role=analytic_role,
                    nullable=operand_comp.nullable,
                )

                operand = Dataset(
                    name=self.regular_aggregation_dataset.name,
                    components=dataset_components,
                    data=None,
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
            if node.order_by is None:
                partitioning = []
            else:
                order_components = [x.component for x in node.order_by]
                partitioning = [
                    x for x in operand.get_identifiers_names() if x not in order_components
                ]

        if node.partition_op == "except all":
            partitioning = []
        elif node.partition_op == "except":
            listed = set(partitioning or [])
            partitioning = [i for i in operand.get_identifiers_names() if i not in listed]

        params = []
        if node.params is not None:
            for param in node.params:
                if isinstance(param, AST.Constant):
                    params.append(param.value)
                elif isinstance(param, AST.VarID):
                    resolved = self.visit(param)
                    params.append(resolved.value if hasattr(resolved, "value") else resolved)
                else:
                    params.append(param)

        window = node.window
        if window is not None and (
            isinstance(window.start, AST.VarID) or isinstance(window.stop, AST.VarID)
        ):
            window = copy(window)
            if isinstance(window.start, AST.VarID):
                start = self.visit(window.start)
                window.start = start.value if hasattr(start, "value") else start
            if isinstance(window.stop, AST.VarID):
                stop = self.visit(window.stop)
                window.stop = stop.value if hasattr(stop, "value") else stop

        result = ANALYTIC_MAPPING[node.op].validate(
            operand=operand,
            partitioning=partitioning,
            ordering=ordering,
            window=window,
            params=params,  # type: ignore[arg-type]
            component_name=analytic_component_name,
        )
        if not self.is_from_regular_aggregation:
            return result

        # # Extracting the component we need (only measure)
        if analytic_component_name is None or node.op == COUNT:
            measure_name = result.get_measures_names()[0]
        else:
            measure_name = analytic_component_name
        output_name = (
            component_name if operand_id_collision and component_name is not None else measure_name
        )
        return DataComponent(
            name=output_name,
            data=None,
            data_type=result.components[measure_name].data_type,
            role=result.components[measure_name].role,
            nullable=result.components[measure_name].nullable,
        )

    def visit_MulOp(self, node: AST.MulOp) -> Any:
        if node.op == BETWEEN:
            operand_element = self.visit(node.children[0])
            from_element = self.visit(node.children[1])
            to_element = self.visit(node.children[2])

            return Between.validate(operand_element, from_element, to_element)

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

            return ExistIn.validate(dataset_1, dataset_2, retain_element)

        elif node.op in SET_MAPPING:
            datasets = []
            for child in node.children:
                datasets.append(self.visit(child))

            for ds in datasets:
                if not isinstance(ds, Dataset):
                    raise ValueError(f"Expected dataset, got {type(ds).__name__}")

            return SET_MAPPING[node.op].validate(datasets)

        elif node.op == CURRENT_DATE:
            return Current_Date.validate()

        else:
            raise SemanticError("1-3-5", op_type="MulOp", node_op=node.op)

    def visit_VarID(self, node: AST.VarID) -> Any:  # noqa: C901
        if self.is_from_assignment:
            return node.value
        # Having takes precedence as it is lower in the AST
        if self.udo_params is not None and node.value in self.udo_params[-1]:
            udo_element = copy(self.udo_params[-1][node.value])
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
            return DataComponent(
                name=node.value,
                data=None,
                data_type=self.aggregation_dataset.components[node.value].data_type,
                role=self.aggregation_dataset.components[node.value].role,
                nullable=self.aggregation_dataset.components[node.value].nullable,
            )
        if self.is_from_regular_aggregation:
            if self.is_from_join and node.value in self.datasets:
                return copy(self.datasets[node.value])
            if self.regular_aggregation_dataset is not None:
                if self.scalars is not None and node.value in self.scalars:
                    if node.value in self.regular_aggregation_dataset.components:
                        raise SemanticError("1-1-6-11", comp_name=node.value)
                    return copy(self.scalars[node.value])
                if (
                    self.is_from_join
                    and node.value not in self.regular_aggregation_dataset.get_components_names()
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
                return DataComponent(
                    name=node.value,
                    data=None,
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
            return DataComponent(
                name=comp_name,
                data=None,
                data_type=self.ruleset_dataset.components[comp_name].data_type,
                role=self.ruleset_dataset.components[comp_name].role,
                nullable=self.ruleset_dataset.components[comp_name].nullable,
            )
        if self.scalars and node.value in self.scalars:
            return copy(self.scalars[node.value])
        if node.value not in self.datasets:
            raise SemanticError("2-3-6", dataset_name=node.value)

        return copy(self.datasets[node.value])

    def visit_Collection(self, node: AST.Collection) -> Any:
        if node.kind == "Set":
            elements = []
            scalar_data_type = None
            duplicates = []
            for child in node.children:
                ref_element = child.children[1] if isinstance(child, AST.ParamOp) else child
                if ref_element in elements:
                    duplicates.append(ref_element)
                scalar = self.visit(child)
                elements.append(scalar.value)
                if scalar_data_type is None:
                    scalar_data_type = scalar.data_type
            if len(duplicates) > 0:
                raise SemanticError("1-2-5", duplicates=duplicates)
            for element in elements:
                if type(element) is not type(elements[0]):
                    raise Exception("All elements in a set must be of the same type")
            if len(elements) == 0:
                raise Exception("A set must contain at least one element")
            if not any(e is None for e in elements) and len(elements) != len(set(elements)):
                raise Exception("A set must not contain duplicates")
            set_type = scalar_data_type or BASIC_TYPES[type(elements[0])]
            return ScalarSet(data_type=set_type, values=elements)
        elif node.kind == "ValueDomain":
            if self.value_domains is None:
                raise SemanticError("2-3-10", comp_type="Value Domains")
            if node.name not in self.value_domains:
                raise SemanticError("1-2-8", name=node.name)
            vd = self.value_domains[node.name]
            return ScalarSet(data_type=vd.type, values=vd.setlist)
        else:
            raise SemanticError("1-2-9", name=node.name)

    def visit_RegularAggregation(self, node: AST.RegularAggregation) -> None:  # noqa: C901
        operands = []
        dataset = self.visit(node.dataset)
        if isinstance(dataset, Scalar):
            raise SemanticError("1-1-1-20", op=node.op)
        self.regular_aggregation_dataset = dataset
        if node.op == APPLY:
            op_map = BINARY_MAPPING
            return REGULAR_AGGREGATION_MAPPING[node.op].validate(dataset, node.children, op_map)
        for child in node.children:
            self.is_from_regular_aggregation = True
            operands.append(self.visit(child))
            self.is_from_regular_aggregation = False
        if node.op == CALC and any(isinstance(operand, Dataset) for operand in operands):
            raise SemanticError("1-2-14", op=node.op)
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
            aux_operands = []
            for operand in operands:
                measure = operand.get_component(operand.get_measures_names()[0])
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
                        data=None,
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
                    data=None,
                    data_type=operands[0].components[measure].data_type,
                    role=operands[0].components[measure].role,
                    nullable=operands[0].components[measure].nullable,
                )
            return REGULAR_AGGREGATION_MAPPING[node.op].validate(operands[0], dataset)
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
            result = REGULAR_AGGREGATION_MAPPING[node.op].validate(operands, dataset)
            if node.isLast:
                result.components = {
                    comp_name[comp_name.find("#") + 1 :]: comp
                    for comp_name, comp in result.components.items()
                }
                for comp in result.components.values():
                    comp.name = comp.name[comp.name.find("#") + 1 :]
                self.is_from_join = False
            return result
        return REGULAR_AGGREGATION_MAPPING[node.op].validate(operands, dataset)

    def visit_If(self, node: AST.If) -> Any:
        condition = self.visit(node.condition)

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
            return self.visit(node.thenOp if condition.value else node.elseOp)

        # Validate condition is a boolean dataset/component
        self._validate_boolean_condition(condition)

        thenOp = self.visit(node.thenOp)
        elseOp = self.visit(node.elseOp)

        return If.validate(condition, thenOp, elseOp)

    def visit_Case(self, node: AST.Case) -> Any:
        conditions: List[Any] = []
        thenOps: List[Any] = []

        for case in node.cases:
            conditions.append(self.visit(case.condition))
            if isinstance(conditions[-1], Scalar):
                thenOps.append(self.visit(case.thenOp))
                continue

            # Validate condition is a boolean dataset/component
            self._validate_boolean_condition(conditions[-1])
            thenOps.append(self.visit(case.thenOp))

        elseOp = self.visit(node.elseOp)

        return Case.validate(conditions, thenOps, elseOp)

    def _validate_boolean_condition(
        self,
        condition: Union[Dataset, DataComponent],
    ) -> None:
        if isinstance(condition, Dataset):
            measures = condition.get_measures()
            if len(measures) != 1:
                raise SemanticError("1-1-1-4", op="condition")
            elif measures[0].data_type != BASIC_TYPES[bool]:
                raise SemanticError("2-1-9-5", op="condition", name=condition.name)
        elif condition.data_type != BASIC_TYPES[bool]:
            raise SemanticError("2-1-9-4", op="condition", name=condition.name)

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
            nullable=node.value is None,
        )

    def visit_JoinOp(self, node: AST.JoinOp) -> Any:
        clause_elements = []
        for clause in node.clauses:
            clause_elements.append(self.visit(clause))
            if hasattr(clause, "op") and clause.op == AS:
                # TODO: We need to delete somewhere the join datasets with alias that are added here
                self.datasets[clause_elements[-1].name] = clause_elements[-1]

        nvl_defaults: Optional[Dict[str, Any]] = None
        if node.nvl:
            nvl_defaults = {pair.component: pair.default.value for pair in node.nvl}

        # No need to check using, regular aggregation is executed afterwards
        self.is_from_join = True
        return JOIN_MAPPING[node.op].validate(clause_elements, node.using, nvl_defaults)

    def visit_ParamConstant(self, node: AST.ParamConstant) -> str:
        return node.value

    def visit_ParamOp(self, node: AST.ParamOp) -> Any:  # noqa: C901
        if node.op == ROUND:
            op_element = self.visit(node.children[0])
            param_element = self.visit(node.params[0]) if len(node.params) != 0 else None
            return Round.validate(op_element, param_element)

        # Numeric Operator
        elif node.op == TRUNC:
            op_element = self.visit(node.children[0])
            param_element = None
            if len(node.params) != 0:
                param_element = self.visit(node.params[0])

            return Trunc.validate(op_element, param_element)

        elif node.op == STRING_DISTANCE:
            method = self.visit(node.params[0])
            s1 = self.visit(node.children[0])
            s2 = self.visit(node.children[1])
            return DISTANCE_DISPATCH[method].validate(s1, s2)

        elif node.op == SUBSTR or node.op == REPLACE or node.op == INSTR:
            params = [None, None, None]
            op_element = self.visit(node.children[0])
            for i, node_param in enumerate(node.params):
                params[i] = self.visit(node_param)
            param1, param2, param3 = tuple(params)
            if node.op == SUBSTR:
                return Substr.validate(op_element, param1, param2)
            elif node.op == REPLACE:
                return Replace.validate(op_element, param1, param2)
            elif node.op == INSTR:
                return Instr.validate(op_element, param1, param2, param3)
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
                    if comp_name in self.aggregation_grouping
                    or comp.role in [Role.MEASURE, Role.VIRAL_ATTRIBUTE]
                }
            result = self.visit(node.params)
            measure = result.get_measures()[0]
            if measure.data_type != Boolean:
                raise SemanticError("1-1-2-3", type=SCALAR_TYPES_CLASS_REVERSE[Boolean])
            return None
        elif node.op == FILL_TIME_SERIES:
            mode = self.visit(node.params[0]) if len(node.params) == 1 else "all"
            return Fill_time_series.validate(self.visit(node.children[0]), mode)
        elif node.op == DATE_ADD:
            params = [self.visit(node.params[0]), self.visit(node.params[1])]
            return Date_Add.validate(self.visit(node.children[0]), params)  # type: ignore[arg-type]
        elif node.op == CAST:
            operand = self.visit(node.children[0])
            scalar_type = node.children[1]
            mask = None
            if len(node.params) > 0:
                mask = self.visit(node.params[0])
            return Cast.validate(operand, scalar_type, mask)  # type: ignore[arg-type]

        raise SemanticError("1-3-5", op_type="ParamOp", node_op=node.op)

    def _get_hr_mode_values(self, node: AST.HROperation) -> Tuple[str, str, str]:
        """Extract mode values with defaults for HROperation."""
        mode = node.validation_mode.value if node.validation_mode else "non_null"
        if node.op == HIERARCHY:
            input_ = node.input_mode.value if node.input_mode else "rule"
            output = node.output.value if node.output else "computed"
        else:  # CHECK_HIERARCHY
            input_ = node.input_mode.value if node.input_mode else "dataset"
            output = node.output.value if node.output else "invalid"
        return mode, input_, output

    def visit_HROperation(self, node: AST.HROperation) -> Any:  # noqa: C901
        conditions = node.conditions or []
        has_conditions = len(conditions) > 0
        dataset = deepcopy(self.visit(node.dataset)) if has_conditions else self.visit(node.dataset)
        component: Optional[str] = self.visit(node.rule_component) if node.rule_component else None
        hr_name = node.ruleset_name
        cond_components = [self.visit(c) for c in conditions] if has_conditions else []

        mode, _, output = self._get_hr_mode_values(node)

        if self.hrs is None:
            raise SemanticError("1-2-6", node_type="Hierarchical Rulesets", node_value="")
        if hr_name not in self.hrs:
            raise SemanticError("1-2-6", node_type="Hierarchical Ruleset", node_value=hr_name)

        if not isinstance(dataset, Dataset):
            raise SemanticError("1-1-1-20", op=node.op)

        hr_info = self.hrs[hr_name]

        if hr_info is not None:
            if len(cond_components) != len(hr_info["condition"]):
                raise SemanticError("1-1-10-2", op=node.op)

            if hr_info["node"].signature_type == "variable" and not names_equal(
                hr_info["signature"], component
            ):
                raise SemanticError(
                    "1-1-10-3",
                    op=node.op,
                    found=component,
                    expected=hr_info["signature"],
                )
            elif hr_info["node"].signature_type == "valuedomain" and component is None:
                raise SemanticError("1-1-10-4", op=node.op)
            elif component is None:
                raise NotImplementedError(
                    "Hierarchical Ruleset handling without component "
                    "and signature type variable is not implemented yet."
                )

            cond_info = {}
            for i, cond_comp in enumerate(hr_info["condition"]):
                if hr_info["node"].signature_type == "variable" and cond_components[i] != cond_comp:
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

                if len(aux) == 0:
                    raise SemanticError("1-1-10-5")

                left_parts = []
                for rule in aux:
                    left_part = rule.rule.left if rule.rule.op == EQ else rule.rule.right.left
                    if left_part in left_parts:
                        raise SemanticError("1-1-10-10", ruleset=hr_name, rule=left_part.value)
                    left_parts.append(left_part)

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

            # Set up interpreter state for rule processing
            self.ruleset_dataset = dataset
            self.ruleset_signature = CaseInsensitiveDict(
                {**{"RULE_COMPONENT": component}, **cond_info}
            )
            self.ruleset_mode = mode
            rule_output_values = {}

            if node.op == HIERARCHY:
                self.is_from_hr_agg = True
                for rule in hr_info["rules"]:
                    self.visit(rule)
                self.is_from_hr_agg = False
            else:
                for rule in hr_info["rules"]:
                    rule_output_values[rule.name] = {
                        "errorcode": rule.erCode,
                        "errorlevel": rule.erLevel,
                        "output": self.visit(rule),
                    }

            self.ruleset_signature = None
            self.ruleset_dataset = None
            self.ruleset_mode = None

            if node.op == CHECK_HIERARCHY:
                return Check_Hierarchy.validate(
                    dataset_element=dataset,
                    rule_info=rule_output_values,
                    output=output,
                )
            return Hierarchy.validate(dataset, output)

        raise SemanticError("1-3-5", op_type="HROperation", node_op=node.op)

    def visit_DPValidation(self, node: AST.DPValidation) -> Any:
        """Handle check_datapoint operator."""
        if self.dprs is None:
            raise SemanticError("1-2-6", node_type="Datapoint Rulesets", node_value="")

        dpr_name = node.ruleset_name
        if dpr_name not in self.dprs:
            raise SemanticError("1-2-6", node_type="Datapoint Ruleset", node_value=dpr_name)
        dpr_info = self.dprs[dpr_name]

        # Extract dataset
        dataset_element = self.visit(node.dataset)
        if not isinstance(dataset_element, Dataset):
            raise SemanticError("1-1-1-20", op=CHECK_DATAPOINT)

        # Check component list validity
        if node.components:
            for comp_name in node.components:
                if comp_name not in dataset_element.components:
                    raise SemanticError(
                        "1-1-1-10",
                        comp_name=comp_name,
                        dataset_name=dataset_element.name,
                    )
            if dpr_info is not None and dpr_info["signature_type"] == "variable":
                for i, comp_name in enumerate(node.components):
                    if not names_equal(comp_name, dpr_info["params"][i]):
                        raise SemanticError(
                            "1-1-10-3",
                            op=CHECK_DATAPOINT,
                            expected=dpr_info["params"][i],
                            found=comp_name,
                        )

        # Get output mode with default
        output = node.output.value if node.output else "invalid"

        if dpr_info is None:
            dpr_info = {}

        rule_output_values = {}
        self.ruleset_dataset = dataset_element
        self.ruleset_signature = dpr_info.get("signature")
        self.ruleset_mode = output

        # Gather rule data
        if dpr_info:
            for rule in dpr_info["rules"]:
                rule_output_values[rule.name] = {
                    "errorcode": rule.erCode,
                    "errorlevel": rule.erLevel,
                    "output": self.visit(rule),
                }

        self.ruleset_mode = None
        self.ruleset_signature = None
        self.ruleset_dataset = None

        # Final evaluation
        return Check_Datapoint.validate(
            dataset_element=dataset_element,
            rule_info=rule_output_values,
            output=output,
        )

    def visit_DPRule(self, node: AST.DPRule) -> Any:
        self.is_from_rule = True
        validation_data = self.visit(node.rule)
        self.is_from_rule = False
        return None if isinstance(validation_data, DataComponent) else validation_data

    def visit_HRule(self, node: AST.HRule) -> Any:
        self.is_from_rule = True
        rule_result = self.visit(node.rule)
        self.is_from_rule = False
        return rule_result if self.is_from_hr_agg else None

    def visit_HRBinOp(self, node: AST.HRBinOp) -> Any:
        if node.op == WHEN:
            # Visit both operands for semantic validation (type checks, component checks)
            self.visit(node.left)
            self.visit(node.right)
            # Semantic mode: no data filtering required for WHEN
            return None

        left_operand = self.visit(node.left)
        right_operand = self.visit(node.right)
        if isinstance(right_operand, Dataset):
            right_operand = get_measure_from_dataset(right_operand, node.right.value)

        if node.op in HR_COMP_MAPPING:
            op = HAAssignment if self.is_from_hr_agg else HR_COMP_MAPPING[node.op]
            return op.validate(left_operand, right_operand)
        if isinstance(left_operand, Dataset):
            left_operand = get_measure_from_dataset(left_operand, node.left.value)
        return HR_NUM_BINARY_MAPPING[node.op].validate(left_operand, right_operand)

    def visit_HRUnOp(self, node: AST.HRUnOp) -> None:
        operand = self.visit(node.operand)
        return HR_UNARY_MAPPING[node.op].validate(operand)

    def visit_Validation(self, node: AST.Validation) -> Dataset:
        validation_element = self.visit(node.validation)
        if not isinstance(validation_element, Dataset):
            raise ValueError(f"Expected dataset, got {type(validation_element).__name__}")

        imbalance_element = None
        if node.imbalance is not None:
            imbalance_element = self.visit(node.imbalance)
            if not isinstance(imbalance_element, Dataset):
                raise ValueError(f"Expected dataset, got {type(validation_element).__name__}")

        return Check.validate(
            validation_element=validation_element,
            imbalance_element=imbalance_element,
            error_code=node.error_code,
            error_level=node.error_level,
            invalid=node.invalid,
        )

    def visit_EvalOp(self, node: AST.EvalOp) -> Dataset:
        if node.language not in EXTERNAL:
            raise SemanticError(code="1-3-6", language=node.language)

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
        return Eval.validate(operands, external_routine, output_to_check)  # type: ignore[arg-type]

    def visit_Identifier(self, node: AST.Identifier) -> Union[AST.AST, Dataset, str]:
        if self.udo_params is not None and node.value in self.udo_params[-1]:
            return self.udo_params[-1][node.value]

        if node.value in self.datasets:
            if self.is_from_assignment:
                return copy(self.datasets[node.value].name)
            return copy(self.datasets[node.value])
        return node.value

    def visit_DefIdentifier(self, node: AST.DefIdentifier) -> Any:
        if not (self.is_from_rule and node.kind == "CodeItemID"):
            return node.value

        ruleset_ds = self.ruleset_dataset
        if ruleset_ds is None:
            raise SemanticError("2-3-7")

        result_components = {c.name: c for c in ruleset_ds.get_components()}
        condition = getattr(node, "_right_condition", None)
        if condition is not None:
            self.visit(condition)
        return Dataset(name=node.value, components=result_components, data=None)

    def visit_UDOCall(self, node: AST.UDOCall) -> None:  # noqa: C901
        if self.udos is None:
            raise SemanticError("2-3-10", comp_type="User Defined Operators")
        elif node.op not in self.udos:
            raise SemanticError("1-2-3", node_op=node.op, op_type="User Defined Operator")
        if self.signature_values is None:
            self.signature_values = {}

        operator = self.udos[node.op]
        signature_values = {}

        if operator is None:
            raise SemanticError("1-2-3", node_op=node.op, op_type="User Defined Operator")
        if operator["output"] == "Component" and not (
            self.is_from_regular_aggregation or self.is_from_rule
        ):
            raise SemanticError("1-2-12", op=node.op)

        for i, param in enumerate(operator["params"]):
            if i >= len(node.params):
                if "default" in param:
                    value = self.visit(param["default"]).value
                    signature_values[param["name"]] = Scalar(
                        name=str(value),
                        value=value,
                        data_type=BASIC_TYPES[type(value)],
                        nullable=value is None,
                    )
                else:
                    raise SemanticError(
                        "1-2-11",
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
                                        "1-3-1-1",
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
                                    "1-3-1-1",
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
                            "1-3-1-1",
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
                        nullable=param_element.nullable,
                    )
                else:
                    raise NotImplementedError

        # We set it here to a list to start the stack of UDO params
        if self.udo_params is None:
            self.udo_params = []

        # Adding parameters to the stack
        for k, v in signature_values.items():
            if hasattr(v, "name"):
                v = v.name  # type: ignore[assignment]
            if v in self.signature_values:
                signature_values[k] = self.signature_values[v]  # type: ignore[index]
        self.signature_values.update(signature_values)
        self.udo_params.append(signature_values)

        # Calling the UDO AST, we use deepcopy to avoid changing the original UDO AST
        result = self.visit(deepcopy(operator["expression"]))

        if self.is_from_regular_aggregation or self.is_from_rule:
            result_type = "Component" if isinstance(result, DataComponent) else "Scalar"
        else:
            result_type = "Scalar" if isinstance(result, Scalar) else "Dataset"

        if result_type != operator["output"]:
            raise SemanticError(
                "1-3-1-1",
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

    def visit_TimeAggregation(self, node: AST.TimeAggregation) -> Any:
        period_to = node.period_to
        if node.period_to_ref is not None:
            saved_grouping = self.is_from_grouping
            saved_having = self.is_from_having
            saved_regular = self.is_from_regular_aggregation
            self.is_from_grouping = False
            self.is_from_having = False
            self.is_from_regular_aggregation = False
            try:
                resolved = self.visit(node.period_to_ref)
            finally:
                self.is_from_grouping = saved_grouping
                self.is_from_having = saved_having
                self.is_from_regular_aggregation = saved_regular
            period_to = resolved.value if hasattr(resolved, "value") else resolved
        if node.operand is not None:
            operand = self.visit(node.operand)
            return Time_Aggregation.validate(
                operand=operand,
                period_from=node.period_from,
                period_to=period_to,  # type: ignore[arg-type]
                conf=node.conf,
            )
        # The aggregation dataset is mandatory here as is part of a group_all statement.
        # If not, a 1-3-2-4 error is raised in AST creation
        if self.aggregation_dataset is None:
            raise SemanticError("1-3-2-4")
        if period_to is None:
            raise SemanticError("1-3-2-4")
        return Time_Aggregation._execute_without_operand(
            aggregation_dataset=self.aggregation_dataset,
            period_from=node.period_from,
            period_to=period_to,
            conf=node.conf,
        )
