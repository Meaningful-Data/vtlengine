from copy import copy, deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

import AST
from AST.ASTTemplate import ASTTemplate
from AST.Grammar.tokens import AGGREGATE, ALL, APPLY, AS, BETWEEN, CHECK_DATAPOINT, DROP, EXISTS_IN, \
    EXTERNAL, FILTER, HAVING, INSTR, KEEP, MEMBERSHIP, REPLACE, ROUND, SUBSTR, TRUNC, WHEN
from DataTypes import BASIC_TYPES
from Model import DataComponent, Dataset, ExternalRoutine, Role, Scalar, ScalarSet, Component
from Operators.Aggregation import extract_grouping_identifiers
from Operators.Assignment import Assignment
from Operators.Comparison import Between, ExistIn
from Operators.General import Eval
from Operators.Conditional import If
from Operators.Numeric import Round, Trunc
from Operators.String import Instr, Replace, Substr
from Operators.Validation import Check, Check_Datapoint
from Utils import AGGREGATION_MAPPING, ANALYTIC_MAPPING, BINARY_MAPPING, JOIN_MAPPING, \
    REGULAR_AGGREGATION_MAPPING, ROLE_SETTER_MAPPING, SET_MAPPING, UNARY_MAPPING, THEN_ELSE


# noinspection PyTypeChecker
@dataclass


class InterpreterAnalyzer(ASTTemplate):
    # Model elements
    datasets: Dict[str, Dataset]
    external_routines: Optional[Dict[str, ExternalRoutine]] = None
    # Flags to change behavior
    is_from_assignment: bool = False
    is_from_regular_aggregation: bool = False
    is_from_having: bool = False
    is_from_rule: bool = False
    is_from_join: bool = False
    is_from_condition: bool = False
    if_stack: Optional[List[str]] = None
    # Handlers for simplicity
    regular_aggregation_dataset: Optional[Dataset] = None
    aggregation_grouping: Optional[List[str]] = None
    aggregation_dataset: Optional[Dataset] = None
    then_condition_dataset: Optional[List[pd.DataFrame]] = None
    else_condition_dataset: Optional[List[pd.DataFrame]] = None
    ruleset_dataset: Optional[Dataset] = None
    rule_data: Optional[pd.DataFrame] = None
    ruleset_signature: Dict[str, str] = None
    # DL
    dprs: Dict[str, Dict[str, Any]] = None

    def visit_Start(self, node: AST.Start) -> Any:
        results = {}
        for child in node.children:
            result = self.visit(child)
            # TODO: Execute collected operations from Spark and add explain
            if isinstance(result, Union[Dataset, Scalar]):
                self.datasets[result.name] = result
                results[result.name] = result
        return results

    # Definition Language

    def visit_DPRuleset(self, node: AST.DPRuleset) -> None:

        # Rule names are optional, if not provided, they are generated.
        # If provided, all must be provided
        rule_names = [rule.name for rule in node.rules if rule.name is not None]
        if len(rule_names) != 0 and len(node.rules) != len(rule_names):
            raise ValueError("All rules must have a name, or none of them")
        if len(rule_names) == 0:
            for i, rule in enumerate(node.rules):
                rule.name = i + 1

        # Signature has the actual parameters names or aliases if provided
        signature_actual_names = {}
        for param in node.params:
            if param.alias is not None:
                signature_actual_names[param.alias] = param.value
            else:
                signature_actual_names[param.value] = param.value

        ruleset_data = {
            'rules': node.rules,
            'signature': signature_actual_names,
            'params': node.params
        }

        # Adding the ruleset to the dprs dictionary
        if self.dprs is None:
            self.dprs = {}
        elif node.name in self.dprs:
            raise ValueError(f"Datapoint Ruleset {node.name} already exists")

        self.dprs[node.name] = ruleset_data

    # Execution Language
    def visit_Assignment(self, node: AST.Assignment) -> Any:
        self.is_from_assignment = True
        left_operand: str = self.visit(node.left)
        self.is_from_assignment = False
        right_operand: Union[Dataset, DataComponent] = self.visit(node.right)
        return Assignment.evaluate(left_operand, right_operand)

    def visit_PersistentAssignment(self, node: AST.PersistentAssignment) -> Any:
        return self.visit_Assignment(node)

    def visit_BinOp(self, node: AST.BinOp) -> None:
        if self.is_from_join and node.op in [MEMBERSHIP, AGGREGATE]:
            left_operand = self.regular_aggregation_dataset
            right_operand = self.visit(node.left).name + '#' + self.visit(node.right)
        else:
            left_operand = self.visit(node.left)
            right_operand = self.visit(node.right)
        if node.op != '#' and not self.is_from_condition and self.if_stack is not None and len(self.if_stack) > 0:
            left_operand, right_operand = self.merge_then_else_datasets(left_operand, right_operand)
        if node.op not in BINARY_MAPPING:
            raise NotImplementedError
        return BINARY_MAPPING[node.op].evaluate(left_operand, right_operand)

    def visit_UnaryOp(self, node: AST.UnaryOp) -> None:
        operand = self.visit(node.operand)
        if node.op not in UNARY_MAPPING and node.op not in ROLE_SETTER_MAPPING:
            raise NotImplementedError
        if self.is_from_regular_aggregation and node.op in ROLE_SETTER_MAPPING:
            data_size = len(self.regular_aggregation_dataset.data)
            return ROLE_SETTER_MAPPING[node.op].evaluate(operand, data_size)
        return UNARY_MAPPING[node.op].evaluate(operand)

    def visit_Aggregation(self, node: AST.Aggregation) -> None:
        # Having takes precedence as it is lower in the AST
        if self.is_from_having:
            operand = self.aggregation_dataset
        elif self.is_from_regular_aggregation:
            operand = self.regular_aggregation_dataset
        else:
            operand = self.visit(node.operand)
        groupings = []
        having = None
        grouping_op = node.grouping_op
        if node.grouping is not None:
            for x in node.grouping:
                groupings.append(self.visit(x))
            if node.having_clause is not None:
                self.aggregation_dataset = Dataset(name=operand.name,
                                                   components=operand.components,
                                                   data=operand.data.copy())
                self.aggregation_grouping = extract_grouping_identifiers(
                    operand.get_identifiers_names(),
                    node.grouping_op,
                    groupings)
                self.is_from_having = True
                having = self.visit(node.having_clause)
                # Reset to default values
                self.is_from_having = False
                self.aggregation_grouping = None
                self.aggregation_dataset = None
        elif self.is_from_having:
            groupings = self.aggregation_grouping
            # Setting here group by as we have already selected the identifiers we need
            grouping_op = 'group by'

        return AGGREGATION_MAPPING[node.op].evaluate(operand, grouping_op, groupings, having)

    def visit_Analytic(self, node: AST.Analytic) -> None:
        if self.is_from_regular_aggregation:
            if node.operand is None:
                operand = self.regular_aggregation_dataset
            else:
                operand_comp = self.visit(node.operand)
                measure_names = self.regular_aggregation_dataset.get_measures_names()
                dataset_components = self.regular_aggregation_dataset.components.copy()
                for name in measure_names:
                    if name != operand_comp.name:
                        dataset_components.pop(name)

                operand = Dataset(name=self.regular_aggregation_dataset.name,
                                  components=dataset_components,
                                  data=self.regular_aggregation_dataset.data[
                                      dataset_components.keys()])

        else:
            operand: Dataset = self.visit(node.operand)
        partitioning = node.partition_by
        ordering = node.order_by if node.order_by is not None else []
        if not isinstance(operand, Dataset):
            raise Exception("Analytic operator must have a dataset as operand")
        if node.partition_by is None:
            order_components = [x.component for x in node.order_by]
            partitioning = [x for x in operand.get_identifiers_names() if x not in order_components]

        params = []
        if node.params is not None:
            for param in node.params:
                if isinstance(param, AST.Constant):
                    params.append(param.value)
                else:
                    params.append(param)

        result = ANALYTIC_MAPPING[node.op].evaluate(operand=operand,
                                                    partitioning=partitioning,
                                                    ordering=ordering,
                                                    window=node.window,
                                                    params=params)
        if not self.is_from_regular_aggregation:
            return result

        # TODO: Review this as the components on calc are not in correct order (Rank test)
        # Extracting the component we need (only measure)
        measure_name = result.get_measures_names()[0]
        return DataComponent(name=measure_name,
                             data=result.data[measure_name],
                             data_type=result.components[measure_name].data_type,
                             role=result.components[measure_name].role)

    def visit_MulOp(self, node: AST.MulOp):
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

            return Between.evaluate(operand_element, from_element, to_element)

        # Comparison Operators
        elif node.op == EXISTS_IN:
            dataset_1 = self.visit(node.children[0])
            if not isinstance(dataset_1, Dataset):
                raise Exception("First operand must be a dataset")
            dataset_2 = self.visit(node.children[1])
            if not isinstance(dataset_2, Dataset):
                raise Exception("Second operand must be a dataset")

            retain_element = None
            if len(node.children) == 3:
                retain_element = self.visit(node.children[2])
                if isinstance(retain_element, Scalar):
                    retain_element = retain_element.value
                if retain_element == ALL:
                    retain_element = None

            return ExistIn.evaluate(dataset_1, dataset_2, retain_element)

        # Set Operators.
        elif node.op in SET_MAPPING:
            datasets = []
            for child in node.children:
                datasets.append(self.visit(child))

            for ds in datasets:
                if not isinstance(ds, Dataset):
                    raise ValueError(f"Expected dataset, got {type(ds).__name__}")

            return SET_MAPPING[node.op].evaluate(datasets)

        raise NotImplementedError

    def visit_VarID(self, node: AST.VarID) -> Any:
        if self.is_from_assignment:
            return node.value
        # Having takes precedence as it is lower in the AST
        if self.is_from_having:
            return DataComponent(name=node.value,
                                 data=self.aggregation_dataset.data[node.value],
                                 data_type=self.aggregation_dataset.components[
                                     node.value].data_type,
                                 role=self.aggregation_dataset.components[node.value].role,
                                 nullable=self.aggregation_dataset.components[node.value].nullable)
        if self.is_from_regular_aggregation:
            if self.is_from_join and node.value in self.datasets.keys():
                return self.datasets[node.value]
            return DataComponent(name=node.value,
                                 data=self.regular_aggregation_dataset.data[node.value],
                                 data_type=
                                 self.regular_aggregation_dataset.components[
                                     node.value].data_type,
                                 role=self.regular_aggregation_dataset.components[
                                     node.value].role,
                                 nullable=self.regular_aggregation_dataset.components[
                                     node.value].nullable)
        if self.is_from_rule:
            if node.value not in self.ruleset_signature:
                raise Exception(f"Component {node.value} not found in ruleset signature")
            comp_name = self.ruleset_signature[node.value]
            if comp_name not in self.ruleset_dataset.components:
                raise Exception(f"Component {comp_name} not found in dataset "
                                f"{self.ruleset_dataset.name}")
            return DataComponent(name=comp_name,
                                 data=self.rule_data[comp_name],
                                 data_type=self.ruleset_dataset.components[comp_name].data_type,
                                 role=self.ruleset_dataset.components[comp_name].role,
                                 nullable=self.ruleset_dataset.components[comp_name].nullable)

        if node.value not in self.datasets:
            raise Exception(f"Dataset {node.value} not found, please check input datastructures")
        return self.datasets[node.value]

    def visit_Collection(self, node: AST.Collection) -> Any:
        if node.kind == 'Set':
            elements = []
            for child in node.children:
                elements.append(self.visit(child).value)
            for element in elements:
                if type(element) != type(elements[0]):
                    raise Exception("All elements in a set must be of the same type")
            if len(elements) == 0:
                raise Exception("A set must contain at least one element")
            if len(elements) != len(set(elements)):
                raise Exception("A set must not contain duplicates")
            return ScalarSet(data_type=BASIC_TYPES[type(elements[0])], values=elements)
        raise NotImplementedError

    def visit_RegularAggregation(self, node: AST.RegularAggregation) -> None:
        if node.op not in REGULAR_AGGREGATION_MAPPING:
            raise NotImplementedError
        operands = []
        dataset = self.visit(node.dataset)
        if isinstance(dataset, Scalar):
            raise Exception(f"Scalar {dataset.name} cannot be used with clause operators")
        self.regular_aggregation_dataset = dataset
        if node.op == APPLY:
            op_map = BINARY_MAPPING
            return REGULAR_AGGREGATION_MAPPING[node.op].evaluate(dataset, node.children, op_map)
        for child in node.children:
            self.is_from_regular_aggregation = True
            operands.append(self.visit(child))
            self.is_from_regular_aggregation = False
        if node.op == AGGREGATE:
            dataset = copy(operands[0])
            dataset.name = self.regular_aggregation_dataset.name
            dataset.components = {comp_name: comp for comp_name, comp in dataset.components.items()
                                  if comp.role != Role.MEASURE}
            if dataset.data is not None:
                dataset.data = dataset.data[dataset.get_identifiers_names()]
            aux_operands = []
            for operand in operands:
                measure = operand.get_component(operand.get_measures_names()[0])
                data = operand.data[measure.name] if operand.data is not None else None
                aux_operands.append(DataComponent(name=operand.name,
                                                  data=data,
                                                  data_type=measure.data_type,
                                                  role=measure.role,
                                                  nullable=measure.nullable))
            operands = aux_operands
        self.regular_aggregation_dataset = None
        if node.op == FILTER:
            if not isinstance(operands[0], DataComponent):
                measure = child.left.value
                operands[0] = DataComponent(name=measure,
                                            data=operands[0].data[measure],
                                            data_type=operands[0].components[measure].data_type,
                                            role=operands[0].components[measure].role,
                                            nullable=operands[0].components[measure].nullable)
            return REGULAR_AGGREGATION_MAPPING[node.op].evaluate(operands[0], dataset)
        if self.is_from_join:
            if node.op in [DROP, KEEP]:
                operands = [operand.get_measures_names() if isinstance(operand,
                                                                       Dataset) else operand.name if
                isinstance(operand, DataComponent) and operand.role is not Role.IDENTIFIER else
                operand for operand in operands]
                operands = list(set([item for sublist in operands for item in
                                     (sublist if isinstance(sublist, list) else [sublist])]))
            result = REGULAR_AGGREGATION_MAPPING[node.op].evaluate(operands, dataset)
            if node.isLast:
                result.data.rename(
                    columns={col: col[col.find('#') + 1:] for col in result.data.columns},
                    inplace=True)
                result.components = {comp_name[comp_name.find('#') + 1:]: comp for comp_name, comp
                                     in
                                     result.components.items()}
                for comp in result.components.values():
                    comp.name = comp.name[comp.name.find('#') + 1:]
                result.data.reset_index(drop=True, inplace=True)
            return result
        return REGULAR_AGGREGATION_MAPPING[node.op].evaluate(operands, dataset)

    def visit_If(self, node: AST.If) -> Dataset:

        self.is_from_condition = True
        condition = self.visit(node.condition)
        self.is_from_condition = False

        if isinstance(condition, Scalar):
            if condition.value:
                return self.visit(node.thenOp)
            else:
                return self.visit(node.elseOp)

        # Analysis for data component and dataset
        else:
            if self.if_stack is None:
                self.if_stack = []
            if self.then_condition_dataset is None:
                self.then_condition_dataset = []
            if self.else_condition_dataset is None:
                self.else_condition_dataset = []
            self.generate_then_else_datasets(condition)

        self.if_stack.append(THEN_ELSE['then'])
        thenOp = self.visit(node.thenOp)
        if isinstance(thenOp, Scalar) or not isinstance(node.thenOp, AST.BinOp):
            self.then_condition_dataset.pop()
            self.if_stack.pop()

        self.if_stack.append(THEN_ELSE['else'])
        elseOp = self.visit(node.elseOp)
        if isinstance(elseOp, Scalar) or not isinstance(node.elseOp, AST.BinOp):
            self.else_condition_dataset.pop()
            self.if_stack.pop()

        return If.evaluate(condition, thenOp, elseOp)

    def visit_RenameNode(self, node: AST.RenameNode) -> Any:
        return node

    def visit_Constant(self, node: AST.Constant) -> Any:
        return Scalar(name=str(node.value), value=node.value,
                      data_type=BASIC_TYPES[type(node.value)])

    def visit_JoinOp(self, node: AST.JoinOp) -> None:
        clause_elements = []
        for clause in node.clauses:
            clause_elements.append(self.visit(clause))
            if hasattr(clause, 'op') and clause.op == AS:
                self.datasets[clause_elements[-1].name] = clause_elements[-1]

        # No need to check using, regular aggregation is executed afterwards
        self.is_from_join = True
        return JOIN_MAPPING[node.op].evaluate(clause_elements, node.using)

    def visit_ParamConstant(self, node: AST.ParamConstant) -> str:
        return node.value

    def visit_ParamOp(self, node: AST.ParamOp) -> None:
        if node.op == ROUND:
            op_element = self.visit(node.children[0])
            if len(node.params) != 0:
                param_element = self.visit(node.params[0])
            else:
                param_element = None

            return Round.evaluate(op_element, param_element)

        # Numeric Operator
        elif node.op == TRUNC:
            op_element = self.visit(node.children[0])
            param_element = None
            if len(node.params) != 0:
                param_element = self.visit(node.params[0])

            return Trunc.evaluate(op_element, param_element)

        elif node.op == SUBSTR or node.op == REPLACE or node.op == INSTR:
            params = [None, None, None]
            op_element = self.visit(node.children[0])
            for i, node_param in enumerate(node.params):
                params[i] = self.visit(node_param)
            param1, param2, param3 = tuple(params)
            if node.op == SUBSTR:
                return Substr.evaluate(op_element, param1, param2)
            elif node.op == REPLACE:
                return Replace.evaluate(op_element, param1, param2)
            elif node.op == INSTR:
                return Instr.evaluate(op_element, param1, param2, param3)
            else:
                raise NotImplementedError
        elif node.op == HAVING:
            for id_name in self.aggregation_grouping:
                if id_name not in self.aggregation_dataset.components:
                    raise ValueError(f"Component {id_name} not found in dataset")
            if len(self.aggregation_dataset.get_measures()) != 1:
                raise ValueError("Only one measure is allowed")
            # Deepcopy is necessary for components to avoid changing the original dataset
            self.aggregation_dataset.components = {comp_name: deepcopy(comp) for comp_name, comp in
                                                   self.aggregation_dataset.components.items()
                                                   if comp_name in self.aggregation_grouping
                                                   or comp.role == Role.MEASURE}
            self.aggregation_dataset.data = self.aggregation_dataset.data[
                self.aggregation_dataset.get_identifiers_names() +
                self.aggregation_dataset.get_measures_names()]
            result = self.visit(node.params)
            # We get only the identifiers we need that have true values when grouped
            measure_name = result.get_measures_names()[0]
            result.data = result.data[result.data[measure_name]]
            # result.data.drop(columns=[measure_name], inplace=True)
            result.data.drop(columns=[measure_name])
            return result.data

        elif node.op == CHECK_DATAPOINT:
            # Checking if ruleset exists
            dpr_name = node.children[1]
            if dpr_name in self.dprs:
                dpr_info = self.dprs[dpr_name]
            else:
                raise Exception(f"Datapoint Ruleset {dpr_name} not found")
            # Extracting dataset
            dataset_element = self.visit(node.children[0])
            # Checking if list of components supplied is valid
            if len(node.children) > 2:
                for comp_name in node.children[2:]:
                    if comp_name not in dataset_element.components:
                        raise ValueError(
                            f"Component {comp_name} not found in dataset {dataset_element.name}")

            output = node.params[0]  # invalid, all_measures, all

            rule_output_values = {}
            self.ruleset_dataset = dataset_element
            self.ruleset_signature = dpr_info['signature']
            # Gather rule data, adding the ruleset dataset to the interpreter
            for rule in dpr_info['rules']:
                rule_output_values[rule.name] = {
                    "error_code": rule.erCode,
                    "error_level": rule.erLevel,
                    "output": self.visit(rule)
                }
            self.ruleset_signature = None
            self.ruleset_dataset = None

            # Datapoint Ruleset final evaluation
            return Check_Datapoint.evaluate(dataset_element=dataset_element,
                                            rule_info=rule_output_values,
                                            output=output)

    def visit_DPRule(self, node: AST.DPRule) -> None:
        self.is_from_rule = True
        self.rule_data = self.ruleset_dataset.data.copy()
        validation_data = self.visit(node.rule)
        self.rule_data = None
        self.is_from_rule = False
        return validation_data

    def visit_HRBinOp(self, node: AST.HRBinOp) -> None:
        if node.op == WHEN:
            filter_comp = self.visit(node.left)
            filtering_indexes = filter_comp.data[filter_comp.data].index
            non_filtering_indexes = filter_comp.data[~filter_comp.data].index
            original_data = self.rule_data.copy()
            self.rule_data = self.rule_data.iloc[filtering_indexes].reset_index(drop=True)
            result_validation = self.visit(node.right)
            self.rule_data['bool_var'] = result_validation.data
            original_data = original_data.merge(self.rule_data, how='left',
                                                on=original_data.columns.tolist())
            original_data.loc[non_filtering_indexes, 'bool_var'] = True
            return original_data

    def visit_Validation(self, node: AST.Validation) -> Dataset:

        validation_element = self.visit(node.validation)
        if not isinstance(validation_element, Dataset):
            raise ValueError(f"Expected dataset, got {type(validation_element).__name__}")

        imbalance_element = None
        if node.imbalance is not None:
            imbalance_element = self.visit(node.imbalance)
            if not isinstance(imbalance_element, Dataset):
                raise ValueError(f"Expected dataset, got {type(validation_element).__name__}")

        return Check.evaluate(validation_element=validation_element,
                              imbalance_element=imbalance_element,
                              error_code=node.error_code,
                              error_level=node.error_level,
                              invalid=node.invalid)

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

        if node.name not in self.external_routines:
            raise Exception(f"External Routine {node.name} not found")
        external_routine = self.external_routines[node.name]
        operand = self.visit(node.operand)
        output_to_check = node.output
        return Eval.evaluate(operand, external_routine, output_to_check)

    def generate_then_else_datasets(self, condition):
        if isinstance(condition, Dataset):
            if len(condition.get_measures()) != 1 or condition.get_measures()[0].data_type != BASIC_TYPES[bool]:
                raise ValueError("Only one boolean measure is allowed on condition dataset")
            name = condition.get_measures_names()[0]
            data = condition.data[name]
        else:
            if condition.data_type != BASIC_TYPES[bool]:
                raise ValueError("Only boolean scalars are allowed on data component condition")
            name = condition.name
            data = condition.data
        data.fillna(False, inplace=True)
        then_index = pd.DataFrame({name: [i for i, data in enumerate(data) if data]})
        else_index = pd.DataFrame({name: [i for i, data in enumerate(data) if not data]})
        component = Component(name=name, data_type=BASIC_TYPES[int], role=Role.MEASURE, nullable=True)
        self.then_condition_dataset.append(
            Dataset(name=name, components={name: component}, data=then_index))
        self.else_condition_dataset.append(
            Dataset(name=name, components={name: component}, data=else_index))

    def merge_then_else_datasets(self, left_operand: Dataset | DataComponent, right_operand):
        merge_dataset = self.then_condition_dataset.pop() if self.if_stack.pop() == THEN_ELSE['then'] else (
            self.else_condition_dataset.pop())
        merge_index = merge_dataset.data[merge_dataset.get_measures_names()[0]].to_list()
        if isinstance(left_operand, Dataset | DataComponent):
            if isinstance(left_operand, Dataset):
                left_operand.get_measures()[0].data_type = BASIC_TYPES[int]
                left = left_operand.data[left_operand.get_measures_names()[0]]
                left_operand.data[left_operand.get_measures_names()[0]] = left.reindex(merge_index, fill_value=None)
            else:
                left_operand.data_type = BASIC_TYPES[int]
                left = left_operand.data
                left_operand.data = left.reindex(merge_index, fill_value=None)
        if isinstance(right_operand, Dataset | DataComponent):
            if isinstance(right_operand, Dataset):
                right_operand.get_measures()[0].data_type = BASIC_TYPES[int]
                right = right_operand.data[right_operand.get_measures_names()[0]]
                right_operand.data[right_operand.get_measures_names()[0]] = right.reindex(merge_index, fill_value=None)
            else:
                right_operand.data_type = BASIC_TYPES[int]
                right = right_operand.data
                right_operand.data = right.reindex(merge_index, fill_value=None)
        return left_operand, right_operand
