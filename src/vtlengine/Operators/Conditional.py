from copy import copy
from typing import Any, List, Union

import numpy as np

# if os.environ.get("SPARK", False):
#     import pyspark.pandas as pd
# else:
#     import pandas as pd
import pandas as pd

from vtlengine.DataTypes import (
    COMP_NAME_MAPPING,
    SCALAR_TYPES_CLASS_REVERSE,
    Boolean,
    Null,
    binary_implicit_promotion,
)
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import DataComponent, Dataset, Role, Scalar
from vtlengine.Operators import Binary, Operator


class If(Operator):
    """
    If class:
        `If-then-else <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=225&zoom=100,72,142>`_ operator
        inherits from Operator, a superclass that contains general validate and evaluate class methods.
        It has the following class methods:
    Class methods:
        evaluate: Evaluates if the operation is well constructed, checking the actual condition and
        dropping a boolean result.
        The result will depend on the data class, such as datacomponent and dataset.

        component_level_evaluation: Returns a pandas dataframe with data to set the condition

        dataset_level_evaluation: Sets the dataset and evaluates its correct schema to be able to perform the condition.

        validate: Class method that has two branches so datacomponent and datasets can be validated. With datacomponent,
        the code reviews if it is actually a Measure and if it is a binary operation. Dataset branch reviews if the
        identifiers are the same in 'if', 'then' and 'else'.
    """  # noqa E501

    @classmethod
    def evaluate(cls, condition: Any, true_branch: Any, false_branch: Any) -> Any:
        result = cls.validate(condition, true_branch, false_branch)
        if not isinstance(result, Scalar):
            if isinstance(condition, DataComponent):
                result.data = cls.component_level_evaluation(condition, true_branch, false_branch)
            if isinstance(condition, Dataset):
                result = cls.dataset_level_evaluation(result, condition, true_branch, false_branch)
        return result

    @classmethod
    def component_level_evaluation(
        cls, condition: DataComponent, true_branch: Any, false_branch: Any
    ) -> Any:
        result = None
        if condition.data is not None:
            if isinstance(true_branch, Scalar):
                true_data = pd.Series(true_branch.value, index=condition.data.index)
            else:
                true_data = true_branch.data.reindex(condition.data.index)
            if isinstance(false_branch, Scalar):
                false_data = pd.Series(false_branch.value, index=condition.data.index)
            else:
                false_data = false_branch.data.reindex(condition.data.index)
            result = np.where(condition.data, true_data, false_data)

        return pd.Series(result, index=condition.data.index)  # type: ignore[union-attr]

    @classmethod
    def dataset_level_evaluation(
        cls, result: Any, condition: Any, true_branch: Any, false_branch: Any
    ) -> Dataset:
        ids = condition.get_identifiers_names()
        condition_measure = condition.get_measures_names()[0]
        true_data = condition.data[condition.data[condition_measure] == True]
        false_data = condition.data[condition.data[condition_measure] != True].fillna(False)

        if isinstance(true_branch, Dataset):
            if len(true_data) > 0 and true_branch.data is not None:
                true_data = pd.merge(
                    true_data,
                    true_branch.data,
                    on=ids,
                    how="right",
                    suffixes=("_condition", ""),
                )
            else:
                true_data = pd.DataFrame(columns=true_branch.get_components_names())
        else:
            true_data[condition_measure] = true_data[condition_measure].apply(
                lambda x: true_branch.value
            )
        if isinstance(false_branch, Dataset):
            if len(false_data) > 0 and false_branch.data is not None:
                false_data = pd.merge(
                    false_data,
                    false_branch.data,
                    on=ids,
                    how="right",
                    suffixes=("_condition", ""),
                )
            else:
                false_data = pd.DataFrame(columns=false_branch.get_components_names())
        else:
            false_data[condition_measure] = false_data[condition_measure].apply(
                lambda x: false_branch.value
            )

        result.data = (
            pd.concat([true_data, false_data], ignore_index=True)
            .drop_duplicates()
            .sort_values(by=ids)
        )
        if isinstance(result, Dataset):
            drop_columns = [
                column for column in result.data.columns if column not in result.components
            ]
            result.data = result.data.dropna(subset=drop_columns).drop(columns=drop_columns)
        if isinstance(true_branch, Scalar) and isinstance(false_branch, Scalar):
            result.get_measures()[0].data_type = true_branch.data_type
            result.get_measures()[0].name = COMP_NAME_MAPPING[true_branch.data_type]
            if result.data is not None:
                result.data = result.data.rename(
                    columns={condition_measure: result.get_measures()[0].name}
                )
        return result

    @classmethod
    def validate(  # noqa: C901
        cls, condition: Any, true_branch: Any, false_branch: Any
    ) -> Union[Scalar, DataComponent, Dataset]:
        nullable = False
        left = true_branch
        right = false_branch
        if true_branch.__class__ != false_branch.__class__:
            if (isinstance(true_branch, DataComponent) and isinstance(false_branch, Dataset)) or (
                isinstance(true_branch, Dataset) and isinstance(false_branch, DataComponent)
            ):
                raise ValueError(
                    "If then and else operands cannot be dataset and component respectively"
                )
            if isinstance(true_branch, Scalar):
                left = false_branch
                right = true_branch

        # Datacomponent
        if isinstance(condition, DataComponent):
            if not condition.data_type == Boolean:
                raise SemanticError(
                    "1-1-9-11",
                    op=cls.op,
                    type=SCALAR_TYPES_CLASS_REVERSE[condition.data_type],
                )
            if not isinstance(left, Scalar) or not isinstance(right, Scalar):
                nullable = condition.nullable
            else:
                if left.data_type == Null or right.data_type == Null:
                    nullable = True
            if isinstance(left, DataComponent):
                nullable |= left.nullable
            if isinstance(right, DataComponent):
                nullable |= right.nullable
            return DataComponent(
                name="result",
                data=None,
                data_type=binary_implicit_promotion(left.data_type, right.data_type),
                role=Role.MEASURE,
                nullable=nullable,
            )

        # Dataset
        if isinstance(left, Scalar) and isinstance(right, Scalar):
            raise SemanticError(
                "1-1-9-12", op=cls.op, then_symbol=left.name, else_symbol=right.name
            )
        if isinstance(left, DataComponent):
            raise SemanticError(
                "1-1-9-12", op=cls.op, then_symbol=left.name, else_symbol=right.name
            )
        if isinstance(left, Scalar):
            left.data_type = right.data_type = binary_implicit_promotion(
                left.data_type, right.data_type
            )
            return Dataset(name="result", components=copy(condition.components), data=None)
        if left.get_identifiers() != condition.get_identifiers():
            raise SemanticError("1-1-9-10", op=cls.op, clause=left.name)
        if isinstance(right, Scalar):
            for component in left.get_measures():
                if component.data_type != right.data_type:
                    component.data_type = binary_implicit_promotion(
                        component.data_type, right.data_type
                    )
        if isinstance(right, Dataset):
            if left.get_identifiers() != condition.get_identifiers():
                raise SemanticError("1-1-9-10", op=cls.op, clause=right.name)
            if left.get_components_names() != right.get_components_names():
                raise SemanticError("1-1-9-13", op=cls.op, then=left.name, else_clause=right.name)
            for component in left.get_measures():
                if component.data_type != right.components[component.name].data_type:
                    component.data_type = right.components[component.name].data_type = (
                        binary_implicit_promotion(
                            component.data_type,
                            right.components[component.name].data_type,
                        )
                    )
        if isinstance(condition, Dataset):
            if len(condition.get_measures()) != 1:
                raise SemanticError("1-1-9-4", op=cls.op, name=condition.name)
            if condition.get_measures()[0].data_type != Boolean:
                raise SemanticError(
                    "1-1-9-5",
                    op=cls.op,
                    type=SCALAR_TYPES_CLASS_REVERSE[condition.get_measures()[0].data_type],
                )
            if left.get_identifiers() != condition.get_identifiers():
                raise SemanticError("1-1-9-6", op=cls.op)
        result_components = {comp_name: copy(comp) for comp_name, comp in left.components.items()}
        return Dataset(name="result", components=result_components, data=None)


class Nvl(Binary):
    """
    Null class:
        `Nvl <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=229&zoom=100,72,370>`_operator class.
        It has the following class methods:

    Class methods:
        Validate: Class method that validates if the operation at scalar,
        datacomponent or dataset level can be performed.
        Evaluate: Evaluates the actual operation, returning the result.
    """  # noqa E501

    @classmethod
    def evaluate(cls, left: Any, right: Any) -> Union[Scalar, DataComponent, Dataset]:
        result = cls.validate(left, right)

        if isinstance(left, Scalar) and isinstance(result, Scalar):
            if left.data_type is Null:
                result.value = right.value
            else:
                result.value = left.value
        else:
            if not isinstance(result, Scalar):
                if isinstance(right, Scalar):
                    result.data = left.data.fillna(right.value)
                else:
                    result.data = left.data.fillna(right.data)
                if isinstance(result, Dataset):
                    result.data = result.data[result.get_components_names()]
        return result

    @classmethod
    def validate(cls, left: Any, right: Any) -> Union[Scalar, DataComponent, Dataset]:
        result_components = {}
        if isinstance(left, Scalar):
            if not isinstance(right, Scalar):
                raise ValueError(
                    "Nvl operation at scalar level must have scalar "
                    "types on right (applicable) side"
                )
            cls.type_validation(left.data_type, right.data_type)
            return Scalar(name="result", value=None, data_type=left.data_type)
        if isinstance(left, DataComponent):
            if isinstance(right, Dataset):
                raise ValueError(
                    "Nvl operation at component level cannot have "
                    "dataset type on right (applicable) side"
                )
            cls.type_validation(left.data_type, right.data_type)
            return DataComponent(
                name="result",
                data=pd.Series(dtype=object),
                data_type=left.data_type,
                role=Role.MEASURE,
                nullable=False,
            )
        if isinstance(left, Dataset):
            if isinstance(right, DataComponent):
                raise ValueError(
                    "Nvl operation at dataset level cannot have component "
                    "type on right (applicable) side"
                )
            if isinstance(right, Scalar):
                for component in left.get_measures():
                    cls.type_validation(component.data_type, right.data_type)
            if isinstance(right, Dataset):
                for component in left.get_measures():
                    cls.type_validation(
                        component.data_type, right.components[component.name].data_type
                    )
            result_components = {
                comp_name: copy(comp)
                for comp_name, comp in left.components.items()
                if comp.role != Role.ATTRIBUTE
            }
            for comp in result_components.values():
                comp.nullable = False
        return Dataset(name="result", components=result_components, data=None)


class Case(Operator):
    @classmethod
    def evaluate(
        cls, conditions: List[Any], thenOps: List[Any], elseOp: Any
    ) -> Union[Scalar, DataComponent, Dataset]:
        result = cls.validate(conditions, thenOps, elseOp)
        for condition in conditions:
            if isinstance(condition, Dataset) and condition.data is not None:
                condition.data.fillna(False, inplace=True)
                condition_measure = condition.get_measures_names()[0]
                if condition.data[condition_measure].dtype != bool:
                    condition.data[condition_measure] = condition.data[condition_measure].astype(
                        bool
                    )
            elif (
                isinstance(
                    condition,
                    DataComponent,
                )
                and condition.data is not None
            ):
                condition.data.fillna(False, inplace=True)
                if condition.data.dtype != bool:
                    condition.data = condition.data.astype(bool)
            elif isinstance(condition, Scalar) and condition.value is None:
                condition.value = False

        if isinstance(result, Scalar):
            result.value = elseOp.value
            for i in range(len(conditions)):
                if conditions[i].value:
                    result.value = thenOps[i].value

        if isinstance(result, DataComponent):
            result.data = pd.Series(None, index=conditions[0].data.index)

            for i, condition in enumerate(conditions):
                value = thenOps[i].value if isinstance(thenOps[i], Scalar) else thenOps[i].data
                result.data = np.where(
                    condition.data.notna(),
                    np.where(condition.data, value, result.data),
                    result.data,
                )

            condition_mask_else = ~np.any([condition.data for condition in conditions], axis=0)
            else_value = elseOp.value if isinstance(elseOp, Scalar) else elseOp.data
            result.data = pd.Series(
                np.where(condition_mask_else, else_value, result.data),
                index=conditions[0].data.index,
            )

        if isinstance(result, Dataset):
            identifiers = result.get_identifiers_names()
            columns = [col for col in result.get_components_names() if col not in identifiers]
            result.data = (
                conditions[0].data[identifiers]
                if conditions[0].data is not None
                else pd.DataFrame(columns=identifiers)
            )

            for i in range(len(conditions)):
                condition = conditions[i]
                bool_col = next(x.name for x in condition.get_measures() if x.data_type == Boolean)
                condition_mask = condition.data[bool_col]

                result.data.loc[condition_mask, columns] = (
                    thenOps[i].value
                    if isinstance(thenOps[i], Scalar)
                    else thenOps[i].data.loc[condition_mask, columns]
                )

            condition_mask_else = ~np.logical_or.reduce(
                [
                    condition.data[
                        next(x.name for x in condition.get_measures() if x.data_type == Boolean)
                    ].astype(bool)
                    for condition in conditions
                ]
            )

            result.data.loc[condition_mask_else, columns] = (  # type: ignore[index, unused-ignore]
                elseOp.value
                if isinstance(elseOp, Scalar)
                else elseOp.data.loc[condition_mask_else, columns]
            )

        return result

    @classmethod
    def validate(
        cls, conditions: List[Any], thenOps: List[Any], elseOp: Any
    ) -> Union[Scalar, DataComponent, Dataset]:
        if len(set(map(type, conditions))) > 1:
            raise SemanticError("2-1-9-1", op=cls.op)

        ops = thenOps + [elseOp]
        then_else_types = set(map(type, ops))
        condition_type = type(conditions[0])

        if condition_type is Scalar:
            for condition in conditions:
                if condition.data_type != Boolean:
                    raise SemanticError("2-1-9-2", op=cls.op, name=condition.name)
            if list(then_else_types) != [Scalar]:
                raise SemanticError("2-1-9-3", op=cls.op)

            # The output data type is the data type of the last then operation that has a true
            # condition, defaulting to the data type of the else operation if no condition is true
            output_data_type = elseOp.data_type
            for i in range(len(conditions)):
                if conditions[i].value:
                    output_data_type = thenOps[i].data_type

            return Scalar(
                name="result",
                value=None,
                data_type=output_data_type,
            )

        elif condition_type is DataComponent:
            for condition in conditions:
                if not condition.data_type == Boolean:
                    raise SemanticError("2-1-9-4", op=cls.op, name=condition.name)

            nullable = any(
                (thenOp.nullable if isinstance(thenOp, DataComponent) else thenOp.data_type == Null)
                for thenOp in ops
            )
            nullable |= any(condition.nullable for condition in conditions)

            data_type = ops[0].data_type
            for op in ops[1:]:
                data_type = binary_implicit_promotion(data_type, op.data_type)

            return DataComponent(
                name="result",
                data=None,
                data_type=data_type,
                role=Role.MEASURE,
                nullable=nullable,
            )

        # Dataset
        for condition in conditions:
            if len(condition.get_measures_names()) != 1:
                raise SemanticError("1-1-1-4", op=cls.op)
            if condition.get_measures()[0].data_type != Boolean:
                raise SemanticError("2-1-9-5", op=cls.op, name=condition.name)

        if Dataset not in then_else_types:
            raise SemanticError("2-1-9-6", op=cls.op)

        components = next(op for op in ops if isinstance(op, Dataset)).components
        comp_names = [comp.name for comp in components.values()]
        for op in ops:
            if isinstance(op, Dataset) and op.get_components_names() != comp_names:
                raise SemanticError("2-1-9-7", op=cls.op)

        return Dataset(name="result", components=components, data=None)
