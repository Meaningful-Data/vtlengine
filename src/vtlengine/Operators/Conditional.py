import os
from copy import copy

from vtlengine.DataTypes import Boolean, COMP_NAME_MAPPING, binary_implicit_promotion, \
    SCALAR_TYPES_CLASS_REVERSE, Null
from vtlengine.Operators import Operator, Binary

from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Scalar, DataComponent, Dataset, Role

if os.environ.get("SPARK", False):
    import pyspark.pandas as pd
else:
    import pandas as pd


class If(Operator):
    """
    If class:
        `If-then-else <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=225&zoom=100,72,142>`_ operator
        inherits from Operator, a superclass that contains general validate and evaluate class methods.
        It has the following class methods:
    Class methods:
        evaluate: Evaluates if the operation is well constructed, checking the actual condition and dropping a boolean
        result. The result will depend on the data class, such as datacomponent and dataset.

        component_level_evaluation: Returns a pandas dataframe with data to set the condition

        dataset_level_evaluation: Sets the dataset and evaluates its correct schema to be able to perform the condition.

        validate: Class method that has two branches so datacomponent and datasets can be validated. With datacomponent,
        the code reviews if it is actually a Measure and if it is a binary operation. Dataset branch reviews if the
        identifiers are the same in 'if', 'then' and 'else'.
    """

    @classmethod
    def evaluate(cls, condition, true_branch, false_branch):
        result = cls.validate(condition, true_branch, false_branch)
        if isinstance(condition, DataComponent):
            result.data = cls.component_level_evaluation(condition, true_branch, false_branch)
        if isinstance(condition, Dataset):
            result = cls.dataset_level_evaluation(result, condition, true_branch, false_branch)
        return result

    @classmethod
    def component_level_evaluation(cls, condition, true_branch, false_branch):
        data = []
        for i, row in enumerate(condition.data):
            if row == True:
                if isinstance(true_branch, Scalar):
                    data.append(true_branch.value)
                elif i in true_branch.data.index:
                    data.append(true_branch.data[i])
                else:
                    data.append(None)
            else:
                if isinstance(false_branch, Scalar):
                    data.append(false_branch.value)
                elif i in false_branch.data.index:
                    data.append(false_branch.data[i])
                else:
                    data.append(None)
        return pd.Series(data, dtype=object).dropna()

    @classmethod
    def dataset_level_evaluation(cls, result, condition, true_branch, false_branch):
        ids = condition.get_identifiers_names()
        condition_measure = condition.get_measures_names()[0]
        true_data = condition.data[condition.data[condition_measure] == True]
        false_data = condition.data[condition.data[condition_measure] != True].fillna(False)

        if isinstance(true_branch, Dataset):
            if len(true_data) > 0:
                true_data = pd.merge(true_data, true_branch.data, on=ids, how='right',
                                     suffixes=('_condition', ''))
            else:
                true_data = pd.DataFrame(columns=true_branch.get_components_names())
        else:
            true_data[condition_measure] = true_data[condition_measure].apply(
                lambda x: true_branch.value)
        if isinstance(false_branch, Dataset):
            if len(false_data) > 0:
                false_data = pd.merge(false_data, false_branch.data, on=ids, how='right',
                                      suffixes=('_condition', ''))
            else:
                false_data = pd.DataFrame(columns=false_branch.get_components_names())
        else:
            false_data[condition_measure] = false_data[condition_measure].apply(
                lambda x: false_branch.value)

        result.data = pd.concat([true_data, false_data],
                                ignore_index=True).drop_duplicates().sort_values(by=ids)
        if isinstance(result, Dataset):
            drop_columns = [column for column in result.data.columns if
                            column not in result.components.keys()]
            result.data = result.data.dropna(subset=drop_columns).drop(columns=drop_columns)
        if isinstance(true_branch, Scalar) and isinstance(false_branch, Scalar):
            result.get_measures()[0].data_type = true_branch.data_type
            result.get_measures()[0].name = COMP_NAME_MAPPING[true_branch.data_type]
            result.data = result.data.rename(
                columns={condition_measure: result.get_measures()[0].name})
        return result

    @classmethod
    def validate(cls, condition, true_branch, false_branch) -> Scalar | DataComponent | Dataset:
        nullable = False
        left = true_branch
        right = false_branch
        if true_branch.__class__ != false_branch.__class__:
            if (isinstance(true_branch, DataComponent) and isinstance(false_branch, Dataset)) or \
                    (isinstance(true_branch, Dataset) and isinstance(false_branch, DataComponent)):
                raise ValueError(
                    "If then and else operands cannot be dataset and component respectively")
            if isinstance(true_branch, Scalar):
                left = false_branch
                right = true_branch

        # Datacomponent
        if isinstance(condition, DataComponent):
            if not condition.data_type == Boolean:
                raise SemanticError("1-1-9-11", op=cls.op,
                                    type=SCALAR_TYPES_CLASS_REVERSE[condition.data_type])
            if not isinstance(left, Scalar) or not isinstance(right, Scalar):
                nullable = condition.nullable
            else:
                if isinstance(left, Scalar) and left.data_type == Null:
                    nullable = True
                if isinstance(right, Scalar) and right.data_type == Null:
                    nullable = True
            if isinstance(left, DataComponent):
                nullable |= left.nullable
            if isinstance(right, DataComponent):
                nullable |= right.nullable
            return DataComponent(name='result', data=None,
                                 data_type=binary_implicit_promotion(left.data_type,
                                                                     right.data_type),
                                 role=Role.MEASURE, nullable=nullable)

        # Dataset
        if isinstance(left, Scalar) and isinstance(right, Scalar):
            raise SemanticError("1-1-9-12", op=cls.op, then_symbol=left.name,
                                else_symbol=right.name)
        if isinstance(left, DataComponent):
            raise SemanticError("1-1-9-12", op=cls.op, then_symbol=left.name,
                                else_symbol=right.name)
        if isinstance(left, Scalar):
            left.data_type = right.data_type = binary_implicit_promotion(left.data_type,
                                                                         right.data_type)
            return Dataset(name='result', components=copy(condition.components), data=None)
        if left.get_identifiers() != condition.get_identifiers():
            raise SemanticError("1-1-9-10", op=cls.op, clause=left.name)
        if isinstance(right, Scalar):
            for component in left.get_measures():
                if component.data_type != right.data_type:
                    component.data_type = binary_implicit_promotion(component.data_type,
                                                                    right.data_type)
        if isinstance(right, Dataset):
            if left.get_identifiers() != condition.get_identifiers():
                raise SemanticError("1-1-9-10", op=cls.op, clause=right.name)
            if left.get_components_names() != right.get_components_names():
                raise SemanticError("1-1-9-13", op=cls.op, then=left.name, else_clause=right.name)
            for component in left.get_measures():
                if component.data_type != right.components[component.name].data_type:
                    component.data_type = right.components[component.name].data_type = \
                        binary_implicit_promotion(component.data_type,
                                                  right.components[component.name].data_type)
        if isinstance(condition, Dataset):
            if len(condition.get_measures()) != 1:
                raise SemanticError("1-1-9-4", op=cls.op, name=condition.name)
            if condition.get_measures()[0].data_type != Boolean:
                raise SemanticError("1-1-9-5", op=cls.op, type=SCALAR_TYPES_CLASS_REVERSE[
                    condition.get_measures()[0].data_type])
            if left.get_identifiers() != condition.get_identifiers():
                raise SemanticError("1-1-9-6", op=cls.op)
        result_components = {comp_name: copy(comp) for comp_name, comp in left.components.items()}
        return Dataset(name='result', components=result_components, data=None)


class Nvl(Binary):
    """
    Null class:
        `Nvl <https://sdmx.org/wp-content/uploads/VTL-2.1-Reference-Manual.pdf#page=229&zoom=100,72,370>`_operator class.
        It has the following class methods:

    Class methods:
        Validate: Class method that validates if the operation at scalar, datacomponent or dataset level can be performed.
        Evaluate: Evaluates the actual operation, returning the result.
    """

    @classmethod
    def evaluate(cls, left, right):
        result = cls.validate(left, right)

        if isinstance(left, Scalar):
            if pd.isnull(left):
                result.value = right.value
            else:
                result.value = left.value
            return result
        if isinstance(right, Scalar):
            result.data = left.data.fillna(right.value)
        if isinstance(right, Dataset) or isinstance(right, DataComponent):
            result.data = left.data.fillna(right.data)
        if isinstance(result, Dataset):
            result.data = result.data[result.get_components_names()]
        return result

    @classmethod
    def validate(cls, left, right) -> Scalar | DataComponent | Dataset:
        if isinstance(left, Scalar):
            if not isinstance(right, Scalar):
                raise ValueError(
                    "Nvl operation at scalar level must have scalar types on right (applicable) side")
            cls.type_validation(left.data_type, right.data_type)
            return Scalar(name='result', value=None, data_type=left.data_type)
        if isinstance(left, DataComponent):
            if isinstance(right, Dataset):
                raise ValueError(
                    "Nvl operation at component level cannot have dataset type on right (applicable) side")
            cls.type_validation(left.data_type, right.data_type)
            return DataComponent(name='result', data=pd.Series(dtype=object),
                                 data_type=left.data_type,
                                 role=Role.MEASURE, nullable=False)
        if isinstance(left, Dataset):
            if isinstance(right, DataComponent):
                raise ValueError(
                    "Nvl operation at dataset level cannot have component type on right (applicable) side")
            if isinstance(right, Scalar):
                for component in left.get_measures():
                    cls.type_validation(component.data_type, right.data_type)
            if isinstance(right, Dataset):
                for component in left.get_measures():
                    cls.type_validation(component.data_type,
                                        right.components[component.name].data_type)
            result_components = {comp_name: copy(comp) for comp_name, comp in
                                 left.components.items()
                                 if comp.role != Role.ATTRIBUTE}
            for comp in result_components.values():
                comp.nullable = False
            return Dataset(name='result', components=result_components, data=None)
