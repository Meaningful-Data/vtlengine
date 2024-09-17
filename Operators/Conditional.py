import os
from copy import copy

from DataTypes import Boolean, COMP_NAME_MAPPING, binary_implicit_promotion
from Model import Scalar, DataComponent, Dataset, Role
from Operators import Operator, Binary

if os.environ.get("SPARK", False):
    import pyspark.pandas as pd
else:
    import pandas as pd


class If(Operator):

    @classmethod
    def evaluate(cls, condition, true_branch, false_branch):
        result = cls.validate(condition, true_branch, false_branch)
        if isinstance(condition, DataComponent):
            result.data = cls.component_level_evaluation(condition, true_branch, false_branch)
            if result.role != Role.IDENTIFIER:
                result.nullable = True
        if isinstance(condition, Dataset):
            result = cls.dataset_level_evaluation(result, condition, true_branch, false_branch)
        return result

    @classmethod
    def component_level_evaluation(cls, condition, true_branch, false_branch):
        data = []
        for i, row in enumerate(condition.data):
            if row:
                data.append(true_branch.value if isinstance(true_branch, Scalar) else true_branch.data[i])
            else:
                data.append(false_branch.value if isinstance(false_branch, Scalar) else false_branch.data[i])
        return pd.Series(data)

    @classmethod
    def dataset_level_evaluation(cls, result, condition, true_branch, false_branch):
        ids = condition.get_identifiers_names()
        condition_measure = condition.get_measures_names()[0]

        true_data = condition.data[condition.data[condition_measure]]
        false_data = condition.data[~condition.data[condition_measure]]

        if isinstance(true_branch, Dataset):
            if len(true_data) > 0:
                true_data = pd.merge(true_data, true_branch.data, on=ids, how='right', suffixes=('_condition', ''))
                # true_data = true_data.dropna(subset=[condition_measure])
            else:
                true_data = pd.DataFrame(columns=true_branch.get_components_names())
        else:
            true_data[condition_measure] = true_data[condition_measure].apply(lambda x: true_branch.value)
        if isinstance(false_branch, Dataset):
            if len(false_data) > 0:
                false_data = pd.merge(false_data, false_branch.data, on=ids, how='right', suffixes=('_condition', ''))
                # false_data.dropna(subset=[condition_measure])
            else:
                false_data = pd.DataFrame(columns=false_branch.get_components_names())
        else:
            false_data[condition_measure] = false_data[condition_measure].apply(lambda x: false_branch.value)

        result.data = pd.concat([true_data, false_data], ignore_index=True).drop_duplicates().sort_values(by=ids)
        if isinstance(result, Dataset):
            drop_columns = [column for column in result.data.columns if column not in result.components.keys()]
            result.data = result.data.dropna(subset=drop_columns).drop(columns=drop_columns)
        if isinstance(true_branch, Scalar) and isinstance(false_branch, Scalar):
            result.get_measures()[0].data_type = true_branch.data_type
            result.get_measures()[0].name = COMP_NAME_MAPPING[true_branch.data_type]
            result.data = result.data.rename(columns={condition_measure: result.get_measures()[0].name})
        return result

    @classmethod
    def validate(cls, condition, true_branch, false_branch) -> Scalar | DataComponent | Dataset:
        left = true_branch
        right = false_branch
        if true_branch.__class__ != false_branch.__class__:
            if (isinstance(true_branch, DataComponent) and isinstance(false_branch, Dataset)) or \
                    (isinstance(true_branch, Dataset) and isinstance(false_branch, DataComponent)):
                raise ValueError("If then and else operands cannot be dataset and component respectively")
            if isinstance(true_branch, Scalar):
                left = false_branch
                right = true_branch

        # Datacomponent
        if isinstance(condition, DataComponent):
            return DataComponent(name='result', data=None,
                                 data_type=binary_implicit_promotion(left.data_type, right.data_type),
                                 role=Role.MEASURE, nullable=False)

        # Dataset
        if isinstance(left, DataComponent):
            raise ValueError("If operation at dataset level cannot have component type on left (condition) side")
        if isinstance(left, Scalar):
            left.data_type = right.data_type = binary_implicit_promotion(left.data_type, right.data_type)
            return Dataset(name='result', components=condition.components.copy(), data=None)
        if isinstance(right, Scalar):
            for component in left.get_measures():
                if component.data_type != right.data_type:
                    component.data_type = binary_implicit_promotion(component.data_type, right.data_type)
        if isinstance(right, Dataset):
            if left.get_components_names() != right.get_components_names():
                raise ValueError("If operands at dataset level must have the same components")
            for component in left.get_measures():
                if component.data_type != right.components[component.name].data_type:
                    component.data_type = right.components[component.name].data_type = \
                        binary_implicit_promotion(component.data_type, right.components[component.name].data_type)
        if isinstance(condition, Dataset):
            if len(condition.get_measures()) != 1 or condition.get_measures()[0].data_type != Boolean:
                raise ValueError("If operation at dataset level condition side must be a dataset having an unique "
                                 "boolean measure")
            if left.get_identifiers() != condition.get_identifiers():
                raise ValueError("If operands at dataset level must have the same identifiers as condition")
        result_components = {comp_name: copy(comp) for comp_name, comp in left.components.items()}
        return Dataset(name='result', components=result_components, data=None)


class Nvl(Binary):

    @classmethod
    def evaluate(cls, left, right):
        result = cls.validate(left, right)

        if isinstance(left, Scalar):
            if pd.isnull(left):
                result.value = right.value
            else:
                result.value = left.value
            return result
        if isinstance(left, Dataset):
            for component in result.get_measures():
                component.nullable = False
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
                raise ValueError("Nvl operation at scalar level must have scalar types on right (applicable) side")
            if left.data_type != right.data_type:
                left.data_type = cls.type_validation(left.data_type, right.data_type)
            return Scalar(name='result', value=None, data_type=left.data_type)
        if isinstance(left, DataComponent):
            if isinstance(right, Dataset):
                raise ValueError("Nvl operation at component level cannot have dataset type on right (applicable) side")
            if left.data_type != right.data_type:
                left.data_type = cls.type_validation(left.data_type, right.data_type)
            return DataComponent(name='result', data=pd.Series(), data_type=left.data_type,
                                 role=Role.MEASURE, nullable=False)
        if isinstance(left, Dataset):
            if isinstance(right, DataComponent):
                raise ValueError("Nvl operation at dataset level cannot have component type on right (applicable) side")
            if isinstance(right, Scalar):
                for component in left.get_measures():
                    if component.data_type != right.data_type:
                        component.data_type = cls.type_validation(component.data_type, right.data_type)
            if isinstance(right, Dataset):
                for component in left.get_measures():
                    if component.data_type != right.components[component.name].data_type:
                        component.data_type = cls.type_validation(component.data_type,
                                                                  right.components[component.name].data_type)
            result_components = {comp_name: copy(comp) for comp_name, comp in left.components.items()
                                 if comp.role != Role.ATTRIBUTE}
            return Dataset(name='result', components=result_components, data=None)
