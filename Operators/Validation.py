from copy import copy
from typing import Any, Dict, Optional

import pandas as pd

from DataTypes import Boolean, Number, String
from Model import Component, Dataset, Role
from Operators import Operator


# noinspection PyTypeChecker
class Check(Operator):

    @classmethod
    def validate(cls, validation_element: Dataset, imbalance_element: Optional[Dataset],
                 error_code: Optional[str], error_level: Optional[int], invalid: bool) -> Dataset:
        if len(validation_element.get_measures()) != 1:
            raise Exception("The validation operand must have exactly one measure of type Boolean")
        measure = validation_element.get_measures()[0]
        if measure.data_type != Boolean:
            raise Exception("The validation operand must have exactly one measure of type Boolean")

        imbalance_measure = None
        if imbalance_element is not None:
            operand_identifiers = validation_element.get_identifiers_names()
            imbalance_identifiers = imbalance_element.get_identifiers_names()
            if operand_identifiers != imbalance_identifiers:
                raise Exception(
                    "The validation and imbalance operands must have the same identifiers")
            if len(imbalance_element.get_measures()) != 1:
                raise Exception(
                    "The imbalance operand must have exactly one measure of type Numeric")

            imbalance_measure = imbalance_element.get_measures()[0]
            if imbalance_measure.data_type != Number:
                raise Exception(
                    "The imbalance operand must have exactly one measure of type Numeric")

        # Generating the result dataset components
        result_components = {comp.name: comp for comp in validation_element.components.values()
                             if comp.role in [Role.IDENTIFIER, Role.MEASURE]}
        if imbalance_measure is None:
            result_components['imbalance'] = Component(name='imbalance', data_type=Number,
                                                       role=Role.MEASURE, nullable=False)
        else:
            result_components['imbalance'] = copy(imbalance_measure)
            result_components['imbalance'].name = 'imbalance'

        nullable_error_code = error_code is None
        nullable_error_level = error_level is None

        result_components['errorcode'] = Component(name='errorcode', data_type=String,
                                                   role=Role.MEASURE, nullable=nullable_error_code)
        result_components['errorlevel'] = Component(name='errorlevel', data_type=Number,
                                                    role=Role.MEASURE,
                                                    nullable=nullable_error_level)

        return Dataset(name="result", components=result_components, data=None)

    @classmethod
    def evaluate(cls, validation_element: Dataset, imbalance_element: Optional[Dataset],
                 error_code: Optional[str], error_level: Optional[int], invalid: bool) -> Dataset:
        result = cls.validate(validation_element, imbalance_element, error_code, error_level,
                              invalid)
        columns_to_keep = (validation_element.get_identifiers_names() +
                           validation_element.get_measures_names())
        result.data = validation_element.data[columns_to_keep]
        if imbalance_element is not None:
            imbalance_measure_name = imbalance_element.get_measures_names()[0]
            result.data['imbalance'] = imbalance_element.data[imbalance_measure_name]
        else:
            result.data['imbalance'] = None

        result.data['errorcode'] = error_code
        result.data['errorlevel'] = error_level
        if invalid:
            # TODO: Is this always bool_var?? In any case this does the trick for more use cases
            validation_measure_name = validation_element.get_measures_names()[0]
            result.data = result.data[result.data[validation_measure_name] == False]
            result.data.reset_index(drop=True, inplace=True)
        return result


# noinspection PyTypeChecker
class Check_Datapoint(Operator):

    @classmethod
    def validate(cls, dataset_element: Dataset, rule_info: Dict[str, Any], output: str) -> Dataset:
        result_components = {comp.name: comp for comp in dataset_element.get_identifiers()}
        result_components['ruleid'] = Component(name='ruleid', data_type=String,
                                                role=Role.IDENTIFIER, nullable=False)
        if output == 'invalid':
            result_components = {**result_components,
                                 **{comp.name: copy(comp) for comp in
                                    dataset_element.get_measures()}
                                 }
        elif output == 'all':
            result_components['bool_var'] = Component(name='bool_var', data_type=Boolean,
                                                      role=Role.MEASURE, nullable=True)
        else:  # output == 'all_measures'
            result_components = {**result_components,
                                 **{comp.name: copy(comp) for comp in
                                    dataset_element.get_measures()},
                                 'bool_var': Component(name='bool_var', data_type=Boolean,
                                                       role=Role.MEASURE, nullable=True)}
        result_components['errorcode'] = Component(name='errorcode', data_type=String,
                                                    role=Role.MEASURE, nullable=True)
        result_components['errorlevel'] = Component(name='errorlevel', data_type=Number,
                                                    role=Role.MEASURE, nullable=True)

        return Dataset(name="result", components=result_components, data=None)

    @classmethod
    def evaluate(cls, dataset_element: Dataset, rule_info: Dict[str, Any], output: str) -> Dataset:
        result = cls.validate(dataset_element, rule_info, output)
        for rule_name, rule_data in rule_info.items():
            rule_df = rule_data['output']
            rule_df['ruleid'] = rule_name
            rule_df['errorcode'] = rule_df['bool_var'].map({False: rule_data['error_code']})
            rule_df['errorlevel'] = rule_df['bool_var'].map({False: rule_data['error_level']})
            if result.data is None:
                result.data = rule_df
            else:
                result.data = pd.concat([result.data, rule_df], ignore_index=True)

        result.data = result.data.drop_duplicates(subset=result.get_identifiers_names() + ['ruleid'])
        if output == 'invalid':
            result.data = result.data[result.data['bool_var'] == False]
            result.data = result.data.drop(columns=['bool_var'])
            result.data.reset_index(drop=True, inplace=True)
        elif output == 'all':
            result.data = result.data[result.get_identifiers_names() + ['bool_var', 'errorcode', 'errorlevel']]
        else:  # output == 'all_measures'
            result.data = result.data[result.get_identifiers_names() + dataset_element.get_measures_names() + ['bool_var', 'errorcode', 'errorlevel']]

        return result
