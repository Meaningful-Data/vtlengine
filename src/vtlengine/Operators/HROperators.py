import operator
from copy import copy
from typing import List, Optional

import vtlengine.Operators as Operators
from vtlengine.AST.Grammar.tokens import HIERARCHY
from vtlengine.DataTypes import Boolean, Number
from vtlengine.Model import Component, DataComponent, Dataset, Role
from vtlengine.Utils.__Virtual_Assets import VirtualCounter
from vtlengine.ViralPropagation import require_rules


def get_measure_from_dataset(dataset: Dataset, code_item: str) -> DataComponent:
    measure_name = dataset.get_measures_names()[0]
    return DataComponent(
        name=code_item,
        data=None,
        data_type=dataset.components[measure_name].data_type,
        role=dataset.components[measure_name].role,
        nullable=dataset.components[measure_name].nullable,
    )


class HRBinOp(Operators.Binary):
    pass


class HRComparison(HRBinOp):
    @classmethod
    def validate(cls, left_operand: Dataset, right_operand: DataComponent) -> Dataset:
        result_components = {
            comp_name: copy(comp)
            for comp_name, comp in left_operand.components.items()
            if comp.role == Role.IDENTIFIER
        }
        result_components["bool_var"] = Component(
            name="bool_var", data_type=Boolean, role=Role.MEASURE, nullable=True
        )
        result_components["imbalance"] = Component(
            name="imbalance", data_type=Number, role=Role.MEASURE, nullable=True
        )
        return Dataset(
            name=f"{left_operand.name}{cls.op}{right_operand.name}",
            components=result_components,
            data=None,
        )


class HREqual(HRComparison):
    op = "="
    py_op = operator.eq


class HRGreater(HRComparison):
    op = ">"
    py_op = operator.gt


class HRGreaterEqual(HRComparison):
    op = ">="
    py_op = operator.ge


class HRLess(HRComparison):
    op = "<"
    py_op = operator.lt


class HRLessEqual(HRComparison):
    op = "<="
    py_op = operator.le


class HRBinNumeric(HRBinOp):
    pass


class HRBinPlus(HRBinNumeric):
    op = "+"
    py_op = operator.add


class HRBinMinus(HRBinNumeric):
    op = "-"
    py_op = operator.sub


class HRUnNumeric(Operators.Unary):
    pass


class HRUnPlus(HRUnNumeric):
    op = "+"
    py_op = operator.pos


class HRUnMinus(HRUnNumeric):
    op = "-"
    py_op = operator.neg


class HAAssignment(Operators.Binary):
    @classmethod
    def validate(cls, left: Dataset, right: DataComponent) -> Dataset:
        result_components = {comp_name: copy(comp) for comp_name, comp in left.components.items()}
        return Dataset(name=f"{left.name}", components=result_components, data=None)


class Hierarchy(Operators.Operator):
    op = HIERARCHY

    @classmethod
    def validate(
        cls,
        dataset: Dataset,
        output: str,
        viral_components: Optional[List[Component]] = None,
    ) -> Dataset:
        dataset_name = VirtualCounter._new_ds_name()
        result_components = {
            comp_name: copy(comp) for comp_name, comp in dataset.components.items()
        }
        # Viral attributes propagate to the hierarchy result (issue #877).
        for viral_comp in viral_components or []:
            result_components[viral_comp.name] = copy(viral_comp)
        # The roll-up combines child nodes into each computed node, so the combined viral
        # attributes require a propagation rule (issue #906).
        require_rules(viral_components or [])
        return Dataset(name=dataset_name, components=result_components, data=None)
