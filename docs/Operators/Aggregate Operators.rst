*********************
Aggregation Operators
*********************

This module contains the necessary operators to perform aggregate operations.

.. py:currentmodule:: vtlengine.Operators.Aggregation

The main characteristic of this kind of operation is the use of the expressions 'group by' and
'group except' to extract the identifiers. Also, the use of pandas dataframe is the main method to perform it.

.. autoclass:: Aggregation

The Aggregation main class has two class methods: validate and evaluate. The first one validates if the structure of the Dataset and the second one
evaluates the data within the dataframe.

For each aggregation operand, there is a class to perform them. These operators are following:

.. autoclass:: Max

.. autoclass:: Min

.. autoclass:: Sum

.. autoclass:: Count

.. autoclass:: Avg

.. autoclass:: Median

.. autoclass:: PopulationStandardDeviation

.. autoclass:: SampleStandardDeviation

.. autoclass:: PopulationVariance

.. autoclass:: SampleVariance

Each operator has a TOKEN that specifies the operator and the type of data that is allowed to perform it. Also, the use of specific pandas functions are integrated.