******************
Analytic Operators
******************

This module contains the necessary tools to perform analytic operations. It performs using the library duckdb, which is similar to pandas but has a database background.

.. py:currentmodule:: vtl-engine-spark.Analytic

Analytic's main class inherits from Operators.Unary. Also, it has the following class methods:

.. autoclass:: Analytic

The method validate, validates if the structure of the Dataset is correct, the evaluate method evaluates the data within the dataframe,
and the analytic function orders the measures and identifiers within the dataframe

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

.. autoclass:: FirstValue

.. autoclass:: LastValue

.. autoclass:: Lag

.. autoclass:: Lead

.. autoclass:: Rank

.. autoclass:: RatioToReport



