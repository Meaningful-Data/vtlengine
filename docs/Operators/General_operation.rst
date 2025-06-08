*****************************************
General operation of vtl engine operators
*****************************************

Vtl engine has a superclass containing the main params to execute the different operation available in this language.
To do it, many class methods are created to indentify what type of data are treating, such as datasets, datacomponents
or even the type of data the operator is using.


.. py:currentmodule:: vtlengine.Operators

Operator class
--------------

.. autoclass:: Operator
    :members:
    :show-inheritance:

Operator class has two subclasses:

Binary
......
This class is prepared to support those operations where two variables are operated.
There are different methods supporting this class, allowing the engine to perform all kind of operations that
vtl language supports.

To distinguish the kind of operator and its role, there are validation methods that verifies what type of data the operand is, focusing on its components and its compatibility.
Also, there are evaluate methods to ensure the type of data is the correct one to operate with in a determined operation.

.. autoclass:: Binary


Unary
.....
This class allows the engine to perform the operations that only have one operand. As binary class, it is supported with validation and evaluation methods.

.. autoclass:: Unary