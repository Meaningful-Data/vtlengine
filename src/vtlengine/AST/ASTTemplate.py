"""
AST.ASTTemplate.py
==================

Description
-----------
Template to start a new visitor for the AST.
"""

from typing import Any

import vtlengine.AST as AST
from vtlengine.AST.ASTVisitor import NodeVisitor


class ASTTemplate(NodeVisitor):
    """
    Template to start a new visitor for the AST.
    """

    def __init__(self):
        pass

    """______________________________________________________________________________________


                                Start of visiting nodes.

    _______________________________________________________________________________________"""

    def visit_Start(self, node: AST.Start) -> Any:
        """
        Start: (children)

        Basic usage:

            for child in node.children:
                self.visit(child)
        """
        for child in node.children:
            self.visit(child)

    def visit_Assignment(self, node: AST.Assignment) -> Any:
        """
        Assignment: (left, op, right)

        op: :=

        Basic usage:

            self.visit(node.left)
            self.visit(node.right)
        """
        self.visit(node.left)
        self.visit(node.right)

    def visit_PersistentAssignment(self, node: AST.PersistentAssignment) -> Any:
        """
        PersistentAssignment: (left, op, right)

        op: <-

        Basic usage:

            self.visit(node.left)
            self.visit(node.right)
        """
        self.visit(node.left)
        self.visit(node.right)

    def visit_VarID(self, node: AST.VarID) -> Any:
        """
        VarID: (token, value)

        Basic usage:

            return node.value
        """
        return node.value

    def visit_UnaryOp(self, node: AST.UnaryOp) -> Any:
        """
        UnaryOp: (op, operand)

        op:

        Basic usage:

            self.visit(node.operand)
        """
        self.visit(node.operand)

    def visit_BinOp(self, node: AST.BinOp) -> None:
        """
        BinOp: (left, op, right)

        op:

        Basic usage:

            self.visit(node.left)
            self.visit(node.right)
        """
        self.visit(node.left)
        self.visit(node.right)

    def visit_MulOp(self, node: AST.MulOp) -> None:
        """
        MulOp: (op, children)

        op:

        Basic usage:

            for child in node.children:
                self.visit(child)
        """
        for child in node.children:
            self.visit(child)

    def visit_ParamOp(self, node: AST.ParamOp) -> None:
        """
        ParamOp: (op, children, params)

        op:

        Basic usage:

            for child in node.children:
                self.visit(child)
            for param in node.params:
                self.visit(param)
        """
        for child in node.children:
            if isinstance(child, AST.AST):
                self.visit(child)
        for param in node.params:
            if isinstance(param, AST.AST):
                self.visit(param)

    def visit_JoinOp(self, node: AST.JoinOp) -> None:
        """
        JoinOp: (op, clauses, using)

        op:

        Basic usage:

            for clause in node.clauses:
                self.visit(clause)

            if node.using != None:
                self.visit(node.body)
        """
        for clause in node.clauses:
            self.visit(clause)

        if node.using is not None:
            self.visit(node.using)

    def visit_Constant(self, node: AST.Constant) -> AST.AST:
        """
        Constant: (type, value)

        Basic usage:

            return node.value
        """
        return node.value

    def visit_ParamConstant(self, node: AST.ParamConstant) -> Any:
        """
        Constant: (type, value)

        Basic usage:

            return node.value
        """
        return node.value

    def visit_Identifier(self, node: AST.Identifier) -> Any:
        """
        Identifier: (value)

        Basic usage:

            return node.value
        """
        return node.value

    def visit_Optional(self, node: AST.Optional) -> AST.AST:
        """
        Optional: (value)

        Basic usage:

            return node.value
        """
        return node.value

    def visit_ID(self, node: AST.ID) -> AST.AST:
        """
        ID: (type, value)

        Basic usage:

            return node.value
        """
        if node.value == "_":
            return
        return node.value

    def visit_Role(self, node: AST.Role) -> AST.AST:
        """
        Role: (role)

        Basic usage:

            return node.role
        """
        return node.role

    def visit_Collection(self, node: AST.Collection) -> None:
        """
        Collection: (name, type, children)

        Basic usage:

            for child in node.children:
                self.visit(child)
        """
        for child in node.children:
            self.visit(child)

    def visit_RegularAggregation(self, node: AST.RegularAggregation) -> None:
        """
        RegularAggregation: (dataset, op, children)

        op:

        Basic usage:

            self.visit(node.dataset)
            for child in node.children:
                self.visit(child)
        """
        self.visit(node.dataset)
        for child in node.children:
            self.visit(child)

    def visit_Aggregation(self, node: AST.Aggregation) -> None:
        """
        Aggregation: (op, operand, grouping_op, grouping)

        op: SUM, AVG , COUNT, MEDIAN, MIN, MAX, STDDEV_POP, STDDEV_SAMP,
              VAR_POP, VAR_SAMP

        grouping types: 'group by', 'group except', 'group all'.

        Basic usage:

            self.visit(node.operand)

            if node.params != None:
                for param in node.params:
                    self.visit(param)
            if node.grouping != None:
                for group in node.grouping:
                    self.visit(group)
        """
        self.visit(node.operand)

        if node.grouping is not None:
            for group in node.grouping:
                self.visit(group)

    def visit_Analytic(self, node: AST.Analytic) -> None:
        """ """
        if node.operand is not None:
            self.visit(node.operand)

    def visit_TimeAggregation(self, node: AST.TimeAggregation) -> None:
        """
        TimeAggregation: (op, operand, params, conf)

        op types: TIME_AGG

        Basic usage:

            if node.operand != None:
            self.visit(node.operand)

            if node.params != None:
                for param in node.params:
                    self.visit(param)
        """
        if node.operand is not None:
            self.visit(node.operand)

    def visit_If(self, node: AST.If) -> Any:
        """
        If: (condition, thenOp, elseOp)

        Basic usage:

            self.visit(node.condition)
            self.visit(node.thenOp)
            self.visit(node.elseOp)
        """
        self.visit(node.condition)
        self.visit(node.thenOp)
        self.visit(node.elseOp)

    def visit_Case(self, node: AST.Case) -> Any:
        """
        Case: (conditions, thenOp, elseOp)

        Basic usage:

            for condition in node.conditions:
                self.visit(condition)
            self.visit(node.thenOp)
            self.visit(node.elseOp)
        """
        for case in node.cases:
            self.visit(case.condition)
            self.visit(case.thenOp)
        self.visit(node.elseOp)

    def visit_CaseObj(self, node: AST.CaseObj) -> Any:
        """
        CaseObj: (condition, thenOp)

        Basic usage:

            self.visit(node.condition)
            self.visit(node.thenOp)
        """
        self.visit(node.condition)
        self.visit(node.thenOp)

    def visit_Validation(self, node: AST.Validation) -> Any:
        """
        Validation: (op, validation, params, inbalance, invalid)

        Basic usage:

            self.visit(node.validation)
            for param in node.params:
                self.visit(param)

            if node.inbalance!=None:
                self.visit(node.inbalance)

        """
        self.visit(node.validation)

        if node.imbalance is not None:
            self.visit(node.imbalance)

    def visit_Operator(self, node: AST.Operator) -> None:
        """
        Operator: (operator, parameters, outputType, expresion)

        Basic usage:

            for parameter in node.parameters:
                self.visit(parameter)

            self.visit(node.expresion)
        """
        for parameter in node.parameters:
            self.visit(parameter)

        self.visit(node.expression)

    def visit_Types(self, node: AST.Types) -> None:
        """
        Types: (kind, type_, constraints, nullable)

        Basic usage:

            for constraint in node.constraints:
                self.visit(constraint)
        """
        if node.constraints is not None:
            for constraint in node.constraints:
                self.visit(constraint)

    def visit_Argument(self, node: AST.Argument) -> None:
        """
        Argument: (name, type_, default)

        Basic usage:

            self.visit(node.type_)
            if default != None:
                self.visit(node.default)
        """
        self.visit(node.type_)
        if node.default is not None:
            self.visit(node.default)

    def visit_HRuleset(self, node: AST.HRuleset) -> None:
        """
        HRuleset: (name, element, rules)

        Basic usage:

            self.visit(node.element)
            for rule in node.rules:
                self.visit(rule)
        """
        self.visit(node.element)
        for rule in node.rules:
            self.visit(rule)

    def visit_DPRuleset(self, node: AST.DPRuleset) -> None:
        """
        DPRuleset: (name, element, rules)

        Basic usage:

            self.visit(node.element)
            for rule in node.rules:
                self.visit(rule)
        """
        for element in node.params:
            self.visit(element)
        for rule in node.rules:
            self.visit(rule)

    def visit_HRule(self, node: AST.HRule) -> None:
        """
        HRule: (name, rule, erCode, erLevel)

        Basic usage:

            self.visit(node.rule)
            if node.erCode != None:
                self.visit(node.erCode)
            if node.erLevel != None:
                self.visit(node.erLevel)
        """
        self.visit(node.rule)

    def visit_DPRule(self, node: AST.DPRule) -> None:
        """
        DPRule: (name, rule, erCode, erLevel)

        Basic usage:

            self.visit(node.rule)
            if node.erCode != None:
                self.visit(node.erCode)
            if node.erLevel != None:
                self.visit(node.erLevel)
        """
        self.visit(node.rule)

    def visit_HRBinOp(self, node: AST.HRBinOp) -> None:
        """
        HRBinOp: (left, op, right)

        Basic usage:

            self.visit(node.left)
            self.visit(node.right)
        """
        self.visit(node.left)
        self.visit(node.right)

    def visit_HRUnOp(self, node: AST.HRUnOp) -> None:
        """
        HRUnOp: (op, operand)

        Basic usage:

            self.visit(node.operand)
        """
        self.visit(node.operand)

    def visit_DefIdentifier(self, node: AST.DefIdentifier) -> AST.AST:
        """
        DefIdentifier: (value, kind)

        Basic usage:

            return node.value
        """
        return node.value

    def visit_DPRIdentifier(self, node: AST.DPRIdentifier) -> str:
        """
        DefIdentifier: (value, kind)

        Basic usage:

            return node.value
        """
        return node.value

    def visit_EvalOp(self, node: AST.EvalOp) -> Any:
        """
        EvalOp: (name, children, output, language)

        Basic usage:

            for child in node.children:
                self.visit(child)
            if node.output != None:
                self.visit(node.output)

        """
        for child in node.operands:
            self.visit(child)

    def visit_ParFunction(self, node: AST.ParFunction) -> None:
        """
        ParFunction: (operand)
        """
        self.visit(node.operand)

    def visit_NoOp(self, node: AST.NoOp) -> None:  # pylint: disable=unused-argument
        """
        NoOp: ()

        Basic usage:

            pass
        """
        ...

    def visit_RenameNode(self, node: AST.RenameNode) -> None:
        """
        RenameNode: (name, to)
        """

    def visit_UDOCall(self, node: AST.UDOCall) -> None:
        """
        UDOCall: (name, children, params)
        """
        for param in node.params:
            self.visit(param)

    def visit_Windowing(self, node: AST.Windowing) -> None:
        """
        Windowing: (type_, start, start_mode, stop, stop_mode)
        """

    def visit_Comment(self, node: AST.Comment) -> None:
        """
        Comment: (value)
        """
