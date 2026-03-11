"""
AST.ASTConstructor.py
=====================

Description
-----------
Node Creator.
"""

from vtlengine.AST.Grammar._cpp_parser import vtl_cpp_parser
from vtlengine.AST import (
    Argument,
    Assignment,
    DefIdentifier,
    DPRule,
    DPRuleset,
    HRBinOp,
    HRule,
    HRuleset,
    HRUnOp,
    Operator,
    PersistentAssignment,
    Start,
)
from vtlengine.AST.Grammar._cpp_parser._rule_constants import RC
from vtlengine.AST.ASTConstructorModules import extract_token_info
from vtlengine.AST.ASTConstructorModules.Expr import Expr
from vtlengine.AST.ASTConstructorModules.ExprComponents import ExprComp
from vtlengine.AST.ASTConstructorModules.Terminals import Terminals
from vtlengine.AST.ASTDataExchange import de_ruleset_elements
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Component, Dataset, Scalar

# pylint: disable=unreachable,expression-not-assigned
# pylint: disable=assignment-from-no-return


class ASTVisitor:
    """
    This class walks the parse tree (CTS) and transform the structure
    to an AST which nodes are defined at AST.AST.py
    """

    """______________________________________________________________________________________


                                    Start Definition.

        _______________________________________________________________________________________"""

    def visitStart(self, ctx: vtl_cpp_parser.ParseNode) -> Start:
        """
        start:
            (statement  EOL)* EOF
        """
        ctx_list = ctx.children

        statements_nodes = []
        statements = [
            statement
            for statement in ctx_list
            if not statement.is_terminal and statement.rule_index == 1
        ]
        if len(statements) != 0:
            for statement in statements:
                statements_nodes.append(self.visitStatement(statement))

        token_info = extract_token_info(ctx)
        # For the Start node, use the last statement's stop position instead of EOF
        if statements:
            last_stmt_info = extract_token_info(statements[-1])
            token_info["line_stop"] = last_stmt_info["line_stop"]
            token_info["column_stop"] = last_stmt_info["column_stop"]

        start_node = Start(children=statements_nodes, **token_info)

        return start_node

    def visitStatement(self, ctx):
        """
        statement:
            varID ASSIGN expr                # temporaryAssignment
            | varID PUT_SYMBOL expr          # persistAssignment
            | defOperators                   # defineExpression
        ;
        """
        ctx_list = ctx.children
        c = ctx_list[0]

        if ctx.ctx_id == RC.TEMPORARY_ASSIGNMENT:
            return self.visitTemporaryAssignment(ctx)

        elif ctx.ctx_id == RC.PERSIST_ASSIGNMENT:
            return self.visitPersistAssignment(ctx)

        elif ctx.ctx_id == RC.DEFINE_EXPRESSION:
            return self.visitDefineExpression(c)
        else:
            raise NotImplementedError

    # varID ASSIGN expr                # temporaryAssignment
    def visitTemporaryAssignment(self, ctx):
        ctx_list = ctx.children

        left_node = Terminals().visitVarID(ctx_list[0])
        op_node = ctx_list[1].text

        right_node = Expr().visitExpr(ctx_list[2])

        token_info = extract_token_info(ctx)
        assignment_node = Assignment(left=left_node, op=op_node, right=right_node, **token_info)
        return assignment_node

    #     | varID PUT_SYMBOL expr          # persistAssignment
    def visitPersistAssignment(self, ctx):
        """
        persistentAssignment: varID PUT_SYMBOL expr;
        """
        ctx_list = ctx.children

        left_node = Terminals().visitVarID(ctx_list[0])
        op_node = ctx_list[1].text

        right_node = Expr().visitExpr(ctx_list[2])

        token_info = extract_token_info(ctx)
        persistent_assignment_node = PersistentAssignment(
            left=left_node, op=op_node, right=right_node, **token_info
        )
        return persistent_assignment_node

    """______________________________________________________________________________________


                                Artefacts Definition.

    _______________________________________________________________________________________"""

    def visitDefineExpression(self, ctx):
        """
        defExpr: defOperator
               | defDatapoint
               | defHierarchical
               ;
        """
        if ctx.ctx_id == RC.DEF_OPERATOR:
            return self.visitDefOperator(ctx)

        elif ctx.ctx_id == RC.DEF_DATAPOINT_RULESET:
            return self.visitDefDatapointRuleset(ctx)

        elif ctx.ctx_id == RC.DEF_HIERARCHICAL:
            return self.visitDefHierarchical(ctx)

    def visitDefOperator(self, ctx):
        """
        DEFINE OPERATOR operatorID LPAREN (parameterItem (COMMA parameterItem)*)? RPAREN
        (RETURNS outputParameterType)? IS (expr) END OPERATOR        # defOperator"""
        ctx_list = ctx.children

        operator = Terminals().visitOperatorID(ctx_list[2])
        parameters = [
            self.visitParameterItem(parameter)
            for parameter in ctx_list
            if not parameter.is_terminal and parameter.rule_index == 58
        ]
        return_ = [
            Terminals().visitOutputParameterType(datatype)
            for datatype in ctx_list
            if not datatype.is_terminal and datatype.rule_index == 59
        ]
        # Here should be modified if we want to include more than one expr per function.
        expr = [
            Expr().visitExpr(expr)
            for expr in ctx_list
            if not expr.is_terminal and expr.rule_index == 2
        ][0]

        if len(return_) == 0:
            raise SemanticError("1-3-2-2", op=operator)
        else:
            return_node = return_[0]

        token_info = extract_token_info(ctx)

        return Operator(
            op=operator,
            parameters=parameters,
            output_type=return_node,
            expression=expr,
            **token_info,
        )

    """
                        -----------------------------------
                                Define Datapoint
                        -----------------------------------
    """

    def visitDefDatapointRuleset(self, ctx):
        """
        DEFINE DATAPOINT RULESET rulesetID LPAREN rulesetSignature RPAREN IS ruleClauseDatapoint
        END DATAPOINT RULESET                            # defDatapointRuleset
        """

        ctx_list = ctx.children

        ruleset_name = Terminals().visitRulesetID(ctx_list[3])
        signature_type, ruleset_elements = self.visitRulesetSignature(ctx_list[5])
        ruleset_rules = self.visitRuleClauseDatapoint(ctx_list[8])

        token_info = extract_token_info(ctx)

        return DPRuleset(
            name=ruleset_name,
            params=ruleset_elements,
            rules=ruleset_rules,
            signature_type=signature_type,
            **token_info,
        )

    def visitRulesetSignature(self, ctx):
        """
        rulesetSignature: (VALUE_DOMAIN|VARIABLE) varSignature (',' varSignature)* ;
        """
        ctx_list = ctx.children
        signature_type = ctx_list[0].text

        value_domains = [
            value_domain
            for value_domain in ctx_list
            if value_domain.is_terminal and value_domain.symbol_type == vtl_cpp_parser.VALUE_DOMAIN
        ]
        kind = ""
        if len(value_domains) != 0:
            kind = "ValuedomainID"

        variables = [
            variable
            for variable in ctx_list
            if variable.is_terminal and variable.symbol_type == vtl_cpp_parser.VARIABLE
        ]
        if len(variables) != 0:
            kind = "ComponentID"

        component_nodes = [
            Terminals().visitSignature(component, kind)
            for component in ctx_list
            if not component.is_terminal and component.rule_index == 73
        ]

        return signature_type, component_nodes

    def visitRuleClauseDatapoint(self, ctx):
        """
        ruleClauseDatapoint: ruleItemDatapoint (';' ruleItemDatapoint)* ;
        """
        ctx_list = ctx.children

        ruleset_rules = [
            self.visitRuleItemDatapoint(ruleId)
            for ruleId in ctx_list
            if not ruleId.is_terminal and ruleId.rule_index == 75
        ]
        return ruleset_rules

    def visitRuleItemDatapoint(self, ctx):
        """
        ruleItemDatapoint: (IDENTIFIER ':')? ( WHEN expr THEN )? expr (erCode)? (erLevel)? ;
        """
        ctx_list = ctx.children

        when = [
            whens
            for whens in ctx_list
            if whens.is_terminal and whens.symbol_type == vtl_cpp_parser.WHEN
        ]
        rule_name = [
            (
                rule_name.text
                if rule_name.is_terminal and rule_name.symbol_type == vtl_cpp_parser.IDENTIFIER
                else None
            )
            for rule_name in ctx_list
        ][0]
        expr_node = [
            ExprComp().visitExprComponent(rule_node)
            for rule_node in ctx_list
            if not rule_node.is_terminal and rule_node.rule_index == 3
        ]

        if len(when) != 0:
            token_info = extract_token_info(when[0])
            rule_node = HRBinOp(
                left=expr_node[0], op=when[0].text, right=expr_node[1], **token_info
            )

        else:
            rule_node = expr_node[0]

        er_code = [
            Terminals().visitErCode(erCode_name)
            for erCode_name in ctx_list
            if not erCode_name.is_terminal and erCode_name.rule_index == 98
        ]
        er_code = None if len(er_code) == 0 else er_code[0]
        er_level = [
            Terminals().visitErLevel(erLevel_name)
            for erLevel_name in ctx_list
            if not erLevel_name.is_terminal and erLevel_name.rule_index == 99
        ]
        er_level = None if len(er_level) == 0 else er_level[0]

        token_info = extract_token_info(ctx)
        return DPRule(
            name=rule_name, rule=rule_node, erCode=er_code, erLevel=er_level, **token_info
        )

    def visitParameterItem(self, ctx):
        """
        parameterItem: varID dataType (DEFAULT constant)? ;
        """
        ctx_list = ctx.children

        argument_name = [
            Terminals().visitVarID(element)
            for element in ctx_list
            if not element.is_terminal and element.rule_index == 94
        ][0]
        argument_type = [
            Terminals().visitInputParameterType(element)
            for element in ctx_list
            if not element.is_terminal and element.rule_index == 61
        ][0]
        argument_default = [
            Terminals().visitScalarItem(element)
            for element in ctx_list
            if not element.is_terminal and element.rule_index == 43
        ]
        argument_default = None if len(argument_default) == 0 else argument_default[0]

        if isinstance(argument_type, (Dataset, Component, Scalar)):
            argument_type.name = argument_name.value
        token_info = extract_token_info(ctx)
        return Argument(
            name=argument_name.value, type_=argument_type, default=argument_default, **token_info
        )

    """
                        -----------------------------------
                                Define Hierarchical
                        -----------------------------------
    """

    def visitDefHierarchical(self, ctx):
        """
        DEFINE DATAPOINT RULESET rulesetID LPAREN rulesetSignature RPAREN IS ruleClauseDatapoint
        END DATAPOINT RULESET                            # defDatapointRuleset
        """

        ctx_list = ctx.children

        ruleset_name = Terminals().visitRulesetID(ctx_list[3])
        signature_type, ruleset_elements = self.visitHierRuleSignature(ctx_list[5])
        if signature_type == "variable" and isinstance(ruleset_elements, list):
            unique_id_names = list(set([elto.value for elto in ruleset_elements]))
            if len(ruleset_elements) > 2 or len(unique_id_names) < 1:
                raise SemanticError("1-1-10-9", ruleset=ruleset_name)
        ruleset_rules = self.visitRuleClauseHierarchical(ctx_list[8])
        # Keep k,v for the hierarchical rulesets
        de_ruleset_elements[ruleset_name] = ruleset_elements
        if len(ruleset_rules) == 0:
            raise Exception(f"No rules found for the ruleset {ruleset_name}")

        token_info = extract_token_info(ctx)

        return HRuleset(
            signature_type=signature_type,
            name=ruleset_name,
            element=ruleset_elements,
            rules=ruleset_rules,
            **token_info,
        )

    # TODO Add support for value Domains.
    def visitHierRuleSignature(self, ctx):
        """
        hierRuleSignature: (VALUE_DOMAIN|VARIABLE) valueDomainSignature? RULE IDENTIFIER ;
        """
        ctx_list = ctx.children

        signature_type = ctx_list[0].text

        value_domain = [
            valueDomain
            for valueDomain in ctx_list
            if valueDomain.is_terminal and valueDomain.symbol_type == vtl_cpp_parser.VALUE_DOMAIN
        ]
        kind = "ValuedomainID" if len(value_domain) != 0 else "DatasetID"

        conditions = [
            self.visitValueDomainSignature(vtlsig)
            for vtlsig in ctx_list
            if not vtlsig.is_terminal and vtlsig.rule_index == 79
        ]

        dataset = [
            identifier
            for identifier in ctx_list
            if identifier.is_terminal and identifier.symbol_type == vtl_cpp_parser.IDENTIFIER
        ][0]

        token_info = extract_token_info(ctx)
        if conditions:
            identifiers_list = [
                DefIdentifier(
                    value=elto.alias if getattr(elto, "alias", None) else elto.value,
                    kind=kind,
                    **token_info,
                )
                for elto in conditions[0]
            ]
            identifiers_list.append(DefIdentifier(value=dataset.text, kind=kind, **token_info))
            return signature_type, identifiers_list
        else:
            return signature_type, DefIdentifier(value=dataset.text, kind=kind, **token_info)

    # TODO Support for valueDomainSignature.
    def visitValueDomainSignature(self, ctx):
        """
        valueDomainSignature: CONDITION IDENTIFIER (AS IDENTIFIER)? (',' IDENTIFIER (AS IDENTIFIER)?)* ;
        """  # noqa E501
        # AST_ASTCONSTRUCTOR.7
        ctx_list = ctx.children
        component_nodes = [
            Terminals().visitSignature(component)
            for component in ctx_list
            if not component.is_terminal and component.rule_index == 73
        ]
        return component_nodes

    def visitRuleClauseHierarchical(self, ctx):
        """
        ruleClauseHierarchical: ruleItemHierarchical (';' ruleItemHierarchical)* ;
        """
        ctx_list = ctx.children

        rules_nodes = [
            self.visitRuleItemHierarchical(rule)
            for rule in ctx_list
            if not rule.is_terminal and rule.rule_index == 77
        ]
        return rules_nodes

    def visitRuleItemHierarchical(self, ctx):
        """
        ruleItemHierarchical: (ruleName=IDENTIFIER  COLON )? codeItemRelation (erCode)? (erLevel)? ;
        """
        ctx_list = ctx.children

        rule_name = [
            (
                rule_name.text
                if rule_name.is_terminal and rule_name.symbol_type == vtl_cpp_parser.IDENTIFIER
                else None
            )
            for rule_name in ctx_list
        ][0]
        rule_node = [
            self.visitCodeItemRelation(rule_node)
            for rule_node in ctx_list
            if not rule_node.is_terminal and rule_node.rule_index == 80
        ][0]

        er_code = [
            Terminals().visitErCode(erCode_name)
            for erCode_name in ctx_list
            if not erCode_name.is_terminal and erCode_name.rule_index == 98
        ]
        er_code = None if len(er_code) == 0 else er_code[0]
        er_level = [
            Terminals().visitErLevel(erLevel_name)
            for erLevel_name in ctx_list
            if not erLevel_name.is_terminal and erLevel_name.rule_index == 99
        ]
        er_level = None if len(er_level) == 0 else er_level[0]

        token_info = extract_token_info(ctx)

        return HRule(name=rule_name, rule=rule_node, erCode=er_code, erLevel=er_level, **token_info)

    def visitCodeItemRelation(self, ctx):
        """
        codeItemRelation: ( WHEN expr THEN )? codeItemRef codeItemRelationClause (codeItemRelationClause)* ;
                        ( WHEN exprComponent THEN )? codetemRef=valueDomainValue comparisonOperand? codeItemRelationClause (codeItemRelationClause)*

        """  # noqa E501

        ctx_list = ctx.children

        when = None

        if ctx_list[0].is_terminal:
            when = ctx_list[0].text
            vd_value = Terminals().visitValueDomainValue(ctx_list[3])
            op = Terminals().visitComparisonOperand(ctx_list[4])
            token_info_value = extract_token_info(ctx_list[3])
            token_info_op = extract_token_info(ctx_list[4])
        else:
            vd_value = Terminals().visitValueDomainValue(ctx_list[0])
            op = Terminals().visitComparisonOperand(ctx_list[1])
            token_info_value = extract_token_info(ctx_list[0])
            token_info_op = extract_token_info(ctx_list[1])

        rule_node = HRBinOp(
            left=DefIdentifier(value=vd_value, kind="CodeItemID", **token_info_value),
            op=op,
            right=None,
            **token_info_op,
        )
        items = [item for item in ctx_list if not item.is_terminal and item.rule_index == 81]
        token_info = extract_token_info(items[0])
        # Means that no concatenations of operations is needed for that rule.
        if len(items) == 1:
            cir_node = self.visitCodeItemRelationClause(items[0])
            if isinstance(cir_node, HRBinOp):
                rule_node.right = HRUnOp(op=cir_node.op, operand=cir_node.right, **token_info)

            elif isinstance(cir_node, DefIdentifier):
                rule_node.right = cir_node

        # Concatenate all the the elements except the last one.
        else:
            previous_node = self.visitCodeItemRelationClause(items[0])
            if isinstance(previous_node, HRBinOp):
                previous_node = HRUnOp(
                    op=previous_node.op, operand=previous_node.right, **token_info
                )

            for item in items[1:]:
                item_node = self.visitCodeItemRelationClause(item)
                item_node.left = previous_node

                previous_node = item_node

            rule_node.right = previous_node

        if when is not None:
            expr_node = ExprComp().visitExprComponent(ctx_list[1])
            token_when_info = extract_token_info(ctx_list[1])
            rule_node = HRBinOp(left=expr_node, op=when, right=rule_node, **token_when_info)

        return rule_node

    def visitCodeItemRelationClause(self, ctx):
        """
        (opAdd=( PLUS | MINUS  ))? rightCodeItem=valueDomainValue ( QLPAREN  rightCondition=exprComponent  QRPAREN )?
        """  # noqa E501
        ctx_list = ctx.children

        expr = [expr for expr in ctx_list if not expr.is_terminal and expr.rule_index == 2]
        if len(expr) != 0:
            # AST_ASTCONSTRUCTOR.8
            raise NotImplementedError

        right_condition = [
            ExprComp().visitExprComponent(right_condition)
            for right_condition in ctx_list
            if not right_condition.is_terminal and right_condition.ctx_id == RC.COMPARISON_EXPR_COMP
        ]

        if ctx_list[0].is_terminal:
            op = ctx_list[0].text
            value = Terminals().visitValueDomainValue(ctx_list[1])

            code_item = DefIdentifier(
                value=value, kind="CodeItemID", **extract_token_info(ctx_list[1])
            )
            if right_condition:
                code_item._right_condition = right_condition[0]

            return HRBinOp(left=None, op=op, right=code_item, **extract_token_info(ctx_list[0]))
        else:
            value = Terminals().visitValueDomainValue(ctx_list[0])
            code_item = DefIdentifier(
                value=value, kind="CodeItemID", **extract_token_info(ctx_list[0])
            )
            if right_condition:
                code_item._right_condition = right_condition[0]

            return code_item
