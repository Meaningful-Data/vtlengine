"""
AST.ASTConstructor.py
=====================

Description
-----------
Node Creator.
"""

from antlr4.tree.Tree import TerminalNodeImpl

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
from vtlengine.AST.ASTConstructorModules import extract_token_info
from vtlengine.AST.ASTConstructorModules.Expr import Expr
from vtlengine.AST.ASTConstructorModules.ExprComponents import ExprComp
from vtlengine.AST.ASTConstructorModules.Terminals import Terminals
from vtlengine.AST.ASTDataExchange import de_ruleset_elements
from vtlengine.AST.Grammar.parser import Parser
from vtlengine.AST.VtlVisitor import VtlVisitor
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Component, Dataset, Scalar

# pylint: disable=unreachable,expression-not-assigned
# pylint: disable=assignment-from-no-return


class ASTVisitor(VtlVisitor):
    """
    This class walks the parse tree (CTS) and transform the structure
    to an AST which nodes are defined at AST.AST.py
    """

    """______________________________________________________________________________________


                                    Start Definition.

        _______________________________________________________________________________________"""

    def visitStart(self, ctx: Parser.StartContext):
        """
        start:
            (statement  EOL)* EOF
        """
        ctx_list = list(ctx.getChildren())

        statements_nodes = []
        statements = [
            statement for statement in ctx_list if isinstance(statement, Parser.StatementContext)
        ]
        if len(statements) != 0:
            for statement in statements:
                statements_nodes.append(self.visitStatement(statement))

        token_info = extract_token_info(ctx)

        start_node = Start(children=statements_nodes, **token_info)

        return start_node

    def visitStatement(self, ctx: Parser.StatementContext):
        """
        statement:
            varID ASSIGN expr                # temporaryAssignment
            | varID PUT_SYMBOL expr          # persistAssignment
            | defOperators                   # defineExpression
        ;
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        if isinstance(ctx, Parser.TemporaryAssignmentContext):
            return self.visitTemporaryAssignment(ctx)

        elif isinstance(ctx, Parser.PersistAssignmentContext):
            return self.visitPersistAssignment(ctx)

        elif isinstance(ctx, Parser.DefineExpressionContext):
            return self.visitDefineExpression(c)
        else:
            raise NotImplementedError

    # varID ASSIGN expr                # temporaryAssignment
    def visitTemporaryAssignment(self, ctx: Parser.TemporaryAssignmentContext):
        ctx_list = list(ctx.getChildren())

        left_node = Terminals().visitVarID(ctx_list[0])
        op_node = ctx_list[1].getSymbol().text

        right_node = Expr().visitExpr(ctx_list[2])

        token_info = extract_token_info(ctx)
        assignment_node = Assignment(left=left_node, op=op_node, right=right_node, **token_info)
        return assignment_node

    #     | varID PUT_SYMBOL expr          # persistAssignment
    def visitPersistAssignment(self, ctx: Parser.PersistAssignmentContext):
        """
        persistentAssignment: varID PUT_SYMBOL expr;
        """
        ctx_list = list(ctx.getChildren())

        left_node = Terminals().visitVarID(ctx_list[0])
        op_node = ctx_list[1].getSymbol().text

        right_node = Expr().visitExpr(ctx_list[2])

        token_info = extract_token_info(ctx)
        persistent_assignment_node = PersistentAssignment(
            left=left_node, op=op_node, right=right_node, **token_info
        )
        return persistent_assignment_node

    """______________________________________________________________________________________


                                Artefacts Definition.

    _______________________________________________________________________________________"""

    def visitDefineExpression(self, ctx: Parser.DefineExpressionContext):
        """
        defExpr: defOperator
               | defDatapoint
               | defHierarchical
               ;
        """
        if isinstance(ctx, Parser.DefOperatorContext):
            return self.visitDefOperator(ctx)

        elif isinstance(ctx, Parser.DefDatapointRulesetContext):
            return self.visitDefDatapointRuleset(ctx)

        elif isinstance(ctx, Parser.DefHierarchicalContext):
            return self.visitDefHierarchical(ctx)

    def visitDefOperator(self, ctx: Parser.DefOperatorContext):
        """
        DEFINE OPERATOR operatorID LPAREN (parameterItem (COMMA parameterItem)*)? RPAREN
        (RETURNS outputParameterType)? IS (expr) END OPERATOR        # defOperator"""
        ctx_list = list(ctx.getChildren())

        operator = Terminals().visitOperatorID(ctx_list[2])
        parameters = [
            self.visitParameterItem(parameter)
            for parameter in ctx_list
            if isinstance(parameter, Parser.ParameterItemContext)
        ]
        return_ = [
            Terminals().visitOutputParameterType(datatype)
            for datatype in ctx_list
            if isinstance(datatype, Parser.OutputParameterTypeContext)
        ]
        # Here should be modified if we want to include more than one expr per function.
        expr = [
            Expr().visitExpr(expr) for expr in ctx_list if isinstance(expr, Parser.ExprContext)
        ][0]

        if len(return_) == 0:
            raise SemanticError("1-4-2-5", op=operator)
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

    def visitDefDatapointRuleset(self, ctx: Parser.DefDatapointRulesetContext):
        """
        DEFINE DATAPOINT RULESET rulesetID LPAREN rulesetSignature RPAREN IS ruleClauseDatapoint
        END DATAPOINT RULESET                            # defDatapointRuleset
        """

        ctx_list = list(ctx.getChildren())

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

    def visitRulesetSignature(self, ctx: Parser.RulesetSignatureContext):
        """
        rulesetSignature: (VALUE_DOMAIN|VARIABLE) varSignature (',' varSignature)* ;
        """
        ctx_list = list(ctx.getChildren())
        signature_type = ctx_list[0].getSymbol().text

        value_domains = [
            value_domain
            for value_domain in ctx_list
            if isinstance(value_domain, TerminalNodeImpl)
            and value_domain.getSymbol().type == Parser.VALUE_DOMAIN
        ]
        kind = ""
        if len(value_domains) != 0:
            kind = "ValuedomainID"

        variables = [
            variable
            for variable in ctx_list
            if isinstance(variable, TerminalNodeImpl)
            and variable.getSymbol().type == Parser.VARIABLE
        ]
        if len(variables) != 0:
            kind = "ComponentID"

        component_nodes = [
            Terminals().visitSignature(component, kind)
            for component in ctx_list
            if isinstance(component, Parser.SignatureContext)
        ]

        return signature_type, component_nodes

    def visitRuleClauseDatapoint(self, ctx: Parser.RuleClauseDatapointContext):
        """
        ruleClauseDatapoint: ruleItemDatapoint (';' ruleItemDatapoint)* ;
        """
        ctx_list = list(ctx.getChildren())

        ruleset_rules = [
            self.visitRuleItemDatapoint(ruleId)
            for ruleId in ctx_list
            if isinstance(ruleId, Parser.RuleItemDatapointContext)
        ]
        return ruleset_rules

    def visitRuleItemDatapoint(self, ctx: Parser.RuleItemDatapointContext):
        """
        ruleItemDatapoint: (IDENTIFIER ':')? ( WHEN expr THEN )? expr (erCode)? (erLevel)? ;
        """
        ctx_list = list(ctx.getChildren())

        when = [
            whens
            for whens in ctx_list
            if isinstance(whens, TerminalNodeImpl) and whens.getSymbol().type == Parser.WHEN
        ]
        rule_name = [
            (
                rule_name.getSymbol().text
                if isinstance(rule_name, TerminalNodeImpl)
                and rule_name.getSymbol().type == Parser.IDENTIFIER
                else None
            )
            for rule_name in ctx_list
        ][0]
        expr_node = [
            ExprComp().visitExprComponent(rule_node)
            for rule_node in ctx_list
            if isinstance(rule_node, Parser.ExprComponentContext)
        ]

        if len(when) != 0:
            token_info = extract_token_info(when[0].getSymbol())
            rule_node = HRBinOp(
                left=expr_node[0], op=when[0].getSymbol().text, right=expr_node[1], **token_info
            )

        else:
            rule_node = expr_node[0]

        er_code = [
            Terminals().visitErCode(erCode_name)
            for erCode_name in ctx_list
            if isinstance(erCode_name, Parser.ErCodeContext)
        ]
        er_code = None if len(er_code) == 0 else er_code[0]
        er_level = [
            Terminals().visitErLevel(erLevel_name)
            for erLevel_name in ctx_list
            if isinstance(erLevel_name, Parser.ErLevelContext)
        ]
        er_level = None if len(er_level) == 0 else er_level[0]

        token_info = extract_token_info(ctx)
        return DPRule(
            name=rule_name, rule=rule_node, erCode=er_code, erLevel=er_level, **token_info
        )

    def visitParameterItem(self, ctx: Parser.ParameterItemContext):
        """
        parameterItem: varID dataType (DEFAULT constant)? ;
        """
        ctx_list = list(ctx.getChildren())

        argument_name = [
            Terminals().visitVarID(element)
            for element in ctx_list
            if isinstance(element, Parser.VarIDContext)
        ][0]
        argument_type = [
            Terminals().visitInputParameterType(element)
            for element in ctx_list
            if isinstance(element, Parser.InputParameterTypeContext)
        ][0]
        argument_default = [
            Terminals().visitScalarItem(element)
            for element in ctx_list
            if isinstance(element, Parser.ScalarItemContext)
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

    def visitDefHierarchical(self, ctx: Parser.DefHierarchicalContext):
        """
        DEFINE DATAPOINT RULESET rulesetID LPAREN rulesetSignature RPAREN IS ruleClauseDatapoint
        END DATAPOINT RULESET                            # defDatapointRuleset
        """

        ctx_list = list(ctx.getChildren())

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
    def visitHierRuleSignature(self, ctx: Parser.HierRuleSignatureContext):
        """
        hierRuleSignature: (VALUE_DOMAIN|VARIABLE) valueDomainSignature? RULE IDENTIFIER ;
        """
        ctx_list = list(ctx.getChildren())

        signature_type = ctx_list[0].getSymbol().text

        value_domain = [
            valueDomain
            for valueDomain in ctx_list
            if isinstance(valueDomain, TerminalNodeImpl)
            and valueDomain.getSymbol().type == Parser.VALUE_DOMAIN
        ]
        kind = "ValuedomainID" if len(value_domain) != 0 else "DatasetID"

        conditions = [
            self.visitValueDomainSignature(vtlsig)
            for vtlsig in ctx_list
            if isinstance(vtlsig, Parser.ValueDomainSignatureContext)
        ]

        dataset = [
            identifier
            for identifier in ctx_list
            if isinstance(identifier, TerminalNodeImpl)
            and identifier.getSymbol().type == Parser.IDENTIFIER
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
            identifiers_list.append(
                DefIdentifier(value=dataset.getSymbol().text, kind=kind, **token_info)
            )
            return signature_type, identifiers_list
        else:
            return signature_type, DefIdentifier(
                value=dataset.getSymbol().text, kind=kind, **token_info
            )

    # TODO Support for valueDomainSignature.
    def visitValueDomainSignature(self, ctx: Parser.ValueDomainSignatureContext):
        """
        valueDomainSignature: CONDITION IDENTIFIER (AS IDENTIFIER)? (',' IDENTIFIER (AS IDENTIFIER)?)* ;
        """  # noqa E501
        # AST_ASTCONSTRUCTOR.7
        ctx_list = list(ctx.getChildren())
        component_nodes = [
            Terminals().visitSignature(component)
            for component in ctx_list
            if isinstance(component, Parser.SignatureContext)
        ]
        return component_nodes

    def visitRuleClauseHierarchical(self, ctx: Parser.RuleClauseHierarchicalContext):
        """
        ruleClauseHierarchical: ruleItemHierarchical (';' ruleItemHierarchical)* ;
        """
        ctx_list = list(ctx.getChildren())

        rules_nodes = [
            self.visitRuleItemHierarchical(rule)
            for rule in ctx_list
            if isinstance(rule, Parser.RuleItemHierarchicalContext)
        ]
        return rules_nodes

    def visitRuleItemHierarchical(self, ctx: Parser.RuleItemHierarchicalContext):
        """
        ruleItemHierarchical: (ruleName=IDENTIFIER  COLON )? codeItemRelation (erCode)? (erLevel)? ;
        """
        ctx_list = list(ctx.getChildren())

        rule_name = [
            (
                rule_name.getSymbol().text
                if isinstance(rule_name, TerminalNodeImpl)
                and rule_name.getSymbol().type == Parser.IDENTIFIER
                else None
            )
            for rule_name in ctx_list
        ][0]
        rule_node = [
            self.visitCodeItemRelation(rule_node)
            for rule_node in ctx_list
            if isinstance(rule_node, Parser.CodeItemRelationContext)
        ][0]

        er_code = [
            Terminals().visitErCode(erCode_name)
            for erCode_name in ctx_list
            if isinstance(erCode_name, Parser.ErCodeContext)
        ]
        er_code = None if len(er_code) == 0 else er_code[0]
        er_level = [
            Terminals().visitErLevel(erLevel_name)
            for erLevel_name in ctx_list
            if isinstance(erLevel_name, Parser.ErLevelContext)
        ]
        er_level = None if len(er_level) == 0 else er_level[0]

        token_info = extract_token_info(ctx)

        return HRule(name=rule_name, rule=rule_node, erCode=er_code, erLevel=er_level, **token_info)

    def visitCodeItemRelation(self, ctx: Parser.CodeItemRelationContext):
        """
        codeItemRelation: ( WHEN expr THEN )? codeItemRef codeItemRelationClause (codeItemRelationClause)* ;
                        ( WHEN exprComponent THEN )? codetemRef=valueDomainValue comparisonOperand? codeItemRelationClause (codeItemRelationClause)*

        """  # noqa E501

        ctx_list = list(ctx.getChildren())

        when = None

        if isinstance(ctx_list[0], TerminalNodeImpl):
            when = ctx_list[0].getSymbol().text
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
        items = [
            item for item in ctx_list if isinstance(item, Parser.CodeItemRelationClauseContext)
        ]
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

    def visitCodeItemRelationClause(self, ctx: Parser.CodeItemRelationClauseContext):
        """
        (opAdd=( PLUS | MINUS  ))? rightCodeItem=valueDomainValue ( QLPAREN  rightCondition=exprComponent  QRPAREN )?
        """  # noqa E501
        ctx_list = list(ctx.getChildren())

        expr = [expr for expr in ctx_list if isinstance(expr, Parser.ExprContext)]
        if len(expr) != 0:
            # AST_ASTCONSTRUCTOR.8
            raise NotImplementedError

        right_condition = [
            ExprComp().visitExprComponent(right_condition)
            for right_condition in ctx_list
            if isinstance(right_condition, Parser.ComparisonExprCompContext)
        ]

        if isinstance(ctx_list[0], TerminalNodeImpl):
            op = ctx_list[0].getSymbol().text
            value = Terminals().visitValueDomainValue(ctx_list[1])

            code_item = DefIdentifier(
                value=value, kind="CodeItemID", **extract_token_info(ctx_list[1])
            )
            if right_condition:
                code_item._right_condition = right_condition[0]

            return HRBinOp(
                left=None, op=op, right=code_item, **extract_token_info(ctx_list[0].getSymbol())
            )
        else:
            value = Terminals().visitValueDomainValue(ctx_list[0])
            code_item = DefIdentifier(
                value=value, kind="CodeItemID", **extract_token_info(ctx_list[0])
            )
            if right_condition:
                code_item._right_condition = right_condition[0]

            return code_item
