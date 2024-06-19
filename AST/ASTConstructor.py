"""
AST.ASTConstructor.py
=====================

Description
-----------
Node Creator.
"""

from antlr4.tree.Tree import TerminalNodeImpl

from AST import (
    Start,
    Assignment,
    PersistentAssignment,
    Operator,
    Argument, DPRuleset, HRBinOp, DPRule, HRuleset, DefIdentifier, HRule, HRUnOp,
)
from AST.ASTConstructorModules.handlers.expr import expr_handler
from AST.ASTConstructorModules.handlers.expr_comp import expr_comp_handler
from AST.ASTConstructorModules.handlers.terminals import terminal_handler
from AST.VtlVisitor import VtlVisitor
from Grammar.parser import Parser


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
        statements = [statement for statement in ctx_list if isinstance(statement, Parser.StatementContext)]
        if len(statements) != 0:
            for statement in statements:
                statements_nodes.append(self.visitStatement(statement))

        start_node = Start(statements_nodes)

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

        left_node = terminal_handler.visitVarID(ctx_list[0])
        op_node = ctx_list[1].getSymbol().text

        right_node = expr_handler.visitExpr(ctx_list[2])

        assignment_node = Assignment(left_node, op_node, right_node)
        return assignment_node

    #     | varID PUT_SYMBOL expr          # persistAssignment
    def visitPersistAssignment(self, ctx: Parser.PersistAssignmentContext):
        """
        persistentAssignment: varID PUT_SYMBOL expr;
        """
        ctx_list = list(ctx.getChildren())

        left_node = terminal_handler.visitVarID(ctx_list[0])
        op_node = ctx_list[1].getSymbol().text

        right_node = expr_handler.visitExpr(ctx_list[2])

        persistent_assignment_node = PersistentAssignment(left_node, op_node, right_node)
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
            (RETURNS outputParameterType)? IS (expr) END OPERATOR        # defOperator        """
        ctx_list = list(ctx.getChildren())

        operator = terminal_handler.visitOperatorID(ctx_list[2])
        parameters = [self.visitParameterItem(parameter) for parameter in ctx_list if
                      isinstance(parameter, Parser.ParameterItemContext)]
        return_ = [
            terminal_handler.visitOutputParameterType(datatype)
            for datatype in ctx_list
            if isinstance(datatype, Parser.OutputParameterTypeContext)
        ]
        # Here should be modify if we want to include more than one expr per function.
        expr = [expr_handler.visitExpr(expr) for expr in ctx_list if isinstance(expr, Parser.ExprContext)][0]

        if len(return_) == 0:
            return_node = None
        else:
            return_node = return_[0]

        return Operator(operator=operator, parameters=parameters, outputType=return_node, expresion=expr)

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

        ruleset_name = terminal_handler.visitRulesetID(ctx_list[3])
        ruleset_elements = self.visitRulesetSignature(ctx_list[5])
        ruleset_rules = self.visitRuleClauseDatapoint(ctx_list[8])

        return DPRuleset(name=ruleset_name, element=ruleset_elements, rules=ruleset_rules)

    def visitRulesetSignature(self, ctx: Parser.RulesetSignatureContext):
        """
        rulesetSignature: (VALUE_DOMAIN|VARIABLE) varSignature (',' varSignature)* ;
        """
        ctx_list = list(ctx.getChildren())

        value_domains = [value_domain for value_domain in ctx_list if isinstance(value_domain, TerminalNodeImpl) and
                         value_domain.getSymbol().type == Parser.VALUE_DOMAIN]
        if len(value_domains) != 0:
            kind = 'ValuedomainID'

        variables = [variable for variable in ctx_list if isinstance(variable, TerminalNodeImpl) and
                     variable.getSymbol().type == Parser.VARIABLE]
        if len(variables) != 0:
            kind = 'ComponentID'

        component_nodes = [terminal_handler.visitSignature(component, kind) for component in ctx_list if
                           isinstance(component, Parser.SignatureContext)]

        return component_nodes

    def visitRuleClauseDatapoint(self, ctx: Parser.RuleClauseDatapointContext):
        """
        ruleClauseDatapoint: ruleItemDatapoint (';' ruleItemDatapoint)* ;
        """
        ctx_list = list(ctx.getChildren())

        ruleset_rules = [self.visitRuleItemDatapoint(ruleId) for ruleId in ctx_list if
                         isinstance(ruleId, Parser.RuleItemDatapointContext)]
        return ruleset_rules

    def visitRuleItemDatapoint(self, ctx: Parser.RuleItemDatapointContext):
        """
        ruleItemDatapoint: (IDENTIFIER ':')? ( WHEN expr THEN )? expr (erCode)? (erLevel)? ;
        """
        ctx_list = list(ctx.getChildren())

        when = [whens for whens in ctx_list if
                isinstance(whens, TerminalNodeImpl) and whens.getSymbol().type == Parser.WHEN]
        rule_name = [rule_name.getSymbol().text
                     if isinstance(rule_name, TerminalNodeImpl) and rule_name.getSymbol().type == Parser.IDENTIFIER
                     else None
                     for rule_name in ctx_list][0]
        expr_node = [expr_comp_handler.visitExprComponent(rule_node) for rule_node in ctx_list if
                     isinstance(rule_node, Parser.ExprComponentContext)]

        if len(when) != 0:
            rule_node = HRBinOp(left=expr_node[0], op=when[0].getSymbol().text, right=expr_node[1])

        else:
            rule_node = expr_node[0]

        er_code = [terminal_handler.visitErCode(erCode_name) for erCode_name in ctx_list if
                   isinstance(erCode_name, Parser.ErCodeContext)]
        if len(er_code) == 0:
            er_code = None
        else:
            er_code = er_code[0]
        er_level = [terminal_handler.visitErLevel(erLevel_name) for erLevel_name in ctx_list if
                    isinstance(erLevel_name, Parser.ErLevelContext)]
        if len(er_level) == 0:
            er_level = None
        else:
            er_level = er_level[0]

        return DPRule(name=rule_name, rule=rule_node, erCode=er_code, erLevel=er_level)

    def visitParameterItem(self, ctx: Parser.ParameterItemContext):
        """
        parameterItem: varID dataType (DEFAULT constant)? ;
        """
        ctx_list = list(ctx.getChildren())

        argument_name = [terminal_handler.visitVarID(element) for element in ctx_list
                         if isinstance(element, Parser.VarIDContext)][0]
        argument_type = [terminal_handler.visitInputParameterType(element) for element in ctx_list
                         if isinstance(element, Parser.InputParameterTypeContext)][0]
        argument_default = [terminal_handler.visitScalarItem(element) for element in ctx_list
                            if isinstance(element, Parser.ScalarItemContext)]

        if len(argument_default) == 0:
            argument_default = None
        else:
            argument_default = argument_default[0]

        argument_type.name = argument_name.value
        return Argument(name=argument_name.value, type=argument_type, default=argument_default)

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

        ruleset_name = terminal_handler.visitRulesetID(ctx_list[3])
        ruleset_elements = self.visitHierRuleSignature(ctx_list[5])
        ruleset_rules = self.visitRuleClauseHierarchical(ctx_list[8])
        # Keep k,v for the hierarchical rulesets
        de_ruleset_elements[ruleset_name] = ruleset_elements

        return HRuleset(name=ruleset_name, element=ruleset_elements, rules=ruleset_rules)

    # TODO Add support for value Domains.
    def visitHierRuleSignature(self, ctx: Parser.HierRuleSignatureContext):
        """
        hierRuleSignature: (VALUE_DOMAIN|VARIABLE) valueDomainSignature? RULE IDENTIFIER ;
        """
        ctx_list = list(ctx.getChildren())

        value_domain = [valueDomain for valueDomain in ctx_list
                        if isinstance(valueDomain, TerminalNodeImpl) and
                        valueDomain.getSymbol().type == Parser.VALUE_DOMAIN]
        if len(value_domain) != 0:
            kind = 'ValuedomainID'
        else:
            kind = 'DatasetID'

        conditions = [
            self.visitValueDomainSignature(vtlsig) for vtlsig in ctx_list if
            isinstance(vtlsig, Parser.ValueDomainSignatureContext)]

        dataset = [identifier for identifier in ctx_list if
                   isinstance(identifier, TerminalNodeImpl) and identifier.getSymbol().type == Parser.IDENTIFIER][0]

        if conditions:
            identifiers_list = [
                DefIdentifier(value=elto.alias if getattr(elto, "alias", None) else elto.value, kind=kind) for elto in conditions[0]]
            identifiers_list.append(DefIdentifier(value=dataset.getSymbol().text, kind=kind))
            return identifiers_list
        else:
            return DefIdentifier(value=dataset.getSymbol().text, kind=kind)

    # TODO Support for valueDomainSignature.
    def visitValueDomainSignature(self, ctx: Parser.ValueDomainSignatureContext):
        """
        valueDomainSignature: CONDITION IDENTIFIER (AS IDENTIFIER)? (',' IDENTIFIER (AS IDENTIFIER)?)* ;
        """
        # AST_ASTCONSTRUCTOR.7
        ctx_list = list(ctx.getChildren())
        component_nodes = [terminal_handler.visitSignature(component) for component in ctx_list if
                           isinstance(component, Parser.SignatureContext)]
        return component_nodes

    def visitRuleClauseHierarchical(self, ctx: Parser.RuleClauseHierarchicalContext):
        """
        ruleClauseHierarchical: ruleItemHierarchical (';' ruleItemHierarchical)* ;
        """
        ctx_list = list(ctx.getChildren())

        rules_nodes = [self.visitRuleItemHierarchical(rule) for rule in ctx_list if
                       isinstance(rule, Parser.RuleItemHierarchicalContext)]
        return rules_nodes

    def visitRuleItemHierarchical(self, ctx: Parser.RuleItemHierarchicalContext):
        """
        ruleItemHierarchical: (ruleName=IDENTIFIER  COLON )? codeItemRelation (erCode)? (erLevel)? ;
        """
        ctx_list = list(ctx.getChildren())

        rule_name = [rule_name.getSymbol().text
                     if isinstance(rule_name, TerminalNodeImpl) and rule_name.getSymbol().type == Parser.IDENTIFIER
                     else None
                     for rule_name in ctx_list][0]
        rule_node = [self.visitCodeItemRelation(rule_node) for rule_node in ctx_list if
                     isinstance(rule_node, Parser.CodeItemRelationContext)][0]

        er_code = [terminal_handler.visitErCode(erCode_name) for erCode_name in ctx_list if
                   isinstance(erCode_name, Parser.ErCodeContext)]
        if len(er_code) == 0:
            er_code = None
        else:
            er_code = er_code[0]
        er_level = [terminal_handler.visitErLevel(erLevel_name) for erLevel_name in ctx_list if
                    isinstance(erLevel_name, Parser.ErLevelContext)]
        if len(er_level) == 0:
            er_level = None
        else:
            er_level = er_level[0]

        return HRule(name=rule_name, rule=rule_node, erCode=er_code, erLevel=er_level)

    def visitCodeItemRelation(self, ctx: Parser.CodeItemRelationContext):
        """
        codeItemRelation: ( WHEN expr THEN )? codeItemRef codeItemRelationClause (codeItemRelationClause)* ;
                        ( WHEN exprComponent THEN )? codetemRef=valueDomainValue comparisonOperand? codeItemRelationClause (codeItemRelationClause)*

        """

        ctx_list = list(ctx.getChildren())

        when = None

        if isinstance(ctx_list[0], TerminalNodeImpl):
            when = ctx_list[0].getSymbol().text
            vd_value = terminal_handler.visitValueDomainValue(ctx_list[3])
            op = terminal_handler.visitComparisonOperand(ctx_list[4])
        else:
            vd_value = terminal_handler.visitValueDomainValue(ctx_list[0])
            op = terminal_handler.visitComparisonOperand(ctx_list[1])

        rule_node = HRBinOp(left=DefIdentifier(value=vd_value, kind='CodeItemID'), op=op, right=None)
        items = [item for item in ctx_list if isinstance(item, Parser.CodeItemRelationClauseContext)]

        # Means that no concatenations of operations is needed for that rule.
        if len(items) == 1:
            cir_node = self.visitCodeItemRelationClause(items[0])
            if isinstance(cir_node, HRBinOp):
                rule_node.right = HRUnOp(op=cir_node.op, operand=cir_node.right)

            elif isinstance(cir_node, DefIdentifier):
                rule_node.right = cir_node

        # Concatenate all the the elements except the last one.
        else:
            previous_node = self.visitCodeItemRelationClause(items[0])
            if isinstance(previous_node, HRBinOp):
                previous_node = HRUnOp(op=previous_node.op, operand=previous_node.right)

            for item in items[1:]:
                item_node = self.visitCodeItemRelationClause(item)
                item_node.left = previous_node

                previous_node = item_node

            rule_node.right = previous_node

        if when is not None:
            expr_node = expr_comp_handler.visitExprComponent(ctx_list[1])
            rule_node = HRBinOp(left=expr_node, op=when, right=rule_node)

        return rule_node

    def visitCodeItemRelationClause(self, ctx: Parser.CodeItemRelationClauseContext):
        """
        (opAdd=( PLUS | MINUS  ))? rightCodeItem=valueDomainValue ( QLPAREN  rightCondition=exprComponent  QRPAREN )?
        """
        ctx_list = list(ctx.getChildren())

        expr = [expr for expr in ctx_list if isinstance(expr, Parser.ExprContext)]
        if len(expr) != 0:
            # AST_ASTCONSTRUCTOR.8
            raise NotImplementedError

        right_condition = [
            expr_comp_handler.visitExprComponent(right_condition) for right_condition in ctx_list if isinstance(right_condition, Parser.ComparisonExprCompContext)
        ]

        if isinstance(ctx_list[0], TerminalNodeImpl):
            op = ctx_list[0].getSymbol().text
            value = terminal_handler.visitValueDomainValue(ctx_list[1])

            code_item = DefIdentifier(value=value, kind='CodeItemID')
            if right_condition:
                setattr(code_item, "_right_condition", right_condition[0])

            return HRBinOp(left=None, op=op,
                           right=code_item)
        else:
            value = terminal_handler.visitValueDomainValue(ctx_list[0])
            code_item = DefIdentifier(value=value, kind='CodeItemID')
            if right_condition:
                setattr(code_item, "_right_condition", right_condition[0])

            return code_item
