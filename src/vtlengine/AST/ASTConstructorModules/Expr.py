import re
from copy import copy
from typing import Any

from antlr4.tree.Tree import TerminalNodeImpl

from vtlengine.AST import (
    ID,
    Aggregation,
    Analytic,
    Assignment,
    BinOp,
    Case,
    CaseObj,
    Constant,
    EvalOp,
    Identifier,
    If,
    JoinOp,
    MulOp,
    ParamConstant,
    ParamOp,
    ParFunction,
    RegularAggregation,
    RenameNode,
    TimeAggregation,
    UDOCall,
    UnaryOp,
    Validation,
    VarID,
    Windowing,
)
from vtlengine.AST.ASTConstructorModules import extract_token_info
from vtlengine.AST.ASTConstructorModules.ExprComponents import ExprComp
from vtlengine.AST.ASTConstructorModules.Terminals import Terminals
from vtlengine.AST.ASTDataExchange import de_ruleset_elements
from vtlengine.AST.Grammar.parser import Parser
from vtlengine.AST.Grammar.tokens import DATASET_PRIORITY
from vtlengine.AST.VtlVisitor import VtlVisitor
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Role


class Expr(VtlVisitor):
    """______________________________________________________________________________________


                                Expr Definition.

    _______________________________________________________________________________________
    """

    def visitExpr(self, ctx: Parser.ExprContext):
        """
        expr:
            LPAREN expr RPAREN											            # parenthesisExpr
               | functions                                                             # functionsExpression
               | dataset=expr  QLPAREN  clause=datasetClause  QRPAREN                  # clauseExpr
               | expr MEMBERSHIP simpleComponentId                                     # membershipExpr
               | op=(PLUS|MINUS|NOT) right=expr                                        # unaryExpr
               | left=expr op=(MUL|DIV) right=expr                                     # arithmeticExpr
               | left=expr op=(PLUS|MINUS|CONCAT) right=expr                           # arithmeticExprOrConcat
               | left=expr op=comparisonOperand  right=expr                            # comparisonExpr
               | left=expr op=(IN|NOT_IN)(lists|valueDomainID)                         # inNotInExpr
               | left=expr op=AND right=expr                                           # booleanExpr
               | left=expr op=(OR|XOR) right=expr							            # booleanExpr
               | IF  conditionalExpr=expr  THEN thenExpr=expr ELSE elseExpr=expr       # ifExpr
               | CASE WHEN expr THEN expr ELSE expr END                             # caseExpr
               | constant														        # constantExpr
               | varID															        # varIdExpr
        ;
        """  # noqa E501
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        if isinstance(ctx, Parser.ParenthesisExprContext):
            return self.visitParenthesisExpr(ctx)

        elif isinstance(ctx, Parser.MembershipExprContext):
            return self.visitMembershipExpr(ctx)

        # dataset=expr  QLPAREN  clause=datasetClause  QRPAREN                  # clauseExpr
        elif isinstance(ctx, Parser.ClauseExprContext):
            return self.visitClauseExpr(ctx)

        # functions
        elif isinstance(ctx, Parser.FunctionsExpressionContext):
            return self.visitFunctionsExpression(c)

        # op=(PLUS|MINUS|NOT) right=expr # unary expression
        elif isinstance(ctx, Parser.UnaryExprContext):
            return self.visitUnaryExpr(ctx)

        # | left=expr op=(MUL|DIV) right=expr               # arithmeticExpr
        elif isinstance(ctx, Parser.ArithmeticExprContext):
            return self.visitArithmeticExpr(ctx)

        # | left=expr op=(PLUS|MINUS|CONCAT) right=expr     # arithmeticExprOrConcat
        elif isinstance(ctx, Parser.ArithmeticExprOrConcatContext):
            return self.visitArithmeticExprOrConcat(ctx)

        # | left=expr op=comparisonOperand  right=expr      # comparisonExpr
        elif isinstance(ctx, Parser.ComparisonExprContext):
            return self.visitComparisonExpr(ctx)

        # | left=expr op=(IN|NOT_IN)(lists|valueDomainID)   # inNotInExpr
        elif isinstance(ctx, Parser.InNotInExprContext):
            return self.visitInNotInExpr(ctx)

        # | left=expr op=AND right=expr                                           # booleanExpr
        # | left=expr op=(OR|XOR) right=expr
        elif isinstance(ctx, Parser.BooleanExprContext):
            return self.visitBooleanExpr(ctx)

        # IF  conditionalExpr=expr  THEN thenExpr=expr ELSE elseExpr=expr       # ifExpr
        elif isinstance(c, TerminalNodeImpl) and (c.getSymbol().type == Parser.IF):
            condition_node = self.visitExpr(ctx_list[1])
            then_op_node = self.visitExpr(ctx_list[3])
            else_op_node = self.visitExpr(ctx_list[5])

            if_node = If(
                condition=condition_node,
                thenOp=then_op_node,
                elseOp=else_op_node,
                **extract_token_info(ctx),
            )

            return if_node

        # CASE WHEN expr THEN expr ELSE expr END                             # caseExpr
        elif isinstance(c, TerminalNodeImpl) and (c.getSymbol().type == Parser.CASE):
            if len(ctx_list) % 4 != 3:
                raise ValueError("Syntax error.")

            else_node = self.visitExpr(ctx_list[-1])
            ctx_list = ctx_list[1:-2]
            cases = []

            for i in range(0, len(ctx_list), 4):
                condition = self.visitExpr(ctx_list[i + 1])
                thenOp = self.visitExpr(ctx_list[i + 3])
                case_obj = CaseObj(
                    condition=condition, thenOp=thenOp, **extract_token_info(ctx_list[i + 1])
                )
                cases.append(case_obj)

            case_node = Case(cases=cases, elseOp=else_node, **extract_token_info(ctx))

            return case_node

        # constant
        elif isinstance(ctx, Parser.ConstantExprContext):
            return Terminals().visitConstant(c)

        # varID
        elif isinstance(ctx, Parser.VarIdExprContext):
            return Terminals().visitVarIdExpr(c)

        else:
            # AST_ASTCONSTRUCTOR.3
            raise NotImplementedError

    def bin_op_creator(self, ctx: Parser.ExprContext):
        ctx_list = list(ctx.getChildren())
        left_node = self.visitExpr(ctx_list[0])
        if isinstance(ctx_list[1], Parser.ComparisonOperandContext):
            op = list(ctx_list[1].getChildren())[0].getSymbol().text
        else:
            op = ctx_list[1].getSymbol().text
        right_node = self.visitExpr(ctx_list[2])
        token_info = extract_token_info(ctx)
        bin_op_node = BinOp(left=left_node, op=op, right=right_node, **token_info)

        return bin_op_node

    def visitArithmeticExpr(self, ctx: Parser.ArithmeticExprContext):
        return self.bin_op_creator(ctx)

    def visitArithmeticExprOrConcat(self, ctx: Parser.ArithmeticExprOrConcatContext):
        return self.bin_op_creator(ctx)

    def visitComparisonExpr(self, ctx: Parser.ComparisonExprContext):
        return self.bin_op_creator(ctx)

    def visitInNotInExpr(self, ctx: Parser.InNotInExprContext):
        ctx_list = list(ctx.getChildren())
        left_node = self.visitExpr(ctx_list[0])
        op = ctx_list[1].symbol.text

        if isinstance(ctx_list[2], Parser.ListsContext):
            right_node = Terminals().visitLists(ctx_list[2])
        elif isinstance(ctx_list[2], Parser.ValueDomainIDContext):
            right_node = Terminals().visitValueDomainID(ctx_list[2])
        else:
            raise NotImplementedError
        bin_op_node = BinOp(left=left_node, op=op, right=right_node, **extract_token_info(ctx))

        return bin_op_node

    def visitBooleanExpr(self, ctx: Parser.BooleanExprContext):
        return self.bin_op_creator(ctx)

    def visitParenthesisExpr(self, ctx: Parser.ParenthesisExprContext):
        operand = self.visitExpr(list(ctx.getChildren())[1])
        return ParFunction(operand=operand, **extract_token_info(ctx))

    def visitUnaryExpr(self, ctx: Parser.UnaryExprContext):
        c_list = list(ctx.getChildren())
        op = c_list[0].getSymbol().text
        right = self.visitExpr(c_list[1])

        return UnaryOp(op=op, operand=right, **extract_token_info(ctx))

    def visitMembershipExpr(self, ctx: Parser.MembershipExprContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]
        membership = [
            componentID
            for componentID in ctx_list
            if isinstance(componentID, Parser.SimpleComponentIdContext)
        ]

        previous_node = self.visitExpr(c)

        # It is only possible to put a membership at the end so go the last one.
        if len(membership) != 0:
            previous_node = BinOp(
                left=previous_node,
                op="#",
                right=Terminals().visitSimpleComponentId(membership[0]),
                **extract_token_info(ctx),
            )

        return previous_node

    def visitClauseExpr(self, ctx: Parser.ClauseExprContext):
        ctx_list = list(ctx.getChildren())

        dataset = self.visitExpr(ctx_list[0])

        dataset_clause = self.visitDatasetClause(ctx_list[2])

        dataset_clause.dataset = dataset

        return dataset_clause

    """______________________________________________________________________________________


                                    Functions Definition.

        _______________________________________________________________________________________"""

    def visitFunctionsExpression(self, ctx: Parser.FunctionsExpressionContext):
        """
        functions:
            joinOperators                       # joinFunctions
            | genericOperators                  # genericFunctions
            | stringOperators                   # stringFunctions
            | numericOperators                  # numericFunctions
            | comparisonOperators               # comparisonFunctions
            | timeOperators                     # timeFunctions
            | setOperators                      # setFunctions
            | hierarchyOperators                # hierarchyFunctions
            | validationOperators               # validationFunctions
            | conditionalOperators              # conditionalFunctions
            | aggrOperatorsGrouping             # aggregateFunctions
            | anFunction                        # analyticFunctions
        ;
        """
        c = ctx.children[0]

        if isinstance(ctx, Parser.JoinFunctionsContext):
            return self.visitJoinFunctions(c)

        elif isinstance(ctx, Parser.GenericFunctionsContext):
            return self.visitGenericFunctions(c)

        elif isinstance(ctx, Parser.StringFunctionsContext):
            return self.visitStringFunctions(c)

        elif isinstance(ctx, Parser.NumericFunctionsContext):
            return self.visitNumericFunctions(c)

        elif isinstance(ctx, Parser.ComparisonFunctionsContext):
            return self.visitComparisonFunctions(c)

        elif isinstance(ctx, Parser.TimeFunctionsContext):
            return self.visitTimeFunctions(c)

        elif isinstance(ctx, Parser.SetFunctionsContext):
            return self.visitSetFunctions(c)

        elif isinstance(ctx, Parser.HierarchyFunctionsContext):
            return self.visitHierarchyFunctions(c)

        elif isinstance(ctx, Parser.ValidationFunctionsContext):
            return self.visitValidationFunctions(c)

        elif isinstance(ctx, Parser.ConditionalFunctionsContext):
            return self.visitConditionalFunctions(c)

        elif isinstance(ctx, Parser.AggregateFunctionsContext):
            return self.visitAggregateFunctions(c)

        elif isinstance(ctx, Parser.AnalyticFunctionsContext):
            return self.visitAnalyticFunctions(c)

        else:
            raise NotImplementedError

    """
                        -----------------------------------
                                Join Functions
                        -----------------------------------
    """

    def visitJoinFunctions(self, ctx: Parser.JoinFunctionsContext):
        ctx_list = list(ctx.getChildren())

        using_node = None

        op_node = ctx_list[0].getSymbol().text

        if op_node in ["inner_join", "left_join"]:
            clause_node, using_node = self.visitJoinClause(ctx_list[2])
        else:
            clause_node = self.visitJoinClauseWithoutUsing(ctx_list[2])

        body_node = self.visitJoinBody(ctx_list[3])

        token_info = extract_token_info(ctx)

        if len(body_node) != 0:
            previous_node = JoinOp(op=op_node, clauses=clause_node, using=using_node, **token_info)
            regular_aggregation = None
            for body in body_node:
                regular_aggregation = body
                regular_aggregation.dataset = previous_node
                previous_node = regular_aggregation

            # set the last of the body clauses (ie dataclauses).
            previous_node.isLast = True

            return regular_aggregation

        else:
            join_node = JoinOp(op=op_node, clauses=clause_node, using=using_node, **token_info)
            join_node.isLast = True
            return join_node

    def visitJoinClauseItem(self, ctx: Parser.JoinClauseItemContext):
        ctx_list = list(ctx.getChildren())
        left_node = self.visitExpr(ctx_list[0])
        if len(ctx_list) == 1:
            return left_node

        token_info = extract_token_info(ctx)
        intop_node = ctx_list[1].getSymbol().text
        right_node = Identifier(
            value=Terminals().visitAlias(ctx_list[2]),
            kind="DatasetID",
            **extract_token_info(ctx_list[1].getSymbol()),
        )
        return BinOp(left=left_node, op=intop_node, right=right_node, **token_info)

    def visitJoinClause(self, ctx: Parser.JoinClauseContext):
        """
        JoinClauseItem (COMMA joinClauseItem)* (USING componentID (COMMA componentID)*)?
        """
        ctx_list = list(ctx.getChildren())

        clause_nodes = []
        component_nodes = []
        using = None

        items = [item for item in ctx_list if isinstance(item, Parser.JoinClauseItemContext)]
        components = [
            component for component in ctx_list if isinstance(component, Parser.ComponentIDContext)
        ]

        for item in items:
            clause_nodes.append(self.visitJoinClauseItem(item))

        if len(components) != 0:
            for component in components:
                component_nodes.append(Terminals().visitComponentID(component).value)
            using = component_nodes

        return clause_nodes, using

    def visitJoinClauseWithoutUsing(self, ctx: Parser.JoinClauseWithoutUsingContext):
        """
        joinClause: joinClauseItem (COMMA joinClauseItem)* (USING componentID (COMMA componentID)*)? ;
        """  # noqa E501
        ctx_list = list(ctx.getChildren())

        clause_nodes = []

        items = [item for item in ctx_list if isinstance(item, Parser.JoinClauseItemContext)]

        for item in items:
            clause_nodes.append(self.visitJoinClauseItem(item))

        return clause_nodes

    def visitJoinBody(self, ctx: Parser.JoinBodyContext):
        """
        joinBody: filterClause? (calcClause|joinApplyClause|aggrClause)? (keepOrDropClause)? renameClause?
        """  # noqa E501
        ctx_list = list(ctx.getChildren())

        body_nodes = []

        for c in ctx_list:
            if isinstance(c, Parser.FilterClauseContext):
                body_nodes.append(self.visitFilterClause(c))
            elif isinstance(c, Parser.CalcClauseContext):
                body_nodes.append(self.visitCalcClause(c))
            elif isinstance(c, Parser.JoinApplyClauseContext):
                body_nodes.append(self.visitJoinApplyClause(c))
            elif isinstance(c, Parser.AggrClauseContext):
                body_nodes.append(self.visitAggrClause(c))
            elif isinstance(c, Parser.KeepOrDropClauseContext):
                body_nodes.append(self.visitKeepOrDropClause(c))
            elif isinstance(c, Parser.RenameClauseContext):
                body_nodes.append(self.visitRenameClause(c))
            else:
                raise NotImplementedError

        return body_nodes

    # TODO Unary Op here?
    def visitJoinApplyClause(self, ctx: Parser.JoinApplyClauseContext):
        """
        joinApplyClause: APPLY expr ;
        """
        ctx_list = list(ctx.getChildren())
        op_node = ctx_list[0].getSymbol().text
        operand_nodes = [self.visitExpr(ctx_list[1])]

        return RegularAggregation(op=op_node, children=operand_nodes, **extract_token_info(ctx))

    """
                        -----------------------------------
                                Generic Functions
                        -----------------------------------
    """

    def visitGenericFunctions(self, ctx: Parser.GenericFunctionsContext):
        if isinstance(ctx, Parser.CallDatasetContext):
            return self.visitCallDataset(ctx)
        elif isinstance(ctx, Parser.EvalAtomContext):
            return self.visitEvalAtom(ctx)
        elif isinstance(ctx, Parser.CastExprDatasetContext):
            return self.visitCastExprDataset(ctx)
        else:
            raise NotImplementedError

    def visitCallDataset(self, ctx: Parser.CallDatasetContext):
        """
        callFunction: operatorID LPAREN (parameter (COMMA parameter)*)? RPAREN   ;
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        op = Terminals().visitOperatorID(c)
        param_nodes = [
            self.visitParameter(element)
            for element in ctx_list
            if isinstance(element, Parser.ParameterContext)
        ]

        return UDOCall(op=op, params=param_nodes, **extract_token_info(ctx))

    def visitEvalAtom(self, ctx: Parser.EvalAtomContext):
        """
        | EVAL LPAREN routineName LPAREN (varID|scalarItem)? (COMMA (varID|scalarItem))* RPAREN (LANGUAGE STRING_CONSTANT)? (RETURNS evalDatasetType)? RPAREN     # evalAtom
        """  # noqa E501
        ctx_list = list(ctx.getChildren())

        routine_name = Terminals().visitRoutineName(ctx_list[2])

        # Think of a way to maintain the order, for now its not necessary.
        var_ids_nodes = [
            Terminals().visitVarID(varID)
            for varID in ctx_list
            if isinstance(varID, Parser.VarIDContext)
        ]
        constant_nodes = [
            Terminals().visitScalarItem(scalar)
            for scalar in ctx_list
            if isinstance(scalar, Parser.ScalarItemContext)
        ]
        children_nodes = var_ids_nodes + constant_nodes

        # Reference manual says it is mandatory.
        language_name = [
            language
            for language in ctx_list
            if isinstance(language, TerminalNodeImpl)
            and language.getSymbol().type == Parser.STRING_CONSTANT
        ]
        if len(language_name) == 0:
            # AST_ASTCONSTRUCTOR.12
            raise SemanticError("1-4-2-1", option="language")
        # Reference manual says it is mandatory.
        output_node = [
            Terminals().visitEvalDatasetType(output)
            for output in ctx_list
            if isinstance(output, Parser.EvalDatasetTypeContext)
        ]
        if len(output_node) == 0:
            # AST_ASTCONSTRUCTOR.13
            raise SemanticError("1-4-2-1", option="output")

        return EvalOp(
            name=routine_name,
            operands=children_nodes,
            output=output_node[0],
            language=language_name[0].getSymbol().text,
            **extract_token_info(ctx),
        )

    def visitCastExprDataset(self, ctx: Parser.CastExprDatasetContext):
        """
        | CAST LPAREN expr COMMA (basicScalarType|valueDomainName) (COMMA STRING_CONSTANT)? RPAREN        # castExprDataset
        """  # noqa E501
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        token = c.getSymbol()

        op = token.text
        expr_node = [
            self.visitExpr(expr) for expr in ctx_list if isinstance(expr, Parser.ExprContext)
        ]
        basic_scalar_type = [
            Terminals().visitBasicScalarType(type_)
            for type_ in ctx_list
            if isinstance(type_, Parser.BasicScalarTypeContext)
        ]

        [
            Terminals().visitValueDomainName(valueD)
            for valueD in ctx_list
            if isinstance(valueD, Parser.ValueDomainNameContext)
        ]

        if len(ctx_list) > 6:
            param_node = [
                ParamConstant(
                    type_="PARAM_CAST",
                    value=str_.symbol.text.strip('"'),
                    **extract_token_info(str_.getSymbol()),
                )
                for str_ in ctx_list
                if isinstance(str_, TerminalNodeImpl)
                and str_.getSymbol().type == Parser.STRING_CONSTANT
            ]
        else:
            param_node = []

        if len(basic_scalar_type) == 1:
            children_nodes = expr_node + basic_scalar_type

            return ParamOp(
                op=op, children=children_nodes, params=param_node, **extract_token_info(ctx)
            )

        else:
            # AST_ASTCONSTRUCTOR.14
            raise NotImplementedError

    def visitParameter(self, ctx: Parser.ParameterContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        if isinstance(c, Parser.ExprContext):
            return self.visitExpr(c)
        elif isinstance(c, TerminalNodeImpl):
            return ID(
                type_="OPTIONAL", value=c.getSymbol().text, **extract_token_info(c.getSymbol())
            )
        else:
            raise NotImplementedError

    """
                        -----------------------------------
                                String Functions
                        -----------------------------------
    """

    def visitStringFunctions(self, ctx: Parser.StringFunctionsContext):
        if isinstance(ctx, Parser.UnaryStringFunctionContext):
            return self.visitUnaryStringFunction(ctx)
        elif isinstance(ctx, Parser.SubstrAtomContext):
            return self.visitSubstrAtom(ctx)
        elif isinstance(ctx, Parser.ReplaceAtomContext):
            return self.visitReplaceAtom(ctx)
        elif isinstance(ctx, Parser.InstrAtomContext):
            return self.visitInstrAtom(ctx)
        else:
            raise NotImplementedError

    def visitUnaryStringFunction(self, ctx: Parser.UnaryStringFunctionContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        token = c.getSymbol()
        op_node = token.text
        operand_node = self.visitExpr(ctx_list[2])
        return UnaryOp(op=op_node, operand=operand_node, **extract_token_info(ctx))

    def visitSubstrAtom(self, ctx: Parser.SubstrAtomContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        params_nodes = []
        children_nodes = []

        token = c.getSymbol()
        childrens = [expr for expr in ctx_list if isinstance(expr, Parser.ExprContext)]
        params = [param for param in ctx_list if isinstance(param, Parser.OptionalExprContext)]

        op_node = token.text
        for children in childrens:
            children_nodes.append(self.visitExpr(children))

        if len(params) != 0:
            for param in params:
                params_nodes.append(self.visitOptionalExpr(param))

        return ParamOp(
            op=op_node, children=children_nodes, params=params_nodes, **extract_token_info(ctx)
        )

    def visitReplaceAtom(self, ctx: Parser.ReplaceAtomContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        token = c.getSymbol()
        expressions = [
            self.visitExpr(expr) for expr in ctx_list if isinstance(expr, Parser.ExprContext)
        ]
        params = [
            self.visitOptionalExpr(param)
            for param in ctx_list
            if isinstance(param, Parser.OptionalExprContext)
        ]

        op_node = token.text

        children_nodes = [expressions[0]]
        params_nodes = [expressions[1]] + params

        return ParamOp(
            op=op_node, children=children_nodes, params=params_nodes, **extract_token_info(ctx)
        )

    def visitInstrAtom(self, ctx: Parser.InstrAtomContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        token = c.getSymbol()
        expressions = [
            self.visitExpr(expr) for expr in ctx_list if isinstance(expr, Parser.ExprContext)
        ]
        params = [
            self.visitOptionalExpr(param)
            for param in ctx_list
            if isinstance(param, Parser.OptionalExprContext)
        ]

        op_node = token.text

        children_nodes = [expressions[0]]
        params_nodes = [expressions[1]] + params

        return ParamOp(
            op=op_node, children=children_nodes, params=params_nodes, **extract_token_info(ctx)
        )

    """
                        -----------------------------------
                                Numeric Functions
                        -----------------------------------
    """

    def visitNumericFunctions(self, ctx: Parser.NumericFunctionsContext):
        if isinstance(ctx, Parser.UnaryNumericContext):
            return self.visitUnaryNumeric(ctx)
        elif isinstance(ctx, Parser.UnaryWithOptionalNumericContext):
            return self.visitUnaryWithOptionalNumeric(ctx)
        elif isinstance(ctx, Parser.BinaryNumericContext):
            return self.visitBinaryNumeric(ctx)
        else:
            raise NotImplementedError

    def visitUnaryNumeric(self, ctx: Parser.UnaryNumericContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        token = c.getSymbol()
        op_node = token.text
        operand_node = self.visitExpr(ctx_list[2])
        return UnaryOp(op=op_node, operand=operand_node, **extract_token_info(ctx))

    def visitUnaryWithOptionalNumeric(self, ctx: Parser.UnaryWithOptionalNumericContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        params_nodes = []
        children_nodes = []

        token = c.getSymbol()
        childrens = [expr for expr in ctx_list if isinstance(expr, Parser.ExprContext)]
        params = [param for param in ctx_list if isinstance(param, Parser.OptionalExprContext)]

        op_node = token.text
        for children in childrens:
            children_nodes.append(self.visitExpr(children))

        if len(params) != 0:
            for param in params:
                params_nodes.append(self.visitOptionalExpr(param))

        return ParamOp(
            op=op_node, children=children_nodes, params=params_nodes, **extract_token_info(ctx)
        )

    def visitBinaryNumeric(self, ctx: Parser.BinaryNumericContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        token = c.getSymbol()

        left_node = self.visitExpr(ctx_list[2])
        op_node = token.text
        right_node = self.visitExpr(ctx_list[4])
        return BinOp(left=left_node, op=op_node, right=right_node, **extract_token_info(ctx))

    """
                        -----------------------------------
                                Comparison Functions
                        -----------------------------------
    """

    def visitComparisonFunctions(self, ctx: Parser.ComparisonFunctionsContext):
        if isinstance(ctx, Parser.BetweenAtomContext):
            return self.visitBetweenAtom(ctx)
        elif isinstance(ctx, Parser.CharsetMatchAtomContext):
            return self.visitCharsetMatchAtom(ctx)
        elif isinstance(ctx, Parser.IsNullAtomContext):
            return self.visitIsNullAtom(ctx)
        elif isinstance(ctx, Parser.ExistInAtomContext):
            return self.visitExistInAtom(ctx)
        else:
            raise NotImplementedError

    def visitBetweenAtom(self, ctx: Parser.BetweenAtomContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        children_nodes = []

        token = c.getSymbol()
        childrens = [expr for expr in ctx_list if isinstance(expr, Parser.ExprContext)]

        op_node = token.text
        for children in childrens:
            children_nodes.append(self.visitExpr(children))

        return MulOp(op=op_node, children=children_nodes, **extract_token_info(ctx))

    def visitCharsetMatchAtom(self, ctx: Parser.CharsetMatchAtomContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]
        token = c.getSymbol()

        left_node = self.visitExpr(ctx_list[2])
        op_node = token.text
        right_node = self.visitExpr(ctx_list[4])
        return BinOp(left=left_node, op=op_node, right=right_node, **extract_token_info(ctx))

    def visitIsNullAtom(self, ctx: Parser.IsNullAtomContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]
        token = c.getSymbol()
        op_node = token.text
        operand_node = self.visitExpr(ctx_list[2])
        return UnaryOp(op=op_node, operand=operand_node, **extract_token_info(ctx))

    def visitExistInAtom(self, ctx: Parser.ExistInAtomContext):
        ctx_list = list(ctx.getChildren())
        token = ctx_list[0].getSymbol()
        op = token.text

        operand_nodes = [
            self.visitExpr(expr) for expr in ctx_list if isinstance(expr, Parser.ExprContext)
        ]
        retain_nodes = [
            Terminals().visitRetainType(retain)
            for retain in ctx_list
            if isinstance(retain, Parser.RetainTypeContext)
        ]

        return MulOp(op=op, children=operand_nodes + retain_nodes, **extract_token_info(ctx))

    """
                            -----------------------------------
                                    Time Functions
                            -----------------------------------
        """

    def visitTimeFunctions(self, ctx: Parser.TimeFunctionsContext):
        if isinstance(ctx, Parser.PeriodAtomContext):
            # return self.visitPeriodAtom(ctx)
            return self.visitTimeUnaryAtom(ctx)
        elif isinstance(ctx, Parser.FillTimeAtomContext):
            return self.visitFillTimeAtom(ctx)
        elif isinstance(ctx, Parser.FlowAtomContext):
            return self.visitFlowAtom(ctx)
        elif isinstance(ctx, Parser.TimeShiftAtomContext):
            return self.visitTimeShiftAtom(ctx)
        elif isinstance(ctx, Parser.TimeAggAtomContext):
            return self.visitTimeAggAtom(ctx)
        elif isinstance(ctx, Parser.CurrentDateAtomContext):
            return self.visitCurrentDateAtom(ctx)
        elif isinstance(ctx, Parser.DateDiffAtomContext):
            return self.visitTimeDiffAtom(ctx)
        elif isinstance(ctx, Parser.DateAddAtomContext):
            return self.visitTimeAddAtom(ctx)
        elif isinstance(
            ctx,
            (
                Parser.YearAtomContext,
                Parser.MonthAtomContext,
                Parser.DayOfMonthAtomContext,
                Parser.DayOfYearAtomContext,
                Parser.DayToYearAtomContext,
                Parser.DayToMonthAtomContext,
                Parser.YearTodayAtomContext,
                Parser.MonthTodayAtomContext,
            ),
        ):
            return self.visitTimeUnaryAtom(ctx)
        else:
            raise NotImplementedError

    def visitTimeUnaryAtom(self, ctx: Any):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        op = c.getSymbol().text
        operand_node = [
            self.visitExpr(operand)
            for operand in ctx_list
            if isinstance(operand, Parser.ExprContext)
        ]

        if len(operand_node) == 0:
            # AST_ASTCONSTRUCTOR.15
            raise NotImplementedError

        return UnaryOp(op=op, operand=operand_node[0], **extract_token_info(ctx))

    def visitTimeShiftAtom(self, ctx: Parser.TimeShiftAtomContext):
        """
        timeShiftExpr: TIMESHIFT '(' expr ',' INTEGER_CONSTANT ')' ;
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        op = c.getSymbol().text
        left_node = self.visitExpr(ctx_list[2])
        right_node = Constant(
            type_="INTEGER_CONSTANT",
            value=Terminals().visitSignedInteger(ctx_list[4]),
            **extract_token_info(ctx_list[4]),
        )

        return BinOp(left=left_node, op=op, right=right_node, **extract_token_info(ctx))

    def visitFillTimeAtom(self, ctx: Parser.FillTimeAtomContext):
        """
        timeSeriesExpr: FILL_TIME_SERIES '(' expr (',' (SINGLE|ALL))? ')' ;
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        op = c.getSymbol().text
        children_node = [self.visitExpr(ctx_list[2])]

        if len(ctx_list) > 4:
            param_constant_node = [
                ParamConstant(
                    type_="PARAM_TIMESERIES",
                    value=ctx_list[4].getSymbol().text,
                    **extract_token_info(ctx_list[4].getSymbol()),
                )
            ]
        else:
            param_constant_node = []

        return ParamOp(
            op=op, children=children_node, params=param_constant_node, **extract_token_info(ctx)
        )

    def visitTimeAggAtom(self, ctx: Parser.TimeAggAtomContext):
        """
        TIME_AGG LPAREN periodIndTo=STRING_CONSTANT (COMMA periodIndFrom=(STRING_CONSTANT| OPTIONAL ))? (COMMA op=optionalExpr)? (COMMA (FIRST|LAST))? RPAREN     # timeAggAtom
        """  # noqa E501
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        op = c.getSymbol().text
        period_to = str(ctx.periodIndTo.text)[1:-1]
        period_from = None

        if ctx.periodIndFrom is not None and ctx.periodIndFrom.type != Parser.OPTIONAL:
            period_from = str(ctx.periodIndFrom.text)[1:-1]

        conf = [
            str_.getSymbol().text
            for str_ in ctx_list
            if isinstance(str_, TerminalNodeImpl)
            and str_.getSymbol().type in [Parser.FIRST, Parser.LAST]
        ]

        conf = None if len(conf) == 0 else conf[0]

        if ctx.op is not None:
            operand_node = self.visitOptionalExpr(ctx.op)
            if isinstance(operand_node, ID):
                operand_node = None
            elif isinstance(operand_node, Identifier):
                operand_node = VarID(value=operand_node.value, **extract_token_info(ctx))
        else:
            operand_node = None

        if operand_node is None:
            # AST_ASTCONSTRUCTOR.17
            raise Exception("Optional as expression node is not allowed in Time Aggregation")
        return TimeAggregation(
            op=op,
            operand=operand_node,
            period_to=period_to,
            period_from=period_from,
            conf=conf,
            **extract_token_info(ctx),
        )

    def visitFlowAtom(self, ctx: Parser.FlowAtomContext):
        ctx_list = list(ctx.getChildren())

        op_node = ctx_list[0].getSymbol().text
        operand_node = self.visitExpr(ctx_list[2])
        return UnaryOp(op=op_node, operand=operand_node, **extract_token_info(ctx))

    def visitCurrentDateAtom(self, ctx: Parser.CurrentDateAtomContext):
        c = list(ctx.getChildren())[0]
        return MulOp(op=c.getSymbol().text, children=[], **extract_token_info(ctx))

    def visitTimeDiffAtom(self, ctx: Parser.TimeShiftAtomContext):
        """ """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        op = c.getSymbol().text
        left_node = self.visitExpr(ctx_list[2])
        right_node = self.visitExpr(ctx_list[4])

        return BinOp(left=left_node, op=op, right=right_node, **extract_token_info(ctx))

    def visitTimeAddAtom(self, ctx: Parser.TimeShiftAtomContext):
        """ """

        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        op = c.getSymbol().text
        children_node = [self.visitExpr(ctx_list[2])]

        param_constant_node = []

        if len(ctx_list) > 4:
            param_constant_node = [self.visitExpr(ctx_list[4])]
            if len(ctx_list) > 6:
                param_constant_node.append(self.visitExpr(ctx_list[6]))

        return ParamOp(
            op=op, children=children_node, params=param_constant_node, **extract_token_info(ctx)
        )

    """
                            -----------------------------------
                                    Conditional Functions
                            -----------------------------------
    """

    def visitConditionalFunctions(self, ctx: Parser.ConditionalFunctionsContext):
        if isinstance(ctx, Parser.NvlAtomContext):
            return self.visitNvlAtom(ctx)
        else:
            raise NotImplementedError

    def visitNvlAtom(self, ctx: Parser.NvlAtomContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        token = c.getSymbol()

        left_node = self.visitExpr(ctx_list[2])
        op_node = token.text
        right_node = self.visitExpr(ctx_list[4])
        return BinOp(left=left_node, op=op_node, right=right_node, **extract_token_info(ctx))

    """
                            -----------------------------------
                                    Set Functions
                            -----------------------------------
    """

    def visitSetFunctions(self, ctx: Parser.SetFunctionsContext):
        """
        setExpr:     UNION LPAREN left=expr (COMMA expr)+ RPAREN                             # unionAtom
                    | INTERSECT LPAREN left=expr (COMMA expr)+ RPAREN                       # intersectAtom
                    | op=(SETDIFF|SYMDIFF) LPAREN left=expr COMMA right=expr RPAREN         # setOrSYmDiffAtom
        """  # noqa E501
        if isinstance(ctx, Parser.UnionAtomContext):
            return self.visitUnionAtom(ctx)
        elif isinstance(ctx, Parser.IntersectAtomContext):
            return self.visitIntersectAtom(ctx)
        elif isinstance(ctx, Parser.SetOrSYmDiffAtomContext):
            return self.visitSetOrSYmDiffAtom(ctx)
        else:
            raise NotImplementedError

    def visitUnionAtom(self, ctx: Parser.UnionAtomContext):
        ctx_list = list(ctx.getChildren())
        exprs_nodes = [
            self.visitExpr(expr) for expr in ctx_list if isinstance(expr, Parser.ExprContext)
        ]

        return MulOp(
            op=ctx_list[0].getSymbol().text, children=exprs_nodes, **extract_token_info(ctx)
        )

    def visitIntersectAtom(self, ctx: Parser.IntersectAtomContext):
        ctx_list = list(ctx.getChildren())
        exprs_nodes = [
            self.visitExpr(expr) for expr in ctx_list if isinstance(expr, Parser.ExprContext)
        ]

        return MulOp(
            op=ctx_list[0].getSymbol().text, children=exprs_nodes, **extract_token_info(ctx)
        )

    def visitSetOrSYmDiffAtom(self, ctx: Parser.SetOrSYmDiffAtomContext):
        ctx_list = list(ctx.getChildren())
        exprs_nodes = [
            self.visitExpr(expr) for expr in ctx_list if isinstance(expr, Parser.ExprContext)
        ]

        return MulOp(
            op=ctx_list[0].getSymbol().text, children=exprs_nodes, **extract_token_info(ctx)
        )

    """
                            -----------------------------------
                                    Hierarchy Functions
                            -----------------------------------
    """

    def visitHierarchyFunctions(self, ctx: Parser.HierarchyFunctionsContext):
        """
        HIERARCHY LPAREN op=expr COMMA hrName=IDENTIFIER (conditionClause)? (RULE ruleComponent=componentID)? (validationMode)? (inputModeHierarchy)? outputModeHierarchy? RPAREN
        """  # noqa E501
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        op = c.getSymbol().text
        dataset_node = self.visitExpr(ctx_list[2])
        rule_name_node = Identifier(
            value=ctx_list[4].getSymbol().text,
            kind="RuleID",
            **extract_token_info(ctx_list[4].getSymbol()),
        )

        conditions = []
        modes = "non_null"
        inputs = "rule"
        retains = "computed"
        rule_comp = None

        for c in ctx_list:
            if isinstance(c, Parser.ConditionClauseContext):
                conditions.append(Terminals().visitConditionClause(c))
            elif isinstance(c, Parser.ComponentIDContext):
                rule_comp = Terminals().visitComponentID(c)
            elif isinstance(c, Parser.ValidationModeContext):
                modes = Terminals().visitValidationMode(c)
            elif isinstance(c, Parser.InputModeHierarchyContext):
                inputs = Terminals().visitInputModeHierarchy(c)
            elif isinstance(c, Parser.OutputModeHierarchyContext):
                retains = Terminals().visitOutputModeHierarchy(c)

        if len(conditions) != 0:
            # AST_ASTCONSTRUCTOR.22
            conditions = conditions[0]

        if inputs == DATASET_PRIORITY:
            raise NotImplementedError("Dataset Priority input mode on HR is not implemented")
        param_constant_node = []

        param_constant_node.append(
            ParamConstant(type_="PARAM_MODE", value=modes, **extract_token_info(ctx))
        )
        param_constant_node.append(
            ParamConstant(type_="PARAM_INPUT", value=inputs, **extract_token_info(ctx))
        )
        param_constant_node.append(
            ParamConstant(type_="PARAM_OUTPUT", value=retains, **extract_token_info(ctx))
        )

        if not rule_comp:
            if isinstance(de_ruleset_elements[rule_name_node.value], list):
                rule_element = de_ruleset_elements[rule_name_node.value][-1]
            else:
                rule_element = de_ruleset_elements[rule_name_node.value]
            if rule_element.kind == "DatasetID":
                check_hierarchy_rule = rule_element.value
                rule_comp = Identifier(
                    value=check_hierarchy_rule, kind="ComponentID", **extract_token_info(ctx)
                )
            else:  # ValuedomainID
                raise SemanticError("1-1-10-4", op=op)

        return ParamOp(
            op=op,
            children=[dataset_node, rule_comp, rule_name_node, *conditions],
            params=param_constant_node,
            **extract_token_info(ctx),
        )

    """
                            -----------------------------------
                                    Validation Functions
                            -----------------------------------
    """

    def visitValidationFunctions(self, ctx: Parser.ValidationFunctionsContext):
        if isinstance(ctx, Parser.ValidateDPrulesetContext):
            return self.visitValidateDPruleset(ctx)
        elif isinstance(ctx, Parser.ValidateHRrulesetContext):
            return self.visitValidateHRruleset(ctx)
        elif isinstance(ctx, Parser.ValidationSimpleContext):
            return self.visitValidationSimple(ctx)

    def visitValidateDPruleset(self, ctx: Parser.ValidateDPrulesetContext):
        """
        validationDatapoint: CHECK_DATAPOINT '(' expr ',' IDENTIFIER (COMPONENTS componentID (',' componentID)*)? (INVALID|ALL_MEASURES|ALL)? ')' ;
        """  # noqa E501
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        op = c.getSymbol().text

        operand_node = self.visitExpr(ctx_list[2])
        rule_name = ctx_list[4].getSymbol().text

        components = [
            Terminals().visitComponentID(comp)
            for comp in ctx_list
            if isinstance(comp, Parser.ComponentIDContext)
        ]
        aux_components = []
        for x in components:
            if isinstance(x, BinOp):
                aux_components.append(x.right.value)
            else:
                aux_components.append(x.value)

        components = aux_components

        # Default value for output is invalid.
        output = "invalid"

        if isinstance(ctx_list[-2], Parser.ValidationOutputContext):
            output = Terminals().visitValidationOutput(ctx_list[-2])

        return ParamOp(
            op=op,
            children=[operand_node, rule_name, *components],
            params=[output],
            **extract_token_info(ctx),
        )

    # TODO Not fully implemented only basic usage available.
    def visitValidateHRruleset(self, ctx: Parser.ValidateHRrulesetContext):
        """
        CHECK_HIERARCHY LPAREN op=expr COMMA hrName=IDENTIFIER conditionClause? (RULE componentID)? validationMode? inputMode? validationOutput? RPAREN     # validateHRruleset
        """  # noqa E501

        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        op = c.getSymbol().text

        dataset_node = self.visitExpr(ctx_list[2])
        rule_name_node = Identifier(
            value=ctx_list[4].getSymbol().text,
            kind="RuleID",
            **extract_token_info(ctx_list[4].getSymbol()),
        )

        conditions = []
        # Default values
        modes = "non_null"
        inputs = "dataset"
        retains = "invalid"
        rule_comp = None

        for c in ctx_list:
            if isinstance(c, Parser.ConditionClauseContext):
                conditions.append(Terminals().visitConditionClause(c))
            elif isinstance(c, Parser.ComponentIDContext):
                rule_comp = Terminals().visitComponentID(c)
            elif isinstance(c, Parser.ValidationModeContext):
                modes = Terminals().visitValidationMode(c)
            elif isinstance(c, Parser.InputModeContext):
                inputs = Terminals().visitInputMode(c)
            elif isinstance(c, Parser.ValidationOutputContext):
                retains = Terminals().visitValidationOutput(c)

        if len(conditions) != 0:
            # AST_ASTCONSTRUCTOR.22
            conditions = conditions[0]

        param_constant_node = []

        if inputs == DATASET_PRIORITY:
            raise NotImplementedError("Dataset Priority input mode on HR is not implemented")

        param_constant_node.append(
            ParamConstant(type_="PARAM_MODE", value=modes, **extract_token_info(ctx))
        )
        param_constant_node.append(
            ParamConstant(type_="PARAM_INPUT", value=inputs, **extract_token_info(ctx))
        )
        param_constant_node.append(
            ParamConstant(type_="PARAM_OUTPUT", value=retains, **extract_token_info(ctx))
        )

        if not rule_comp:
            if isinstance(de_ruleset_elements[rule_name_node.value], list):
                rule_element = de_ruleset_elements[rule_name_node.value][-1]
            else:
                rule_element = de_ruleset_elements[rule_name_node.value]

            if rule_element.kind == "DatasetID":
                check_hierarchy_rule = rule_element.value
                rule_comp = Identifier(
                    value=check_hierarchy_rule,
                    kind="ComponentID",
                    **extract_token_info(ctx),
                )
            else:  # ValuedomainID
                raise SemanticError("1-1-10-4", op=op)

        return ParamOp(
            op=op,
            children=[dataset_node, rule_comp, rule_name_node, *conditions],
            params=param_constant_node,
            **extract_token_info(ctx),
        )

    def visitValidationSimple(self, ctx: Parser.ValidationSimpleContext):
        """
        | CHECK LPAREN op=expr (codeErr=erCode)? (levelCode=erLevel)? imbalanceExpr?  output=(INVALID|ALL)? RPAREN	        # validationSimple
        """  # noqa E501
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]
        token = c.getSymbol()

        validation_node = self.visitExpr(ctx_list[2])

        inbalance_node = None
        error_code = None
        error_level = None
        for param in ctx_list:
            if isinstance(param, Parser.ErCodeContext):
                error_code = Terminals().visitErCode(param)
            elif isinstance(param, Parser.ErLevelContext):
                error_level = Terminals().visitErLevel(param)
            elif isinstance(param, Parser.ImbalanceExprContext):
                inbalance_node = self.visitImbalanceExpr(param)

        invalid = ctx_list[-2] if isinstance(ctx_list[-2], TerminalNodeImpl) else None
        invalid_value = False if invalid is None else invalid.getSymbol().text == "invalid"

        return Validation(
            op=token.text,
            validation=validation_node,
            error_code=error_code,
            error_level=error_level,
            imbalance=inbalance_node,
            invalid=invalid_value,
            **extract_token_info(ctx),
        )

    def visitImbalanceExpr(self, ctx: Parser.ImbalanceExprContext):
        ctx_list = list(ctx.getChildren())
        return self.visitExpr(ctx_list[1])

    """
                            -----------------------------------
                                    Aggregate Functions
                            -----------------------------------
    """

    # TODO Count function count() without parameters. Used at least in aggregations at having.
    def visitAggregateFunctions(self, ctx: Parser.AggregateFunctionsContext):
        """
        aggrFunction: SUM '(' expr ')'
                    | AVG '(' expr ')'
                    | COUNT '(' expr? ')'
                    | MEDIAN '(' expr ')'
                    | MIN '(' expr ')'
                    | MAX '(' expr ')'
                    | RANK '(' expr ')'
                    | STDDEV_POP '(' expr ')'
                    | STDDEV_SAMP '(' expr ')'
                    | VAR_POP '(' expr ')'
                    | VAR_SAMP '(' expr ')'
                    ;
        """
        if isinstance(ctx, Parser.AggrDatasetContext):
            return self.visitAggrDataset(ctx)
        else:
            raise NotImplementedError

    def visitAggrDataset(self, ctx: Parser.AggrDatasetContext):
        ctx_list = list(ctx.getChildren())
        # c = ctx_list[0]

        grouping_op = None
        group_node = None
        have_node = None

        groups = [group for group in ctx_list if isinstance(group, Parser.GroupingClauseContext)]
        haves = [have for have in ctx_list if isinstance(have, Parser.HavingClauseContext)]

        op_node = ctx_list[0].getSymbol().text
        operand = self.visitExpr(ctx_list[2])

        if len(groups) != 0:
            grouping_op, group_node = self.visitGroupingClause(groups[0])
        if len(haves) != 0:
            have_node, expr = self.visitHavingClause(haves[0])
            have_node.expr = expr

        return Aggregation(
            op=op_node,
            operand=operand,
            grouping_op=grouping_op,
            grouping=group_node,
            having_clause=have_node,
            **extract_token_info(ctx),
        )

    """
                            -----------------------------------
                                    Analytic Functions
                            -----------------------------------
    """

    def visitAnalyticFunctions(self, ctx: Parser.AnalyticFunctionsContext):
        # ctx_list = list(ctx.getChildren())

        if isinstance(ctx, Parser.AnSimpleFunctionContext):
            return self.visitAnSimpleFunction(ctx)
        elif isinstance(ctx, Parser.LagOrLeadAnContext):
            return self.visitLagOrLeadAn(ctx)
        elif isinstance(ctx, Parser.RatioToReportAnContext):
            return self.visitRatioToReportAn(ctx)
        else:
            raise NotImplementedError

    def visitAnSimpleFunction(self, ctx: Parser.AnSimpleFunctionContext):
        ctx_list = list(ctx.getChildren())

        window = None
        partition_by = None
        order_by = None

        op_node = ctx_list[0].getSymbol().text
        operand = self.visitExpr(ctx_list[2])

        for c in ctx_list[5:-2]:
            if isinstance(c, Parser.PartitionByClauseContext):
                partition_by = Terminals().visitPartitionByClause(c)
                continue
            elif isinstance(c, Parser.OrderByClauseContext):
                order_by = Terminals().visitOrderByClause(c)
                continue
            elif isinstance(c, Parser.WindowingClauseContext):
                window = Terminals().visitWindowingClause(c)
                continue
            else:
                raise NotImplementedError

        if window is None:
            window = Windowing(
                type_="data",
                start=-1,
                stop=0,
                start_mode="preceding",
                stop_mode="current",
                **extract_token_info(ctx),
            )

        return Analytic(
            op=op_node,
            operand=operand,
            partition_by=partition_by,
            order_by=order_by,
            window=window,
            **extract_token_info(ctx),
        )

    def visitLagOrLeadAn(self, ctx: Parser.LagOrLeadAnContext):
        ctx_list = list(ctx.getChildren())

        params = None
        partition_by = None
        order_by = None

        op_node = ctx_list[0].getSymbol().text
        operand = self.visitExpr(ctx_list[2])

        for c in ctx_list[4:-2]:
            if isinstance(c, Parser.PartitionByClauseContext):
                partition_by = Terminals().visitPartitionByClause(c)
                continue
            elif isinstance(c, Parser.OrderByClauseContext):
                order_by = Terminals().visitOrderByClause(c)
                continue
            elif isinstance(c, (Parser.SignedIntegerContext, Parser.ScalarItemContext)):
                if params is None:
                    params = []
                if isinstance(c, Parser.SignedIntegerContext):
                    params.append(Terminals().visitSignedInteger(c))
                else:
                    params.append(Terminals().visitScalarItem(c))
                continue

        if len(params) == 0:
            # AST_ASTCONSTRUCTOR.16
            raise Exception(f"{op_node} requires an offset parameter.")

        return Analytic(
            op=op_node,
            operand=operand,
            partition_by=partition_by,
            order_by=order_by,
            params=params,
            **extract_token_info(ctx),
        )

    def visitRatioToReportAn(self, ctx: Parser.RatioToReportAnContext):
        ctx_list = list(ctx.getChildren())

        # params = None
        order_by = None

        op_node = ctx_list[0].getSymbol().text
        operand = self.visitExpr(ctx_list[2])

        partition_by = Terminals().visitPartitionByClause(ctx_list[5])

        return Analytic(
            op=op_node,
            operand=operand,
            partition_by=partition_by,
            order_by=order_by,
            **extract_token_info(ctx),
        )

    """______________________________________________________________________________________


                                        Clause Definition.

       _______________________________________________________________________________________"""

    def visitDatasetClause(self, ctx: Parser.DatasetClauseContext):
        """
        datasetClause:
                    renameClause
                      | aggrClause
                      | filterClause
                      | calcClause
                      | keepClause
                      | dropClause
                      | pivotExpr
                      | unpivotExpr
                      | subspaceExpr
                      ;
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        # RENAME renameClause
        if isinstance(c, Parser.RenameClauseContext):
            return self.visitRenameClause(c)

        # aggrClause
        elif isinstance(c, Parser.AggrClauseContext):
            return self.visitAggrClause(c)

        # filterClause
        elif isinstance(c, Parser.FilterClauseContext):
            return self.visitFilterClause(c)

        # calcClause
        elif isinstance(c, Parser.CalcClauseContext):
            return self.visitCalcClause(c)

        # keepClause
        elif isinstance(c, Parser.KeepOrDropClauseContext):
            return self.visitKeepOrDropClause(c)

        # pivotExpr
        elif isinstance(c, Parser.PivotOrUnpivotClauseContext):
            return self.visitPivotOrUnpivotClause(c)

        # subspaceExpr
        elif isinstance(c, Parser.SubspaceClauseContext):
            return self.visitSubspaceClause(c)

    """
                    -----------------------------------
                            Rename Clause
                    -----------------------------------
    """

    def visitRenameClause(self, ctx: Parser.RenameClauseContext):
        """
        renameClause: RENAME renameClauseItem (COMMA renameClauseItem)*;
        """
        ctx_list = list(ctx.getChildren())

        renames = [
            ctx_child
            for ctx_child in ctx_list
            if isinstance(ctx_child, Parser.RenameClauseItemContext)
        ]
        rename_nodes = []

        for ctx_rename in renames:
            rename_nodes.append(self.visitRenameClauseItem(ctx_rename))

        return RegularAggregation(
            op=ctx_list[0].getSymbol().text, children=rename_nodes, **extract_token_info(ctx)
        )

    def visitRenameClauseItem(self, ctx: Parser.RenameClauseItemContext):
        """
        renameClauseItem: fromName=componentID TO toName=componentID;
        """
        ctx_list = list(ctx.getChildren())

        left_node = Terminals().visitComponentID(ctx_list[0])
        if isinstance(left_node, BinOp):
            left_node = f"{left_node.left.value}{left_node.op}{left_node.right.value}"
        else:
            left_node = left_node.value

        right_node = Terminals().visitVarID(ctx_list[2]).value

        return RenameNode(old_name=left_node, new_name=right_node, **extract_token_info(ctx))

    """
                    -----------------------------------
                            Aggregate Clause
                    -----------------------------------
    """

    def visitAggregateClause(self, ctx: Parser.AggregateClauseContext):
        """
        aggregateClause: aggrFunctionClause (',' aggrFunctionClause)* ;
        """
        ctx_list = list(ctx.getChildren())

        aggregates_nodes = []

        aggregates = [
            aggregate
            for aggregate in ctx_list
            if isinstance(aggregate, Parser.AggrFunctionClauseContext)
        ]

        for agg in aggregates:
            aggregates_nodes.append(self.visitAggrFunctionClause(agg))

        return aggregates_nodes

    def visitAggrFunctionClause(self, ctx: Parser.AggrFunctionClauseContext):
        """
        aggrFunctionClause: (componentRole)? componentID ':=' aggrFunction ;
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        if isinstance(c, Parser.ComponentRoleContext):
            role = Terminals().visitComponentRole(c)
            base_index = 1
        else:
            base_index = 0
            role = Role.MEASURE

        left_node = Terminals().visitSimpleComponentId(ctx_list[base_index])
        op_node = ":="
        right_node = ExprComp().visitAggregateFunctionsComponents(ctx_list[base_index + 2])
        # Encoding the role information inside the Assignment for easiness and simplicity.
        # Cannot find another way with less lines of code
        left_node.role = role

        return Assignment(left=left_node, op=op_node, right=right_node, **extract_token_info(ctx))

    def visitAggrClause(self, ctx: Parser.AggrClauseContext):
        """
        aggrClause: AGGREGATE aggregateClause (groupingClause havingClause?)? ;
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        op_node = c.getSymbol().text
        group_node = None
        grouping_op = None
        have_node = None

        groups = [group for group in ctx_list if isinstance(group, Parser.GroupingClauseContext)]
        haves = [have for have in ctx_list if isinstance(have, Parser.HavingClauseContext)]

        aggregate_nodes = self.visitAggregateClause(ctx_list[1])

        children = []

        if len(groups) != 0:
            grouping_op, group_node = self.visitGroupingClause(groups[0])
        if len(haves) > 0:
            have_node, expr = self.visitHavingClause(haves[0])
            have_node.expr = expr
        for element in aggregate_nodes:
            element.right = Aggregation(
                op=element.right.op,
                operand=element.right.operand,
                grouping_op=grouping_op,
                grouping=group_node,
                having_clause=have_node,
                **extract_token_info(ctx_list[1]),
            )
            children.append(copy(element))

        return RegularAggregation(op=op_node, children=children, **extract_token_info(ctx))

    def visitGroupingClause(self, ctx: Parser.GroupingClauseContext):
        """
        groupingClause:
            GROUP op=(BY | EXCEPT) componentID (COMMA componentID)*     # groupByOrExcept
            | GROUP ALL exprComponent                                   # groupAll
          ;
        """
        if isinstance(ctx, Parser.GroupByOrExceptContext):
            return self.visitGroupByOrExcept(ctx)
        elif isinstance(ctx, Parser.GroupAllContext):
            return self.visitGroupAll(ctx)
        else:
            raise NotImplementedError

    def visitHavingClause(self, ctx: Parser.HavingClauseContext):
        """
        havingClause: HAVING exprComponent ;
        """
        ctx_list = list(ctx.getChildren())
        op_node = ctx_list[0].getSymbol().text

        text = ctx_list[1].start.source[1].strdata
        expr = re.split("having", text)[1]
        expr = "having " + expr[:-2].strip()

        if "]" in expr:
            index = expr.index("]")
            expr = expr[:index]
        if "end" in expr:
            index = expr.index("end")
            expr = expr[:index]
        if expr.count(")") > expr.count("("):
            index = expr.rindex(")")
            expr = expr[:index]

        if "{" in expr or "}" in expr:
            expr = expr.replace("{", "(")
            expr = expr.replace("}", ")")
        if "not_in" in expr:
            expr = expr.replace("not_in", "not in")
        if '"' in expr:
            expr = expr.replace('"', "'")

        if isinstance(ctx_list[1], Parser.ComparisonExprCompContext):
            param_nodes = ExprComp().visitComparisonExprComp(ctx_list[1])
        elif isinstance(ctx_list[1], Parser.InNotInExprCompContext):
            param_nodes = ExprComp().visitInNotInExprComp(ctx_list[1])
        elif isinstance(ctx_list[1], Parser.BooleanExprCompContext):
            param_nodes = ExprComp().visitBooleanExprComp(ctx_list[1])
        else:
            raise NotImplementedError

        return ParamOp(
            op=op_node, children=None, params=param_nodes, **extract_token_info(ctx)
        ), expr

    def visitGroupByOrExcept(self, ctx: Parser.GroupByOrExceptContext):
        ctx_list = list(ctx.getChildren())

        token_left = ctx_list[0].getSymbol().text
        token_right = ctx_list[1].getSymbol().text

        op_node = token_left + " " + token_right

        children_nodes = [
            Terminals().visitComponentID(identifier)
            for identifier in ctx_list
            if isinstance(identifier, Parser.ComponentIDContext)
        ]

        return op_node, children_nodes

    def visitGroupAll(self, ctx: Parser.GroupAllContext):
        ctx_list = list(ctx.getChildren())

        token_left = ctx_list[0].getSymbol().text
        token_right = ctx_list[1].getSymbol().text

        op_node = token_left + " " + token_right

        children_nodes = [ExprComp().visitExprComponent(ctx_list[2])]

        return op_node, children_nodes

    """
                    -----------------------------------
                            Filter Clause
                    -----------------------------------
    """

    def visitFilterClause(self, ctx: Parser.FilterClauseContext):
        """
        filterClause: FILTER expr;
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]
        token = c.getSymbol()

        op_node = token.text
        operand_nodes = []
        operand_nodes.append(ExprComp().visitExprComponent(ctx_list[1]))

        return RegularAggregation(op=op_node, children=operand_nodes, **extract_token_info(ctx))

    """
                    -----------------------------------
                            Calc Clause
                    -----------------------------------
    """

    def visitCalcClause(self, ctx: Parser.CalcClauseContext):
        """
        calcClause: CALC calcClauseItem (',' calcClauseItem)*;
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        calcClauseItems = [
            calcClauseItem
            for calcClauseItem in ctx_list
            if isinstance(calcClauseItem, Parser.CalcClauseItemContext)
        ]
        calcClauseItems_nodes = []

        op_node = c.getSymbol().text
        for calcClauseItem in calcClauseItems:
            result = self.visitCalcClauseItem(calcClauseItem)
            calcClauseItems_nodes.append(result)

        return RegularAggregation(
            op=op_node, children=calcClauseItems_nodes, **extract_token_info(ctx)
        )

    def visitCalcClauseItem(self, ctx: Parser.CalcClauseItemContext):
        """
        calcClauseItem: (componentRole)? componentID  ASSIGN  exprComponent;
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        if isinstance(c, Parser.ComponentRoleContext):
            role = Terminals().visitComponentRole(c)

            left_node = Terminals().visitComponentID(ctx_list[1])
            op_node = ":="
            right_node = ExprComp().visitExprComponent(ctx_list[3])
            operand_node = Assignment(
                left=left_node, op=op_node, right=right_node, **extract_token_info(ctx)
            )
            if role is None:
                return UnaryOp(
                    op=Role.MEASURE.value.lower(), operand=operand_node, **extract_token_info(c)
                )
            return UnaryOp(op=role.value.lower(), operand=operand_node, **extract_token_info(c))
        else:
            left_node = Terminals().visitSimpleComponentId(c)
            op_node = ":="
            right_node = ExprComp().visitExprComponent(ctx_list[2])

            operand_node = Assignment(
                left=left_node, op=op_node, right=right_node, **extract_token_info(ctx)
            )
            return UnaryOp(
                op=Role.MEASURE.value.lower(), operand=operand_node, **extract_token_info(ctx)
            )

    def visitKeepOrDropClause(self, ctx: Parser.KeepOrDropClauseContext):
        """
        keepOrDropClause: op = (KEEP | DROP) componentID (COMMA componentID)* ;
        """

        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        items = [item for item in ctx_list if isinstance(item, Parser.ComponentIDContext)]
        nodes = []

        op_node = c.getSymbol().text
        for item in items:
            nodes.append(Terminals().visitComponentID(item))

        return RegularAggregation(op=op_node, children=nodes, **extract_token_info(ctx))

    """
                    -----------------------------------
                            Pivot/Unpivot Clause
                    -----------------------------------
    """

    def visitPivotOrUnpivotClause(self, ctx: Parser.PivotOrUnpivotClauseContext):
        """
        pivotOrUnpivotClause: op=(PIVOT|UNPIVOT) id_=componentID COMMA mea=componentID ;
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]
        token = c.getSymbol()

        op_node = token.text
        children_nodes = []
        children_nodes.append(Terminals().visitComponentID(ctx_list[1]))
        children_nodes.append(Terminals().visitComponentID(ctx_list[3]))

        return RegularAggregation(op=op_node, children=children_nodes, **extract_token_info(ctx))

    """
                    -----------------------------------
                            Subspace Clause
                    -----------------------------------
    """

    def visitSubspaceClause(self, ctx: Parser.SubspaceClauseContext):
        """
        subspaceClause: SUBSPACE subspaceClauseItem (COMMA subspaceClauseItem)*;"""
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        subspace_nodes = []
        subspaces = [
            subspace
            for subspace in ctx_list
            if isinstance(subspace, Parser.SubspaceClauseItemContext)
        ]

        for subspace in subspaces:
            subspace_nodes.append(self.visitSubspaceClauseItem(subspace))

        op_node = c.getSymbol().text
        return RegularAggregation(op=op_node, children=subspace_nodes, **extract_token_info(ctx))

    def visitSubspaceClauseItem(self, ctx: Parser.SubspaceClauseItemContext):
        ctx_list = list(ctx.getChildren())

        left_node = Terminals().visitVarID(ctx_list[0])
        op_node = ctx_list[1].getSymbol().text
        right_node = Terminals().visitScalarItem(ctx_list[2])
        return BinOp(left=left_node, op=op_node, right=right_node, **extract_token_info(ctx))

    def visitOptionalExpr(self, ctx: Parser.OptionalExprContext):
        """
        optionalExpr: expr
                    | OPTIONAL ;
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        if isinstance(c, Parser.ExprContext):
            return self.visitExpr(c)

        elif isinstance(c, TerminalNodeImpl):
            token = c.getSymbol()
            opt = token.text
            return ID(type_="OPTIONAL", value=opt, **extract_token_info(ctx))
