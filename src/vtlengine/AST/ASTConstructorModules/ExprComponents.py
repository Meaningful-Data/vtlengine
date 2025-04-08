from antlr4.tree.Tree import TerminalNodeImpl

from vtlengine.AST import (
    ID,
    Aggregation,
    Analytic,
    BinOp,
    Case,
    CaseObj,
    Constant,
    EvalOp,
    Identifier,
    If,
    MulOp,
    ParamConstant,
    ParamOp,
    ParFunction,
    TimeAggregation,
    UDOCall,
    UnaryOp,
    VarID,
)
from vtlengine.AST.ASTConstructorModules import extract_token_info
from vtlengine.AST.ASTConstructorModules.Terminals import Terminals
from vtlengine.AST.Grammar.parser import Parser
from vtlengine.AST.VtlVisitor import VtlVisitor
from vtlengine.Exceptions import SemanticError


class ExprComp(VtlVisitor):
    """______________________________________________________________________________________


                                ExprComponent Definition.

    _______________________________________________________________________________________
    """

    def visitExprComponent(self, ctx: Parser.ExprComponentContext):
        """
        exprComponent:
            LPAREN exprComponent RPAREN                                                                             # parenthesisExprComp
            | functionsComponents                                                                                   # functionsExpressionComp
            | op=(PLUS|MINUS|NOT) right=exprComponent                                                               # unaryExprComp
            | left=exprComponent op=(MUL|DIV) right=exprComponent                                                   # arithmeticExprComp
            | left=exprComponent op=(PLUS|MINUS|CONCAT) right=exprComponent                                         # arithmeticExprOrConcatComp
            | left=exprComponent comparisonOperand right=exprComponent                                              # comparisonExprComp
            | left=exprComponent op=(IN|NOT_IN)(lists|valueDomainID)                                                # inNotInExprComp
            | left=exprComponent op=AND right=exprComponent                                                         # booleanExprComp
            | left=exprComponent op=(OR|XOR) right=exprComponent                                                    # booleanExprComp
            | IF  conditionalExpr=exprComponent  THEN thenExpr=exprComponent ELSE elseExpr=exprComponent            # ifExprComp
            | constant                                                                                              # constantExprComp
            | componentID                                                                                           # compId
        ;
        """  # noqa E501
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        if isinstance(ctx, Parser.ParenthesisExprCompContext):
            return self.visitParenthesisExprComp(ctx)

        # functions
        elif isinstance(ctx, Parser.FunctionsExpressionCompContext):
            return self.visitFunctionsComponents(c)

        # op=(PLUS|MINUS|NOT) right=expr # unary expression
        elif isinstance(ctx, Parser.UnaryExprCompContext):
            return self.visitUnaryExprComp(ctx)

        # | left=expr op=(MUL|DIV) right=expr               # arithmeticExpr
        elif isinstance(ctx, Parser.ArithmeticExprCompContext):
            return self.visitArithmeticExprComp(ctx)

        # | left=expr op=(PLUS|MINUS|CONCAT) right=expr     # arithmeticExprOrConcat
        elif isinstance(ctx, Parser.ArithmeticExprOrConcatCompContext):
            return self.visitArithmeticExprOrConcatComp(ctx)

        # | left=expr op=comparisonOperand  right=expr      # comparisonExpr
        elif isinstance(ctx, Parser.ComparisonExprCompContext):
            return self.visitComparisonExprComp(ctx)

        # | left=expr op=(IN|NOT_IN)(lists|valueDomainID)   # inNotInExpr
        elif isinstance(ctx, Parser.InNotInExprCompContext):
            return self.visitInNotInExprComp(ctx)

        # | left=expr op=AND right=expr                                           # booleanExpr
        # | left=expr op=(OR|XOR) right=expr
        elif isinstance(ctx, Parser.BooleanExprCompContext):
            return self.visitBooleanExprComp(ctx)

        # IF  conditionalExpr=expr  THEN thenExpr=expr ELSE elseExpr=expr       # ifExpr
        elif isinstance(ctx, Parser.IfExprCompContext):
            return self.visitIfExprComp(ctx)

        # CASE WHEN conditionalExpr=expr THEN thenExpr=expr ELSE elseExpr=expr END # caseExpr
        elif isinstance(ctx, Parser.CaseExprCompContext):
            return self.visitCaseExprComp(ctx)

        # constant
        elif isinstance(ctx, Parser.ConstantExprCompContext):
            return Terminals().visitConstant(c)

        # componentID
        # TODO Changed to pass more tests. Original code: return Terminals().visitComponentID(c)
        elif isinstance(ctx, Parser.CompIdContext):
            if len(c.children) > 1:
                return Terminals().visitComponentID(c)
            token = c.children[0].getSymbol()
            # check token text
            has_scaped_char = token.text.find("'") != -1
            if has_scaped_char:
                token.text = str(token.text.replace("'", ""))
            var_id_node = VarID(value=token.text, **extract_token_info(ctx))
            return var_id_node

        else:
            # AST_ASTCONSTRUCTOR.3
            raise NotImplementedError

    def bin_op_creator_comp(self, ctx: Parser.ExprComponentContext):
        ctx_list = list(ctx.getChildren())
        left_node = self.visitExprComponent(ctx_list[0])
        if isinstance(ctx_list[1], Parser.ComparisonOperandContext):
            op = list(ctx_list[1].getChildren())[0].getSymbol().text
        else:
            op = ctx_list[1].getSymbol().text
        right_node = self.visitExprComponent(ctx_list[2])

        bin_op_node = BinOp(left=left_node, op=op, right=right_node, **extract_token_info(ctx))

        return bin_op_node

    def visitArithmeticExprComp(self, ctx: Parser.ArithmeticExprContext):
        return self.bin_op_creator_comp(ctx)

    def visitArithmeticExprOrConcatComp(self, ctx: Parser.ArithmeticExprOrConcatContext):
        return self.bin_op_creator_comp(ctx)

    def visitComparisonExprComp(self, ctx: Parser.ComparisonExprContext):
        return self.bin_op_creator_comp(ctx)

    def visitInNotInExprComp(self, ctx: Parser.InNotInExprContext):
        ctx_list = list(ctx.getChildren())
        left_node = self.visitExprComponent(ctx_list[0])
        op = ctx_list[1].symbol.text

        if isinstance(ctx_list[2], Parser.ListsContext):
            right_node = Terminals().visitLists(ctx_list[2])
        elif isinstance(ctx_list[2], Parser.ValueDomainIDContext):
            right_node = Terminals().visitValueDomainID(ctx_list[2])
        else:
            raise NotImplementedError
        bin_op_node = BinOp(left=left_node, op=op, right=right_node, **extract_token_info(ctx))

        return bin_op_node

    def visitBooleanExprComp(self, ctx: Parser.BooleanExprContext):
        return self.bin_op_creator_comp(ctx)

    def visitParenthesisExprComp(self, ctx: Parser.ParenthesisExprContext):
        operand = self.visitExprComponent(list(ctx.getChildren())[1])
        return ParFunction(operand=operand, **extract_token_info(ctx))

    def visitUnaryExprComp(self, ctx: Parser.UnaryExprContext):
        c_list = list(ctx.getChildren())
        op = c_list[0].getSymbol().text
        right = self.visitExprComponent(c_list[1])

        return UnaryOp(op=op, operand=right, **extract_token_info(ctx))

    def visitIfExprComp(self, ctx: Parser.IfExprCompContext):
        ctx_list = list(ctx.getChildren())

        condition_node = self.visitExprComponent(ctx_list[1])
        then_op_node = self.visitExprComponent(ctx_list[3])
        else_op_node = self.visitExprComponent(ctx_list[5])

        if_node = If(
            condition=condition_node,
            thenOp=then_op_node,
            elseOp=else_op_node,
            **extract_token_info(ctx),
        )

        return if_node

    def visitCaseExprComp(self, ctx: Parser.CaseExprCompContext):
        ctx_list = list(ctx.getChildren())

        if len(ctx_list) % 4 != 3:
            raise ValueError("Syntax error.")

        else_node = self.visitExprComponent(ctx_list[-1])
        ctx_list = ctx_list[1:-2]
        cases = []

        for i in range(0, len(ctx_list), 4):
            condition = self.visitExprComponent(ctx_list[i + 1])
            thenOp = self.visitExprComponent(ctx_list[i + 3])
            case_obj = CaseObj(
                condition=condition, thenOp=thenOp, **extract_token_info(ctx_list[i + 1])
            )
            cases.append(case_obj)

        case_node = Case(cases=cases, elseOp=else_node, **extract_token_info(ctx))

        return case_node

    def visitOptionalExprComponent(self, ctx: Parser.OptionalExprComponentContext):
        """
        optionalExpr: expr
                    | OPTIONAL ;
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        if isinstance(c, Parser.ExprComponentContext):
            return self.visitExprComponent(c)

        elif isinstance(c, TerminalNodeImpl):
            token = c.getSymbol()
            opt = token.text
            return ID(type_="OPTIONAL", value=opt, **extract_token_info(ctx))

    """____________________________________________________________________________________


                                    FunctionsComponents Definition.

      _____________________________________________________________________________________"""

    def visitFunctionsComponents(self, ctx: Parser.FunctionsComponentsContext):
        """
        functionsComponents:
            genericOperatorsComponent           # genericFunctionsComponents
           | stringOperatorsComponent           # stringFunctionsComponents
           | numericOperatorsComponent          # numericFunctionsComponents
           | comparisonOperatorsComponent       # comparisonFunctionsComponents
           | timeOperatorsComponent             # timeFunctionsComponents
           | conditionalOperatorsComponent      # conditionalFunctionsComponents
           | aggrOperators                      # aggregateFunctionsComponents
           | anFunctionComponent                # analyticFunctionsComponents

        ;
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        if isinstance(ctx, Parser.GenericFunctionsComponentsContext):
            return self.visitGenericFunctionsComponents(c)

        elif isinstance(ctx, Parser.StringFunctionsComponentsContext):
            return self.visitStringFunctionsComponents(c)

        elif isinstance(ctx, Parser.NumericFunctionsComponentsContext):
            return self.visitNumericFunctionsComponents(c)

        elif isinstance(ctx, Parser.ComparisonFunctionsComponentsContext):
            return self.visitComparisonFunctionsComponents(c)

        elif isinstance(ctx, Parser.TimeFunctionsComponentsContext):
            return self.visitTimeFunctionsComponents(c)

        elif isinstance(ctx, Parser.ConditionalFunctionsComponentsContext):
            return self.visitConditionalFunctionsComponents(c)

        elif isinstance(ctx, Parser.AggregateFunctionsComponentsContext):
            return self.visitAggregateFunctionsComponents(c)

        elif isinstance(ctx, Parser.AnalyticFunctionsComponentsContext):
            return self.visitAnalyticFunctionsComponents(c)
        else:
            raise NotImplementedError

    """
                                -----------------------------------
                                    Generic Functions Components
                                -----------------------------------
    """

    def visitGenericFunctionsComponents(self, ctx: Parser.GenericFunctionsComponentsContext):
        if isinstance(ctx, Parser.CallComponentContext):
            return self.visitCallComponent(ctx)
        elif isinstance(ctx, Parser.EvalAtomComponentContext):
            return self.visitEvalAtomComponent(ctx)
        elif isinstance(ctx, Parser.CastExprComponentContext):
            return self.visitCastExprComponent(ctx)
        else:
            raise NotImplementedError

    def visitCallComponent(self, ctx: Parser.CallComponentContext):
        """
        callFunction: operatorID LPAREN (parameterComponent (COMMA parameterComponent)*)? RPAREN    # callComponent
        """  # noqa E501
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        op = Terminals().visitOperatorID(c)
        param_nodes = [
            self.visitParameterComponent(element)
            for element in ctx_list
            if isinstance(element, Parser.ParameterComponentContext)
        ]

        return UDOCall(op=op, params=param_nodes, **extract_token_info(ctx))

    def visitEvalAtomComponent(self, ctx: Parser.EvalAtomComponentContext):
        """
        | EVAL LPAREN routineName LPAREN (componentID|scalarItem)? (COMMA (componentID|scalarItem))* RPAREN (LANGUAGE STRING_CONSTANT)? (RETURNS outputParameterTypeComponent)? RPAREN      # evalAtomComponent
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

        if len(children_nodes) > 1:
            raise Exception("Only one operand is allowed in Eval")

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
            Terminals().visitOutputParameterTypeComponent(output)
            for output in ctx_list
            if isinstance(output, Parser.OutputParameterTypeComponentContext)
        ]
        if len(output_node) == 0:
            # AST_ASTCONSTRUCTOR.13
            raise SemanticError("1-4-2-1", option="output")

        return EvalOp(
            name=routine_name,
            operands=children_nodes[0],
            output=output_node[0],
            language=language_name[0].getSymbol().text,
            **extract_token_info(ctx),
        )

    def visitCastExprComponent(self, ctx: Parser.CastExprComponentContext):
        """
        | CAST LPAREN exprComponent COMMA (basicScalarType|valueDomainName) (COMMA STRING_CONSTANT)? RPAREN         # castExprComponent
        """  # noqa E501
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        token = c.getSymbol()

        op = token.text
        expr_node = [
            self.visitExprComponent(expr)
            for expr in ctx_list
            if isinstance(expr, Parser.ExprComponentContext)
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

    def visitParameterComponent(self, ctx: Parser.ParameterComponentContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        if isinstance(c, Parser.ExprComponentContext):
            return self.visitExprComponent(c)
        elif isinstance(c, TerminalNodeImpl):
            return ID(type_="OPTIONAL", value=c.getSymbol().text, **extract_token_info(ctx))
        else:
            raise NotImplementedError

    """
                            -----------------------------------
                                    String Functions
                            -----------------------------------
        """

    def visitStringFunctionsComponents(self, ctx: Parser.StringFunctionsComponentsContext):
        if isinstance(ctx, Parser.UnaryStringFunctionComponentContext):
            return self.visitUnaryStringFunctionComponent(ctx)
        elif isinstance(ctx, Parser.SubstrAtomComponentContext):
            return self.visitSubstrAtomComponent(ctx)
        elif isinstance(ctx, Parser.ReplaceAtomComponentContext):
            return self.visitReplaceAtomComponent(ctx)
        elif isinstance(ctx, Parser.InstrAtomComponentContext):
            return self.visitInstrAtomComponent(ctx)
        else:
            raise NotImplementedError

    def visitUnaryStringFunctionComponent(self, ctx: Parser.UnaryStringFunctionComponentContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        token = c.getSymbol()
        op_node = token.text
        operand_node = self.visitExprComponent(ctx_list[2])
        return UnaryOp(op=op_node, operand=operand_node, **extract_token_info(ctx))

    def visitSubstrAtomComponent(self, ctx: Parser.SubstrAtomComponentContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        params_nodes = []
        children_nodes = []

        token = c.getSymbol()
        childrens = [expr for expr in ctx_list if isinstance(expr, Parser.ExprComponentContext)]
        params = [
            param for param in ctx_list if isinstance(param, Parser.OptionalExprComponentContext)
        ]

        op_node = token.text
        for children in childrens:
            children_nodes.append(self.visitExprComponent(children))

        if len(params) != 0:
            for param in params:
                params_nodes.append(self.visitOptionalExprComponent(param))

        return ParamOp(
            op=op_node, children=children_nodes, params=params_nodes, **extract_token_info(ctx)
        )

    def visitReplaceAtomComponent(self, ctx: Parser.ReplaceAtomComponentContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        token = c.getSymbol()
        expressions = [
            self.visitExprComponent(expr)
            for expr in ctx_list
            if isinstance(expr, Parser.ExprComponentContext)
        ]
        params = [
            self.visitOptionalExprComponent(param)
            for param in ctx_list
            if isinstance(param, Parser.OptionalExprComponentContext)
        ]

        op_node = token.text

        children_nodes = [expressions[0]]
        params_nodes = [expressions[1]] + params

        return ParamOp(
            op=op_node, children=children_nodes, params=params_nodes, **extract_token_info(ctx)
        )

    def visitInstrAtomComponent(self, ctx: Parser.InstrAtomComponentContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        token = c.getSymbol()
        expressions = [
            self.visitExprComponent(expr)
            for expr in ctx_list
            if isinstance(expr, Parser.ExprComponentContext)
        ]
        params = [
            self.visitOptionalExprComponent(param)
            for param in ctx_list
            if isinstance(param, Parser.OptionalExprComponentContext)
        ]

        op_node = token.text

        children_nodes = [expressions[0]]
        params_nodes = [expressions[1]] + params

        return ParamOp(
            op=op_node, children=children_nodes, params=params_nodes, **extract_token_info(ctx)
        )

    """
                        -----------------------------------
                                Numeric Component Functions
                        -----------------------------------
    """

    def visitNumericFunctionsComponents(self, ctx: Parser.NumericFunctionsComponentsContext):
        if isinstance(ctx, Parser.UnaryNumericComponentContext):
            return self.visitUnaryNumericComponent(ctx)
        elif isinstance(ctx, Parser.UnaryWithOptionalNumericComponentContext):
            return self.visitUnaryWithOptionalNumericComponent(ctx)
        elif isinstance(ctx, Parser.BinaryNumericComponentContext):
            return self.visitBinaryNumericComponent(ctx)
        else:
            raise NotImplementedError

    def visitUnaryNumericComponent(self, ctx: Parser.UnaryNumericComponentContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        token = c.getSymbol()
        op_node = token.text
        operand_node = self.visitExprComponent(ctx_list[2])
        return UnaryOp(op=op_node, operand=operand_node, **extract_token_info(ctx))

    def visitUnaryWithOptionalNumericComponent(
        self, ctx: Parser.UnaryWithOptionalNumericComponentContext
    ):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        params_nodes = []
        children_nodes = []

        token = c.getSymbol()
        childrens = [expr for expr in ctx_list if isinstance(expr, Parser.ExprComponentContext)]
        params = [
            param for param in ctx_list if isinstance(param, Parser.OptionalExprComponentContext)
        ]

        op_node = token.text
        for children in childrens:
            children_nodes.append(self.visitExprComponent(children))

        if len(params) != 0:
            for param in params:
                params_nodes.append(self.visitOptionalExprComponent(param))

        return ParamOp(
            op=op_node, children=children_nodes, params=params_nodes, **extract_token_info(ctx)
        )

    def visitBinaryNumericComponent(self, ctx: Parser.BinaryNumericComponentContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        token = c.getSymbol()

        left_node = self.visitExprComponent(ctx_list[2])
        op_node = token.text
        right_node = self.visitExprComponent(ctx_list[4])
        return BinOp(left=left_node, op=op_node, right=right_node, **extract_token_info(ctx))

    """
                            -----------------------------------
                                    Time Functions
                            -----------------------------------
        """

    def visitTimeFunctionsComponents(self, ctx: Parser.TimeFunctionsComponentsContext):
        if isinstance(ctx, Parser.PeriodAtomComponentContext):
            return self.visitTimeUnaryAtomComponent(ctx)
        elif isinstance(ctx, Parser.FillTimeAtomComponentContext):
            return self.visitFillTimeAtomComponent(ctx)
        elif isinstance(ctx, Parser.FlowAtomComponentContext):
            raise SemanticError("1-1-19-7", op=ctx.op.text)
        elif isinstance(ctx, Parser.TimeShiftAtomComponentContext):
            return self.visitTimeShiftAtomComponent(ctx)
        elif isinstance(ctx, Parser.TimeAggAtomComponentContext):
            return self.visitTimeAggAtomComponent(ctx)
        elif isinstance(ctx, Parser.CurrentDateAtomComponentContext):
            return self.visitCurrentDateAtomComponent(ctx)
        elif isinstance(ctx, Parser.DateDiffAtomComponentContext):
            return self.visitDateDiffAtomComponent(ctx)
        elif isinstance(ctx, Parser.DateAddAtomComponentContext):
            return self.visitDateAddAtomComponentContext(ctx)
        elif isinstance(
            ctx,
            (
                Parser.YearAtomComponentContext,
                Parser.MonthAtomComponentContext,
                Parser.DayOfMonthAtomComponentContext,
                Parser.DayOfYearAtomComponentContext,
                Parser.DayToYearAtomComponentContext,
                Parser.DayToMonthAtomComponentContext,
                Parser.YearToDayAtomComponentContext,
                Parser.MonthToDayAtomComponentContext,
            ),
        ):
            return self.visitTimeUnaryAtomComponent(ctx)
        else:
            raise NotImplementedError

    def visitTimeUnaryAtomComponent(self, ctx: Parser.PeriodAtomComponentContext):
        """
        periodExpr: PERIOD_INDICATOR '(' expr? ')' ;
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        op = c.getSymbol().text
        operand_node = [
            self.visitExprComponent(operand)
            for operand in ctx_list
            if isinstance(operand, Parser.ExprComponentContext)
        ]

        if len(operand_node) == 0:
            # AST_ASTCONSTRUCTOR.15
            raise NotImplementedError

        return UnaryOp(op=op, operand=operand_node[0], **extract_token_info(ctx))

    def visitTimeShiftAtomComponent(self, ctx: Parser.TimeShiftAtomComponentContext):
        """
        timeShiftExpr: TIMESHIFT '(' expr ',' INTEGER_CONSTANT ')' ;
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        op = c.getSymbol().text
        left_node = self.visitExprComponent(ctx_list[2])
        right_node = Constant(
            type_="INTEGER_CONSTANT",
            value=int(ctx_list[4].getSymbol().text),
            **extract_token_info(ctx),
        )

        return BinOp(left=left_node, op=op, right=right_node, **extract_token_info(ctx))

    def visitFillTimeAtomComponent(self, ctx: Parser.FillTimeAtomComponentContext):
        """
        timeSeriesExpr: FILL_TIME_SERIES '(' expr (',' (SINGLE|ALL))? ')' ;
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        op = c.getSymbol().text
        children_node = [self.visitExprComponent(ctx_list[2])]

        if len(ctx_list) > 4:
            param_constant_node = [
                ParamConstant(
                    type_="PARAM_TIMESERIES",
                    value=ctx_list[4].getSymbol().text,
                    **extract_token_info(ctx),
                )
            ]
        else:
            param_constant_node = []

        return ParamOp(
            op=op, children=children_node, params=param_constant_node, **extract_token_info(ctx)
        )

    def visitTimeAggAtomComponent(self, ctx: Parser.TimeAggAtomComponentContext):
        """
        TIME_AGG LPAREN periodIndTo=STRING_CONSTANT (COMMA periodIndFrom=(STRING_CONSTANT| OPTIONAL ))?
        (COMMA op=optionalExprComponent)? (COMMA (FIRST|LAST))? RPAREN    # timeAggAtomComponent;
        """  # noqa E501
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        op = c.getSymbol().text
        period_to = str(ctx.periodIndTo.text)[1:-1]
        period_from = None

        if ctx.periodIndFrom is not None and ctx.periodIndFrom.type != Parser.OPTIONAL:
            # raise SemanticError("periodIndFrom is not allowed in Time_agg")
            period_from = str(ctx.periodIndFrom.text)[1:-1]

        conf = [
            str_.getSymbol().text
            for str_ in ctx_list
            if isinstance(str_, TerminalNodeImpl)
            and str_.getSymbol().type in [Parser.FIRST, Parser.LAST]
        ]

        conf = None if len(conf) == 0 else conf[0]

        if ctx.op is not None:
            operand_node = self.visitOptionalExprComponent(ctx.op)
            if isinstance(operand_node, ID):
                operand_node = None
            elif isinstance(operand_node, Identifier):
                operand_node = VarID(
                    value=operand_node.value, **extract_token_info(ctx.op)
                )  # Converting Identifier to VarID
        else:
            operand_node = None

        return TimeAggregation(
            op=op,
            operand=operand_node,
            period_to=period_to,
            period_from=period_from,
            conf=conf,
            **extract_token_info(ctx),
        )

    def visitCurrentDateAtomComponent(self, ctx: Parser.CurrentDateAtomComponentContext):
        c = list(ctx.getChildren())[0]
        return MulOp(op=c.getSymbol().text, children=[], **extract_token_info(ctx))

    def visitDateDiffAtomComponent(self, ctx: Parser.TimeShiftAtomComponentContext):
        """ """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        op = c.getSymbol().text
        left_node = self.visitExprComponent(ctx_list[2])
        right_node = self.visitExprComponent(ctx_list[4])

        return BinOp(left=left_node, op=op, right=right_node, **extract_token_info(ctx))

    def visitDateAddAtomComponentContext(self, ctx: Parser.DateAddAtomComponentContext):
        """ """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        op = c.getSymbol().text
        children_node = [self.visitExprComponent(ctx_list[2])]

        param_constant_node = []

        if len(ctx_list) > 4:
            param_constant_node = [self.visitExprComponent(ctx_list[4])]
            if len(ctx_list) > 6:
                param_constant_node.append(self.visitExprComponent(ctx_list[6]))

        return ParamOp(
            op=op, children=children_node, params=param_constant_node, **extract_token_info(ctx)
        )

    """
                            -----------------------------------
                                    Conditional Functions
                            -----------------------------------
    """

    def visitConditionalFunctionsComponents(
        self, ctx: Parser.ConditionalFunctionsComponentsContext
    ):
        if isinstance(ctx, Parser.NvlAtomComponentContext):
            return self.visitNvlAtomComponent(ctx)
        else:
            raise NotImplementedError

    def visitNvlAtomComponent(self, ctx: Parser.NvlAtomComponentContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        token = c.getSymbol()

        left_node = self.visitExprComponent(ctx_list[2])
        op_node = token.text
        right_node = self.visitExprComponent(ctx_list[4])
        return BinOp(left=left_node, op=op_node, right=right_node, **extract_token_info(ctx))

    """
                        -----------------------------------
                            Comparison Components Functions
                        -----------------------------------
    """

    def visitComparisonFunctionsComponents(self, ctx: Parser.ComparisonFunctionsComponentsContext):
        if isinstance(ctx, Parser.BetweenAtomComponentContext):
            return self.visitBetweenAtomComponent(ctx)
        elif isinstance(ctx, Parser.CharsetMatchAtomComponentContext):
            return self.visitCharsetMatchAtomComponent(ctx)
        elif isinstance(ctx, Parser.IsNullAtomComponentContext):
            return self.visitIsNullAtomComponent(ctx)
        else:
            raise NotImplementedError

    def visitBetweenAtomComponent(self, ctx: Parser.BetweenAtomComponentContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        children_nodes = []

        token = c.getSymbol()
        childrens = [expr for expr in ctx_list if isinstance(expr, Parser.ExprComponentContext)]

        op_node = token.text
        for children in childrens:
            children_nodes.append(self.visitExprComponent(children))

        return MulOp(op=op_node, children=children_nodes, **extract_token_info(ctx))

    def visitCharsetMatchAtomComponent(self, ctx: Parser.CharsetMatchAtomComponentContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]
        token = c.getSymbol()

        left_node = self.visitExprComponent(ctx_list[2])
        op_node = token.text
        right_node = self.visitExprComponent(ctx_list[4])
        return BinOp(left=left_node, op=op_node, right=right_node, **extract_token_info(ctx))

    def visitIsNullAtomComponent(self, ctx: Parser.IsNullAtomComponentContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]
        token = c.getSymbol()
        op_node = token.text
        operand_node = self.visitExprComponent(ctx_list[2])
        return UnaryOp(op=op_node, operand=operand_node, **extract_token_info(ctx))

    """
                            -----------------------------------
                                Aggregate Components Functions
                            -----------------------------------
        """

    def visitAggregateFunctionsComponents(self, ctx: Parser.AggregateFunctionsComponentsContext):
        if isinstance(ctx, Parser.AggrCompContext):
            return self.visitAggrComp(ctx)
        elif isinstance(ctx, Parser.CountAggrCompContext):
            return self.visitCountAggrComp(ctx)
        else:
            raise NotImplementedError

    def visitAggrComp(self, ctx: Parser.AggrCompContext):
        ctx_list = list(ctx.getChildren())
        op_node = ctx_list[0].getSymbol().text
        operand_node = self.visitExprComponent(ctx_list[2])
        return Aggregation(op=op_node, operand=operand_node, **extract_token_info(ctx))

    def visitCountAggrComp(self, ctx: Parser.CountAggrCompContext):
        ctx_list = list(ctx.getChildren())
        op_node = ctx_list[0].getSymbol().text
        return Aggregation(op=op_node, **extract_token_info(ctx))

    """
                                -----------------------------------
                                    Analytic Components Functions
                                -----------------------------------
    """

    def visitAnalyticFunctionsComponents(self, ctx: Parser.AnalyticFunctionsComponentsContext):
        if isinstance(ctx, Parser.AnSimpleFunctionComponentContext):
            return self.visitAnSimpleFunctionComponent(ctx)
        elif isinstance(ctx, Parser.LagOrLeadAnComponentContext):
            return self.visitLagOrLeadAnComponent(ctx)
        elif isinstance(ctx, Parser.RankAnComponentContext):
            return self.visitRankAnComponent(ctx)
        elif isinstance(ctx, Parser.RatioToReportAnComponentContext):
            return self.visitRatioToReportAnComponent(ctx)
        else:
            raise NotImplementedError

    def visitAnSimpleFunctionComponent(self, ctx: Parser.AnSimpleFunctionComponentContext):
        ctx_list = list(ctx.getChildren())

        params = None
        partition_by = None
        order_by = None

        op_node = ctx_list[0].getSymbol().text
        operand = self.visitExprComponent(ctx_list[2])

        for c in ctx_list[5:-2]:
            if isinstance(c, Parser.PartitionByClauseContext):
                partition_by = Terminals().visitPartitionByClause(c)
                continue
            elif isinstance(c, Parser.OrderByClauseContext):
                order_by = Terminals().visitOrderByClause(c)
                continue
            elif isinstance(c, Parser.WindowingClauseContext):
                params = Terminals().visitWindowingClause(c)
                continue
            else:
                raise NotImplementedError

        return Analytic(
            op=op_node,
            operand=operand,
            partition_by=partition_by,
            order_by=order_by,
            window=params,
            **extract_token_info(ctx),
        )

    def visitLagOrLeadAnComponent(self, ctx: Parser.LagOrLeadAnComponentContext):
        ctx_list = list(ctx.getChildren())

        params = None
        partition_by = None
        order_by = None

        op_node = ctx_list[0].getSymbol().text
        operand = self.visitExprComponent(ctx_list[2])

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

        return Analytic(
            op=op_node,
            operand=operand,
            partition_by=partition_by,
            order_by=order_by,
            params=params,
            **extract_token_info(ctx),
        )

    def visitRankAnComponent(self, ctx: Parser.RankAnComponentContext):
        ctx_list = list(ctx.getChildren())

        partition_by = None
        order_by = None

        op_node = ctx_list[0].getSymbol().text

        for c in ctx_list[4:-2]:
            if isinstance(c, Parser.PartitionByClauseContext):
                partition_by = Terminals().visitPartitionByClause(c)
                continue
            elif isinstance(c, Parser.OrderByClauseContext):
                order_by = Terminals().visitOrderByClause(c)
                continue

        return Analytic(
            op=op_node,
            operand=None,
            partition_by=partition_by,
            order_by=order_by,
            window=None,
            **extract_token_info(ctx),
        )

    def visitRatioToReportAnComponent(self, ctx: Parser.RatioToReportAnComponentContext):
        ctx_list = list(ctx.getChildren())

        params = None
        order_by = None

        op_node = ctx_list[0].getSymbol().text
        operand = self.visitExprComponent(ctx_list[2])

        partition_by = Terminals().visitPartitionByClause(ctx_list[5])

        return Analytic(
            op=op_node,
            operand=operand,
            partition_by=partition_by,
            order_by=order_by,
            window=params,
            **extract_token_info(ctx),
        )
