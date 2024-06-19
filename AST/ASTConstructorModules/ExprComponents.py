from antlr4.tree.Tree import TerminalNodeImpl

from AST import If, BinOp, UnaryOp, ID, ParamOp, MulOp, Constant, ParamConstant, TimeAggregation, \
    Identifier, EvalOp, Types, VarID, Analytic, AggregationComp
from AST.ASTConstructorModules.handlers.terminals import terminal_handler
from AST.VtlVisitor import VtlVisitor
from Grammar.parser import Parser


class ExprComp(VtlVisitor):
    """______________________________________________________________________________________


                                    ExprComponent Definition.

        _______________________________________________________________________________________"""

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
        """
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

        # constant
        elif isinstance(ctx, Parser.ConstantExprCompContext):
            return terminal_handler.visitConstant(c)

        # componentID
        # TODO Changed to pass more tests. Original code: return terminal_handler.visitComponentID(c)
        elif isinstance(ctx, Parser.CompIdContext):
            if len(c.children) > 1:
                return terminal_handler.visitComponentID(c)
            token = c.children[0].getSymbol()
            # check token text
            has_scaped_char = token.text.find("\'") != -1
            if has_scaped_char:
                token.text = str(token.text.replace("\'", ""))
            var_id_node = VarID(token.text)
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

        bin_op_node = BinOp(left_node, op, right_node)

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
            right_node = terminal_handler.visitLists(ctx_list[2])
        elif isinstance(ctx_list[2], Parser.ValueDomainIDContext):
            right_node = terminal_handler.visitValueDomainID(ctx_list[2])
        else:
            raise NotImplementedError
        bin_op_node = BinOp(left_node, op, right_node)

        return bin_op_node

    def visitBooleanExprComp(self, ctx: Parser.BooleanExprContext):
        return self.bin_op_creator_comp(ctx)

    def visitParenthesisExprComp(self, ctx: Parser.ParenthesisExprContext):
        return self.visitExprComponent(list(ctx.getChildren())[1])

    def visitUnaryExprComp(self, ctx: Parser.UnaryExprContext):
        c_list = list(ctx.getChildren())
        op = c_list[0].getSymbol().text
        right = self.visitExprComponent(c_list[1])

        return UnaryOp(op, right)

    def visitIfExprComp(self, ctx: Parser.IfExprCompContext):
        ctx_list = list(ctx.getChildren())

        condition_node = self.visitExprComponent(ctx_list[1])
        then_op_node = self.visitExprComponent(ctx_list[3])
        else_op_node = self.visitExprComponent(ctx_list[5])

        if_node = If(condition_node, then_op_node, else_op_node)

        return if_node

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
            return ID('OPTIONAL', opt)

    """______________________________________________________________________________________


                                        FunctionsComponents Definition.

            _______________________________________________________________________________________"""

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
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        op = terminal_handler.visitOperatorID(c)
        operator_node = Identifier(op, kind='OperatorID')
        param_nodes = [self.visitParameterComponent(element) for element in ctx_list if
                       isinstance(element, Parser.ParameterComponentContext)]

        return ParamOp(op=op, children=[operator_node], params=param_nodes)

    def visitEvalAtomComponent(self, ctx: Parser.EvalAtomComponentContext):
        """
            | EVAL LPAREN routineName LPAREN (componentID|scalarItem)? (COMMA (componentID|scalarItem))* RPAREN (LANGUAGE STRING_CONSTANT)? (RETURNS outputParameterTypeComponent)? RPAREN      # evalAtomComponent
        """
        ctx_list = list(ctx.getChildren())

        routine_name = terminal_handler.visitRoutineName(ctx_list[2])

        # Think of a way to maintain the order, for now its not necessary.
        var_ids_nodes = [terminal_handler.visitVarID(varID) for varID in ctx_list if
                         isinstance(varID, Parser.VarIDContext)]
        constant_nodes = [terminal_handler.visitScalarItem(scalar) for scalar in ctx_list if
                          isinstance(scalar, Parser.ScalarItemContext)]
        children_nodes = var_ids_nodes + constant_nodes

        # Reference manual says it is mandatory.
        language_name = [language for language in ctx_list if
                         isinstance(language, TerminalNodeImpl) and language.getSymbol().type == Parser.STRING_CONSTANT]
        if len(language_name) == 0:
            # AST_ASTCONSTRUCTOR.12
            raise SemanticError("1-4-2-1", option='language')
        # Reference manual says it is mandatory.
        output_node = [terminal_handler.visitOutputParameterTypeComponent(output) for output in ctx_list if
                       isinstance(output, Parser.OutputParameterTypeComponentContext)]
        if len(output_node) == 0:
            # AST_ASTCONSTRUCTOR.13
            raise SemanticError("1-4-2-1", option='output')

        return EvalOp(name=routine_name, children=children_nodes, output=output_node[0],
                      language=language_name[0].getSymbol().text)

    def visitCastExprComponent(self, ctx: Parser.CastExprComponentContext):
        """
            | CAST LPAREN exprComponent COMMA (basicScalarType|valueDomainName) (COMMA STRING_CONSTANT)? RPAREN         # castExprComponent
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        token = c.getSymbol()

        op = token.text
        expr_node = [self.visitExprComponent(expr) for expr in ctx_list if
                     isinstance(expr, Parser.ExprComponentContext)]
        basic_scalar_type = [terminal_handler.visitBasicScalarType(type_) for type_ in ctx_list if
                             isinstance(type_, Parser.BasicScalarTypeContext)]

        [terminal_handler.visitValueDomainName(valueD) for valueD in ctx_list if
         isinstance(valueD, Parser.ValueDomainNameContext)]

        if len(ctx_list) > 6:
            param_node = [ParamConstant('PARAM_CAST', str_.symbol.text.strip('"')) for str_ in ctx_list if
                          isinstance(str_, TerminalNodeImpl) and str_.getSymbol().type == Parser.STRING_CONSTANT]
        else:
            param_node = []

        if len(basic_scalar_type) == 1:
            basic_scalar_type_node = [Types(kind='Scalar', type_=basic_scalar_type[0], constraints=[], nullable=None)]
            children_nodes = expr_node + basic_scalar_type_node

            return ParamOp(op=op, children=children_nodes, params=param_node)

        else:
            # AST_ASTCONSTRUCTOR.14
            raise NotImplementedError

    def visitParameterComponent(self, ctx: Parser.ParameterComponentContext):

        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        if isinstance(c, Parser.ExprComponentContext):
            return self.visitExprComponent(c)
        elif isinstance(c, TerminalNodeImpl):
            return ID('OPTIONAL', c.getSymbol().text)
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
        return UnaryOp(op_node, operand_node)

    def visitSubstrAtomComponent(self, ctx: Parser.SubstrAtomComponentContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        params_nodes = []
        children_nodes = []

        token = c.getSymbol()
        childrens = [expr for expr in ctx_list if isinstance(expr, Parser.ExprComponentContext)]
        params = [param for param in ctx_list if isinstance(param, Parser.OptionalExprComponentContext)]

        op_node = token.text
        for children in childrens:
            children_nodes.append(self.visitExprComponent(children))

        if len(params) != 0:
            for param in params:
                params_nodes.append(self.visitOptionalExprComponent(param))

        return ParamOp(op_node, children_nodes, params_nodes)

    def visitReplaceAtomComponent(self, ctx: Parser.ReplaceAtomComponentContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        token = c.getSymbol()
        expressions = [self.visitExprComponent(expr) for expr in ctx_list if
                       isinstance(expr, Parser.ExprComponentContext)]
        params = [self.visitOptionalExprComponent(param) for param in ctx_list if
                  isinstance(param, Parser.OptionalExprComponentContext)]

        op_node = token.text

        children_nodes = [expressions[0]]
        params_nodes = [expressions[1]] + params

        return ParamOp(op_node, children_nodes, params_nodes)

    def visitInstrAtomComponent(self, ctx: Parser.InstrAtomComponentContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        token = c.getSymbol()
        expressions = [self.visitExprComponent(expr) for expr in ctx_list if
                       isinstance(expr, Parser.ExprComponentContext)]
        params = [self.visitOptionalExprComponent(param) for param in ctx_list if
                  isinstance(param, Parser.OptionalExprComponentContext)]

        op_node = token.text

        children_nodes = [expressions[0]]
        params_nodes = [expressions[1]] + params

        return ParamOp(op_node, children_nodes, params_nodes)

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
        return UnaryOp(op_node, operand_node)

    def visitUnaryWithOptionalNumericComponent(self, ctx: Parser.UnaryWithOptionalNumericComponentContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        params_nodes = []
        children_nodes = []

        token = c.getSymbol()
        childrens = [expr for expr in ctx_list if isinstance(expr, Parser.ExprComponentContext)]
        params = [param for param in ctx_list if isinstance(param, Parser.OptionalExprComponentContext)]

        op_node = token.text
        for children in childrens:
            children_nodes.append(self.visitExprComponent(children))

        if len(params) != 0:
            for param in params:
                params_nodes.append(self.visitOptionalExprComponent(param))

        return ParamOp(op_node, children_nodes, params_nodes)

    def visitBinaryNumericComponent(self, ctx: Parser.BinaryNumericComponentContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        token = c.getSymbol()

        left_node = self.visitExprComponent(ctx_list[2])
        op_node = token.text
        right_node = self.visitExprComponent(ctx_list[4])
        return BinOp(left_node, op_node, right_node)

    """
                            -----------------------------------
                                    Time Functions             
                            -----------------------------------
        """

    def visitTimeFunctionsComponents(self, ctx: Parser.TimeFunctionsComponentsContext):
        if isinstance(ctx, Parser.PeriodAtomComponentContext):
            return self.visitPeriodAtomComponent(ctx)
        elif isinstance(ctx, Parser.FillTimeAtomComponentContext):
            return self.visitFillTimeAtomComponent(ctx)
        elif isinstance(ctx, Parser.FlowAtomComponentContext):
            raise SemanticError("1-1-19-7",op=ctx.op.text)
        elif isinstance(ctx, Parser.TimeShiftAtomComponentContext):
            return self.visitTimeShiftAtomComponent(ctx)
        elif isinstance(ctx, Parser.TimeAggAtomComponentContext):
            return self.visitTimeAggAtomComponent(ctx)
        elif isinstance(ctx, Parser.CurrentDateAtomComponentContext):
            return self.visitCurrentDateAtomComponent(ctx)
        else:
            raise NotImplementedError

    def visitPeriodAtomComponent(self, ctx: Parser.PeriodAtomComponentContext):
        """
        periodExpr: PERIOD_INDICATOR '(' expr? ')' ;
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        op = c.getSymbol().text
        operand_node = [self.visitExprComponent(operand) for operand in ctx_list if
                        isinstance(operand, Parser.ExprComponentContext)]

        if len(operand_node) == 0:
            # AST_ASTCONSTRUCTOR.15
            raise NotImplementedError

        return UnaryOp(op=op, operand=operand_node[0])

    def visitTimeShiftAtomComponent(self, ctx: Parser.TimeShiftAtomComponentContext):
        """
        timeShiftExpr: TIMESHIFT '(' expr ',' INTEGER_CONSTANT ')' ;
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        op = c.getSymbol().text
        left_node = self.visitExprComponent(ctx_list[2])
        right_node = Constant('INTEGER_CONSTANT', int(ctx_list[4].getSymbol().text))

        return BinOp(left=left_node, op=op, right=right_node)

    def visitFillTimeAtomComponent(self, ctx: Parser.FillTimeAtomComponentContext):
        """
        timeSeriesExpr: FILL_TIME_SERIES '(' expr (',' (SINGLE|ALL))? ')' ;
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        op = c.getSymbol().text
        children_node = [self.visitExprComponent(ctx_list[2])]

        if len(ctx_list) > 4:
            param_constant_node = [ParamConstant('PARAM_TIMESERIES', ctx_list[4].getSymbol().text)]
        else:
            param_constant_node = []

        return ParamOp(op=op, children=children_node, params=param_constant_node)

    def visitTimeAggAtomComponent(self, ctx: Parser.TimeAggAtomComponentContext):
        """
        TIME_AGG LPAREN periodIndTo=STRING_CONSTANT (COMMA periodIndFrom=(STRING_CONSTANT| OPTIONAL ))?
        (COMMA op=optionalExprComponent)? (COMMA (FIRST|LAST))? RPAREN    # timeAggAtomComponent;
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        op = c.getSymbol().text
        param_node = [Constant('STRING_CONSTANT', str(ctx.periodIndTo.text)[1:-1])]

        if ctx.periodIndFrom is not None and ctx.periodIndFrom.type != Parser.OPTIONAL:
            # raise SemanticError("periodIndFrom is not allowed in Time_agg")
            periodIndFrom_node = Constant('STRING_CONSTANT', str(ctx.periodIndFrom.text)[1:-1])
            param_node.append(periodIndFrom_node)

        conf = [str_.getSymbol().text for str_ in ctx_list if
                isinstance(str_, TerminalNodeImpl) and str_.getSymbol().type in [Parser.FIRST, Parser.LAST]]

        if len(conf) == 0:
            conf = None
        else:
            conf = conf[0]

        if ctx.op is not None:
            operand_node = self.visitOptionalExprComponent(ctx.op)
            if isinstance(operand_node, ID):
                operand_node = None
        else:
            operand_node = None

        if operand_node is None:
            # AST_ASTCONSTRUCTOR.17
            raise SemanticError("1-4-2-2")
        elif isinstance(operand_node, VarID):
            operand_node = Identifier(operand_node.value, 'ComponentID')
        return TimeAggregation(op=op, operand=operand_node, params=param_node, conf=conf)

    def visitCurrentDateAtomComponent(self, ctx: Parser.CurrentDateAtomComponentContext):
        c = list(ctx.getChildren())[0]
        return MulOp(op=c.getSymbol().text, children=[])

    """
                            -----------------------------------
                                    Conditional Functions             
                            -----------------------------------
    """

    def visitConditionalFunctionsComponents(self, ctx: Parser.ConditionalFunctionsComponentsContext):
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
        return BinOp(left_node, op_node, right_node)

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

        return MulOp(op_node, children_nodes)

    def visitCharsetMatchAtomComponent(self, ctx: Parser.CharsetMatchAtomComponentContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]
        token = c.getSymbol()

        left_node = self.visitExprComponent(ctx_list[2])
        op_node = token.text
        right_node = self.visitExprComponent(ctx_list[4])
        return BinOp(left_node, op_node, right_node)

    def visitIsNullAtomComponent(self, ctx: Parser.IsNullAtomComponentContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]
        token = c.getSymbol()
        op_node = token.text
        operand_node = self.visitExprComponent(ctx_list[2])
        return UnaryOp(op_node, operand_node)

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
        return AggregationComp(op_node, operand_node)

    def visitCountAggrComp(self, ctx: Parser.CountAggrCompContext):
        ctx_list = list(ctx.getChildren())
        op_node = ctx_list[0].getSymbol().text
        param_constant = ParamConstant('Null', None)
        return AggregationComp(op_node, param_constant)

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
                partition_by = terminal_handler.visitPartitionByClause(c)
                continue
            elif isinstance(c, Parser.OrderByClauseContext):
                order_by = terminal_handler.visitOrderByClause(c)
                continue
            elif isinstance(c, Parser.WindowingClauseContext):
                params = terminal_handler.visitWindowingClause(c)
                continue
            else:
                raise NotImplementedError

        return Analytic(op=op_node, operand=operand, partition_by=partition_by, order_by=order_by, params=params)

    def visitLagOrLeadAnComponent(self, ctx: Parser.LagOrLeadAnComponentContext):
        ctx_list = list(ctx.getChildren())

        params = None
        partition_by = None
        order_by = None

        op_node = ctx_list[0].getSymbol().text
        operand = self.visitExprComponent(ctx_list[2])

        for c in ctx_list[4:-2]:
            if isinstance(c, Parser.PartitionByClauseContext):
                partition_by = terminal_handler.visitPartitionByClause(c)
                continue
            elif isinstance(c, Parser.OrderByClauseContext):
                order_by = terminal_handler.visitOrderByClause(c)
                continue
            elif isinstance(c, Parser.SignedIntegerContext) or isinstance(c, Parser.ScalarItemContext):
                if params is None:
                    params = []
                if isinstance(c, Parser.SignedIntegerContext):
                    params.append(terminal_handler.visitSignedInteger(c))
                else:
                    params.append(terminal_handler.visitScalarItem(c))
                continue

        return Analytic(op=op_node, operand=operand, partition_by=partition_by, order_by=order_by, params=params)

    def visitRankAnComponent(self, ctx: Parser.RankAnComponentContext):
        ctx_list = list(ctx.getChildren())

        partition_by = None
        order_by = None

        op_node = ctx_list[0].getSymbol().text

        for c in ctx_list[4:-2]:
            if isinstance(c, Parser.PartitionByClauseContext):
                partition_by = terminal_handler.visitPartitionByClause(c)
                continue
            elif isinstance(c, Parser.OrderByClauseContext):
                order_by = terminal_handler.visitOrderByClause(c)
                continue

        return Analytic(op=op_node, operand=None, partition_by=partition_by, order_by=order_by, params=None)

    def visitRatioToReportAnComponent(self, ctx: Parser.RatioToReportAnComponentContext):
        ctx_list = list(ctx.getChildren())

        params = None
        order_by = None

        op_node = ctx_list[0].getSymbol().text
        operand = self.visitExprComponent(ctx_list[2])

        partition_by = terminal_handler.visitPartitionByClause(ctx_list[5])

        return Analytic(op=op_node, operand=operand, partition_by=partition_by, order_by=order_by, params=params)
