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
from vtlengine.AST.Grammar._cpp_parser import vtl_cpp_parser
from vtlengine.AST.Grammar._cpp_parser._rule_constants import RC
from vtlengine.Exceptions import SemanticError


class ExprComp:
    """______________________________________________________________________________________


                                ExprComponent Definition.

    _______________________________________________________________________________________
    """

    def visitExprComponent(self, ctx):  # type: ignore[no-untyped-def]
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
        ctx_list = ctx.children
        c = ctx_list[0]

        if ctx.ctx_id == RC.PARENTHESIS_EXPR_COMP:
            return self.visitParenthesisExprComp(ctx)

        # functions
        elif ctx.ctx_id == RC.FUNCTIONS_EXPRESSION_COMP:
            return self.visitFunctionsComponents(c)

        # op=(PLUS|MINUS|NOT) right=expr # unary expression
        elif ctx.ctx_id == RC.UNARY_EXPR_COMP:
            return self.visitUnaryExprComp(ctx)

        # | left=expr op=(MUL|DIV) right=expr               # arithmeticExpr
        elif ctx.ctx_id == RC.ARITHMETIC_EXPR_COMP:
            return self.visitArithmeticExprComp(ctx)

        # | left=expr op=(PLUS|MINUS|CONCAT) right=expr     # arithmeticExprOrConcat
        elif ctx.ctx_id == RC.ARITHMETIC_EXPR_OR_CONCAT_COMP:
            return self.visitArithmeticExprOrConcatComp(ctx)

        # | left=expr op=comparisonOperand  right=expr      # comparisonExpr
        elif ctx.ctx_id == RC.COMPARISON_EXPR_COMP:
            return self.visitComparisonExprComp(ctx)

        # | left=expr op=(IN|NOT_IN)(lists|valueDomainID)   # inNotInExpr
        elif ctx.ctx_id == RC.IN_NOT_IN_EXPR_COMP:
            return self.visitInNotInExprComp(ctx)

        # | left=expr op=AND right=expr                                           # booleanExpr
        # | left=expr op=(OR|XOR) right=expr
        elif ctx.ctx_id == RC.BOOLEAN_EXPR_COMP:
            return self.visitBooleanExprComp(ctx)

        # IF  conditionalExpr=expr  THEN thenExpr=expr ELSE elseExpr=expr       # ifExpr
        elif ctx.ctx_id == RC.IF_EXPR_COMP:
            return self.visitIfExprComp(ctx)

        # CASE WHEN conditionalExpr=expr THEN thenExpr=expr ELSE elseExpr=expr END # caseExpr
        elif ctx.ctx_id == RC.CASE_EXPR_COMP:
            return self.visitCaseExprComp(ctx)

        # constant
        elif ctx.ctx_id == RC.CONSTANT_EXPR_COMP:
            return Terminals().visitConstant(c)

        # componentID
        # TODO Changed to pass more tests. Original code: return Terminals().visitComponentID(c)
        elif ctx.ctx_id == RC.COMP_ID:
            if len(c.children) > 1:
                return Terminals().visitComponentID(c)
            token = c.children[0]
            # check token text
            token_text = token.text
            has_scaped_char = token_text.find("'") != -1
            if has_scaped_char:
                token_text = str(token_text.replace("'", ""))
            var_id_node = VarID(value=token_text, **extract_token_info(ctx))
            return var_id_node

        else:
            # AST_ASTCONSTRUCTOR.3
            raise NotImplementedError

    def bin_op_creator_comp(self, ctx):  # type: ignore[no-untyped-def]
        ctx_list = ctx.children
        left_node = self.visitExprComponent(ctx_list[0])
        if not ctx_list[1].is_terminal and ctx_list[1].ctx_id == RC.COMPARISON_OPERAND:
            op = ctx_list[1].children[0].text
        else:
            op = ctx_list[1].text
        right_node = self.visitExprComponent(ctx_list[2])

        bin_op_node = BinOp(left=left_node, op=op, right=right_node, **extract_token_info(ctx))

        return bin_op_node

    def visitArithmeticExprComp(self, ctx):  # type: ignore[no-untyped-def]
        return self.bin_op_creator_comp(ctx)

    def visitArithmeticExprOrConcatComp(self, ctx):  # type: ignore[no-untyped-def]
        return self.bin_op_creator_comp(ctx)

    def visitComparisonExprComp(self, ctx):  # type: ignore[no-untyped-def]
        return self.bin_op_creator_comp(ctx)

    def visitInNotInExprComp(self, ctx):  # type: ignore[no-untyped-def]
        ctx_list = ctx.children
        left_node = self.visitExprComponent(ctx_list[0])
        op = ctx_list[1].text

        if not ctx_list[2].is_terminal and ctx_list[2].ctx_id == RC.LISTS:
            right_node = Terminals().visitLists(ctx_list[2])
        elif not ctx_list[2].is_terminal and ctx_list[2].ctx_id == RC.VALUE_DOMAIN_ID:
            right_node = Terminals().visitValueDomainID(ctx_list[2])
        else:
            raise NotImplementedError
        bin_op_node = BinOp(left=left_node, op=op, right=right_node, **extract_token_info(ctx))

        return bin_op_node

    def visitBooleanExprComp(self, ctx):  # type: ignore[no-untyped-def]
        return self.bin_op_creator_comp(ctx)

    def visitParenthesisExprComp(self, ctx):  # type: ignore[no-untyped-def]
        operand = self.visitExprComponent(ctx.children[1])
        return ParFunction(operand=operand, **extract_token_info(ctx))

    def visitUnaryExprComp(self, ctx):  # type: ignore[no-untyped-def]
        c_list = ctx.children
        op = c_list[0].text
        right = self.visitExprComponent(c_list[1])

        return UnaryOp(op=op, operand=right, **extract_token_info(ctx))

    def visitIfExprComp(self, ctx):  # type: ignore[no-untyped-def]
        ctx_list = ctx.children

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

    def visitCaseExprComp(self, ctx):  # type: ignore[no-untyped-def]
        ctx_list = ctx.children

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

    def visitOptionalExprComponent(self, ctx):  # type: ignore[no-untyped-def]
        """
        optionalExpr: expr
                    | OPTIONAL ;
        """
        ctx_list = ctx.children
        c = ctx_list[0]

        if not c.is_terminal and c.rule_index == 3:
            return self.visitExprComponent(c)

        elif c.is_terminal:
            opt = c.text
            return ID(type_="OPTIONAL", value=opt, **extract_token_info(ctx))

    """____________________________________________________________________________________


                                    FunctionsComponents Definition.

      _____________________________________________________________________________________"""

    def visitFunctionsComponents(self, ctx):  # type: ignore[no-untyped-def]
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
        ctx_list = ctx.children
        c = ctx_list[0]

        if ctx.ctx_id == RC.GENERIC_FUNCTIONS_COMPONENTS:
            return self.visitGenericFunctionsComponents(c)

        elif ctx.ctx_id == RC.STRING_FUNCTIONS_COMPONENTS:
            return self.visitStringFunctionsComponents(c)

        elif ctx.ctx_id == RC.NUMERIC_FUNCTIONS_COMPONENTS:
            return self.visitNumericFunctionsComponents(c)

        elif ctx.ctx_id == RC.COMPARISON_FUNCTIONS_COMPONENTS:
            return self.visitComparisonFunctionsComponents(c)

        elif ctx.ctx_id == RC.TIME_FUNCTIONS_COMPONENTS:
            return self.visitTimeFunctionsComponents(c)

        elif ctx.ctx_id == RC.CONDITIONAL_FUNCTIONS_COMPONENTS:
            return self.visitConditionalFunctionsComponents(c)

        elif ctx.ctx_id == RC.AGGREGATE_FUNCTIONS_COMPONENTS:
            return self.visitAggregateFunctionsComponents(c)

        elif ctx.ctx_id == RC.ANALYTIC_FUNCTIONS_COMPONENTS:
            return self.visitAnalyticFunctionsComponents(c)
        else:
            raise NotImplementedError

    """
                                -----------------------------------
                                    Generic Functions Components
                                -----------------------------------
    """

    def visitGenericFunctionsComponents(self, ctx):  # type: ignore[no-untyped-def]
        if ctx.ctx_id == RC.CALL_COMPONENT:
            return self.visitCallComponent(ctx)
        elif ctx.ctx_id == RC.EVAL_ATOM_COMPONENT:
            return self.visitEvalAtomComponent(ctx)
        elif ctx.ctx_id == RC.CAST_EXPR_COMPONENT:
            return self.visitCastExprComponent(ctx)
        else:
            raise NotImplementedError

    def visitCallComponent(self, ctx):  # type: ignore[no-untyped-def]
        """
        callFunction: operatorID LPAREN (parameterComponent (COMMA parameterComponent)*)? RPAREN    # callComponent
        """  # noqa E501
        ctx_list = ctx.children
        c = ctx_list[0]

        op = Terminals().visitOperatorID(c)
        param_nodes = [
            self.visitParameterComponent(element)
            for element in ctx_list
            if not element.is_terminal and element.ctx_id == RC.PARAMETER_COMPONENT
        ]

        return UDOCall(op=op, params=param_nodes, **extract_token_info(ctx))

    def visitEvalAtomComponent(self, ctx):  # type: ignore[no-untyped-def]
        """
        | EVAL LPAREN routineName LPAREN (componentID|scalarItem)? (COMMA (componentID|scalarItem))* RPAREN (LANGUAGE STRING_CONSTANT)? (RETURNS outputParameterTypeComponent)? RPAREN      # evalAtomComponent
        """  # noqa E501
        ctx_list = ctx.children

        routine_name = Terminals().visitRoutineName(ctx_list[2])

        # Think of a way to maintain the order, for now its not necessary.
        var_ids_nodes = [
            Terminals().visitVarID(varID)
            for varID in ctx_list
            if not varID.is_terminal and varID.ctx_id == RC.VAR_ID
        ]
        constant_nodes = [
            Terminals().visitScalarItem(scalar)
            for scalar in ctx_list
            if not scalar.is_terminal and scalar.rule_index == 43
        ]
        children_nodes = var_ids_nodes + constant_nodes

        if len(children_nodes) > 1:
            raise Exception("Only one operand is allowed in Eval")

        # Reference manual says it is mandatory.
        language_name = [
            language
            for language in ctx_list
            if language.is_terminal and language.symbol_type == vtl_cpp_parser.STRING_CONSTANT
        ]
        if len(language_name) == 0:
            # AST_ASTCONSTRUCTOR.12
            raise SemanticError("1-3-2-1", option="language")
        # Reference manual says it is mandatory.
        output_node = [
            Terminals().visitOutputParameterTypeComponent(output)
            for output in ctx_list
            if not output.is_terminal and output.ctx_id == RC.OUTPUT_PARAMETER_TYPE_COMPONENT
        ]
        if len(output_node) == 0:
            # AST_ASTCONSTRUCTOR.13
            raise SemanticError("1-3-2-1", option="output")

        return EvalOp(
            name=routine_name,
            operands=children_nodes[0],
            output=output_node[0],
            language=language_name[0].text,
            **extract_token_info(ctx),
        )

    def visitCastExprComponent(self, ctx):  # type: ignore[no-untyped-def]
        """
        | CAST LPAREN exprComponent COMMA (basicScalarType|valueDomainName) (COMMA STRING_CONSTANT)? RPAREN         # castExprComponent
        """  # noqa E501
        ctx_list = ctx.children
        c = ctx_list[0]

        op = c.text
        expr_node = [
            self.visitExprComponent(expr)
            for expr in ctx_list
            if not expr.is_terminal and expr.rule_index == 3
        ]
        basic_scalar_type = [
            Terminals().visitBasicScalarType(type_)
            for type_ in ctx_list
            if not type_.is_terminal and type_.ctx_id == RC.BASIC_SCALAR_TYPE
        ]

        [
            Terminals().visitValueDomainName(valueD)
            for valueD in ctx_list
            if not valueD.is_terminal and valueD.ctx_id == RC.VALUE_DOMAIN_NAME
        ]

        if len(ctx_list) > 6:
            param_node = [
                ParamConstant(
                    type_="PARAM_CAST",
                    value=str_.text.strip('"'),
                    **extract_token_info(str_),
                )
                for str_ in ctx_list
                if str_.is_terminal and str_.symbol_type == vtl_cpp_parser.STRING_CONSTANT
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

    def visitParameterComponent(self, ctx):  # type: ignore[no-untyped-def]
        ctx_list = ctx.children
        c = ctx_list[0]

        if not c.is_terminal and c.rule_index == 3:
            return self.visitExprComponent(c)
        elif c.is_terminal:
            return ID(type_="OPTIONAL", value=c.text, **extract_token_info(ctx))
        else:
            raise NotImplementedError

    """
                            -----------------------------------
                                    String Functions
                            -----------------------------------
        """

    def visitStringFunctionsComponents(self, ctx):  # type: ignore[no-untyped-def]
        if ctx.ctx_id == RC.UNARY_STRING_FUNCTION_COMPONENT:
            return self.visitUnaryStringFunctionComponent(ctx)
        elif ctx.ctx_id == RC.SUBSTR_ATOM_COMPONENT:
            return self.visitSubstrAtomComponent(ctx)
        elif ctx.ctx_id == RC.REPLACE_ATOM_COMPONENT:
            return self.visitReplaceAtomComponent(ctx)
        elif ctx.ctx_id == RC.INSTR_ATOM_COMPONENT:
            return self.visitInstrAtomComponent(ctx)
        else:
            raise NotImplementedError

    def visitUnaryStringFunctionComponent(self, ctx):  # type: ignore[no-untyped-def]
        ctx_list = ctx.children
        c = ctx_list[0]

        op_node = c.text
        operand_node = self.visitExprComponent(ctx_list[2])
        return UnaryOp(op=op_node, operand=operand_node, **extract_token_info(ctx))

    def visitSubstrAtomComponent(self, ctx):  # type: ignore[no-untyped-def]
        ctx_list = ctx.children
        c = ctx_list[0]

        params_nodes = []
        children_nodes = []

        childrens = [expr for expr in ctx_list if not expr.is_terminal and expr.rule_index == 3]
        params = [
            param
            for param in ctx_list
            if not param.is_terminal and param.ctx_id == RC.OPTIONAL_EXPR_COMPONENT
        ]

        op_node = c.text
        for children in childrens:
            children_nodes.append(self.visitExprComponent(children))

        if len(params) != 0:
            for param in params:
                params_nodes.append(self.visitOptionalExprComponent(param))

        return ParamOp(
            op=op_node, children=children_nodes, params=params_nodes, **extract_token_info(ctx)
        )

    def visitReplaceAtomComponent(self, ctx):  # type: ignore[no-untyped-def]
        ctx_list = ctx.children
        c = ctx_list[0]

        expressions = [
            self.visitExprComponent(expr)
            for expr in ctx_list
            if not expr.is_terminal and expr.rule_index == 3
        ]
        params = [
            self.visitOptionalExprComponent(param)
            for param in ctx_list
            if not param.is_terminal and param.ctx_id == RC.OPTIONAL_EXPR_COMPONENT
        ]

        op_node = c.text

        children_nodes = [expressions[0]]
        params_nodes = [expressions[1]] + params

        return ParamOp(
            op=op_node, children=children_nodes, params=params_nodes, **extract_token_info(ctx)
        )

    def visitInstrAtomComponent(self, ctx):  # type: ignore[no-untyped-def]
        ctx_list = ctx.children
        c = ctx_list[0]

        expressions = [
            self.visitExprComponent(expr)
            for expr in ctx_list
            if not expr.is_terminal and expr.rule_index == 3
        ]
        params = [
            self.visitOptionalExprComponent(param)
            for param in ctx_list
            if not param.is_terminal and param.ctx_id == RC.OPTIONAL_EXPR_COMPONENT
        ]

        op_node = c.text

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

    def visitNumericFunctionsComponents(self, ctx):  # type: ignore[no-untyped-def]
        if ctx.ctx_id == RC.UNARY_NUMERIC_COMPONENT:
            return self.visitUnaryNumericComponent(ctx)
        elif ctx.ctx_id == RC.UNARY_WITH_OPTIONAL_NUMERIC_COMPONENT:
            return self.visitUnaryWithOptionalNumericComponent(ctx)
        elif ctx.ctx_id == RC.BINARY_NUMERIC_COMPONENT:
            return self.visitBinaryNumericComponent(ctx)
        else:
            raise NotImplementedError

    def visitUnaryNumericComponent(self, ctx):  # type: ignore[no-untyped-def]
        ctx_list = ctx.children
        c = ctx_list[0]

        op_node = c.text
        operand_node = self.visitExprComponent(ctx_list[2])
        return UnaryOp(op=op_node, operand=operand_node, **extract_token_info(ctx))

    def visitUnaryWithOptionalNumericComponent(self, ctx):  # type: ignore[no-untyped-def]
        ctx_list = ctx.children
        c = ctx_list[0]

        params_nodes = []
        children_nodes = []

        childrens = [expr for expr in ctx_list if not expr.is_terminal and expr.rule_index == 3]
        params = [
            param
            for param in ctx_list
            if not param.is_terminal and param.ctx_id == RC.OPTIONAL_EXPR_COMPONENT
        ]

        op_node = c.text
        for children in childrens:
            children_nodes.append(self.visitExprComponent(children))

        if len(params) != 0:
            for param in params:
                params_nodes.append(self.visitOptionalExprComponent(param))

        return ParamOp(
            op=op_node, children=children_nodes, params=params_nodes, **extract_token_info(ctx)
        )

    def visitBinaryNumericComponent(self, ctx):  # type: ignore[no-untyped-def]
        ctx_list = ctx.children
        c = ctx_list[0]

        left_node = self.visitExprComponent(ctx_list[2])
        op_node = c.text
        right_node = self.visitExprComponent(ctx_list[4])
        return BinOp(left=left_node, op=op_node, right=right_node, **extract_token_info(ctx))

    """
                            -----------------------------------
                                    Time Functions
                            -----------------------------------
        """

    def visitTimeFunctionsComponents(self, ctx):  # type: ignore[no-untyped-def]
        if ctx.ctx_id == RC.PERIOD_ATOM_COMPONENT:
            return self.visitTimeUnaryAtomComponent(ctx)
        elif ctx.ctx_id == RC.FILL_TIME_ATOM_COMPONENT:
            return self.visitFillTimeAtomComponent(ctx)
        elif ctx.ctx_id == RC.FLOW_ATOM_COMPONENT:
            raise SemanticError("1-1-19-7", op=ctx.children[0].text)
        elif ctx.ctx_id == RC.TIME_SHIFT_ATOM_COMPONENT:
            return self.visitTimeShiftAtomComponent(ctx)
        elif ctx.ctx_id == RC.TIME_AGG_ATOM_COMPONENT:
            return self.visitTimeAggAtomComponent(ctx)
        elif ctx.ctx_id == RC.CURRENT_DATE_ATOM_COMPONENT:
            return self.visitCurrentDateAtomComponent(ctx)
        elif ctx.ctx_id == RC.DATE_DIFF_ATOM_COMPONENT:
            return self.visitDateDiffAtomComponent(ctx)
        elif ctx.ctx_id == RC.DATE_ADD_ATOM_COMPONENT:
            return self.visitDateAddAtomComponentContext(ctx)
        elif ctx.ctx_id in (
            RC.YEAR_ATOM_COMPONENT,
            RC.MONTH_ATOM_COMPONENT,
            RC.DAY_OF_MONTH_ATOM_COMPONENT,
            RC.DAT_OF_YEAR_ATOM_COMPONENT,
            RC.DAY_TO_YEAR_ATOM_COMPONENT,
            RC.DAY_TO_MONTH_ATOM_COMPONENT,
            RC.YEAR_TODAY_ATOM_COMPONENT,
            RC.MONTH_TODAY_ATOM_COMPONENT,
        ):
            return self.visitTimeUnaryAtomComponent(ctx)
        else:
            raise NotImplementedError

    def visitTimeUnaryAtomComponent(self, ctx):  # type: ignore[no-untyped-def]
        """
        periodExpr: PERIOD_INDICATOR '(' expr? ')' ;
        """
        ctx_list = ctx.children
        c = ctx_list[0]

        op = c.text
        operand_node = [
            self.visitExprComponent(operand)
            for operand in ctx_list
            if not operand.is_terminal and operand.rule_index == 3
        ]

        if len(operand_node) == 0:
            # AST_ASTCONSTRUCTOR.15
            raise NotImplementedError

        return UnaryOp(op=op, operand=operand_node[0], **extract_token_info(ctx))

    def visitTimeShiftAtomComponent(self, ctx):  # type: ignore[no-untyped-def]
        """
        timeShiftExpr: TIMESHIFT '(' expr ',' INTEGER_CONSTANT ')' ;
        """
        ctx_list = ctx.children
        c = ctx_list[0]

        op = c.text
        left_node = self.visitExprComponent(ctx_list[2])
        right_node = Constant(
            type_="INTEGER_CONSTANT",
            value=int(ctx_list[4].text),
            **extract_token_info(ctx),
        )

        return BinOp(left=left_node, op=op, right=right_node, **extract_token_info(ctx))

    def visitFillTimeAtomComponent(self, ctx):  # type: ignore[no-untyped-def]
        """
        timeSeriesExpr: FILL_TIME_SERIES '(' expr (',' (SINGLE|ALL))? ')' ;
        """
        ctx_list = ctx.children
        c = ctx_list[0]

        op = c.text
        children_node = [self.visitExprComponent(ctx_list[2])]

        if len(ctx_list) > 4:
            param_constant_node = [
                ParamConstant(
                    type_="PARAM_TIMESERIES",
                    value=ctx_list[4].text,
                    **extract_token_info(ctx),
                )
            ]
        else:
            param_constant_node = []

        return ParamOp(
            op=op, children=children_node, params=param_constant_node, **extract_token_info(ctx)
        )

    def visitTimeAggAtomComponent(self, ctx):  # type: ignore[no-untyped-def]
        """
        TIME_AGG LPAREN periodIndTo=STRING_CONSTANT (COMMA periodIndFrom=(STRING_CONSTANT| OPTIONAL ))?
        (COMMA op=optionalExprComponent)? (COMMA (FIRST|LAST))? RPAREN    # timeAggAtomComponent;
        """  # noqa E501
        ctx_list = ctx.children
        c = ctx_list[0]

        op = c.text

        # periodIndTo is always at index 2 (TIME_AGG LPAREN periodIndTo)
        period_to = str(ctx_list[2].text)[1:-1]
        period_from = None

        # Find periodIndFrom: look for STRING_CONSTANT or OPTIONAL after the first COMMA
        # Grammar: TIME_AGG LPAREN STRING_CONSTANT (COMMA (STRING_CONSTANT|OPTIONAL))?
        #          (COMMA optionalExprComponent)? (COMMA (FIRST|LAST))? RPAREN
        # We need to scan children to find the named elements positionally.
        # ctx_list[0]=TIME_AGG, [1]=LPAREN, [2]=periodIndTo
        # If there's a periodIndFrom, it's at index 4 (after COMMA at index 3)
        idx = 3  # start after periodIndTo
        period_ind_from_token = None

        if (
            idx < len(ctx_list)
            and ctx_list[idx].is_terminal
            and idx + 1 < len(ctx_list)
            and ctx_list[idx + 1].is_terminal
            and ctx_list[idx + 1].symbol_type
            in (vtl_cpp_parser.STRING_CONSTANT, vtl_cpp_parser.OPTIONAL)
        ):
            period_ind_from_token = ctx_list[idx + 1]
            idx = idx + 2

        if (
            period_ind_from_token is not None
            and period_ind_from_token.symbol_type != vtl_cpp_parser.OPTIONAL
        ):
            period_from = str(period_ind_from_token.text)[1:-1]

        # Find optionalExprComponent (op)
        optional_expr_comp = None
        for child in ctx_list:
            if not child.is_terminal and child.ctx_id == RC.OPTIONAL_EXPR_COMPONENT:
                optional_expr_comp = child
                break

        # Find FIRST/LAST
        conf_list = [
            str_.text
            for str_ in ctx_list
            if str_.is_terminal and str_.symbol_type in (vtl_cpp_parser.FIRST, vtl_cpp_parser.LAST)
        ]

        conf = None if len(conf_list) == 0 else conf_list[0]

        if optional_expr_comp is not None:
            operand_node = self.visitOptionalExprComponent(optional_expr_comp)
            if isinstance(operand_node, ID):
                operand_node = None
            elif isinstance(operand_node, Identifier):
                operand_node = VarID(
                    value=operand_node.value,
                    **extract_token_info(optional_expr_comp),
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

    def visitCurrentDateAtomComponent(self, ctx):  # type: ignore[no-untyped-def]
        c = ctx.children[0]
        return MulOp(op=c.text, children=[], **extract_token_info(ctx))

    def visitDateDiffAtomComponent(self, ctx):  # type: ignore[no-untyped-def]
        """ """
        from vtlengine.AST.ASTConstructorModules.Expr import Expr

        ctx_list = ctx.children
        c = ctx_list[0]

        op = c.text
        left_node = self.visitExprComponent(ctx_list[2])
        # dateTo is 'expr' (not exprComponent) in the new grammar
        right_node = Expr().visitExpr(ctx_list[4])

        return BinOp(left=left_node, op=op, right=right_node, **extract_token_info(ctx))

    def visitDateAddAtomComponentContext(self, ctx):  # type: ignore[no-untyped-def]
        """ """
        ctx_list = ctx.children
        c = ctx_list[0]

        op = c.text
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

    def visitConditionalFunctionsComponents(self, ctx):  # type: ignore[no-untyped-def]
        if ctx.ctx_id == RC.NVL_ATOM_COMPONENT:
            return self.visitNvlAtomComponent(ctx)
        else:
            raise NotImplementedError

    def visitNvlAtomComponent(self, ctx):  # type: ignore[no-untyped-def]
        ctx_list = ctx.children
        c = ctx_list[0]

        left_node = self.visitExprComponent(ctx_list[2])
        op_node = c.text
        right_node = self.visitExprComponent(ctx_list[4])
        return BinOp(left=left_node, op=op_node, right=right_node, **extract_token_info(ctx))

    """
                        -----------------------------------
                            Comparison Components Functions
                        -----------------------------------
    """

    def visitComparisonFunctionsComponents(self, ctx):  # type: ignore[no-untyped-def]
        if ctx.ctx_id == RC.BETWEEN_ATOM_COMPONENT:
            return self.visitBetweenAtomComponent(ctx)
        elif ctx.ctx_id == RC.CHARSET_MATCH_ATOM_COMPONENT:
            return self.visitCharsetMatchAtomComponent(ctx)
        elif ctx.ctx_id == RC.IS_NULL_ATOM_COMPONENT:
            return self.visitIsNullAtomComponent(ctx)
        else:
            raise NotImplementedError

    def visitBetweenAtomComponent(self, ctx):  # type: ignore[no-untyped-def]
        ctx_list = ctx.children
        c = ctx_list[0]

        children_nodes = []

        childrens = [expr for expr in ctx_list if not expr.is_terminal and expr.rule_index == 3]

        op_node = c.text
        for children in childrens:
            children_nodes.append(self.visitExprComponent(children))

        return MulOp(op=op_node, children=children_nodes, **extract_token_info(ctx))

    def visitCharsetMatchAtomComponent(self, ctx):  # type: ignore[no-untyped-def]
        ctx_list = ctx.children
        c = ctx_list[0]

        left_node = self.visitExprComponent(ctx_list[2])
        op_node = c.text
        right_node = self.visitExprComponent(ctx_list[4])
        return BinOp(left=left_node, op=op_node, right=right_node, **extract_token_info(ctx))

    def visitIsNullAtomComponent(self, ctx):  # type: ignore[no-untyped-def]
        ctx_list = ctx.children
        c = ctx_list[0]
        op_node = c.text
        operand_node = self.visitExprComponent(ctx_list[2])
        return UnaryOp(op=op_node, operand=operand_node, **extract_token_info(ctx))

    """
                            -----------------------------------
                                Aggregate Components Functions
                            -----------------------------------
        """

    def visitAggregateFunctionsComponents(self, ctx):  # type: ignore[no-untyped-def]
        if ctx.ctx_id == RC.AGGR_COMP:
            return self.visitAggrComp(ctx)
        elif ctx.ctx_id == RC.COUNT_AGGR_COMP:
            return self.visitCountAggrComp(ctx)
        else:
            raise NotImplementedError

    def visitAggrComp(self, ctx):  # type: ignore[no-untyped-def]
        ctx_list = ctx.children
        op_node = ctx_list[0].text
        operand_node = self.visitExprComponent(ctx_list[2])
        return Aggregation(op=op_node, operand=operand_node, **extract_token_info(ctx))

    def visitCountAggrComp(self, ctx):  # type: ignore[no-untyped-def]
        ctx_list = ctx.children
        op_node = ctx_list[0].text
        return Aggregation(op=op_node, **extract_token_info(ctx))

    """
                                -----------------------------------
                                    Analytic Components Functions
                                -----------------------------------
    """

    def visitAnalyticFunctionsComponents(self, ctx):  # type: ignore[no-untyped-def]
        if ctx.ctx_id == RC.AN_SIMPLE_FUNCTION_COMPONENT:
            return self.visitAnSimpleFunctionComponent(ctx)
        elif ctx.ctx_id == RC.LAG_OR_LEAD_AN_COMPONENT:
            return self.visitLagOrLeadAnComponent(ctx)
        elif ctx.ctx_id == RC.RANK_AN_COMPONENT:
            return self.visitRankAnComponent(ctx)
        elif ctx.ctx_id == RC.RATIO_TO_REPORT_AN_COMPONENT:
            return self.visitRatioToReportAnComponent(ctx)
        else:
            raise NotImplementedError

    def visitAnSimpleFunctionComponent(self, ctx):  # type: ignore[no-untyped-def]
        ctx_list = ctx.children

        params = None
        partition_by = None
        order_by = None

        op_node = ctx_list[0].text
        operand = self.visitExprComponent(ctx_list[2])

        for c in ctx_list[5:-2]:
            if not c.is_terminal and c.ctx_id == RC.PARTITION_BY_CLAUSE:
                partition_by = Terminals().visitPartitionByClause(c)
                continue
            elif not c.is_terminal and c.ctx_id == RC.ORDER_BY_CLAUSE:
                order_by = Terminals().visitOrderByClause(c)
                continue
            elif not c.is_terminal and c.ctx_id == RC.WINDOWING_CLAUSE:
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

    def visitLagOrLeadAnComponent(self, ctx):  # type: ignore[no-untyped-def]
        ctx_list = ctx.children

        params = None
        partition_by = None
        order_by = None

        op_node = ctx_list[0].text
        operand = self.visitExprComponent(ctx_list[2])

        for c in ctx_list[4:-2]:
            if not c.is_terminal and c.ctx_id == RC.PARTITION_BY_CLAUSE:
                partition_by = Terminals().visitPartitionByClause(c)
                continue
            elif not c.is_terminal and c.ctx_id == RC.ORDER_BY_CLAUSE:
                order_by = Terminals().visitOrderByClause(c)
                continue
            elif not c.is_terminal and c.rule_index in (53, 43):
                # SignedInteger (rule 53) or ScalarItem (rule 43)
                if params is None:
                    params = []
                if c.rule_index == 53:
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

    def visitRankAnComponent(self, ctx):  # type: ignore[no-untyped-def]
        ctx_list = ctx.children

        partition_by = None
        order_by = None

        op_node = ctx_list[0].text

        for c in ctx_list[4:-2]:
            if not c.is_terminal and c.ctx_id == RC.PARTITION_BY_CLAUSE:
                partition_by = Terminals().visitPartitionByClause(c)
                continue
            elif not c.is_terminal and c.ctx_id == RC.ORDER_BY_CLAUSE:
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

    def visitRatioToReportAnComponent(self, ctx):  # type: ignore[no-untyped-def]
        ctx_list = ctx.children

        params = None
        order_by = None

        op_node = ctx_list[0].text
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
