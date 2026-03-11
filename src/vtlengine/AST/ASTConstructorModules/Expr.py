import re
from copy import copy
from typing import Any, Optional

from vtlengine.AST import (
    ID,
    Aggregation,
    Analytic,
    Assignment,
    BinOp,
    Case,
    CaseObj,
    CHInputMode,
    Constant,
    DPValidation,
    EvalOp,
    HierarchyOutput,
    HRInputMode,
    HROperation,
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
    ValidationMode,
    ValidationOutput,
    VarID,
    Windowing,
)
from vtlengine.AST.ASTConstructorModules import extract_token_info
from vtlengine.AST.ASTConstructorModules.ExprComponents import ExprComp
from vtlengine.AST.ASTConstructorModules.Terminals import Terminals
from vtlengine.AST.ASTDataExchange import de_ruleset_elements
from vtlengine.AST.Grammar._cpp_parser import vtl_cpp_parser
from vtlengine.AST.Grammar._cpp_parser._rule_constants import RC
from vtlengine.AST.Grammar.tokens import DATASET_PRIORITY
from vtlengine.Exceptions import SemanticError
from vtlengine.Model import Role


class Expr:
    """______________________________________________________________________________________


                                Expr Definition.

    _______________________________________________________________________________________
    """

    def visitExpr(self, ctx: Any) -> Any:
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
        ctx_list = ctx.children
        c = ctx_list[0]

        if ctx.ctx_id == RC.PARENTHESIS_EXPR:
            return self.visitParenthesisExpr(ctx)

        elif ctx.ctx_id == RC.MEMBERSHIP_EXPR:
            return self.visitMembershipExpr(ctx)

        # dataset=expr  QLPAREN  clause=datasetClause  QRPAREN                  # clauseExpr
        elif ctx.ctx_id == RC.CLAUSE_EXPR:
            return self.visitClauseExpr(ctx)

        # functions
        elif ctx.ctx_id == RC.FUNCTIONS_EXPRESSION:
            return self.visitFunctionsExpression(c)

        # op=(PLUS|MINUS|NOT) right=expr # unary expression
        elif ctx.ctx_id == RC.UNARY_EXPR:
            return self.visitUnaryExpr(ctx)

        # | left=expr op=(MUL|DIV) right=expr               # arithmeticExpr
        elif ctx.ctx_id == RC.ARITHMETIC_EXPR:
            return self.visitArithmeticExpr(ctx)

        # | left=expr op=(PLUS|MINUS|CONCAT) right=expr     # arithmeticExprOrConcat
        elif ctx.ctx_id == RC.ARITHMETIC_EXPR_OR_CONCAT:
            return self.visitArithmeticExprOrConcat(ctx)

        # | left=expr op=comparisonOperand  right=expr      # comparisonExpr
        elif ctx.ctx_id == RC.COMPARISON_EXPR:
            return self.visitComparisonExpr(ctx)

        # | left=expr op=(IN|NOT_IN)(lists|valueDomainID)   # inNotInExpr
        elif ctx.ctx_id == RC.IN_NOT_IN_EXPR:
            return self.visitInNotInExpr(ctx)

        # | left=expr op=AND right=expr                                           # booleanExpr
        # | left=expr op=(OR|XOR) right=expr
        elif ctx.ctx_id == RC.BOOLEAN_EXPR:
            return self.visitBooleanExpr(ctx)

        # IF  conditionalExpr=expr  THEN thenExpr=expr ELSE elseExpr=expr       # ifExpr
        elif ctx.ctx_id == RC.IF_EXPR:
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
        elif ctx.ctx_id == RC.CASE_EXPR:
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
        elif ctx.ctx_id == RC.CONSTANT_EXPR:
            return Terminals().visitConstant(c)

        # varID
        elif ctx.ctx_id == RC.VAR_ID_EXPR:
            return Terminals().visitVarIdExpr(c)

        else:
            # AST_ASTCONSTRUCTOR.3
            raise NotImplementedError

    def bin_op_creator(self, ctx: Any) -> BinOp:
        ctx_list = ctx.children
        left_node = self.visitExpr(ctx_list[0])
        if not ctx_list[1].is_terminal and ctx_list[1].rule_index == RC.COMPARISON_OPERAND[0]:
            op = ctx_list[1].children[0].text
        else:
            op = ctx_list[1].text
        right_node = self.visitExpr(ctx_list[2])
        token_info = extract_token_info(ctx)
        bin_op_node = BinOp(left=left_node, op=op, right=right_node, **token_info)

        return bin_op_node

    def visitArithmeticExpr(self, ctx: Any) -> BinOp:
        return self.bin_op_creator(ctx)

    def visitArithmeticExprOrConcat(self, ctx: Any) -> BinOp:
        return self.bin_op_creator(ctx)

    def visitComparisonExpr(self, ctx: Any) -> BinOp:
        return self.bin_op_creator(ctx)

    def visitInNotInExpr(self, ctx: Any) -> BinOp:
        ctx_list = ctx.children
        left_node = self.visitExpr(ctx_list[0])
        op = ctx_list[1].text

        if not ctx_list[2].is_terminal and ctx_list[2].rule_index == RC.LISTS[0]:
            right_node = Terminals().visitLists(ctx_list[2])
        elif not ctx_list[2].is_terminal and ctx_list[2].rule_index == RC.VALUE_DOMAIN_ID[0]:
            right_node = Terminals().visitValueDomainID(ctx_list[2])
        else:
            raise NotImplementedError
        bin_op_node = BinOp(left=left_node, op=op, right=right_node, **extract_token_info(ctx))

        return bin_op_node

    def visitBooleanExpr(self, ctx: Any) -> BinOp:
        return self.bin_op_creator(ctx)

    def visitParenthesisExpr(self, ctx: Any) -> ParFunction:
        operand = self.visitExpr(ctx.children[1])
        return ParFunction(operand=operand, **extract_token_info(ctx))

    def visitUnaryExpr(self, ctx: Any) -> UnaryOp:
        c_list = ctx.children
        op = c_list[0].text
        right = self.visitExpr(c_list[1])

        return UnaryOp(op=op, operand=right, **extract_token_info(ctx))

    def visitMembershipExpr(self, ctx: Any) -> Any:
        ctx_list = ctx.children
        c = ctx_list[0]
        membership = [
            componentID
            for componentID in ctx_list
            if not componentID.is_terminal and componentID.rule_index == RC.SIMPLE_COMPONENT_ID[0]
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

    def visitClauseExpr(self, ctx: Any) -> Any:
        ctx_list = ctx.children

        dataset = self.visitExpr(ctx_list[0])

        dataset_clause = self.visitDatasetClause(ctx_list[2])

        dataset_clause.dataset = dataset

        return dataset_clause

    """______________________________________________________________________________________


                                    Functions Definition.

        _______________________________________________________________________________________"""

    def visitFunctionsExpression(self, ctx: Any) -> Any:
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

        if ctx.ctx_id == RC.JOIN_FUNCTIONS:
            return self.visitJoinFunctions(c)

        elif ctx.ctx_id == RC.GENERIC_FUNCTIONS:
            return self.visitGenericFunctions(c)

        elif ctx.ctx_id == RC.STRING_FUNCTIONS:
            return self.visitStringFunctions(c)

        elif ctx.ctx_id == RC.NUMERIC_FUNCTIONS:
            return self.visitNumericFunctions(c)

        elif ctx.ctx_id == RC.COMPARISON_FUNCTIONS:
            return self.visitComparisonFunctions(c)

        elif ctx.ctx_id == RC.TIME_FUNCTIONS:
            return self.visitTimeFunctions(c)

        elif ctx.ctx_id == RC.SET_FUNCTIONS:
            return self.visitSetFunctions(c)

        elif ctx.ctx_id == RC.HIERARCHY_FUNCTIONS:
            return self.visitHierarchyFunctions(c)

        elif ctx.ctx_id == RC.VALIDATION_FUNCTIONS:
            return self.visitValidationFunctions(c)

        elif ctx.ctx_id == RC.CONDITIONAL_FUNCTIONS:
            return self.visitConditionalFunctions(c)

        elif ctx.ctx_id == RC.AGGREGATE_FUNCTIONS:
            return self.visitAggregateFunctions(c)

        elif ctx.ctx_id == RC.ANALYTIC_FUNCTIONS:
            return self.visitAnalyticFunctions(c)

        else:
            raise NotImplementedError

    """
                        -----------------------------------
                                Join Functions
                        -----------------------------------
    """

    def visitJoinFunctions(self, ctx: Any) -> Any:
        ctx_list = ctx.children

        using_node = None

        op_node = ctx_list[0].text

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

    def visitJoinClauseItem(self, ctx: Any) -> Any:
        ctx_list = ctx.children
        left_node = self.visitExpr(ctx_list[0])
        if len(ctx_list) == 1:
            return left_node

        token_info = extract_token_info(ctx)
        intop_node = ctx_list[1].text
        right_node = Identifier(
            value=Terminals().visitAlias(ctx_list[2]),
            kind="DatasetID",
            **extract_token_info(ctx_list[1]),
        )
        return BinOp(left=left_node, op=intop_node, right=right_node, **token_info)

    def visitJoinClause(self, ctx: Any) -> Any:
        """
        JoinClauseItem (COMMA joinClauseItem)* (USING componentID (COMMA componentID)*)?
        """
        ctx_list = ctx.children

        clause_nodes = []
        component_nodes = []
        using = None

        items = [
            item
            for item in ctx_list
            if not item.is_terminal and item.rule_index == RC.JOIN_CLAUSE_ITEM[0]
        ]
        components = [
            component
            for component in ctx_list
            if not component.is_terminal and component.rule_index == RC.COMPONENT_ID[0]
        ]

        for item in items:
            clause_nodes.append(self.visitJoinClauseItem(item))

        if len(components) != 0:
            for component in components:
                component_nodes.append(Terminals().visitComponentID(component).value)
            using = component_nodes

        return clause_nodes, using

    def visitJoinClauseWithoutUsing(self, ctx: Any) -> Any:
        """
        joinClause: joinClauseItem (COMMA joinClauseItem)* (USING componentID (COMMA componentID)*)? ;
        """  # noqa E501
        ctx_list = ctx.children

        clause_nodes = []

        items = [
            item
            for item in ctx_list
            if not item.is_terminal and item.rule_index == RC.JOIN_CLAUSE_ITEM[0]
        ]

        for item in items:
            clause_nodes.append(self.visitJoinClauseItem(item))

        return clause_nodes

    def visitJoinBody(self, ctx: Any) -> Any:
        """
        joinBody: filterClause? (calcClause|joinApplyClause|aggrClause)? (keepOrDropClause)? renameClause?
        """  # noqa E501
        ctx_list = ctx.children

        body_nodes = []

        for c in ctx_list:
            if c.is_terminal:
                raise NotImplementedError
            elif c.rule_index == RC.FILTER_CLAUSE[0]:
                body_nodes.append(self.visitFilterClause(c))
            elif c.rule_index == RC.CALC_CLAUSE[0]:
                body_nodes.append(self.visitCalcClause(c))
            elif c.rule_index == RC.JOIN_APPLY_CLAUSE[0]:
                body_nodes.append(self.visitJoinApplyClause(c))
            elif c.rule_index == RC.AGGR_CLAUSE[0]:
                body_nodes.append(self.visitAggrClause(c))
            elif c.rule_index == RC.KEEP_OR_DROP_CLAUSE[0]:
                body_nodes.append(self.visitKeepOrDropClause(c))
            elif c.rule_index == RC.RENAME_CLAUSE[0]:
                body_nodes.append(self.visitRenameClause(c))
            else:
                raise NotImplementedError

        return body_nodes

    # TODO Unary Op here?
    def visitJoinApplyClause(self, ctx: Any) -> RegularAggregation:
        """
        joinApplyClause: APPLY expr ;
        """
        ctx_list = ctx.children
        op_node = ctx_list[0].text
        operand_nodes = [self.visitExpr(ctx_list[1])]

        return RegularAggregation(op=op_node, children=operand_nodes, **extract_token_info(ctx))

    """
                        -----------------------------------
                                Generic Functions
                        -----------------------------------
    """

    def visitGenericFunctions(self, ctx: Any) -> Any:
        if ctx.ctx_id == RC.CALL_DATASET:
            return self.visitCallDataset(ctx)
        elif ctx.ctx_id == RC.EVAL_ATOM:
            return self.visitEvalAtom(ctx)
        elif ctx.ctx_id == RC.CAST_EXPR_DATASET:
            return self.visitCastExprDataset(ctx)
        else:
            raise NotImplementedError

    def visitCallDataset(self, ctx: Any) -> UDOCall:
        """
        callFunction: operatorID LPAREN (parameter (COMMA parameter)*)? RPAREN   ;
        """
        ctx_list = ctx.children
        c = ctx_list[0]

        op = Terminals().visitOperatorID(c)
        param_nodes = [
            self.visitParameter(element)
            for element in ctx_list
            if not element.is_terminal and element.rule_index == RC.PARAMETER[0]
        ]

        return UDOCall(op=op, params=param_nodes, **extract_token_info(ctx))

    def visitEvalAtom(self, ctx: Any) -> EvalOp:
        """
        | EVAL LPAREN routineName LPAREN (varID|scalarItem)? (COMMA (varID|scalarItem))* RPAREN (LANGUAGE STRING_CONSTANT)? (RETURNS evalDatasetType)? RPAREN     # evalAtom
        """  # noqa E501
        ctx_list = ctx.children

        routine_name = Terminals().visitRoutineName(ctx_list[2])

        # Think of a way to maintain the order, for now its not necessary.
        var_ids_nodes = [
            Terminals().visitVarID(varID)
            for varID in ctx_list
            if not varID.is_terminal and varID.rule_index == RC.VAR_ID[0]
        ]
        constant_nodes = [
            Terminals().visitScalarItem(scalar)
            for scalar in ctx_list
            if not scalar.is_terminal and scalar.ctx_id in (RC.SIMPLE_SCALAR, RC.SCALAR_WITH_CAST)
        ]
        children_nodes = var_ids_nodes + constant_nodes

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
            Terminals().visitEvalDatasetType(output)
            for output in ctx_list
            if not output.is_terminal and output.rule_index == RC.EVAL_DATASET_TYPE[0]
        ]
        if len(output_node) == 0:
            # AST_ASTCONSTRUCTOR.13
            raise SemanticError("1-3-2-1", option="output")

        return EvalOp(
            name=routine_name,
            operands=children_nodes,
            output=output_node[0],
            language=language_name[0].text,
            **extract_token_info(ctx),
        )

    def visitCastExprDataset(self, ctx: Any) -> Any:
        """
        | CAST LPAREN expr COMMA (basicScalarType|valueDomainName) (COMMA STRING_CONSTANT)? RPAREN        # castExprDataset
        """  # noqa E501
        ctx_list = ctx.children
        c = ctx_list[0]

        op = c.text
        expr_node = [
            self.visitExpr(expr)
            for expr in ctx_list
            if not expr.is_terminal and expr.rule_index == 2
        ]
        basic_scalar_type = [
            Terminals().visitBasicScalarType(type_)
            for type_ in ctx_list
            if not type_.is_terminal and type_.rule_index == RC.BASIC_SCALAR_TYPE[0]
        ]

        [
            Terminals().visitValueDomainName(valueD)
            for valueD in ctx_list
            if not valueD.is_terminal and valueD.rule_index == RC.VALUE_DOMAIN_NAME[0]
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

    def visitParameter(self, ctx: Any) -> Any:
        ctx_list = ctx.children
        c = ctx_list[0]

        if not c.is_terminal and c.rule_index == 2:
            return self.visitExpr(c)
        elif c.is_terminal:
            return ID(type_="OPTIONAL", value=c.text, **extract_token_info(c))
        else:
            raise NotImplementedError

    """
                        -----------------------------------
                                String Functions
                        -----------------------------------
    """

    def visitStringFunctions(self, ctx: Any) -> Any:
        if ctx.ctx_id == RC.UNARY_STRING_FUNCTION:
            return self.visitUnaryStringFunction(ctx)
        elif ctx.ctx_id == RC.SUBSTR_ATOM:
            return self.visitSubstrAtom(ctx)
        elif ctx.ctx_id == RC.REPLACE_ATOM:
            return self.visitReplaceAtom(ctx)
        elif ctx.ctx_id == RC.INSTR_ATOM:
            return self.visitInstrAtom(ctx)
        else:
            raise NotImplementedError

    def visitUnaryStringFunction(self, ctx: Any) -> UnaryOp:
        ctx_list = ctx.children
        c = ctx_list[0]

        op_node = c.text
        operand_node = self.visitExpr(ctx_list[2])
        return UnaryOp(op=op_node, operand=operand_node, **extract_token_info(ctx))

    def visitSubstrAtom(self, ctx: Any) -> ParamOp:
        ctx_list = ctx.children
        c = ctx_list[0]

        params_nodes = []
        children_nodes = []

        childrens = [expr for expr in ctx_list if not expr.is_terminal and expr.rule_index == 2]
        params = [
            param
            for param in ctx_list
            if not param.is_terminal and param.rule_index == RC.OPTIONAL_EXPR[0]
        ]

        op_node = c.text
        for children in childrens:
            children_nodes.append(self.visitExpr(children))

        if len(params) != 0:
            for param in params:
                params_nodes.append(self.visitOptionalExpr(param))

        return ParamOp(
            op=op_node, children=children_nodes, params=params_nodes, **extract_token_info(ctx)
        )

    def visitReplaceAtom(self, ctx: Any) -> ParamOp:
        ctx_list = ctx.children
        c = ctx_list[0]

        expressions = [
            self.visitExpr(expr)
            for expr in ctx_list
            if not expr.is_terminal and expr.rule_index == 2
        ]
        params = [
            self.visitOptionalExpr(param)
            for param in ctx_list
            if not param.is_terminal and param.rule_index == RC.OPTIONAL_EXPR[0]
        ]

        op_node = c.text

        children_nodes = [expressions[0]]
        params_nodes = [expressions[1]] + params

        return ParamOp(
            op=op_node, children=children_nodes, params=params_nodes, **extract_token_info(ctx)
        )

    def visitInstrAtom(self, ctx: Any) -> ParamOp:
        ctx_list = ctx.children
        c = ctx_list[0]

        expressions = [
            self.visitExpr(expr)
            for expr in ctx_list
            if not expr.is_terminal and expr.rule_index == 2
        ]
        params = [
            self.visitOptionalExpr(param)
            for param in ctx_list
            if not param.is_terminal and param.rule_index == RC.OPTIONAL_EXPR[0]
        ]

        op_node = c.text

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

    def visitNumericFunctions(self, ctx: Any) -> Any:
        if ctx.ctx_id == RC.UNARY_NUMERIC:
            return self.visitUnaryNumeric(ctx)
        elif ctx.ctx_id == RC.UNARY_WITH_OPTIONAL_NUMERIC:
            return self.visitUnaryWithOptionalNumeric(ctx)
        elif ctx.ctx_id == RC.BINARY_NUMERIC:
            return self.visitBinaryNumeric(ctx)
        else:
            raise NotImplementedError

    def visitUnaryNumeric(self, ctx: Any) -> UnaryOp:
        ctx_list = ctx.children
        c = ctx_list[0]

        op_node = c.text
        operand_node = self.visitExpr(ctx_list[2])
        return UnaryOp(op=op_node, operand=operand_node, **extract_token_info(ctx))

    def visitUnaryWithOptionalNumeric(self, ctx: Any) -> ParamOp:
        ctx_list = ctx.children
        c = ctx_list[0]

        params_nodes = []
        children_nodes = []

        childrens = [expr for expr in ctx_list if not expr.is_terminal and expr.rule_index == 2]
        params = [
            param
            for param in ctx_list
            if not param.is_terminal and param.rule_index == RC.OPTIONAL_EXPR[0]
        ]

        op_node = c.text
        for children in childrens:
            children_nodes.append(self.visitExpr(children))

        if len(params) != 0:
            for param in params:
                params_nodes.append(self.visitOptionalExpr(param))

        return ParamOp(
            op=op_node, children=children_nodes, params=params_nodes, **extract_token_info(ctx)
        )

    def visitBinaryNumeric(self, ctx: Any) -> BinOp:
        ctx_list = ctx.children
        c = ctx_list[0]

        left_node = self.visitExpr(ctx_list[2])
        op_node = c.text
        right_node = self.visitExpr(ctx_list[4])
        return BinOp(left=left_node, op=op_node, right=right_node, **extract_token_info(ctx))

    """
                        -----------------------------------
                                Comparison Functions
                        -----------------------------------
    """

    def visitComparisonFunctions(self, ctx: Any) -> Any:
        if ctx.ctx_id == RC.BETWEEN_ATOM:
            return self.visitBetweenAtom(ctx)
        elif ctx.ctx_id == RC.CHARSET_MATCH_ATOM:
            return self.visitCharsetMatchAtom(ctx)
        elif ctx.ctx_id == RC.IS_NULL_ATOM:
            return self.visitIsNullAtom(ctx)
        elif ctx.ctx_id == RC.EXIST_IN_ATOM:
            return self.visitExistInAtom(ctx)
        else:
            raise NotImplementedError

    def visitBetweenAtom(self, ctx: Any) -> MulOp:
        ctx_list = ctx.children
        c = ctx_list[0]

        children_nodes = []

        childrens = [expr for expr in ctx_list if not expr.is_terminal and expr.rule_index == 2]

        op_node = c.text
        for children in childrens:
            children_nodes.append(self.visitExpr(children))

        return MulOp(op=op_node, children=children_nodes, **extract_token_info(ctx))

    def visitCharsetMatchAtom(self, ctx: Any) -> BinOp:
        ctx_list = ctx.children
        c = ctx_list[0]

        left_node = self.visitExpr(ctx_list[2])
        op_node = c.text
        right_node = self.visitExpr(ctx_list[4])
        return BinOp(left=left_node, op=op_node, right=right_node, **extract_token_info(ctx))

    def visitIsNullAtom(self, ctx: Any) -> UnaryOp:
        ctx_list = ctx.children
        c = ctx_list[0]
        op_node = c.text
        operand_node = self.visitExpr(ctx_list[2])
        return UnaryOp(op=op_node, operand=operand_node, **extract_token_info(ctx))

    def visitExistInAtom(self, ctx: Any) -> MulOp:
        ctx_list = ctx.children
        op = ctx_list[0].text

        operand_nodes = [
            self.visitExpr(expr)
            for expr in ctx_list
            if not expr.is_terminal and expr.rule_index == 2
        ]
        retain_nodes = [
            Terminals().visitRetainType(retain)
            for retain in ctx_list
            if not retain.is_terminal and retain.rule_index == RC.RETAIN_TYPE[0]
        ]

        return MulOp(op=op, children=operand_nodes + retain_nodes, **extract_token_info(ctx))

    """
                            -----------------------------------
                                    Time Functions
                            -----------------------------------
        """

    def visitTimeFunctions(self, ctx: Any) -> Any:
        if ctx.ctx_id == RC.PERIOD_ATOM:
            return self.visitTimeUnaryAtom(ctx)
        elif ctx.ctx_id == RC.FILL_TIME_ATOM:
            return self.visitFillTimeAtom(ctx)
        elif ctx.ctx_id == RC.FLOW_ATOM:
            return self.visitFlowAtom(ctx)
        elif ctx.ctx_id == RC.TIME_SHIFT_ATOM:
            return self.visitTimeShiftAtom(ctx)
        elif ctx.ctx_id == RC.TIME_AGG_ATOM:
            return self.visitTimeAggAtom(ctx)
        elif ctx.ctx_id == RC.CURRENT_DATE_ATOM:
            return self.visitCurrentDateAtom(ctx)
        elif ctx.ctx_id == RC.DATE_DIFF_ATOM:
            return self.visitTimeDiffAtom(ctx)
        elif ctx.ctx_id == RC.DATE_ADD_ATOM:
            return self.visitTimeAddAtom(ctx)
        elif ctx.ctx_id in (
            RC.YEAR_ATOM,
            RC.MONTH_ATOM,
            RC.DAY_OF_MONTH_ATOM,
            RC.DAY_OF_YEAR_ATOM,
            RC.DAY_TO_YEAR_ATOM,
            RC.DAY_TO_MONTH_ATOM,
            RC.YEAR_TODAY_ATOM,
            RC.MONTH_TODAY_ATOM,
        ):
            return self.visitTimeUnaryAtom(ctx)
        else:
            raise NotImplementedError

    def visitTimeUnaryAtom(self, ctx: Any) -> UnaryOp:
        ctx_list = ctx.children
        c = ctx_list[0]

        op = c.text
        operand_node = [
            self.visitExpr(operand)
            for operand in ctx_list
            if not operand.is_terminal and operand.rule_index == 2
        ]

        if len(operand_node) == 0:
            # AST_ASTCONSTRUCTOR.15
            raise NotImplementedError

        return UnaryOp(op=op, operand=operand_node[0], **extract_token_info(ctx))

    def visitTimeShiftAtom(self, ctx: Any) -> BinOp:
        """
        timeShiftExpr: TIMESHIFT '(' expr ',' INTEGER_CONSTANT ')' ;
        """
        ctx_list = ctx.children
        c = ctx_list[0]

        op = c.text
        left_node = self.visitExpr(ctx_list[2])
        right_node = Constant(
            type_="INTEGER_CONSTANT",
            value=Terminals().visitSignedInteger(ctx_list[4]),
            **extract_token_info(ctx_list[4]),
        )

        return BinOp(left=left_node, op=op, right=right_node, **extract_token_info(ctx))

    def visitFillTimeAtom(self, ctx: Any) -> ParamOp:
        """
        timeSeriesExpr: FILL_TIME_SERIES '(' expr (',' (SINGLE|ALL))? ')' ;
        """
        ctx_list = ctx.children
        c = ctx_list[0]

        op = c.text
        children_node = [self.visitExpr(ctx_list[2])]

        if len(ctx_list) > 4:
            param_constant_node = [
                ParamConstant(
                    type_="PARAM_TIMESERIES",
                    value=ctx_list[4].text,
                    **extract_token_info(ctx_list[4]),
                )
            ]
        else:
            param_constant_node = []

        return ParamOp(
            op=op, children=children_node, params=param_constant_node, **extract_token_info(ctx)
        )

    def visitTimeAggAtom(self, ctx: Any) -> TimeAggregation:
        """
        TIME_AGG LPAREN periodIndTo=STRING_CONSTANT (COMMA periodIndFrom=(STRING_CONSTANT| OPTIONAL ))? (COMMA op=optionalExpr)? (COMMA (FIRST|LAST))? RPAREN     # timeAggAtom
        """  # noqa E501
        ctx_list = ctx.children
        c = ctx_list[0]

        op = c.text

        # Find periodIndTo: first STRING_CONSTANT terminal
        period_to = None
        period_from = None
        optional_expr_node = None
        conf = None
        period_to_found = False

        for child in ctx_list:
            if child.is_terminal:
                if child.symbol_type == vtl_cpp_parser.STRING_CONSTANT:
                    if not period_to_found:
                        period_to = str(child.text)[1:-1]
                        period_to_found = True
                    else:
                        period_from = str(child.text)[1:-1]
                elif child.symbol_type == vtl_cpp_parser.OPTIONAL:
                    pass  # periodIndFrom is OPTIONAL, skip
                elif child.symbol_type in (vtl_cpp_parser.FIRST, vtl_cpp_parser.LAST):
                    conf = child.text
            elif not child.is_terminal and child.rule_index == RC.OPTIONAL_EXPR[0]:
                optional_expr_node = child

        conf_val = None if conf is None else conf

        if optional_expr_node is not None:
            operand_node = self.visitOptionalExpr(optional_expr_node)
            if isinstance(operand_node, ID):
                operand_node = None
            elif isinstance(operand_node, Identifier):
                operand_node = VarID(value=operand_node.value, **extract_token_info(ctx))
        else:
            operand_node = None

        if operand_node is None:
            raise SemanticError("1-3-2-4")
        return TimeAggregation(
            op=op,
            operand=operand_node,
            period_to=period_to,
            period_from=period_from,
            conf=conf_val,
            **extract_token_info(ctx),
        )

    def visitFlowAtom(self, ctx: Any) -> UnaryOp:
        ctx_list = ctx.children

        op_node = ctx_list[0].text
        operand_node = self.visitExpr(ctx_list[2])
        return UnaryOp(op=op_node, operand=operand_node, **extract_token_info(ctx))

    def visitCurrentDateAtom(self, ctx: Any) -> MulOp:
        c = ctx.children[0]
        return MulOp(op=c.text, children=[], **extract_token_info(ctx))

    def visitTimeDiffAtom(self, ctx: Any) -> BinOp:
        """ """
        ctx_list = ctx.children
        c = ctx_list[0]

        op = c.text
        left_node = self.visitExpr(ctx_list[2])
        right_node = self.visitExpr(ctx_list[4])

        return BinOp(left=left_node, op=op, right=right_node, **extract_token_info(ctx))

    def visitTimeAddAtom(self, ctx: Any) -> ParamOp:
        """ """

        ctx_list = ctx.children
        c = ctx_list[0]

        op = c.text
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

    def visitConditionalFunctions(self, ctx: Any) -> Any:
        if ctx.ctx_id == RC.NVL_ATOM:
            return self.visitNvlAtom(ctx)
        else:
            raise NotImplementedError

    def visitNvlAtom(self, ctx: Any) -> BinOp:
        ctx_list = ctx.children
        c = ctx_list[0]

        left_node = self.visitExpr(ctx_list[2])
        op_node = c.text
        right_node = self.visitExpr(ctx_list[4])
        return BinOp(left=left_node, op=op_node, right=right_node, **extract_token_info(ctx))

    """
                            -----------------------------------
                                    Set Functions
                            -----------------------------------
    """

    def visitSetFunctions(self, ctx: Any) -> Any:
        """
        setExpr:     UNION LPAREN left=expr (COMMA expr)+ RPAREN                             # unionAtom
                    | INTERSECT LPAREN left=expr (COMMA expr)+ RPAREN                       # intersectAtom
                    | op=(SETDIFF|SYMDIFF) LPAREN left=expr COMMA right=expr RPAREN         # setOrSYmDiffAtom
        """  # noqa E501
        if ctx.ctx_id == RC.UNION_ATOM:
            return self.visitUnionAtom(ctx)
        elif ctx.ctx_id == RC.INTERSECT_ATOM:
            return self.visitIntersectAtom(ctx)
        elif ctx.ctx_id == RC.SET_OR_SYM_DIFF_ATOM:
            return self.visitSetOrSYmDiffAtom(ctx)
        else:
            raise NotImplementedError

    def visitUnionAtom(self, ctx: Any) -> MulOp:
        ctx_list = ctx.children
        exprs_nodes = [
            self.visitExpr(expr)
            for expr in ctx_list
            if not expr.is_terminal and expr.rule_index == 2
        ]

        return MulOp(op=ctx_list[0].text, children=exprs_nodes, **extract_token_info(ctx))

    def visitIntersectAtom(self, ctx: Any) -> MulOp:
        ctx_list = ctx.children
        exprs_nodes = [
            self.visitExpr(expr)
            for expr in ctx_list
            if not expr.is_terminal and expr.rule_index == 2
        ]

        return MulOp(op=ctx_list[0].text, children=exprs_nodes, **extract_token_info(ctx))

    def visitSetOrSYmDiffAtom(self, ctx: Any) -> MulOp:
        ctx_list = ctx.children
        exprs_nodes = [
            self.visitExpr(expr)
            for expr in ctx_list
            if not expr.is_terminal and expr.rule_index == 2
        ]

        return MulOp(op=ctx_list[0].text, children=exprs_nodes, **extract_token_info(ctx))

    """
                            -----------------------------------
                                    Hierarchy Functions
                            -----------------------------------
    """

    def visitHierarchyFunctions(self, ctx: Any) -> HROperation:
        """
        HIERARCHY LPAREN op=expr COMMA hrName=IDENTIFIER (conditionClause)? (RULE ruleComponent=componentID)? (validationMode)? (inputModeHierarchy)? outputModeHierarchy? RPAREN
        """  # noqa E501
        ctx_list = ctx.children
        c = ctx_list[0]

        op = c.text
        dataset_node = self.visitExpr(ctx_list[2])
        ruleset_name = ctx_list[4].text

        conditions = []
        validation_mode: Optional[ValidationMode] = None
        input_mode: Optional[HRInputMode] = None
        output: Optional[HierarchyOutput] = None
        rule_comp = None

        for c in ctx_list:
            if c.is_terminal:
                continue
            if c.rule_index == RC.CONDITION_CLAUSE[0]:
                conditions.append(Terminals().visitConditionClause(c))
            elif c.rule_index == RC.COMPONENT_ID[0]:
                rule_comp = Terminals().visitComponentID(c)
            elif c.rule_index == RC.VALIDATION_MODE[0]:
                mode_str = Terminals().visitValidationMode(c)
                validation_mode = ValidationMode(mode_str)
            elif c.rule_index == RC.INPUT_MODE_HIERARCHY[0]:
                input_str = Terminals().visitInputModeHierarchy(c)
                if input_str == DATASET_PRIORITY:
                    msg = "Dataset Priority input mode on HR is not implemented"
                    raise NotImplementedError(msg)
                input_mode = HRInputMode(input_str)
            elif c.rule_index == RC.OUTPUT_MODE_HIERARCHY[0]:
                output_str = Terminals().visitOutputModeHierarchy(c)
                output = HierarchyOutput(output_str)

        conditions = conditions[0] if conditions else []

        if not rule_comp and ruleset_name in de_ruleset_elements:
            if isinstance(de_ruleset_elements[ruleset_name], list):
                rule_element = de_ruleset_elements[ruleset_name][-1]
            else:
                rule_element = de_ruleset_elements[ruleset_name]
            if rule_element.kind == "DatasetID":
                check_hierarchy_rule = rule_element.value
                rule_comp = Identifier(
                    value=check_hierarchy_rule, kind="ComponentID", **extract_token_info(ctx)
                )
            else:  # ValuedomainID
                raise SemanticError("1-1-10-4", op=op)

        return HROperation(
            op=op,
            dataset=dataset_node,
            ruleset_name=ruleset_name,
            rule_component=rule_comp,
            conditions=conditions if isinstance(conditions, list) else [conditions],
            validation_mode=validation_mode,
            input_mode=input_mode,
            output=output,
            **extract_token_info(ctx),
        )

    """
                            -----------------------------------
                                    Validation Functions
                            -----------------------------------
    """

    def visitValidationFunctions(self, ctx: Any) -> Any:
        if ctx.ctx_id == RC.VALIDATE_D_PRULESET:
            return self.visitValidateDPruleset(ctx)
        elif ctx.ctx_id == RC.VALIDATE_HR_RULESET:
            return self.visitValidateHRruleset(ctx)
        elif ctx.ctx_id == RC.VALIDATION_SIMPLE:
            return self.visitValidationSimple(ctx)

    def visitValidateDPruleset(self, ctx: Any) -> DPValidation:
        """
        validationDatapoint: CHECK_DATAPOINT '(' expr ',' IDENTIFIER (COMPONENTS componentID (',' componentID)*)? (INVALID|ALL_MEASURES|ALL)? ')' ;
        """  # noqa E501
        ctx_list = ctx.children

        dataset_node = self.visitExpr(ctx_list[2])
        ruleset_name = ctx_list[4].text

        components = [
            Terminals().visitComponentID(comp)
            for comp in ctx_list
            if not comp.is_terminal and comp.rule_index == RC.COMPONENT_ID[0]
        ]
        component_names = []
        for x in components:
            if isinstance(x, BinOp):
                component_names.append(x.right.value)
            else:
                component_names.append(x.value)

        # Default value for output is invalid (None means use default at interpretation)
        output: Optional[ValidationOutput] = None

        if not ctx_list[-2].is_terminal and ctx_list[-2].rule_index == RC.VALIDATION_OUTPUT[0]:
            output_str = Terminals().visitValidationOutput(ctx_list[-2])
            output = ValidationOutput(output_str)

        return DPValidation(
            dataset=dataset_node,
            ruleset_name=ruleset_name,
            components=component_names,
            output=output,
            **extract_token_info(ctx),
        )

    def visitValidateHRruleset(self, ctx: Any) -> HROperation:
        """
        CHECK_HIERARCHY LPAREN op=expr COMMA hrName=IDENTIFIER conditionClause? (RULE componentID)? validationMode? inputMode? validationOutput? RPAREN     # validateHRruleset
        """  # noqa E501

        ctx_list = ctx.children
        c = ctx_list[0]

        op = c.text

        dataset_node = self.visitExpr(ctx_list[2])
        ruleset_name = ctx_list[4].text

        conditions = []
        validation_mode: Optional[ValidationMode] = None
        input_mode: Optional[CHInputMode] = None
        output: Optional[ValidationOutput] = None
        rule_comp = None

        for c in ctx_list:
            if c.is_terminal:
                continue
            if c.rule_index == RC.CONDITION_CLAUSE[0]:
                conditions.append(Terminals().visitConditionClause(c))
            elif c.rule_index == RC.COMPONENT_ID[0]:
                rule_comp = Terminals().visitComponentID(c)
            elif c.rule_index == RC.VALIDATION_MODE[0]:
                mode_str = Terminals().visitValidationMode(c)
                validation_mode = ValidationMode(mode_str)
            elif c.rule_index == RC.INPUT_MODE[0]:
                input_str = Terminals().visitInputMode(c)
                if input_str == DATASET_PRIORITY:
                    msg = "Dataset Priority input mode on HR is not implemented"
                    raise NotImplementedError(msg)
                input_mode = CHInputMode(input_str)
            elif c.rule_index == RC.VALIDATION_OUTPUT[0]:
                output_str = Terminals().visitValidationOutput(c)
                output = ValidationOutput(output_str)

        # AST_ASTCONSTRUCTOR.22
        conditions = conditions[0] if conditions else []

        if not rule_comp and ruleset_name in de_ruleset_elements:
            if isinstance(de_ruleset_elements[ruleset_name], list):
                rule_element = de_ruleset_elements[ruleset_name][-1]
            else:
                rule_element = de_ruleset_elements[ruleset_name]

            if rule_element.kind == "DatasetID":
                check_hierarchy_rule = rule_element.value
                rule_comp = Identifier(
                    value=check_hierarchy_rule,
                    kind="ComponentID",
                    **extract_token_info(ctx),
                )
            else:  # ValuedomainID
                raise SemanticError("1-1-10-4", op=op)

        return HROperation(
            op=op,
            dataset=dataset_node,
            ruleset_name=ruleset_name,
            rule_component=rule_comp,
            conditions=conditions if isinstance(conditions, list) else [conditions],
            validation_mode=validation_mode,
            input_mode=input_mode,
            output=output,
            **extract_token_info(ctx),
        )

    def visitValidationSimple(self, ctx: Any) -> Validation:
        """
        | CHECK LPAREN op=expr (codeErr=erCode)? (levelCode=erLevel)? imbalanceExpr?  output=(INVALID|ALL)? RPAREN	        # validationSimple
        """  # noqa E501
        ctx_list = ctx.children
        c = ctx_list[0]

        validation_node = self.visitExpr(ctx_list[2])

        inbalance_node = None
        error_code = None
        error_level = None
        for param in ctx_list:
            if param.is_terminal:
                continue
            if param.rule_index == RC.ER_CODE[0]:
                error_code = Terminals().visitErCode(param)
            elif param.rule_index == RC.ER_LEVEL[0]:
                error_level = Terminals().visitErLevel(param)
            elif param.rule_index == RC.IMBALANCE_EXPR[0]:
                inbalance_node = self.visitImbalanceExpr(param)

        invalid = ctx_list[-2] if ctx_list[-2].is_terminal else None
        invalid_value = False if invalid is None else invalid.text == "invalid"

        return Validation(
            op=c.text,
            validation=validation_node,
            error_code=error_code,
            error_level=error_level,
            imbalance=inbalance_node,
            invalid=invalid_value,
            **extract_token_info(ctx),
        )

    def visitImbalanceExpr(self, ctx: Any) -> Any:
        ctx_list = ctx.children
        return self.visitExpr(ctx_list[1])

    """
                            -----------------------------------
                                    Aggregate Functions
                            -----------------------------------
    """

    # TODO Count function count() without parameters. Used at least in aggregations at having.
    def visitAggregateFunctions(self, ctx: Any) -> Any:
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
        if ctx.ctx_id == RC.AGGR_DATASET:
            return self.visitAggrDataset(ctx)
        else:
            raise NotImplementedError

    def visitAggrDataset(self, ctx: Any) -> Aggregation:
        ctx_list = ctx.children

        grouping_op = None
        group_node = None
        have_node = None

        groups = [
            group
            for group in ctx_list
            if not group.is_terminal
            and group.rule_index == RC.GROUP_BY_OR_EXCEPT[0]  # GroupingClause rule_index = 56
        ]
        haves = [
            have
            for have in ctx_list
            if not have.is_terminal and have.rule_index == RC.HAVING_CLAUSE[0]
        ]

        op_node = ctx_list[0].text
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

    def visitAnalyticFunctions(self, ctx: Any) -> Any:
        if ctx.ctx_id == RC.AN_SIMPLE_FUNCTION:
            return self.visitAnSimpleFunction(ctx)
        elif ctx.ctx_id == RC.LAG_OR_LEAD_AN:
            return self.visitLagOrLeadAn(ctx)
        elif ctx.ctx_id == RC.RATIO_TO_REPORT_AN:
            return self.visitRatioToReportAn(ctx)
        else:
            raise NotImplementedError

    def visitAnSimpleFunction(self, ctx: Any) -> Analytic:
        ctx_list = ctx.children

        window = None
        partition_by = None
        order_by = None

        op_node = ctx_list[0].text
        operand = self.visitExpr(ctx_list[2])

        for c in ctx_list[5:-2]:
            if not c.is_terminal and c.rule_index == RC.PARTITION_BY_CLAUSE[0]:
                partition_by = Terminals().visitPartitionByClause(c)
                continue
            elif not c.is_terminal and c.rule_index == RC.ORDER_BY_CLAUSE[0]:
                order_by = Terminals().visitOrderByClause(c)
                continue
            elif not c.is_terminal and c.rule_index == RC.WINDOWING_CLAUSE[0]:
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

    def visitLagOrLeadAn(self, ctx: Any) -> Analytic:
        ctx_list = ctx.children

        params = None
        partition_by = None
        order_by = None

        op_node = ctx_list[0].text
        operand = self.visitExpr(ctx_list[2])

        for c in ctx_list[4:-2]:
            if c.is_terminal:
                continue
            if c.rule_index == RC.PARTITION_BY_CLAUSE[0]:
                partition_by = Terminals().visitPartitionByClause(c)
                continue
            elif c.rule_index == RC.ORDER_BY_CLAUSE[0]:
                order_by = Terminals().visitOrderByClause(c)
                continue
            elif c.rule_index == RC.SIGNED_INTEGER[0] or c.ctx_id in (
                RC.SIMPLE_SCALAR,
                RC.SCALAR_WITH_CAST,
            ):
                if params is None:
                    params = []
                if c.rule_index == RC.SIGNED_INTEGER[0]:
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

    def visitRatioToReportAn(self, ctx: Any) -> Analytic:
        ctx_list = ctx.children

        order_by = None

        op_node = ctx_list[0].text
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

    def visitDatasetClause(self, ctx: Any) -> Any:
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
        ctx_list = ctx.children
        c = ctx_list[0]

        # RENAME renameClause
        if not c.is_terminal and c.rule_index == RC.RENAME_CLAUSE[0]:
            return self.visitRenameClause(c)

        # aggrClause
        elif not c.is_terminal and c.rule_index == RC.AGGR_CLAUSE[0]:
            return self.visitAggrClause(c)

        # filterClause
        elif not c.is_terminal and c.rule_index == RC.FILTER_CLAUSE[0]:
            return self.visitFilterClause(c)

        # calcClause
        elif not c.is_terminal and c.rule_index == RC.CALC_CLAUSE[0]:
            return self.visitCalcClause(c)

        # keepClause
        elif not c.is_terminal and c.rule_index == RC.KEEP_OR_DROP_CLAUSE[0]:
            return self.visitKeepOrDropClause(c)

        # pivotExpr
        elif not c.is_terminal and c.rule_index == RC.PIVOT_OR_UNPIVOT_CLAUSE[0]:
            return self.visitPivotOrUnpivotClause(c)

        # subspaceExpr
        elif not c.is_terminal and c.rule_index == RC.SUBSPACE_CLAUSE[0]:
            return self.visitSubspaceClause(c)

    """
                    -----------------------------------
                            Rename Clause
                    -----------------------------------
    """

    def visitRenameClause(self, ctx: Any) -> RegularAggregation:
        """
        renameClause: RENAME renameClauseItem (COMMA renameClauseItem)*;
        """
        ctx_list = ctx.children

        renames = [
            ctx_child
            for ctx_child in ctx_list
            if not ctx_child.is_terminal and ctx_child.rule_index == RC.RENAME_CLAUSE_ITEM[0]
        ]
        rename_nodes = []

        for ctx_rename in renames:
            rename_nodes.append(self.visitRenameClauseItem(ctx_rename))

        return RegularAggregation(
            op=ctx_list[0].text, children=rename_nodes, **extract_token_info(ctx)
        )

    def visitRenameClauseItem(self, ctx: Any) -> RenameNode:
        """
        renameClauseItem: fromName=componentID TO toName=componentID;
        """
        ctx_list = ctx.children

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

    def visitAggregateClause(self, ctx: Any) -> list:
        """
        aggregateClause: aggrFunctionClause (',' aggrFunctionClause)* ;
        """
        ctx_list = ctx.children

        aggregates_nodes = []

        aggregates = [
            aggregate
            for aggregate in ctx_list
            if not aggregate.is_terminal and aggregate.rule_index == RC.AGGR_FUNCTION_CLAUSE[0]
        ]

        for agg in aggregates:
            aggregates_nodes.append(self.visitAggrFunctionClause(agg))

        return aggregates_nodes

    def visitAggrFunctionClause(self, ctx: Any) -> Assignment:
        """
        aggrFunctionClause: (componentRole)? componentID ':=' aggrFunction ;
        """
        ctx_list = ctx.children
        c = ctx_list[0]

        if not c.is_terminal and c.rule_index == RC.COMPONENT_ROLE[0]:
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

    def visitAggrClause(self, ctx: Any) -> RegularAggregation:
        """
        aggrClause: AGGREGATE aggregateClause (groupingClause havingClause?)? ;
        """
        ctx_list = ctx.children
        c = ctx_list[0]

        op_node = c.text
        group_node = None
        grouping_op = None
        have_node = None

        groups = [
            group
            for group in ctx_list
            if not group.is_terminal
            and group.rule_index == RC.GROUP_BY_OR_EXCEPT[0]  # GroupingClause rule_index = 56
        ]
        haves = [
            have
            for have in ctx_list
            if not have.is_terminal and have.rule_index == RC.HAVING_CLAUSE[0]
        ]

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

    def visitGroupingClause(self, ctx: Any) -> Any:
        """
        groupingClause:
            GROUP op=(BY | EXCEPT) componentID (COMMA componentID)*     # groupByOrExcept
            | GROUP ALL exprComponent                                   # groupAll
          ;
        """
        if ctx.ctx_id == RC.GROUP_BY_OR_EXCEPT:
            return self.visitGroupByOrExcept(ctx)
        elif ctx.ctx_id == RC.GROUP_ALL:
            return self.visitGroupAll(ctx)
        else:
            raise NotImplementedError

    def visitHavingClause(self, ctx: Any) -> Any:
        """
        havingClause: HAVING exprComponent ;
        """
        ctx_list = ctx.children
        op_node = ctx_list[0].text

        strdata = vtl_cpp_parser.get_input_text()
        # Extract the relevant substring using position information from the exprComponent node
        expr_component = ctx_list[1]
        # Use the start position of the having keyword to extract the having clause text
        start_line = ctx.start_line
        # Find 'having' in strdata and extract from there
        lines = strdata.split("\n")
        # Build the text from start_line to end
        text_from_having = "\n".join(lines[start_line - 1 :])
        # Find 'having' keyword position within that line
        line_text = lines[start_line - 1]
        having_pos = line_text.lower().find("having")
        if having_pos >= 0:
            text_from_having = line_text[having_pos:] + "\n" + "\n".join(lines[start_line:])

        expr = re.split("having", text_from_having)[1]
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

        if expr_component.ctx_id == RC.COMPARISON_EXPR_COMP:
            param_nodes = ExprComp().visitComparisonExprComp(expr_component)
        elif expr_component.ctx_id == RC.IN_NOT_IN_EXPR_COMP:
            param_nodes = ExprComp().visitInNotInExprComp(expr_component)
        elif expr_component.ctx_id == RC.BOOLEAN_EXPR_COMP:
            param_nodes = ExprComp().visitBooleanExprComp(expr_component)
        else:
            raise NotImplementedError

        return ParamOp(
            op=op_node, children=None, params=param_nodes, **extract_token_info(ctx)
        ), expr

    def visitGroupByOrExcept(self, ctx: Any) -> Any:
        ctx_list = ctx.children

        token_left = ctx_list[0].text
        token_right = ctx_list[1].text

        op_node = token_left + " " + token_right

        children_nodes = [
            Terminals().visitComponentID(identifier)
            for identifier in ctx_list
            if not identifier.is_terminal and identifier.rule_index == RC.COMPONENT_ID[0]
        ]

        return op_node, children_nodes

    def visitGroupAll(self, ctx: Any) -> Any:
        ctx_list = ctx.children

        token_left = ctx_list[0].text
        token_right = ctx_list[1].text

        op_node = token_left + " " + token_right

        children_nodes: list = []

        # Check if TIME_AGG is present (more than just GROUP ALL)
        if len(ctx_list) > 2:
            period_to = None
            period_from = None
            operand_node = None
            conf = None

            for child in ctx_list:
                if child.is_terminal:
                    if child.symbol_type == vtl_cpp_parser.STRING_CONSTANT:
                        if period_to is None:
                            period_to = child.text[1:-1]
                        else:
                            period_from = child.text[1:-1]
                    elif child.symbol_type in (vtl_cpp_parser.FIRST, vtl_cpp_parser.LAST):
                        conf = child.text
                elif not child.is_terminal and child.rule_index == RC.OPTIONAL_EXPR[0]:
                    operand_node = self.visitOptionalExpr(child)
                    if isinstance(operand_node, ID):
                        operand_node = None
                    elif isinstance(operand_node, Identifier):
                        operand_node = VarID(value=operand_node.value, **extract_token_info(child))

            children_nodes = [
                TimeAggregation(
                    op="time_agg",
                    operand=operand_node,
                    period_to=period_to,
                    period_from=period_from,
                    conf=conf,
                    **extract_token_info(ctx),
                )
            ]

        return op_node, children_nodes

    """
                    -----------------------------------
                            Filter Clause
                    -----------------------------------
    """

    def visitFilterClause(self, ctx: Any) -> RegularAggregation:
        """
        filterClause: FILTER expr;
        """
        ctx_list = ctx.children
        c = ctx_list[0]

        op_node = c.text
        operand_nodes = []
        operand_nodes.append(ExprComp().visitExprComponent(ctx_list[1]))

        return RegularAggregation(op=op_node, children=operand_nodes, **extract_token_info(ctx))

    """
                    -----------------------------------
                            Calc Clause
                    -----------------------------------
    """

    def visitCalcClause(self, ctx: Any) -> RegularAggregation:
        """
        calcClause: CALC calcClauseItem (',' calcClauseItem)*;
        """
        ctx_list = ctx.children
        c = ctx_list[0]

        calcClauseItems = [
            calcClauseItem
            for calcClauseItem in ctx_list
            if not calcClauseItem.is_terminal
            and calcClauseItem.rule_index == RC.CALC_CLAUSE_ITEM[0]
        ]
        calcClauseItems_nodes = []

        op_node = c.text
        for calcClauseItem in calcClauseItems:
            result = self.visitCalcClauseItem(calcClauseItem)
            calcClauseItems_nodes.append(result)

        return RegularAggregation(
            op=op_node, children=calcClauseItems_nodes, **extract_token_info(ctx)
        )

    def visitCalcClauseItem(self, ctx: Any) -> UnaryOp:
        """
        calcClauseItem: (componentRole)? componentID  ASSIGN  exprComponent;
        """
        ctx_list = ctx.children
        c = ctx_list[0]

        if not c.is_terminal and c.rule_index == RC.COMPONENT_ROLE[0]:
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

    def visitKeepOrDropClause(self, ctx: Any) -> RegularAggregation:
        """
        keepOrDropClause: op = (KEEP | DROP) componentID (COMMA componentID)* ;
        """

        ctx_list = ctx.children
        c = ctx_list[0]

        items = [
            item
            for item in ctx_list
            if not item.is_terminal and item.rule_index == RC.COMPONENT_ID[0]
        ]
        nodes = []

        op_node = c.text
        for item in items:
            nodes.append(Terminals().visitComponentID(item))

        return RegularAggregation(op=op_node, children=nodes, **extract_token_info(ctx))

    """
                    -----------------------------------
                            Pivot/Unpivot Clause
                    -----------------------------------
    """

    def visitPivotOrUnpivotClause(self, ctx: Any) -> RegularAggregation:
        """
        pivotOrUnpivotClause: op=(PIVOT|UNPIVOT) id_=componentID COMMA mea=componentID ;
        """
        ctx_list = ctx.children
        c = ctx_list[0]

        op_node = c.text
        children_nodes = []
        children_nodes.append(Terminals().visitComponentID(ctx_list[1]))
        children_nodes.append(Terminals().visitComponentID(ctx_list[3]))

        return RegularAggregation(op=op_node, children=children_nodes, **extract_token_info(ctx))

    """
                    -----------------------------------
                            Subspace Clause
                    -----------------------------------
    """

    def visitSubspaceClause(self, ctx: Any) -> RegularAggregation:
        """
        subspaceClause: SUBSPACE subspaceClauseItem (COMMA subspaceClauseItem)*;"""
        ctx_list = ctx.children
        c = ctx_list[0]

        subspace_nodes = []
        subspaces = [
            subspace
            for subspace in ctx_list
            if not subspace.is_terminal and subspace.rule_index == RC.SUBSPACE_CLAUSE_ITEM[0]
        ]

        for subspace in subspaces:
            subspace_nodes.append(self.visitSubspaceClauseItem(subspace))

        op_node = c.text
        return RegularAggregation(op=op_node, children=subspace_nodes, **extract_token_info(ctx))

    def visitSubspaceClauseItem(self, ctx: Any) -> BinOp:
        ctx_list = ctx.children

        left_node = Terminals().visitVarID(ctx_list[0])
        op_node = ctx_list[1].text
        if not ctx_list[2].is_terminal and ctx_list[2].ctx_id == RC.SCALAR_WITH_CAST:
            right_node = Terminals().visitScalarWithCast(ctx_list[2])
        elif not ctx_list[2].is_terminal and ctx_list[2].ctx_id in (
            RC.SIMPLE_SCALAR,
            RC.SCALAR_WITH_CAST,
        ):
            right_node = Terminals().visitScalarItem(ctx_list[2])
        else:
            right_node = Terminals().visitVarID(ctx_list[2])
        return BinOp(left=left_node, op=op_node, right=right_node, **extract_token_info(ctx))

    def visitOptionalExpr(self, ctx: Any) -> Any:
        """
        optionalExpr: expr
                    | OPTIONAL ;
        """
        ctx_list = ctx.children
        c = ctx_list[0]

        if not c.is_terminal and c.rule_index == 2:
            return self.visitExpr(c)

        elif c.is_terminal:
            opt = c.text
            return ID(type_="OPTIONAL", value=opt, **extract_token_info(ctx))
