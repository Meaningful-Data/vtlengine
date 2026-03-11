from vtlengine.AST import (
    BinOp,
    Collection,
    Constant,
    DPRIdentifier,
    Identifier,
    OrderBy,
    ParamConstant,
    ParamOp,
    VarID,
    Windowing,
)
from vtlengine.AST.ASTConstructorModules import extract_token_info
from vtlengine.AST.Grammar._cpp_parser import vtl_cpp_parser
from vtlengine.AST.Grammar._cpp_parser._rule_constants import RC
from vtlengine.DataTypes import (
    Boolean,
    Date,
    Duration,
    Integer,
    Number,
    String,
    TimeInterval,
    TimePeriod,
)
from vtlengine.Model import Component, Dataset, Role, Scalar


def _remove_scaped_characters(text):
    has_scaped_char = text.find("'") != -1
    if has_scaped_char:
        text = str(text.replace("'", ""))
    return text


class Terminals:
    def visitConstant(self, ctx):
        # constant: signedInteger | signedNumber | BOOLEAN_CONSTANT |
        #           STRING_CONSTANT | NULL_CONSTANT
        child = ctx.children[0]
        token_info = extract_token_info(ctx)

        if not child.is_terminal and child.ctx_id == RC.SIGNED_INTEGER:
            constant_node = Constant(
                type_="INTEGER_CONSTANT",
                value=self.visitSignedInteger(child),
                **token_info,
            )

        elif not child.is_terminal and child.ctx_id == RC.SIGNED_NUMBER:
            constant_node = Constant(
                type_="FLOAT_CONSTANT",
                value=self.visitSignedNumber(child),
                **token_info,
            )

        else:
            if child.symbol_type == vtl_cpp_parser.BOOLEAN_CONSTANT:
                if child.text == "true":
                    constant_node = Constant(type_="BOOLEAN_CONSTANT", value=True, **token_info)
                elif child.text == "false":
                    constant_node = Constant(type_="BOOLEAN_CONSTANT", value=False, **token_info)
                else:
                    raise NotImplementedError

            elif child.symbol_type == vtl_cpp_parser.STRING_CONSTANT:
                constant_node = Constant(
                    type_="STRING_CONSTANT", value=child.text[1:-1], **token_info
                )

            elif child.symbol_type == vtl_cpp_parser.NULL_CONSTANT:
                constant_node = Constant(type_="NULL_CONSTANT", value=None, **token_info)

            else:
                raise NotImplementedError

        return constant_node

    def visitVarID(self, ctx):
        token = ctx.children[0]
        text = _remove_scaped_characters(token.text)
        token_info = extract_token_info(token)
        var_id_node = VarID(value=text, **token_info)
        return var_id_node

    def visitVarIdExpr(self, ctx):
        if not ctx.children[0].is_terminal and ctx.children[0].ctx_id == RC.VAR_ID:
            return self.visitVarID(ctx.children[0])

        token = ctx.children[0]
        # check token text
        text = _remove_scaped_characters(token.text)
        token_info = extract_token_info(token)
        var_id_node = VarID(value=text, **token_info)
        return var_id_node

    def visitSimpleComponentId(self, ctx):
        """
        componentID: IDENTIFIER ;
        """
        token = ctx.children[0]
        # check token text
        text = _remove_scaped_characters(token.text)

        return Identifier(value=text, kind="ComponentID", **extract_token_info(ctx))

    def visitComponentID(self, ctx):
        ctx_list = ctx.children

        if len(ctx_list) == 1:
            component_name = ctx_list[0].text
            if component_name.startswith("'") and component_name.endswith(
                "'"
            ):  # The component could be imbalance, errorcode or errorlevel
                component_name = component_name[1:-1]
            return Identifier(
                value=component_name,
                kind="ComponentID",
                **extract_token_info(ctx_list[0]),
            )
        else:
            component_name = ctx_list[2].text
            if component_name.startswith("'") and component_name.endswith(
                "'"
            ):  # The component could be imbalance, errorcode or errorlevel
                component_name = component_name[1:-1]
            op_node = ctx_list[1].text
            return BinOp(
                left=Identifier(
                    value=ctx_list[0].text,
                    kind="DatasetID",
                    **extract_token_info(ctx_list[0]),
                ),
                op=op_node,
                right=Identifier(
                    value=component_name,
                    kind="ComponentID",
                    **extract_token_info(ctx_list[1]),
                ),
                **extract_token_info(ctx),
            )

    def visitOperatorID(self, ctx):
        """
        operatorID: IDENTIFIER ;
        """
        ctx_list = ctx.children
        c = ctx_list[0]
        return c.text

    def visitValueDomainID(self, ctx):
        """
        valueDomainID: IDENTIFIER ;
        """
        return Collection(
            name=ctx.children[0].text,
            children=[],
            kind="ValueDomain",
            type="",
            **extract_token_info(ctx),
        )

    def visitRulesetID(self, ctx):
        """
        rulesetID: IDENTIFIER ;
        """
        ctx_list = ctx.children
        c = ctx_list[0]
        return c.text

    def visitValueDomainName(self, ctx):
        """
        valueDomainName: IDENTIFIER ;
        """
        ctx_list = ctx.children
        # AST_ASTCONSTRUCTOR.48
        raise NotImplementedError(
            "Value Domain '{}' not available for cast operator or scalar type "
            "representation or rulesets.".format(ctx_list[0].text)
        )

    def visitValueDomainValue(self, ctx):
        child = ctx.children[0]
        if not child.is_terminal and child.ctx_id in (RC.SIGNED_INTEGER, RC.SIGNED_NUMBER):
            return child.text
        return _remove_scaped_characters(child.text)

    def visitRoutineName(self, ctx):
        """
        routineName: IDENTIFIER ;
        """
        ctx_list = ctx.children
        c = ctx_list[0]

        return c.text

    def visitBasicScalarType(self, ctx):
        """
        basicScalarType: STRING
                       | INTEGER
                       | NUMBER
                       | BOOLEAN
                       | DATE
                       | TIME_PERIOD
                       | DURATION
                       | SCALAR
                       | TIME
                       ;
        """
        ctx_list = ctx.children
        c = ctx_list[0]

        if c.symbol_type == vtl_cpp_parser.STRING:
            return String
        elif c.symbol_type == vtl_cpp_parser.INTEGER:
            return Integer
        elif c.symbol_type == vtl_cpp_parser.NUMBER:
            return Number
        elif c.symbol_type == vtl_cpp_parser.BOOLEAN:
            return Boolean
        elif c.symbol_type == vtl_cpp_parser.DATE:
            return Date
        elif c.symbol_type == vtl_cpp_parser.TIME_PERIOD:
            return TimePeriod
        elif c.symbol_type == vtl_cpp_parser.DURATION:
            return Duration
        elif c.symbol_type == vtl_cpp_parser.SCALAR:
            return "Scalar"
        elif c.symbol_type == vtl_cpp_parser.TIME:
            return TimeInterval

    def visitComponentRole(self, ctx):
        """
        componentRole: MEASURE
                     |COMPONENT
                     |DIMENSION
                     |ATTRIBUTE
                     |viralAttribute
                     ;
        """
        ctx_list = ctx.children
        c = ctx_list[0]

        if not c.is_terminal and c.ctx_id == RC.VIRAL_ATTRIBUTE:
            return self.visitViralAttribute(c)
        else:
            text = c.text
            if text == "component":
                return None
            # Use upper case on first letter
            text = text[0].upper() + text[1:].lower()
            return Role(text)

    def visitViralAttribute(self, ctx):
        """
        viralAttribute: VIRAL ATTRIBUTE;
        """
        # ctx_list = ctx.children
        # c = ctx_list[0]

        raise NotImplementedError

    def visitLists(self, ctx):
        """
        lists:  GLPAREN  scalarItem (COMMA scalarItem)*  GRPAREN
        """
        ctx_list = ctx.children

        scalar_nodes = []

        scalars = [
            scalar
            for scalar in ctx_list
            if not scalar.is_terminal and scalar.ctx_id == RC.SIMPLE_SCALAR
        ]

        scalars_with_cast = [
            scalar
            for scalar in ctx_list
            if not scalar.is_terminal and scalar.ctx_id == RC.SCALAR_WITH_CAST
        ]

        for scalar in scalars:
            scalar_nodes.append(self.visitSimpleScalar(scalar))

        for scalar_with_cast in scalars_with_cast:
            scalar_nodes.append(self.visitScalarWithCast(scalar_with_cast))

        return Collection(
            name="List", type="Lists", children=scalar_nodes, **extract_token_info(ctx)
        )

    def visitMultModifier(self, ctx):
        """
        multModifier: OPTIONAL  ( PLUS | MUL )?;
        """
        pass

    def visitCompConstraint(self, ctx):
        """
        compConstraint: componentType (componentID|multModifier) ;
        """
        ctx_list = ctx.children

        component_node = [
            self.visitComponentType(component)
            for component in ctx_list
            if not component.is_terminal and component.ctx_id == RC.COMPONENT_TYPE
        ]
        component_name = [
            self.visitComponentID(component).value
            for component in ctx_list
            if not component.is_terminal and component.ctx_id == RC.COMPONENT_ID
        ]
        component_mult = [
            self.visitMultModifier(modifier)
            for modifier in ctx_list
            if not modifier.is_terminal and modifier.ctx_id == RC.MULT_MODIFIER
        ]

        if len(component_mult) != 0:
            # AST_ASTCONSTRUCTOR.51
            raise NotImplementedError

        component_node[0].name = component_name[0]
        return component_node[0]

    def visitSimpleScalar(self, ctx):
        ctx_list = ctx.children
        c = ctx_list[0]
        if not c.is_terminal and c.ctx_id == RC.CONSTANT:
            return self.visitConstant(c)
        else:
            raise NotImplementedError

    def visitScalarType(self, ctx):
        """
        scalarType: (basicScalarType|valueDomainName)scalarTypeConstraint?((NOT)? NULL_CONSTANT)? ;
        """
        ctx_list = ctx.children

        type_ctx_ids = (
            RC.BASIC_SCALAR_TYPE,
            RC.VALUE_DOMAIN_NAME,
            RC.CONDITION_CONSTRAINT,
            RC.RANGE_CONSTRAINT,
        )
        scalartype = [
            scalartype
            for scalartype in ctx_list
            if not scalartype.is_terminal and scalartype.ctx_id in type_ctx_ids
        ][0]

        scalartype_constraint = [
            constraint
            for constraint in ctx_list
            if not constraint.is_terminal
            and constraint.ctx_id in (RC.CONDITION_CONSTRAINT, RC.RANGE_CONSTRAINT)
        ]
        not_ = [
            not_.text
            for not_ in ctx_list
            if not_.is_terminal and not_.symbol_type == vtl_cpp_parser.NOT
        ]
        null_constant = [
            null.text
            for null in ctx_list
            if null.is_terminal and null.symbol_type == vtl_cpp_parser.NULL_CONSTANT
        ]

        if not scalartype.is_terminal and scalartype.ctx_id == RC.BASIC_SCALAR_TYPE:
            if scalartype.children[0].symbol_type == vtl_cpp_parser.SCALAR:
                return Scalar(name="", data_type=None, value=None)
            type_node = self.visitBasicScalarType(scalartype)

        else:
            raise SyntaxError(
                f"Invalid parameter type definition {scalartype.children[0].text} at line "
                f"{ctx.start_line}:{ctx.start_column}."
            )

        if len(scalartype_constraint) != 0:
            # AST_ASTCONSTRUCTOR.45
            raise NotImplementedError

        if len(not_) != 0:
            # AST_ASTCONSTRUCTOR.46
            raise NotImplementedError

        if len(null_constant) != 0:
            # AST_ASTCONSTRUCTOR.47
            raise NotImplementedError

        return type_node

    def visitDatasetType(self, ctx):
        """
        datasetType: DATASET ('{'compConstraint (',' compConstraint)* '}' )? ;
        """
        ctx_list = ctx.children

        components = [
            self.visitCompConstraint(constraint)
            for constraint in ctx_list
            if not constraint.is_terminal and constraint.ctx_id == RC.COMP_CONSTRAINT
        ]
        components = {component.name: component for component in components}

        return Dataset(name="Dataset", components=components, data=None)

    def visitRulesetType(self, ctx):
        """
        rulesetType: RULESET
                   | dpRuleset
                   | hrRuleset
                   ;
        """
        raise NotImplementedError

    def visitDpRuleset(self, ctx):
        """
        DATAPOINT                                                                               # dataPoint
            | DATAPOINT_ON_VD  (GLPAREN  valueDomainName (MUL valueDomainName)*  GRPAREN )?         # dataPointVd
            | DATAPOINT_ON_VAR  (GLPAREN  varID (MUL varID)*  GRPAREN )?                            # dataPointVar
        ;
        """  # noqa E501
        # AST_ASTCONSTRUCTOR.54
        raise NotImplementedError

    def visitHrRuleset(self, ctx):
        """
        hrRuleset: HIERARCHICAL                                                                                                            # hrRulesetType
            | HIERARCHICAL_ON_VD ( GLPAREN  vdName=IDENTIFIER (LPAREN valueDomainName (MUL valueDomainName)* RPAREN)?  GRPAREN )?   # hrRulesetVdType
            | HIERARCHICAL_ON_VAR ( GLPAREN  varName=varID (LPAREN  varID (MUL varID)* RPAREN)?  GRPAREN )?                         # hrRulesetVarType
        ;
        """  # noqa E501
        # AST_ASTCONSTRUCTOR.55
        raise NotImplementedError

    def visitComponentType(self, ctx):
        """
        componentType:  componentRole ( LT   scalarType  MT  )?
        """
        ctx_list = ctx.children

        role_node = self.visitComponentRole(ctx_list[0])
        data_type = [
            self.visitScalarType(constraint)
            for constraint in ctx_list
            if not constraint.is_terminal and constraint.ctx_id == RC.SCALAR_TYPE
        ]
        data_type = data_type[0] if len(data_type) > 0 else String()

        nullable = role_node != Role.IDENTIFIER

        return Component(name="Component", data_type=data_type, role=role_node, nullable=nullable)

    def visitInputParameterType(self, ctx):
        """
        inputParameterType:
            scalarType
            | datasetType
            | scalarSetType
            | rulesetType
            | componentType
        ;
        """
        ctx_list = ctx.children
        c = ctx_list[0]

        if not c.is_terminal and c.ctx_id == RC.SCALAR_TYPE:
            return self.visitScalarType(c)

        elif not c.is_terminal and c.ctx_id == RC.DATASET_TYPE:
            return self.visitDatasetType(c)

        elif not c.is_terminal and c.ctx_id == RC.SCALAR_SET_TYPE:
            return self.visitScalarSetType(c)

        elif not c.is_terminal and c.ctx_id == RC.RULESET_TYPE:
            return self.visitRulesetType(c)

        elif not c.is_terminal and c.ctx_id == RC.COMPONENT_TYPE:
            return self.visitComponentType(c)
        else:
            raise NotImplementedError

    def visitOutputParameterType(self, ctx):
        """
        outputParameterType: scalarType
                           | datasetType
                           | componentType
                           ;
        """
        ctx_list = ctx.children
        c = ctx_list[0]

        if not c.is_terminal and c.ctx_id == RC.SCALAR_TYPE:
            # return self.visitScalarType(c).__class__.__name__
            return "Scalar"

        elif not c.is_terminal and c.ctx_id == RC.DATASET_TYPE:
            return "Dataset"

        elif not c.is_terminal and c.ctx_id == RC.COMPONENT_TYPE:
            return "Component"
        else:
            raise NotImplementedError

    def visitOutputParameterTypeComponent(self, ctx):
        """
        outputParameterType: scalarType
                           | componentType
                           ;
        """
        ctx_list = ctx.children
        c = ctx_list[0]

        if not c.is_terminal and c.ctx_id == RC.SCALAR_TYPE:
            return self.visitScalarType(c)

        elif not c.is_terminal and c.ctx_id == RC.COMPONENT_TYPE:
            return self.visitComponentType(c)
        else:
            raise NotImplementedError

    def visitScalarItem(self, ctx):
        # ctx is a scalarItem node (either simpleScalar or scalarWithCast alternative)
        if ctx.ctx_id == RC.SIMPLE_SCALAR:
            return self.visitSimpleScalar(ctx)
        elif ctx.ctx_id == RC.SCALAR_WITH_CAST:
            return self.visitScalarWithCast(ctx)
        else:
            raise NotImplementedError

    def visitScalarWithCast(self, ctx):
        """
        |  CAST LPAREN constant COMMA (basicScalarType) (COMMA STRING_CONSTANT)? RPAREN    #scalarWithCast  # noqa E501
        """  # noqa E501
        ctx_list = ctx.children
        c = ctx_list[0]

        op = c.text
        const_node = self.visitConstant(ctx_list[2])
        basic_scalar_type = [self.visitBasicScalarType(ctx_list[4])]

        param_node = (
            [
                ParamConstant(
                    type_="PARAM_CAST", value=ctx_list[6], **extract_token_info(ctx_list[6])
                )
            ]
            if len(ctx_list) > 6
            else []
        )

        if len(basic_scalar_type) == 1:
            children_nodes = [const_node, basic_scalar_type[0]]

            return ParamOp(
                op=op, children=children_nodes, params=param_node, **extract_token_info(ctx)
            )

        else:
            # AST_ASTCONSTRUCTOR.14
            raise NotImplementedError

    def visitScalarSetType(self, ctx):
        """
        scalarSetType: SET ('<' scalarType '>')? ;
        """
        # AST_ASTCONSTRUCTOR.60
        raise NotImplementedError

    def visitRetainType(self, ctx):
        """
        retainType: BOOLEAN_CONSTANT
                  | ALL
                  ;
        """
        token = ctx.children[0]

        if token.symbol_type == vtl_cpp_parser.BOOLEAN_CONSTANT:
            if token.text == "true":
                param_constant_node = Constant(
                    type_="BOOLEAN_CONSTANT", value=True, **extract_token_info(token)
                )
            elif token.text == "false":
                param_constant_node = Constant(
                    type_="BOOLEAN_CONSTANT", value=False, **extract_token_info(token)
                )
            else:
                raise NotImplementedError

        elif token.symbol_type == vtl_cpp_parser.ALL:
            param_constant_node = ParamConstant(
                type_="PARAM_CONSTANT", value=token.text, **extract_token_info(token)
            )

        else:
            raise NotImplementedError

        return param_constant_node

    def visitEvalDatasetType(self, ctx):
        ctx_list = ctx.children
        c = ctx_list[0]

        if not c.is_terminal and c.ctx_id == RC.DATASET_TYPE:
            return self.visitDatasetType(c)
        elif not c.is_terminal and c.ctx_id == RC.SCALAR_TYPE:
            return self.visitScalarType(c)
        else:
            raise NotImplementedError

    def visitAlias(self, ctx):
        return ctx.children[0].text

    def visitSignedInteger(self, ctx):
        # signedInteger: (MINUS|PLUS)? INTEGER_CONSTANT
        return int(ctx.text)

    def visitSignedNumber(self, ctx):
        # signedNumber: (MINUS|PLUS)? NUMBER_CONSTANT
        return float(ctx.text)

    def visitComparisonOperand(self, ctx):
        return ctx.children[0].text

    def visitErCode(self, ctx):
        """
        erCode: ERRORCODE  constant;
        """
        ctx_list = ctx.children

        try:
            return str(self.visitConstant(ctx_list[1]).value)
        except Exception:
            raise Exception(f"Error code must be a string, line {ctx_list[1].start_line}")

    def visitErLevel(self, ctx):
        """
        erLevel: ERRORLEVEL  constant;
        """
        ctx_list = ctx.children
        return self.visitConstant(ctx_list[1]).value

    def visitSignature(self, ctx, kind="ComponentID"):
        """
        VarID (AS alias)?
        """
        token_info = extract_token_info(ctx)

        ctx_list = ctx.children
        c = ctx_list[0]

        node_name = self.visitVarID(c).value

        alias_name = None

        if len(ctx_list) == 1:
            return DPRIdentifier(value=node_name, kind=kind, alias=alias_name, **token_info)

        alias_name = self.visitAlias(ctx_list[2])
        return DPRIdentifier(value=node_name, kind=kind, alias=alias_name, **token_info)

    """
        From Hierarchical
    """

    def visitConditionClause(self, ctx):
        ctx_list = ctx.children

        components = [
            self.visitComponentID(c)
            for c in ctx_list
            if not c.is_terminal and c.ctx_id == RC.COMPONENT_ID
        ]

        return components

    def visitValidationMode(self, ctx):
        return ctx.children[0].text

    def visitValidationOutput(self, ctx):
        return ctx.children[0].text

    def visitInputMode(self, ctx):
        return ctx.children[0].text

    def visitInputModeHierarchy(self, ctx):
        return ctx.children[0].text

    def visitOutputModeHierarchy(self, ctx):
        return ctx.children[0].text

    """
        From Analytic
    """

    def visitPartitionByClause(self, ctx):
        ctx_list = ctx.children

        return [
            self.visitComponentID(compID).value
            for compID in ctx_list
            if not compID.is_terminal and compID.ctx_id == RC.COMPONENT_ID
        ]

    def visitOrderByClause(self, ctx):
        ctx_list = ctx.children

        return [
            self.visitOrderByItem(c)
            for c in ctx_list
            if not c.is_terminal and c.ctx_id == RC.ORDER_BY_ITEM
        ]

    def visitWindowingClause(self, ctx):
        ctx_list = ctx.children

        win_mode = ctx_list[0].text  # Windowing mode (data points | range )

        token_info = extract_token_info(ctx)

        if win_mode == "data":
            num_rows_1, mode_1 = self.visitLimitClauseItem(ctx_list[3])
            num_rows_2, mode_2 = self.visitLimitClauseItem(ctx_list[5])
        else:
            num_rows_1, mode_1 = self.visitLimitClauseItem(ctx_list[2])
            num_rows_2, mode_2 = self.visitLimitClauseItem(ctx_list[4])

        first = num_rows_1  # unbounded (default value)
        second = num_rows_2  # current data point (default value)

        if (
            mode_2 == "preceding"
            and mode_1 == "preceding"
            and num_rows_1 == -1
            and num_rows_2 == -1
        ):  # preceding and preceding (error)
            raise Exception(
                f"Cannot have 2 preceding clauses with unbounded in analytic clause, "
                f"line {ctx_list[3].start_line}"
            )

        if (
            mode_1 == "following" and num_rows_1 == -1 and num_rows_2 == -1
        ):  # following and following (error)
            raise Exception(
                f"Cannot have 2 following clauses with unbounded in analytic clause, "
                f"line {ctx_list[3].start_line}"
            )

        if mode_1 == mode_2:
            if mode_1 == "preceding" and first != -1 and second > first:  # 3 and 1: must be [-3:-1]
                return create_windowing(win_mode, [second, first], [mode_2, mode_1], token_info)
            if mode_1 == "preceding" and second == -1:
                return create_windowing(win_mode, [second, first], [mode_2, mode_1], token_info)
            if mode_1 == "following" and second != -1 and second < first:  # 3 and 1: must be [1:3]
                return create_windowing(win_mode, [second, first], [mode_2, mode_1], token_info)
            if mode_1 == "following" and first == -1:
                return create_windowing(win_mode, [second, first], [mode_2, mode_1], token_info)

        return create_windowing(win_mode, [first, second], [mode_1, mode_2], token_info)

    def visitOrderByItem(self, ctx):
        ctx_list = ctx.children

        token_info = extract_token_info(ctx)

        if len(ctx_list) == 1:
            return OrderBy(
                component=self.visitComponentID(ctx_list[0]).value, order="asc", **token_info
            )

        return OrderBy(
            component=self.visitComponentID(ctx_list[0]).value,
            order=ctx_list[1].text,
            **token_info,
        )

    def visitLimitClauseItem(self, ctx):
        # limitClauseItem: signedInteger limitDir=PRECEDING
        #     | signedInteger limitDir=FOLLOWING
        #     | CURRENT DATA POINT
        #     | UNBOUNDED limitDir=PRECEDING
        #     | UNBOUNDED limitDir=FOLLOWING
        ctx_list = ctx.children
        c = ctx_list[0]
        if not c.is_terminal and c.ctx_id == RC.SIGNED_INTEGER:
            result = self.visitSignedInteger(c)
            # limitDir is the last terminal child (PRECEDING or FOLLOWING)
            limit_dir = ctx_list[-1].text
            return result, limit_dir
        elif c.text.lower() == "unbounded":
            limit_dir = ctx_list[-1].text
            return -1, limit_dir
        elif c.text == "current":
            return 0, ctx_list[0].text


def create_windowing(win_mode, values, modes, token_info):
    for e in range(0, 2):
        if values[e] == -1:
            values[e] = "unbounded"
        elif values[e] == 0:
            values[e] = "current row"

    return Windowing(
        type_=win_mode,
        start=values[0],
        stop=values[1],
        start_mode=modes[0],
        stop_mode=modes[1],
        **token_info,
    )
