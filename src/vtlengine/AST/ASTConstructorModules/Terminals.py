from antlr4.tree.Tree import TerminalNodeImpl

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
from vtlengine.AST.Grammar.parser import Parser
from vtlengine.AST.VtlVisitor import VtlVisitor
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


class Terminals(VtlVisitor):
    def visitConstant(self, ctx: Parser.ConstantContext):
        token = ctx.children[0].getSymbol()
        token_info = extract_token_info(token)

        if token.type == Parser.INTEGER_CONSTANT:
            constant_node = Constant(type_="INTEGER_CONSTANT", value=int(token.text), **token_info)

        elif token.type == Parser.NUMBER_CONSTANT:
            constant_node = Constant(type_="FLOAT_CONSTANT", value=float(token.text), **token_info)

        elif token.type == Parser.BOOLEAN_CONSTANT:
            if token.text == "true":
                constant_node = Constant(type_="BOOLEAN_CONSTANT", value=True, **token_info)
            elif token.text == "false":
                constant_node = Constant(type_="BOOLEAN_CONSTANT", value=False, **token_info)
            else:
                raise NotImplementedError

        elif token.type == Parser.STRING_CONSTANT:
            constant_node = Constant(type_="STRING_CONSTANT", value=token.text[1:-1], **token_info)

        elif token.type == Parser.NULL_CONSTANT:
            constant_node = Constant(type_="NULL_CONSTANT", value=None, **token_info)

        else:
            raise NotImplementedError

        return constant_node

    def visitVarID(self, ctx: Parser.VarIDContext):
        token = ctx.children[0].getSymbol()
        token.text = _remove_scaped_characters(token.text)
        token_info = extract_token_info(token)
        var_id_node = VarID(value=token.text, **token_info)
        return var_id_node

    def visitVarIdExpr(self, ctx: Parser.VarIdExprContext):
        if isinstance(ctx.children[0], Parser.VarIDContext):
            return self.visitVarID(ctx.children[0])

        token = ctx.children[0].getSymbol()
        # check token text
        token.text = _remove_scaped_characters(token.text)
        token_info = extract_token_info(token)
        var_id_node = VarID(value=token.text, **token_info)
        return var_id_node

    def visitSimpleComponentId(self, ctx: Parser.SimpleComponentIdContext):
        """
        componentID: IDENTIFIER ;
        """
        token = ctx.children[0].getSymbol()
        # check token text
        token.text = _remove_scaped_characters(token.text)

        return Identifier(value=token.text, kind="ComponentID", **extract_token_info(ctx))

    def visitComponentID(self, ctx: Parser.ComponentIDContext):
        ctx_list = list(ctx.getChildren())

        if len(ctx_list) == 1:
            component_name = ctx_list[0].getSymbol().text
            if component_name.startswith("'") and component_name.endswith(
                "'"
            ):  # The component could be imbalance, errorcode or errorlevel
                component_name = component_name[1:-1]
            return Identifier(
                value=component_name,
                kind="ComponentID",
                **extract_token_info(ctx_list[0].getSymbol()),
            )
        else:
            component_name = ctx_list[2].getSymbol().text
            if component_name.startswith("'") and component_name.endswith(
                "'"
            ):  # The component could be imbalance, errorcode or errorlevel
                component_name = component_name[1:-1]
            op_node = ctx_list[1].getSymbol().text
            return BinOp(
                left=Identifier(
                    value=ctx_list[0].getSymbol().text,
                    kind="DatasetID",
                    **extract_token_info(ctx_list[0].getSymbol()),
                ),
                op=op_node,
                right=Identifier(
                    value=component_name,
                    kind="ComponentID",
                    **extract_token_info(ctx_list[1].getSymbol()),
                ),
                **extract_token_info(ctx),
            )

    def visitOperatorID(self, ctx: Parser.OperatorIDContext):
        """
        operatorID: IDENTIFIER ;
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]
        token = c.getSymbol()
        return token.text

    def visitValueDomainID(self, ctx: Parser.ValueDomainIDContext):
        """
        valueDomainID: IDENTIFIER ;
        """
        return Collection(
            name=ctx.children[0].getSymbol().text,
            children=[],
            kind="ValueDomain",
            type="",
            **extract_token_info(ctx),
        )

    def visitRulesetID(self, ctx: Parser.RulesetIDContext):
        """
        rulesetID: IDENTIFIER ;
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]
        token = c.getSymbol()
        return token.text

    def visitValueDomainName(self, ctx: Parser.ValueDomainNameContext):
        """
        valueDomainName: IDENTIFIER ;
        """
        ctx_list = list(ctx.getChildren())
        # AST_ASTCONSTRUCTOR.48
        raise NotImplementedError(
            "Value Domain '{}' not available for cast operator or scalar type "
            "representation or rulesets.".format(ctx_list[0].getSymbol().text)
        )

    def visitValueDomainValue(self, ctx: Parser.ValueDomainValueContext):
        return _remove_scaped_characters(ctx.children[0].getSymbol().text)

    def visitRoutineName(self, ctx: Parser.RoutineNameContext):
        """
        routineName: IDENTIFIER ;
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]
        token = c.getSymbol()

        return token.text

    def visitBasicScalarType(self, ctx: Parser.BasicScalarTypeContext):
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
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]
        token = c.getSymbol()

        if token.type == Parser.STRING:
            return String
        elif token.type == Parser.INTEGER:
            return Integer
        elif token.type == Parser.NUMBER:
            return Number
        elif token.type == Parser.BOOLEAN:
            return Boolean
        elif token.type == Parser.DATE:
            return Date
        elif token.type == Parser.TIME_PERIOD:
            return TimePeriod
        elif token.type == Parser.DURATION:
            return Duration
        elif token.type == Parser.SCALAR:
            return "Scalar"
        elif token.type == Parser.TIME:
            return TimeInterval

    def visitComponentRole(self, ctx: Parser.ComponentRoleContext):
        """
        componentRole: MEASURE
                     |COMPONENT
                     |DIMENSION
                     |ATTRIBUTE
                     |viralAttribute
                     ;
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        if isinstance(c, Parser.ViralAttributeContext):
            return self.visitViralAttribute(c)
        else:
            token = c.getSymbol()
            text = token.text
            if text == "component":
                return None
            # Use upper case on first letter
            text = text[0].upper() + text[1:].lower()
            return Role(text)

    def visitViralAttribute(self, ctx: Parser.ViralAttributeContext):
        """
        viralAttribute: VIRAL ATTRIBUTE;
        """
        # ctx_list = list(ctx.getChildren())
        # c = ctx_list[0]
        # token = c.getSymbol()

        raise NotImplementedError

    def visitLists(self, ctx: Parser.ListsContext):
        """
        lists:  GLPAREN  scalarItem (COMMA scalarItem)*  GRPAREN
        """
        ctx_list = list(ctx.getChildren())

        scalar_nodes = []

        scalars = [scalar for scalar in ctx_list if isinstance(scalar, Parser.SimpleScalarContext)]

        scalars_with_cast = [
            scalar for scalar in ctx_list if isinstance(scalar, Parser.ScalarWithCastContext)
        ]

        for scalar in scalars:
            scalar_nodes.append(self.visitSimpleScalar(scalar))

        for scalar_with_cast in scalars_with_cast:
            scalar_nodes.append(self.visitScalarWithCast(scalar_with_cast))

        return Collection(
            name="List", type="Lists", children=scalar_nodes, **extract_token_info(ctx)
        )

    def visitMultModifier(self, ctx: Parser.MultModifierContext):
        """
        multModifier: OPTIONAL  ( PLUS | MUL )?;
        """
        pass

    def visitCompConstraint(self, ctx: Parser.CompConstraintContext):
        """
        compConstraint: componentType (componentID|multModifier) ;
        """
        ctx_list = list(ctx.getChildren())

        component_node = [
            self.visitComponentType(component)
            for component in ctx_list
            if isinstance(component, Parser.ComponentTypeContext)
        ]
        component_name = [
            self.visitComponentID(component).value
            for component in ctx_list
            if isinstance(component, Parser.ComponentIDContext)
        ]
        component_mult = [
            self.visitMultModifier(modifier)
            for modifier in ctx_list
            if isinstance(modifier, Parser.MultModifierContext)
        ]

        if len(component_mult) != 0:
            # AST_ASTCONSTRUCTOR.51
            raise NotImplementedError

        component_node[0].name = component_name[0]
        return component_node[0]

    def visitSimpleScalar(self, ctx: Parser.SimpleScalarContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]
        if isinstance(c, Parser.ConstantContext):
            return self.visitConstant(c)
        else:
            raise NotImplementedError

    def visitScalarType(self, ctx: Parser.ScalarTypeContext):
        """
        scalarType: (basicScalarType|valueDomainName)scalarTypeConstraint?((NOT)? NULL_CONSTANT)? ;
        """
        ctx_list = list(ctx.getChildren())

        types = (
            Parser.BasicScalarTypeContext,
            Parser.ValueDomainNameContext,
            Parser.ScalarTypeConstraintContext,
        )
        scalartype = [scalartype for scalartype in ctx_list if isinstance(scalartype, types)][0]

        scalartype_constraint = [
            constraint
            for constraint in ctx_list
            if isinstance(constraint, Parser.ScalarTypeConstraintContext)
        ]
        not_ = [
            not_.getSymbol().text
            for not_ in ctx_list
            if isinstance(not_, TerminalNodeImpl) and not_.getSymbol().type == Parser.NOT
        ]
        null_constant = [
            null.getSymbol().text
            for null in ctx_list
            if isinstance(null, TerminalNodeImpl) and null.getSymbol().type == Parser.NULL_CONSTANT
        ]

        if isinstance(scalartype, Parser.BasicScalarTypeContext):
            if scalartype.children[0].getSymbol().type == Parser.SCALAR:
                return Scalar(name="", data_type=None, value=None)
            type_node = self.visitBasicScalarType(scalartype)

        elif isinstance(scalartype, Parser.ValueDomainNameContext):
            # type_node = self.visitValueDomainName(scalartype)
            raise NotImplementedError
        else:
            raise NotImplementedError

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

    def visitDatasetType(self, ctx: Parser.DatasetTypeContext):
        """
        datasetType: DATASET ('{'compConstraint (',' compConstraint)* '}' )? ;
        """
        ctx_list = list(ctx.getChildren())

        components = [
            self.visitCompConstraint(constraint)
            for constraint in ctx_list
            if isinstance(constraint, Parser.CompConstraintContext)
        ]
        components = {component.name: component for component in components}

        return Dataset(name="Dataset", components=components, data=None)

    def visitRulesetType(self, ctx: Parser.RulesetTypeContext):
        """
        rulesetType: RULESET
                   | dpRuleset
                   | hrRuleset
                   ;
        """
        raise NotImplementedError

    def visitDpRuleset(self, ctx: Parser.DpRulesetContext):
        """
        DATAPOINT                                                                               # dataPoint
            | DATAPOINT_ON_VD  (GLPAREN  valueDomainName (MUL valueDomainName)*  GRPAREN )?         # dataPointVd
            | DATAPOINT_ON_VAR  (GLPAREN  varID (MUL varID)*  GRPAREN )?                            # dataPointVar
        ;
        """  # noqa E501
        # AST_ASTCONSTRUCTOR.54
        raise NotImplementedError

    def visitHrRuleset(self, ctx: Parser.HrRulesetContext):
        """
        hrRuleset: HIERARCHICAL                                                                                                            # hrRulesetType
            | HIERARCHICAL_ON_VD ( GLPAREN  vdName=IDENTIFIER (LPAREN valueDomainName (MUL valueDomainName)* RPAREN)?  GRPAREN )?   # hrRulesetVdType
            | HIERARCHICAL_ON_VAR ( GLPAREN  varName=varID (LPAREN  varID (MUL varID)* RPAREN)?  GRPAREN )?                         # hrRulesetVarType
        ;
        """  # noqa E501
        # AST_ASTCONSTRUCTOR.55
        raise NotImplementedError

    def visitComponentType(self, ctx: Parser.ComponentTypeContext):
        """
        componentType:  componentRole ( LT   scalarType  MT  )?
        """
        ctx_list = list(ctx.getChildren())

        role_node = self.visitComponentRole(ctx_list[0])
        data_type = [
            self.visitScalarType(constraint)
            for constraint in ctx_list
            if isinstance(constraint, Parser.ScalarTypeContext)
        ]
        data_type = data_type[0] if len(data_type) > 0 else String()

        nullable = role_node != Role.IDENTIFIER

        return Component(name="Component", data_type=data_type, role=role_node, nullable=nullable)

    def visitInputParameterType(self, ctx: Parser.InputParameterTypeContext):
        """
        inputParameterType:
            scalarType
            | datasetType
            | scalarSetType
            | rulesetType
            | componentType
        ;
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        if isinstance(c, Parser.ScalarTypeContext):
            return self.visitScalarType(c)

        elif isinstance(c, Parser.DatasetTypeContext):
            return self.visitDatasetType(c)

        elif isinstance(c, Parser.ScalarSetTypeContext):
            return self.visitScalarSetType(c)

        elif isinstance(c, Parser.RulesetTypeContext):
            return self.visitRulesetType(c)

        elif isinstance(c, Parser.ComponentTypeContext):
            return self.visitComponentType(c)
        else:
            raise NotImplementedError

    def visitOutputParameterType(self, ctx: Parser.OutputParameterTypeContext):
        """
        outputParameterType: scalarType
                           | datasetType
                           | componentType
                           ;
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        if isinstance(c, Parser.ScalarTypeContext):
            # return self.visitScalarType(c).__class__.__name__
            return "Scalar"

        elif isinstance(c, Parser.DatasetTypeContext):
            return "Dataset"

        elif isinstance(c, Parser.ComponentTypeContext):
            return "Component"
        else:
            raise NotImplementedError

    def visitOutputParameterTypeComponent(self, ctx: Parser.OutputParameterTypeComponentContext):
        """
        outputParameterType: scalarType
                           | componentType
                           ;
        """
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        if isinstance(c, Parser.ScalarTypeContext):
            return self.visitScalarType(c)

        elif isinstance(c, Parser.ComponentTypeContext):
            return self.visitComponentType(c)
        else:
            raise NotImplementedError

    def visitScalarItem(self, ctx: Parser.ScalarItemContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        if isinstance(c, Parser.ConstantContext):
            return self.visitConstant(c)
        elif isinstance(c, Parser.ScalarWithCastContext):
            return self.visitScalarWithCast(c)
        else:
            raise NotImplementedError

    def visitScalarWithCast(self, ctx: Parser.ScalarWithCastContext):
        """
        |  CAST LPAREN constant COMMA (basicScalarType) (COMMA STRING_CONSTANT)? RPAREN    #scalarWithCast  # noqa E501
        """  # noqa E501
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        token = c.getSymbol()

        op = token.text
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

    def visitScalarSetType(self, ctx: Parser.ScalarSetTypeContext):
        """
        scalarSetType: SET ('<' scalarType '>')? ;
        """
        # AST_ASTCONSTRUCTOR.60
        raise NotImplementedError

    def visitRetainType(self, ctx: Parser.RetainTypeContext):
        """
        retainType: BOOLEAN_CONSTANT
                  | ALL
                  ;
        """
        token = ctx.children[0].getSymbol()

        if token.type == Parser.BOOLEAN_CONSTANT:
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

        elif token.type == Parser.ALL:
            param_constant_node = ParamConstant(
                type_="PARAM_CONSTANT", value=token.text, **extract_token_info(token)
            )

        else:
            raise NotImplementedError

        return param_constant_node

    def visitEvalDatasetType(self, ctx: Parser.EvalDatasetTypeContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]

        if isinstance(c, Parser.DatasetTypeContext):
            return self.visitDatasetType(c)
        elif isinstance(c, Parser.ScalarTypeContext):
            return self.visitScalarType(c)
        else:
            raise NotImplementedError

    def visitAlias(self, ctx: Parser.AliasContext):
        return ctx.children[0].getSymbol().text

    def visitSignedInteger(self, ctx: Parser.SignedIntegerContext):
        return int(ctx.children[0].getSymbol().text)

    def visitComparisonOperand(self, ctx: Parser.ComparisonOperandContext):
        return ctx.children[0].getSymbol().text

    def visitErCode(self, ctx: Parser.ErCodeContext):
        """
        erCode: ERRORCODE  constant;
        """
        ctx_list = list(ctx.getChildren())

        try:
            return str(self.visitConstant(ctx_list[1]).value)
        except Exception:
            raise Exception(f"Error code must be a string, line {ctx_list[1].getSymbol().line}")

    def visitErLevel(self, ctx: Parser.ErLevelContext):
        """
        erLevel: ERRORLEVEL  constant;
        """
        ctx_list = list(ctx.getChildren())

        try:
            return int(self.visitConstant(ctx_list[1]).value)
        except Exception:
            raise Exception(f"Error level must be an integer, line {ctx_list[1].start.line}")

    def visitSignature(self, ctx: Parser.SignatureContext, kind="ComponentID"):
        """
        VarID (AS alias)?
        """
        token_info = extract_token_info(ctx)

        ctx_list = list(ctx.getChildren())
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

    def visitConditionClause(self, ctx: Parser.ConditionClauseContext):
        ctx_list = list(ctx.getChildren())

        components = [
            self.visitComponentID(c) for c in ctx_list if isinstance(c, Parser.ComponentIDContext)
        ]

        return components

    def visitValidationMode(self, ctx: Parser.ValidationModeContext):
        return ctx.children[0].getSymbol().text

    def visitValidationOutput(self, ctx: Parser.ValidationOutputContext):
        return ctx.children[0].getSymbol().text

    def visitInputMode(self, ctx: Parser.InputModeContext):
        return ctx.children[0].getSymbol().text

    def visitInputModeHierarchy(self, ctx: Parser.InputModeHierarchyContext):
        return ctx.children[0].getSymbol().text

    def visitOutputModeHierarchy(self, ctx: Parser.OutputModeHierarchyContext):
        return ctx.children[0].getSymbol().text

    """
        From Analytic
    """

    def visitPartitionByClause(self, ctx: Parser.PartitionByClauseContext):
        ctx_list = list(ctx.getChildren())

        return [
            self.visitComponentID(compID).value
            for compID in ctx_list
            if isinstance(compID, Parser.ComponentIDContext)
        ]

    def visitOrderByClause(self, ctx: Parser.OrderByClauseContext):
        ctx_list = list(ctx.getChildren())

        return [
            self.visitOrderByItem(c) for c in ctx_list if isinstance(c, Parser.OrderByItemContext)
        ]

    def visitWindowingClause(self, ctx: Parser.WindowingClauseContext):
        ctx_list = list(ctx.getChildren())

        win_mode = ctx_list[0].getSymbol().text  # Windowing mode (data points | range )

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
                f"line {ctx_list[3].start.line}"
            )

        if (
            mode_1 == "following" and num_rows_1 == -1 and num_rows_2 == -1
        ):  # following and following (error)
            raise Exception(
                f"Cannot have 2 following clauses with unbounded in analytic clause, "
                f"line {ctx_list[3].start.line}"
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

    def visitOrderByItem(self, ctx: Parser.OrderByItemContext):
        ctx_list = list(ctx.getChildren())

        token_info = extract_token_info(ctx)

        if len(ctx_list) == 1:
            return OrderBy(
                component=self.visitComponentID(ctx_list[0]).value, order="asc", **token_info
            )

        return OrderBy(
            component=self.visitComponentID(ctx_list[0]).value,
            order=ctx_list[1].getSymbol().text,
            **token_info,
        )

    def visitLimitClauseItem(self, ctx: Parser.LimitClauseItemContext):
        ctx_list = list(ctx.getChildren())
        c = ctx_list[0]
        if c.getSymbol().text.lower() == "unbounded":
            result = -1
        elif c.getSymbol().text == "current":
            result = 0
            return result, ctx_list[0].getSymbol().text
        else:
            result = int(c.getSymbol().text)
            if result < 0:
                raise Exception(
                    f"Cannot use negative numbers ({result}) on limitClause, line {c.symbol.line}"
                )

        return result, ctx_list[1].getSymbol().text


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
