"""
AST.ASTConstructor.py
=====================

Description
-----------
Node Creator.
"""

from typing import Any, List, Optional

from vtlengine.AST import (
    AggregateVpClause,
    Argument,
    Assignment,
    DefIdentifier,
    DPRIdentifier,
    DPRule,
    DPRuleset,
    EnumeratedVpClause,
    HRBinOp,
    HRule,
    HRuleset,
    HRUnOp,
    Operator,
    PersistentAssignment,
    SignatureConstraints,
    Start,
    TypeConstraint,
    UnnamedComponent,
    ViralPropagationDef,
)
from vtlengine.AST.ASTConstructorModules import extract_token_info
from vtlengine.AST.ASTConstructorModules.Expr import Expr
from vtlengine.AST.ASTConstructorModules.ExprComponents import ExprComp
from vtlengine.AST.ASTConstructorModules.Terminals import (
    Terminals,
    _remove_scaped_characters,
    is_quoted_identifier,
)
from vtlengine.AST.ASTDataExchange import de_ruleset_elements
from vtlengine.AST.Grammar._cpp_parser import vtl_cpp_parser
from vtlengine.AST.Grammar._cpp_parser._rule_constants import RC
from vtlengine.Exceptions import InputValidationException, SemanticError
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
        # The calls of a script only use the rulesets it defines, not those of an earlier one
        de_ruleset_elements.clear()
        ctx_list = ctx.children

        statements_nodes = []
        statements = [
            statement
            for statement in ctx_list
            if not statement.is_terminal and statement.rule_index == RC.STATEMENT[0]
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

        elif ctx.ctx_id == RC.DEF_VIRAL_PROPAGATION:
            return self.visitDefViralPropagation(ctx)

    def visitDefViralPropagation(self, ctx: Any) -> Any:
        """
        DEFINE VIRAL PROPAGATION varID LPAREN vpSignature RPAREN IS
        vpBody END VIRAL PROPAGATION  # defViralPropagation
        """
        ctx_list = ctx.children

        propagation_name = Terminals().visitVarID(ctx_list[3]).value
        signature_type, target = self.visitVpSignature(ctx_list[5])
        enumerated_clauses, aggregate_clause, default_value, has_default = self.visitVpBody(
            ctx_list[8]
        )

        token_info = extract_token_info(ctx)

        return ViralPropagationDef(
            name=propagation_name,
            signature_type=signature_type,
            target=target,
            enumerated_clauses=enumerated_clauses,
            aggregate_clause=aggregate_clause,
            default_value=default_value,
            has_default=has_default,
            **token_info,
        )

    def visitVpSignature(self, ctx: Any) -> Any:
        """vpSignature: (VARIABLE | VALUE_DOMAIN) IDENTIFIER ;"""
        ctx_list = ctx.children
        signature_type = ctx_list[0].text
        target = _remove_scaped_characters(ctx_list[1].text)
        return signature_type, target

    def visitVpBody(self, ctx: Any) -> Any:
        """vpBody: vpClause (EOL vpClause)* ;"""
        ctx_list = ctx.children
        enumerated_clauses: List[EnumeratedVpClause] = []
        aggregate_clause: Optional[AggregateVpClause] = None
        default_value: Optional[str] = None
        has_default = False

        for child in ctx_list:
            if child.is_terminal:
                continue
            if child.ctx_id == RC.ENUMERATED_VP_CLAUSE:
                enumerated_clauses.append(self.visitEnumeratedVpClause(child))
            elif child.ctx_id == RC.AGGREGATION_VP_CLAUSE:
                aggregate_clause = self.visitAggregationVpClause(child)
            elif child.ctx_id == RC.DEFAULT_VP_CLAUSE:
                default_value = self.visitDefaultVpClause(child)
                has_default = True

        return enumerated_clauses, aggregate_clause, default_value, has_default

    def visitEnumeratedVpClause(self, ctx: Any) -> Any:
        """enumeratedVpClause: (IDENTIFIER COLON)? WHEN vpCondition THEN constant ;"""
        ctx_list = ctx.children
        rule_name: Optional[str] = None
        values: List[str] = []
        result: str = ""

        i = 0
        # Optional rule name: IDENTIFIER COLON
        if ctx_list[i].is_terminal and ctx_list[i].symbol_type == vtl_cpp_parser.IDENTIFIER:
            rule_name = ctx_list[i].text
            i += 2  # skip IDENTIFIER and COLON

        i += 1  # skip WHEN
        # vpCondition
        values = self.visitVpCondition(ctx_list[i])
        i += 1

        i += 1  # skip THEN
        # constant (result)
        result = Terminals().visitConstant(ctx_list[i]).value

        token_info = extract_token_info(ctx)
        return EnumeratedVpClause(name=rule_name, values=values, result=result, **token_info)

    def visitAggregationVpClause(self, ctx: Any) -> Any:
        """aggregationVpClause: AGGREGATE (MIN | MAX | SUM | AVG) ;"""
        ctx_list = ctx.children
        function = ctx_list[1].text
        token_info = extract_token_info(ctx)
        return AggregateVpClause(function=function, **token_info)

    def visitDefaultVpClause(self, ctx: Any) -> Any:
        """defaultVpClause: ELSE constant ;"""
        ctx_list = ctx.children
        return Terminals().visitConstant(ctx_list[1]).value

    def visitVpCondition(self, ctx: Any) -> Any:
        """vpCondition: constant (AND constant)? ;"""
        ctx_list = ctx.children
        values = []
        for child in ctx_list:
            if not child.is_terminal and child.rule_index == RC.CONSTANT[0]:
                values.append(Terminals().visitConstant(child).value)
        return values

    def visitDefOperator(self, ctx: Any) -> Any:
        """
        DEFINE OPERATOR operatorID LPAREN (parameterItem (COMMA parameterItem)*)? RPAREN
        (RETURNS outputParameterType)? IS (expr) END OPERATOR        # defOperator"""
        ctx_list = ctx.children

        operator = Terminals().visitOperatorID(ctx_list[2])
        parameters = [
            self.visitParameterItem(parameter)
            for parameter in ctx_list
            if not parameter.is_terminal and parameter.rule_index == RC.PARAMETER_ITEM[0]
        ]
        return_ = [
            (
                Terminals().visitOutputParameterType(datatype),
                Terminals().visitDeclaredOutputType(datatype, allow_constraints=True),
                self._signature_constraints(datatype),
            )
            for datatype in ctx_list
            if not datatype.is_terminal and datatype.rule_index == RC.OUTPUT_PARAMETER_TYPE[0]
        ]
        # Here should be modified if we want to include more than one expr per function.
        expr = [
            Expr().visitExpr(expr)
            for expr in ctx_list
            if not expr.is_terminal and expr.rule_index == RC.EXPR[0]
        ][0]

        # Without returns, the result is whatever the body gives (issue #1160)
        return_node, return_type, return_constraints = return_[0] if return_ else (None, None, None)
        if return_ and return_type is None:
            raise SemanticError("1-3-2-6", op=operator)

        token_info = extract_token_info(ctx)

        return Operator(
            op=operator,
            parameters=parameters,
            output_type=return_node,
            expression=expr,
            return_type=return_type,
            return_constraints=return_constraints,
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
            if not component.is_terminal and component.rule_index == RC.SIGNATURE[0]
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
            if not ruleId.is_terminal and ruleId.rule_index == RC.RULE_ITEM_DATAPOINT[0]
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
            if not rule_node.is_terminal and rule_node.rule_index == RC.EXPR_COMPONENT[0]
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
            if not erCode_name.is_terminal and erCode_name.rule_index == RC.ER_CODE[0]
        ]
        er_code = None if len(er_code) == 0 else er_code[0]
        er_level = [
            Terminals().visitErLevel(erLevel_name)
            for erLevel_name in ctx_list
            if not erLevel_name.is_terminal and erLevel_name.rule_index == RC.ER_LEVEL[0]
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
            if not element.is_terminal and element.rule_index == RC.VAR_ID[0]
        ][0]
        type_ctx = [
            element
            for element in ctx_list
            if not element.is_terminal and element.rule_index == RC.INPUT_PARAMETER_TYPE[0]
        ][0]
        argument_type = Terminals().visitInputParameterType(type_ctx, allow_constraints=True)
        argument_default = [
            Terminals().visitScalarItem(element)
            for element in ctx_list
            if not element.is_terminal and element.rule_index == RC.SCALAR_ITEM[0]
        ]
        argument_default = None if len(argument_default) == 0 else argument_default[0]

        if isinstance(argument_type, (Dataset, Component, Scalar)):
            argument_type.name = argument_name.value
        token_info = extract_token_info(ctx)
        return Argument(
            name=argument_name.value,
            type_=argument_type,
            default=argument_default,
            constraints=self._signature_constraints(type_ctx),
            **token_info,
        )

    def _signature_constraints(self, ctx: Any) -> Optional[SignatureConstraints]:
        """What the type of a parameter or of the returns of a user defined operator requires
        beyond its model type, or None when it requires nothing more."""
        c = ctx.children[0]
        constraints = SignatureConstraints()
        if not c.is_terminal and c.ctx_id == RC.SCALAR_TYPE:
            constraints.value = self._type_constraint(c)
        elif not c.is_terminal and c.ctx_id == RC.COMPONENT_TYPE:
            constraints.value = self._component_type_constraint(c)
        elif not c.is_terminal and c.ctx_id == RC.DATASET_TYPE:
            for comp in c.children:
                if comp.is_terminal or comp.ctx_id != RC.COMP_CONSTRAINT:
                    continue
                comp_type, name_or_mult = comp.children[0], comp.children[1]
                constraint = self._component_type_constraint(comp_type)
                if name_or_mult.ctx_id == RC.MULT_MODIFIER:
                    component = Terminals().visitComponentType(comp_type, allow_constraints=True)
                    multiplicity = name_or_mult.text.lstrip("_")
                    constraints.unnamed.append(
                        UnnamedComponent(component, multiplicity, constraint)
                    )
                elif constraint is not None:
                    name = Terminals().visitComponentID(name_or_mult).value
                    constraints.components[name] = constraint
        if constraints == SignatureConstraints():
            return None
        return constraints

    def _component_type_constraint(self, ctx: Any) -> Optional[TypeConstraint]:
        """The constraint of the scalar type of a componentType, if it has one."""
        for c in ctx.children:
            if not c.is_terminal and c.ctx_id == RC.SCALAR_TYPE:
                return self._type_constraint(c)
        return None

    @staticmethod
    def _type_constraint(ctx: Any) -> Optional[TypeConstraint]:
        """
        scalarType: (basicScalarType|valueDomainName)scalarTypeConstraint?((NOT)? NULL_CONSTANT)? ;
        scalarTypeConstraint: QLPAREN exprComponent QRPAREN       # conditionConstraint
                            | GLPAREN scalarItem (COMMA scalarItem)* GRPAREN  # rangeConstraint
        """
        constraint = TypeConstraint(
            not_null=any(
                c.is_terminal and c.symbol_type == vtl_cpp_parser.NOT for c in ctx.children
            )
        )
        for c in ctx.children:
            if not c.is_terminal and c.ctx_id == RC.CONDITION_CONSTRAINT:
                constraint.condition = ExprComp().visitExprComponent(c.children[1])
            elif not c.is_terminal and c.ctx_id == RC.RANGE_CONSTRAINT:
                constraint.values = [
                    Terminals().visitScalarItem(item) for item in c.children if not item.is_terminal
                ]
        return None if constraint == TypeConstraint() else constraint

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
        ruleset_rules = self.visitRuleClauseHierarchical(ctx_list[8])
        # Keep k,v for the hierarchical rulesets
        de_ruleset_elements[ruleset_name] = ruleset_elements
        if len(ruleset_rules) == 0:
            raise InputValidationException(code="0-5-1-2", ruleset_name=ruleset_name)

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
            if not vtlsig.is_terminal and vtlsig.rule_index == RC.VALUE_DOMAIN_SIGNATURE[0]
        ]

        dataset = [
            identifier
            for identifier in ctx_list
            if identifier.is_terminal and identifier.symbol_type == vtl_cpp_parser.IDENTIFIER
        ][0]

        token_info = extract_token_info(ctx)
        if conditions:
            # The rules name a condition by its alias, if it has one
            identifiers_list = [
                DPRIdentifier(value=elto.value, kind=kind, alias=elto.alias, **token_info)
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
            if not component.is_terminal and component.rule_index == RC.SIGNATURE[0]
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
            if not rule.is_terminal and rule.rule_index == RC.RULE_ITEM_HIERARCHICAL[0]
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
            if not rule_node.is_terminal and rule_node.rule_index == RC.CODE_ITEM_RELATION[0]
        ][0]

        er_code = [
            Terminals().visitErCode(erCode_name)
            for erCode_name in ctx_list
            if not erCode_name.is_terminal and erCode_name.rule_index == RC.ER_CODE[0]
        ]
        er_code = None if len(er_code) == 0 else er_code[0]
        er_level = [
            Terminals().visitErLevel(erLevel_name)
            for erLevel_name in ctx_list
            if not erLevel_name.is_terminal and erLevel_name.rule_index == RC.ER_LEVEL[0]
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
            vd_quoted = is_quoted_identifier(ctx_list[3])
            op = Terminals().visitComparisonOperand(ctx_list[4])
            token_info_value = extract_token_info(ctx_list[3])
            token_info_op = extract_token_info(ctx_list[4])
        else:
            vd_value = Terminals().visitValueDomainValue(ctx_list[0])
            vd_quoted = is_quoted_identifier(ctx_list[0])
            op = Terminals().visitComparisonOperand(ctx_list[1])
            token_info_value = extract_token_info(ctx_list[0])
            token_info_op = extract_token_info(ctx_list[1])

        rule_node = HRBinOp(
            left=DefIdentifier(
                value=vd_value, kind="CodeItemID", was_quoted=vd_quoted, **token_info_value
            ),
            op=op,
            right=None,
            **token_info_op,
        )
        items = [
            item
            for item in ctx_list
            if not item.is_terminal and item.rule_index == RC.CODE_ITEM_RELATION_CLAUSE[0]
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

    def visitCodeItemRelationClause(self, ctx):
        """
        (opAdd=( PLUS | MINUS  ))? rightCodeItem=valueDomainValue ( QLPAREN  rightCondition=exprComponent  QRPAREN )?
        """  # noqa E501
        ctx_list = ctx.children

        right_condition = [
            ExprComp().visitExprComponent(right_condition)
            for right_condition in ctx_list
            if not right_condition.is_terminal
            and right_condition.rule_index == RC.EXPR_COMPONENT[0]
        ]

        if ctx_list[0].is_terminal:
            op = ctx_list[0].text
            value = Terminals().visitValueDomainValue(ctx_list[1])

            code_item = DefIdentifier(
                value=value,
                kind="CodeItemID",
                was_quoted=is_quoted_identifier(ctx_list[1]),
                **extract_token_info(ctx_list[1]),
            )
            if right_condition:
                code_item._right_condition = right_condition[0]

            return HRBinOp(left=None, op=op, right=code_item, **extract_token_info(ctx_list[0]))
        else:
            value = Terminals().visitValueDomainValue(ctx_list[0])
            code_item = DefIdentifier(
                value=value,
                kind="CodeItemID",
                was_quoted=is_quoted_identifier(ctx_list[0]),
                **extract_token_info(ctx_list[0]),
            )
            if right_condition:
                code_item._right_condition = right_condition[0]

            return code_item
