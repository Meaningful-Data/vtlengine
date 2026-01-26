from antlr4 import ParseTreeVisitor

from vtlengine.AST.Grammar.parser import Parser

# This class defines a complete generic visitor for a parse tree produced by Parser.


class VtlVisitor(ParseTreeVisitor):
    # Visit a parse tree produced by Parser#start.
    def visitStart(self, ctx: Parser.StartContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#temporaryAssignment.
    def visitTemporaryAssignment(self, ctx: Parser.TemporaryAssignmentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#persistAssignment.
    def visitPersistAssignment(self, ctx: Parser.PersistAssignmentContext):
        return self.visitChildren(ctx)

    def visitStatement(self, ctx: Parser.StatementContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#defineExpression.
    def visitDefineExpression(self, ctx: Parser.DefineExpressionContext):
        return self.visitChildren(ctx)

    def visitExpr(self, ctx: Parser.ExprContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#varIdExpr.
    def visitVarIdExpr(self, ctx: Parser.VarIdExprContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#membershipExpr.
    def visitMembershipExpr(self, ctx: Parser.MembershipExprContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#inNotInExpr.
    def visitInNotInExpr(self, ctx: Parser.InNotInExprContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#booleanExpr.
    def visitBooleanExpr(self, ctx: Parser.BooleanExprContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#comparisonExpr.
    def visitComparisonExpr(self, ctx: Parser.ComparisonExprContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#unaryExpr.
    def visitUnaryExpr(self, ctx: Parser.UnaryExprContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#functionsExpression.
    def visitFunctionsExpression(self, ctx: Parser.FunctionsExpressionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#ifExpr.
    def visitIfExpr(self, ctx: Parser.IfExprContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#caseExpr.
    def visitCaseExpr(self, ctx: Parser.CaseExprContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#clauseExpr.
    def visitClauseExpr(self, ctx: Parser.ClauseExprContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#arithmeticExpr.
    def visitArithmeticExpr(self, ctx: Parser.ArithmeticExprContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#parenthesisExpr.
    def visitParenthesisExpr(self, ctx: Parser.ParenthesisExprContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#constantExpr.
    def visitConstantExpr(self, ctx: Parser.ConstantExprContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#arithmeticExprOrConcat.
    def visitArithmeticExprOrConcat(self, ctx: Parser.ArithmeticExprOrConcatContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#arithmeticExprComp.
    def visitArithmeticExprComp(self, ctx: Parser.ArithmeticExprCompContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#ifExprComp.
    def visitIfExprComp(self, ctx: Parser.IfExprCompContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#caseExprComp.
    def visitCaseExprComp(self, ctx: Parser.CaseExprCompContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#comparisonExprComp.
    def visitComparisonExprComp(self, ctx: Parser.ComparisonExprCompContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#functionsExpressionComp.
    def visitFunctionsExpressionComp(self, ctx: Parser.FunctionsExpressionCompContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#compId.
    def visitCompId(self, ctx: Parser.CompIdContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#constantExprComp.
    def visitConstantExprComp(self, ctx: Parser.ConstantExprCompContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#arithmeticExprOrConcatComp.
    def visitArithmeticExprOrConcatComp(self, ctx: Parser.ArithmeticExprOrConcatCompContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#parenthesisExprComp.
    def visitParenthesisExprComp(self, ctx: Parser.ParenthesisExprCompContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#inNotInExprComp.
    def visitInNotInExprComp(self, ctx: Parser.InNotInExprCompContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#unaryExprComp.
    def visitUnaryExprComp(self, ctx: Parser.UnaryExprCompContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#booleanExprComp.
    def visitBooleanExprComp(self, ctx: Parser.BooleanExprCompContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#genericFunctionsComponents.
    def visitGenericFunctionsComponents(self, ctx: Parser.GenericFunctionsComponentsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#stringFunctionsComponents.
    def visitStringFunctionsComponents(self, ctx: Parser.StringFunctionsComponentsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#numericFunctionsComponents.
    def visitNumericFunctionsComponents(self, ctx: Parser.NumericFunctionsComponentsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#comparisonFunctionsComponents.
    def visitComparisonFunctionsComponents(self, ctx: Parser.ComparisonFunctionsComponentsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#timeFunctionsComponents.
    def visitTimeFunctionsComponents(self, ctx: Parser.TimeFunctionsComponentsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#conditionalFunctionsComponents.
    def visitConditionalFunctionsComponents(
        self, ctx: Parser.ConditionalFunctionsComponentsContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#aggregateFunctionsComponents.
    def visitAggregateFunctionsComponents(self, ctx: Parser.AggregateFunctionsComponentsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#analyticFunctionsComponents.
    def visitAnalyticFunctionsComponents(self, ctx: Parser.AnalyticFunctionsComponentsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#joinFunctions.
    def visitJoinFunctions(self, ctx: Parser.JoinFunctionsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#genericFunctions.
    def visitGenericFunctions(self, ctx: Parser.GenericFunctionsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#stringFunctions.
    def visitStringFunctions(self, ctx: Parser.StringFunctionsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#numericFunctions.
    def visitNumericFunctions(self, ctx: Parser.NumericFunctionsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#comparisonFunctions.
    def visitComparisonFunctions(self, ctx: Parser.ComparisonFunctionsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#timeFunctions.
    def visitTimeFunctions(self, ctx: Parser.TimeFunctionsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#setFunctions.
    def visitSetFunctions(self, ctx: Parser.SetFunctionsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#hierarchyFunctions.
    def visitHierarchyFunctions(self, ctx: Parser.HierarchyFunctionsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#validationFunctions.
    def visitValidationFunctions(self, ctx: Parser.ValidationFunctionsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#conditionalFunctions.
    def visitConditionalFunctions(self, ctx: Parser.ConditionalFunctionsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#aggregateFunctions.
    def visitAggregateFunctions(self, ctx: Parser.AggregateFunctionsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#analyticFunctions.
    def visitAnalyticFunctions(self, ctx: Parser.AnalyticFunctionsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#datasetClause.
    def visitDatasetClause(self, ctx: Parser.DatasetClauseContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#renameClause.
    def visitRenameClause(self, ctx: Parser.RenameClauseContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#aggrClause.
    def visitAggrClause(self, ctx: Parser.AggrClauseContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#filterClause.
    def visitFilterClause(self, ctx: Parser.FilterClauseContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#calcClause.
    def visitCalcClause(self, ctx: Parser.CalcClauseContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#keepOrDropClause.
    def visitKeepOrDropClause(self, ctx: Parser.KeepOrDropClauseContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#pivotOrUnpivotClause.
    def visitPivotOrUnpivotClause(self, ctx: Parser.PivotOrUnpivotClauseContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#subspaceClause.
    def visitSubspaceClause(self, ctx: Parser.SubspaceClauseContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#joinExpr.
    def visitJoinExpr(self, ctx: Parser.JoinExprContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#defOperator.
    def visitDefOperator(self, ctx: Parser.DefOperatorContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#defDatapointRuleset.
    def visitDefDatapointRuleset(self, ctx: Parser.DefDatapointRulesetContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#defHierarchical.
    def visitDefHierarchical(self, ctx: Parser.DefHierarchicalContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#callDataset.
    def visitCallDataset(self, ctx: Parser.CallDatasetContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#evalAtom.
    def visitEvalAtom(self, ctx: Parser.EvalAtomContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#castExprDataset.
    def visitCastExprDataset(self, ctx: Parser.CastExprDatasetContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#callComponent.
    def visitCallComponent(self, ctx: Parser.CallComponentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#castExprComponent.
    def visitCastExprComponent(self, ctx: Parser.CastExprComponentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#evalAtomComponent.
    def visitEvalAtomComponent(self, ctx: Parser.EvalAtomComponentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#parameterComponent.
    def visitParameterComponent(self, ctx: Parser.ParameterComponentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#parameter.
    def visitParameter(self, ctx: Parser.ParameterContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#unaryStringFunction.
    def visitUnaryStringFunction(self, ctx: Parser.UnaryStringFunctionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#substrAtom.
    def visitSubstrAtom(self, ctx: Parser.SubstrAtomContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#replaceAtom.
    def visitReplaceAtom(self, ctx: Parser.ReplaceAtomContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#instrAtom.
    def visitInstrAtom(self, ctx: Parser.InstrAtomContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#unaryStringFunctionComponent.
    def visitUnaryStringFunctionComponent(self, ctx: Parser.UnaryStringFunctionComponentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#substrAtomComponent.
    def visitSubstrAtomComponent(self, ctx: Parser.SubstrAtomComponentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#replaceAtomComponent.
    def visitReplaceAtomComponent(self, ctx: Parser.ReplaceAtomComponentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#instrAtomComponent.
    def visitInstrAtomComponent(self, ctx: Parser.InstrAtomComponentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#unaryNumeric.
    def visitUnaryNumeric(self, ctx: Parser.UnaryNumericContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#unaryWithOptionalNumeric.
    def visitUnaryWithOptionalNumeric(self, ctx: Parser.UnaryWithOptionalNumericContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#binaryNumeric.
    def visitBinaryNumeric(self, ctx: Parser.BinaryNumericContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#unaryNumericComponent.
    def visitUnaryNumericComponent(self, ctx: Parser.UnaryNumericComponentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#unaryWithOptionalNumericComponent.
    def visitUnaryWithOptionalNumericComponent(
        self, ctx: Parser.UnaryWithOptionalNumericComponentContext
    ):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#binaryNumericComponent.
    def visitBinaryNumericComponent(self, ctx: Parser.BinaryNumericComponentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#betweenAtom.
    def visitBetweenAtom(self, ctx: Parser.BetweenAtomContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#charsetMatchAtom.
    def visitCharsetMatchAtom(self, ctx: Parser.CharsetMatchAtomContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#isNullAtom.
    def visitIsNullAtom(self, ctx: Parser.IsNullAtomContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#existInAtom.
    def visitExistInAtom(self, ctx: Parser.ExistInAtomContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#betweenAtomComponent.
    def visitBetweenAtomComponent(self, ctx: Parser.BetweenAtomComponentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#charsetMatchAtomComponent.
    def visitCharsetMatchAtomComponent(self, ctx: Parser.CharsetMatchAtomComponentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#isNullAtomComponent.
    def visitIsNullAtomComponent(self, ctx: Parser.IsNullAtomComponentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#periodAtom.
    def visitPeriodAtom(self, ctx: Parser.PeriodAtomContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#fillTimeAtom.
    def visitFillTimeAtom(self, ctx: Parser.FillTimeAtomContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#flowAtom.
    def visitFlowAtom(self, ctx: Parser.FlowAtomContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#timeShiftAtom.
    def visitTimeShiftAtom(self, ctx: Parser.TimeShiftAtomContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#timeAggAtom.
    def visitTimeAggAtom(self, ctx: Parser.TimeAggAtomContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#currentDateAtom.
    def visitCurrentDateAtom(self, ctx: Parser.CurrentDateAtomContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#periodAtomComponent.
    def visitTimeUnaryAtomComponent(self, ctx: Parser.PeriodAtomComponentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#fillTimeAtomComponent.
    def visitFillTimeAtomComponent(self, ctx: Parser.FillTimeAtomComponentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#flowAtomComponent.
    def visitFlowAtomComponent(self, ctx: Parser.FlowAtomComponentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#timeShiftAtomComponent.
    def visitTimeShiftAtomComponent(self, ctx: Parser.TimeShiftAtomComponentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#timeAggAtomComponent.
    def visitTimeAggAtomComponent(self, ctx: Parser.TimeAggAtomComponentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#currentDateAtomComponent.
    def visitCurrentDateAtomComponent(self, ctx: Parser.CurrentDateAtomComponentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#unionAtom.
    def visitUnionAtom(self, ctx: Parser.UnionAtomContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#intersectAtom.
    def visitIntersectAtom(self, ctx: Parser.IntersectAtomContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#setOrSYmDiffAtom.
    def visitSetOrSYmDiffAtom(self, ctx: Parser.SetOrSYmDiffAtomContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#hierarchyOperators.
    def visitHierarchyOperators(self, ctx: Parser.HierarchyOperatorsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#validateDPruleset.
    def visitValidateDPruleset(self, ctx: Parser.ValidateDPrulesetContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#validateHRruleset.
    def visitValidateHRruleset(self, ctx: Parser.ValidateHRrulesetContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#validationSimple.
    def visitValidationSimple(self, ctx: Parser.ValidationSimpleContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#nvlAtom.
    def visitNvlAtom(self, ctx: Parser.NvlAtomContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#nvlAtomComponent.
    def visitNvlAtomComponent(self, ctx: Parser.NvlAtomComponentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#aggrComp.
    def visitAggrComp(self, ctx: Parser.AggrCompContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#countAggrComp.
    def visitCountAggrComp(self, ctx: Parser.CountAggrCompContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#aggrDataset.
    def visitAggrDataset(self, ctx: Parser.AggrDatasetContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#anSimpleFunction.
    def visitAnSimpleFunction(self, ctx: Parser.AnSimpleFunctionContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#lagOrLeadAn.
    def visitLagOrLeadAn(self, ctx: Parser.LagOrLeadAnContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#ratioToReportAn.
    def visitRatioToReportAn(self, ctx: Parser.RatioToReportAnContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#anSimpleFunctionComponent.
    def visitAnSimpleFunctionComponent(self, ctx: Parser.AnSimpleFunctionComponentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#lagOrLeadAnComponent.
    def visitLagOrLeadAnComponent(self, ctx: Parser.LagOrLeadAnComponentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#rankAnComponent.
    def visitRankAnComponent(self, ctx: Parser.RankAnComponentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#ratioToReportAnComponent.
    def visitRatioToReportAnComponent(self, ctx: Parser.RatioToReportAnComponentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#renameClauseItem.
    def visitRenameClauseItem(self, ctx: Parser.RenameClauseItemContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#aggregateClause.
    def visitAggregateClause(self, ctx: Parser.AggregateClauseContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#aggrFunctionClause.
    def visitAggrFunctionClause(self, ctx: Parser.AggrFunctionClauseContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#calcClauseItem.
    def visitCalcClauseItem(self, ctx: Parser.CalcClauseItemContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#subspaceClauseItem.
    def visitSubspaceClauseItem(self, ctx: Parser.SubspaceClauseItemContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#simpleScalar.
    def visitSimpleScalar(self, ctx: Parser.SimpleScalarContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#scalarWithCast.
    def visitScalarWithCast(self, ctx: Parser.ScalarWithCastContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#joinClauseWithoutUsing.
    def visitJoinClauseWithoutUsing(self, ctx: Parser.JoinClauseWithoutUsingContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#joinClause.
    def visitJoinClause(self, ctx: Parser.JoinClauseContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#joinClauseItem.
    def visitJoinClauseItem(self, ctx: Parser.JoinClauseItemContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#joinBody.
    def visitJoinBody(self, ctx: Parser.JoinBodyContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#joinApplyClause.
    def visitJoinApplyClause(self, ctx: Parser.JoinApplyClauseContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#partitionByClause.
    def visitPartitionByClause(self, ctx: Parser.PartitionByClauseContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#orderByClause.
    def visitOrderByClause(self, ctx: Parser.OrderByClauseContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#orderByItem.
    def visitOrderByItem(self, ctx: Parser.OrderByItemContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#windowingClause.
    def visitWindowingClause(self, ctx: Parser.WindowingClauseContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#signedInteger.
    def visitSignedInteger(self, ctx: Parser.SignedIntegerContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#limitClauseItem.
    def visitLimitClauseItem(self, ctx: Parser.LimitClauseItemContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#groupByOrExcept.
    def visitGroupByOrExcept(self, ctx: Parser.GroupByOrExceptContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#groupAll.
    def visitGroupAll(self, ctx: Parser.GroupAllContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#havingClause.
    def visitHavingClause(self, ctx: Parser.HavingClauseContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#parameterItem.
    def visitParameterItem(self, ctx: Parser.ParameterItemContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#outputParameterType.
    def visitOutputParameterType(self, ctx: Parser.OutputParameterTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#outputParameterTypeComponent.
    def visitOutputParameterTypeComponent(self, ctx: Parser.OutputParameterTypeComponentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#inputParameterType.
    def visitInputParameterType(self, ctx: Parser.InputParameterTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#rulesetType.
    def visitRulesetType(self, ctx: Parser.RulesetTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#scalarType.
    def visitScalarType(self, ctx: Parser.ScalarTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#componentType.
    def visitComponentType(self, ctx: Parser.ComponentTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#datasetType.
    def visitDatasetType(self, ctx: Parser.DatasetTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#evalDatasetType.
    def visitEvalDatasetType(self, ctx: Parser.EvalDatasetTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#scalarSetType.
    def visitScalarSetType(self, ctx: Parser.ScalarSetTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#dataPoint.
    def visitDataPoint(self, ctx: Parser.DataPointContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#dataPointVd.
    def visitDataPointVd(self, ctx: Parser.DataPointVdContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#dataPointVar.
    def visitDataPointVar(self, ctx: Parser.DataPointVarContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#hrRulesetType.
    def visitHrRulesetType(self, ctx: Parser.HrRulesetTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#hrRulesetVdType.
    def visitHrRulesetVdType(self, ctx: Parser.HrRulesetVdTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#hrRulesetVarType.
    def visitHrRulesetVarType(self, ctx: Parser.HrRulesetVarTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#valueDomainName.
    def visitValueDomainName(self, ctx: Parser.ValueDomainNameContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#rulesetID.
    def visitRulesetID(self, ctx: Parser.RulesetIDContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#rulesetSignature.
    def visitRulesetSignature(self, ctx: Parser.RulesetSignatureContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#signature.
    def visitSignature(self, ctx: Parser.SignatureContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#ruleClauseDatapoint.
    def visitRuleClauseDatapoint(self, ctx: Parser.RuleClauseDatapointContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#ruleItemDatapoint.
    def visitRuleItemDatapoint(self, ctx: Parser.RuleItemDatapointContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#ruleClauseHierarchical.
    def visitRuleClauseHierarchical(self, ctx: Parser.RuleClauseHierarchicalContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#ruleItemHierarchical.
    def visitRuleItemHierarchical(self, ctx: Parser.RuleItemHierarchicalContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#hierRuleSignature.
    def visitHierRuleSignature(self, ctx: Parser.HierRuleSignatureContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#valueDomainSignature.
    def visitValueDomainSignature(self, ctx: Parser.ValueDomainSignatureContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#codeItemRelation.
    def visitCodeItemRelation(self, ctx: Parser.CodeItemRelationContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#codeItemRelationClause.
    def visitCodeItemRelationClause(self, ctx: Parser.CodeItemRelationClauseContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#valueDomainValue.
    def visitValueDomainValue(self, ctx: Parser.ValueDomainValueContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#conditionConstraint.
    def visitConditionConstraint(self, ctx: Parser.ConditionConstraintContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#rangeConstraint.
    def visitRangeConstraint(self, ctx: Parser.RangeConstraintContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#compConstraint.
    def visitCompConstraint(self, ctx: Parser.CompConstraintContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#multModifier.
    def visitMultModifier(self, ctx: Parser.MultModifierContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#validationOutput.
    def visitValidationOutput(self, ctx: Parser.ValidationOutputContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#validationMode.
    def visitValidationMode(self, ctx: Parser.ValidationModeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#conditionClause.
    def visitConditionClause(self, ctx: Parser.ConditionClauseContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#inputMode.
    def visitInputMode(self, ctx: Parser.InputModeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#imbalanceExpr.
    def visitImbalanceExpr(self, ctx: Parser.ImbalanceExprContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#inputModeHierarchy.
    def visitInputModeHierarchy(self, ctx: Parser.InputModeHierarchyContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#outputModeHierarchy.
    def visitOutputModeHierarchy(self, ctx: Parser.OutputModeHierarchyContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#alias.
    def visitAlias(self, ctx: Parser.AliasContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#varID.
    def visitVarID(self, ctx: Parser.VarIDContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#simpleComponentId.
    def visitSimpleComponentId(self, ctx: Parser.SimpleComponentIdContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#componentID.
    def visitComponentID(self, ctx: Parser.ComponentIDContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#lists.
    def visitLists(self, ctx: Parser.ListsContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#erCode.
    def visitErCode(self, ctx: Parser.ErCodeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#erLevel.
    def visitErLevel(self, ctx: Parser.ErLevelContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#comparisonOperand.
    def visitComparisonOperand(self, ctx: Parser.ComparisonOperandContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#optionalExpr.
    def visitOptionalExpr(self, ctx: Parser.OptionalExprContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#optionalExprComponent.
    def visitOptionalExprComponent(self, ctx: Parser.OptionalExprComponentContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#componentRole.
    def visitComponentRole(self, ctx: Parser.ComponentRoleContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#viralAttribute.
    def visitViralAttribute(self, ctx: Parser.ViralAttributeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#valueDomainID.
    def visitValueDomainID(self, ctx: Parser.ValueDomainIDContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#operatorID.
    def visitOperatorID(self, ctx: Parser.OperatorIDContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#routineName.
    def visitRoutineName(self, ctx: Parser.RoutineNameContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#constant.
    def visitConstant(self, ctx: Parser.ConstantContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#basicScalarType.
    def visitBasicScalarType(self, ctx: Parser.BasicScalarTypeContext):
        return self.visitChildren(ctx)

    # Visit a parse tree produced by Parser#retainType.
    def visitRetainType(self, ctx: Parser.RetainTypeContext):
        return self.visitChildren(ctx)


del Parser
