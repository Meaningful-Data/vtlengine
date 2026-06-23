
// Generated from Vtl.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"
#include "Vtl.h"



/**
 * This class defines an abstract visitor for a parse tree
 * produced by Vtl.
 */
class  VtlVisitor : public antlr4::tree::AbstractParseTreeVisitor {
public:

  /**
   * Visit parse trees produced by Vtl.
   */
    virtual std::any visitStart(Vtl::StartContext *context) = 0;

    virtual std::any visitTemporaryAssignment(Vtl::TemporaryAssignmentContext *context) = 0;

    virtual std::any visitPersistAssignment(Vtl::PersistAssignmentContext *context) = 0;

    virtual std::any visitDefineExpression(Vtl::DefineExpressionContext *context) = 0;

    virtual std::any visitVarIdExpr(Vtl::VarIdExprContext *context) = 0;

    virtual std::any visitMembershipExpr(Vtl::MembershipExprContext *context) = 0;

    virtual std::any visitInNotInExpr(Vtl::InNotInExprContext *context) = 0;

    virtual std::any visitBooleanExpr(Vtl::BooleanExprContext *context) = 0;

    virtual std::any visitComparisonExpr(Vtl::ComparisonExprContext *context) = 0;

    virtual std::any visitUnaryExpr(Vtl::UnaryExprContext *context) = 0;

    virtual std::any visitFunctionsExpression(Vtl::FunctionsExpressionContext *context) = 0;

    virtual std::any visitIfExpr(Vtl::IfExprContext *context) = 0;

    virtual std::any visitClauseExpr(Vtl::ClauseExprContext *context) = 0;

    virtual std::any visitCaseExpr(Vtl::CaseExprContext *context) = 0;

    virtual std::any visitArithmeticExpr(Vtl::ArithmeticExprContext *context) = 0;

    virtual std::any visitParenthesisExpr(Vtl::ParenthesisExprContext *context) = 0;

    virtual std::any visitConstantExpr(Vtl::ConstantExprContext *context) = 0;

    virtual std::any visitArithmeticExprOrConcat(Vtl::ArithmeticExprOrConcatContext *context) = 0;

    virtual std::any visitArithmeticExprComp(Vtl::ArithmeticExprCompContext *context) = 0;

    virtual std::any visitIfExprComp(Vtl::IfExprCompContext *context) = 0;

    virtual std::any visitComparisonExprComp(Vtl::ComparisonExprCompContext *context) = 0;

    virtual std::any visitFunctionsExpressionComp(Vtl::FunctionsExpressionCompContext *context) = 0;

    virtual std::any visitCompId(Vtl::CompIdContext *context) = 0;

    virtual std::any visitConstantExprComp(Vtl::ConstantExprCompContext *context) = 0;

    virtual std::any visitArithmeticExprOrConcatComp(Vtl::ArithmeticExprOrConcatCompContext *context) = 0;

    virtual std::any visitParenthesisExprComp(Vtl::ParenthesisExprCompContext *context) = 0;

    virtual std::any visitInNotInExprComp(Vtl::InNotInExprCompContext *context) = 0;

    virtual std::any visitUnaryExprComp(Vtl::UnaryExprCompContext *context) = 0;

    virtual std::any visitCaseExprComp(Vtl::CaseExprCompContext *context) = 0;

    virtual std::any visitBooleanExprComp(Vtl::BooleanExprCompContext *context) = 0;

    virtual std::any visitGenericFunctionsComponents(Vtl::GenericFunctionsComponentsContext *context) = 0;

    virtual std::any visitStringFunctionsComponents(Vtl::StringFunctionsComponentsContext *context) = 0;

    virtual std::any visitNumericFunctionsComponents(Vtl::NumericFunctionsComponentsContext *context) = 0;

    virtual std::any visitComparisonFunctionsComponents(Vtl::ComparisonFunctionsComponentsContext *context) = 0;

    virtual std::any visitTimeFunctionsComponents(Vtl::TimeFunctionsComponentsContext *context) = 0;

    virtual std::any visitConditionalFunctionsComponents(Vtl::ConditionalFunctionsComponentsContext *context) = 0;

    virtual std::any visitAggregateFunctionsComponents(Vtl::AggregateFunctionsComponentsContext *context) = 0;

    virtual std::any visitAnalyticFunctionsComponents(Vtl::AnalyticFunctionsComponentsContext *context) = 0;

    virtual std::any visitJoinFunctions(Vtl::JoinFunctionsContext *context) = 0;

    virtual std::any visitGenericFunctions(Vtl::GenericFunctionsContext *context) = 0;

    virtual std::any visitStringFunctions(Vtl::StringFunctionsContext *context) = 0;

    virtual std::any visitNumericFunctions(Vtl::NumericFunctionsContext *context) = 0;

    virtual std::any visitComparisonFunctions(Vtl::ComparisonFunctionsContext *context) = 0;

    virtual std::any visitTimeFunctions(Vtl::TimeFunctionsContext *context) = 0;

    virtual std::any visitSetFunctions(Vtl::SetFunctionsContext *context) = 0;

    virtual std::any visitHierarchyFunctions(Vtl::HierarchyFunctionsContext *context) = 0;

    virtual std::any visitValidationFunctions(Vtl::ValidationFunctionsContext *context) = 0;

    virtual std::any visitConditionalFunctions(Vtl::ConditionalFunctionsContext *context) = 0;

    virtual std::any visitAggregateFunctions(Vtl::AggregateFunctionsContext *context) = 0;

    virtual std::any visitAnalyticFunctions(Vtl::AnalyticFunctionsContext *context) = 0;

    virtual std::any visitDatasetClause(Vtl::DatasetClauseContext *context) = 0;

    virtual std::any visitRenameClause(Vtl::RenameClauseContext *context) = 0;

    virtual std::any visitAggrClause(Vtl::AggrClauseContext *context) = 0;

    virtual std::any visitFilterClause(Vtl::FilterClauseContext *context) = 0;

    virtual std::any visitCalcClause(Vtl::CalcClauseContext *context) = 0;

    virtual std::any visitKeepOrDropClause(Vtl::KeepOrDropClauseContext *context) = 0;

    virtual std::any visitPivotOrUnpivotClause(Vtl::PivotOrUnpivotClauseContext *context) = 0;

    virtual std::any visitCustomPivotClause(Vtl::CustomPivotClauseContext *context) = 0;

    virtual std::any visitSubspaceClause(Vtl::SubspaceClauseContext *context) = 0;

    virtual std::any visitInnerJoinExpr(Vtl::InnerJoinExprContext *context) = 0;

    virtual std::any visitLeftJoinExpr(Vtl::LeftJoinExprContext *context) = 0;

    virtual std::any visitFullJoinExpr(Vtl::FullJoinExprContext *context) = 0;

    virtual std::any visitCrossJoinExpr(Vtl::CrossJoinExprContext *context) = 0;

    virtual std::any visitDefOperator(Vtl::DefOperatorContext *context) = 0;

    virtual std::any visitDefDatapointRuleset(Vtl::DefDatapointRulesetContext *context) = 0;

    virtual std::any visitDefHierarchical(Vtl::DefHierarchicalContext *context) = 0;

    virtual std::any visitDefViralPropagation(Vtl::DefViralPropagationContext *context) = 0;

    virtual std::any visitVpSignature(Vtl::VpSignatureContext *context) = 0;

    virtual std::any visitVpBody(Vtl::VpBodyContext *context) = 0;

    virtual std::any visitEnumeratedVpClause(Vtl::EnumeratedVpClauseContext *context) = 0;

    virtual std::any visitAggregationVpClause(Vtl::AggregationVpClauseContext *context) = 0;

    virtual std::any visitDefaultVpClause(Vtl::DefaultVpClauseContext *context) = 0;

    virtual std::any visitVpCondition(Vtl::VpConditionContext *context) = 0;

    virtual std::any visitCallDataset(Vtl::CallDatasetContext *context) = 0;

    virtual std::any visitEvalAtom(Vtl::EvalAtomContext *context) = 0;

    virtual std::any visitCastExprDataset(Vtl::CastExprDatasetContext *context) = 0;

    virtual std::any visitCallComponent(Vtl::CallComponentContext *context) = 0;

    virtual std::any visitCastExprComponent(Vtl::CastExprComponentContext *context) = 0;

    virtual std::any visitEvalAtomComponent(Vtl::EvalAtomComponentContext *context) = 0;

    virtual std::any visitParameterComponent(Vtl::ParameterComponentContext *context) = 0;

    virtual std::any visitParameter(Vtl::ParameterContext *context) = 0;

    virtual std::any visitStringDistanceMethods(Vtl::StringDistanceMethodsContext *context) = 0;

    virtual std::any visitUnaryStringFunction(Vtl::UnaryStringFunctionContext *context) = 0;

    virtual std::any visitSubstrAtom(Vtl::SubstrAtomContext *context) = 0;

    virtual std::any visitReplaceAtom(Vtl::ReplaceAtomContext *context) = 0;

    virtual std::any visitInstrAtom(Vtl::InstrAtomContext *context) = 0;

    virtual std::any visitStringDistanceAtom(Vtl::StringDistanceAtomContext *context) = 0;

    virtual std::any visitUnaryStringFunctionComponent(Vtl::UnaryStringFunctionComponentContext *context) = 0;

    virtual std::any visitSubstrAtomComponent(Vtl::SubstrAtomComponentContext *context) = 0;

    virtual std::any visitReplaceAtomComponent(Vtl::ReplaceAtomComponentContext *context) = 0;

    virtual std::any visitInstrAtomComponent(Vtl::InstrAtomComponentContext *context) = 0;

    virtual std::any visitStringDistanceAtomComponent(Vtl::StringDistanceAtomComponentContext *context) = 0;

    virtual std::any visitUnaryNumeric(Vtl::UnaryNumericContext *context) = 0;

    virtual std::any visitUnaryWithOptionalNumeric(Vtl::UnaryWithOptionalNumericContext *context) = 0;

    virtual std::any visitBinaryNumeric(Vtl::BinaryNumericContext *context) = 0;

    virtual std::any visitUnaryNumericComponent(Vtl::UnaryNumericComponentContext *context) = 0;

    virtual std::any visitUnaryWithOptionalNumericComponent(Vtl::UnaryWithOptionalNumericComponentContext *context) = 0;

    virtual std::any visitBinaryNumericComponent(Vtl::BinaryNumericComponentContext *context) = 0;

    virtual std::any visitBetweenAtom(Vtl::BetweenAtomContext *context) = 0;

    virtual std::any visitCharsetMatchAtom(Vtl::CharsetMatchAtomContext *context) = 0;

    virtual std::any visitIsNullAtom(Vtl::IsNullAtomContext *context) = 0;

    virtual std::any visitExistInAtom(Vtl::ExistInAtomContext *context) = 0;

    virtual std::any visitBetweenAtomComponent(Vtl::BetweenAtomComponentContext *context) = 0;

    virtual std::any visitCharsetMatchAtomComponent(Vtl::CharsetMatchAtomComponentContext *context) = 0;

    virtual std::any visitIsNullAtomComponent(Vtl::IsNullAtomComponentContext *context) = 0;

    virtual std::any visitPeriodAtom(Vtl::PeriodAtomContext *context) = 0;

    virtual std::any visitFillTimeAtom(Vtl::FillTimeAtomContext *context) = 0;

    virtual std::any visitFlowAtom(Vtl::FlowAtomContext *context) = 0;

    virtual std::any visitTimeShiftAtom(Vtl::TimeShiftAtomContext *context) = 0;

    virtual std::any visitTimeAggAtom(Vtl::TimeAggAtomContext *context) = 0;

    virtual std::any visitCurrentDateAtom(Vtl::CurrentDateAtomContext *context) = 0;

    virtual std::any visitDateDiffAtom(Vtl::DateDiffAtomContext *context) = 0;

    virtual std::any visitDateAddAtom(Vtl::DateAddAtomContext *context) = 0;

    virtual std::any visitYearAtom(Vtl::YearAtomContext *context) = 0;

    virtual std::any visitMonthAtom(Vtl::MonthAtomContext *context) = 0;

    virtual std::any visitDayOfMonthAtom(Vtl::DayOfMonthAtomContext *context) = 0;

    virtual std::any visitDayOfYearAtom(Vtl::DayOfYearAtomContext *context) = 0;

    virtual std::any visitDayToYearAtom(Vtl::DayToYearAtomContext *context) = 0;

    virtual std::any visitDayToMonthAtom(Vtl::DayToMonthAtomContext *context) = 0;

    virtual std::any visitYearTodayAtom(Vtl::YearTodayAtomContext *context) = 0;

    virtual std::any visitMonthTodayAtom(Vtl::MonthTodayAtomContext *context) = 0;

    virtual std::any visitPeriodAtomComponent(Vtl::PeriodAtomComponentContext *context) = 0;

    virtual std::any visitFillTimeAtomComponent(Vtl::FillTimeAtomComponentContext *context) = 0;

    virtual std::any visitFlowAtomComponent(Vtl::FlowAtomComponentContext *context) = 0;

    virtual std::any visitTimeShiftAtomComponent(Vtl::TimeShiftAtomComponentContext *context) = 0;

    virtual std::any visitTimeAggAtomComponent(Vtl::TimeAggAtomComponentContext *context) = 0;

    virtual std::any visitCurrentDateAtomComponent(Vtl::CurrentDateAtomComponentContext *context) = 0;

    virtual std::any visitDateDiffAtomComponent(Vtl::DateDiffAtomComponentContext *context) = 0;

    virtual std::any visitDateAddAtomComponent(Vtl::DateAddAtomComponentContext *context) = 0;

    virtual std::any visitYearAtomComponent(Vtl::YearAtomComponentContext *context) = 0;

    virtual std::any visitMonthAtomComponent(Vtl::MonthAtomComponentContext *context) = 0;

    virtual std::any visitDayOfMonthAtomComponent(Vtl::DayOfMonthAtomComponentContext *context) = 0;

    virtual std::any visitDayOfYearAtomComponent(Vtl::DayOfYearAtomComponentContext *context) = 0;

    virtual std::any visitDayToYearAtomComponent(Vtl::DayToYearAtomComponentContext *context) = 0;

    virtual std::any visitDayToMonthAtomComponent(Vtl::DayToMonthAtomComponentContext *context) = 0;

    virtual std::any visitYearTodayAtomComponent(Vtl::YearTodayAtomComponentContext *context) = 0;

    virtual std::any visitMonthTodayAtomComponent(Vtl::MonthTodayAtomComponentContext *context) = 0;

    virtual std::any visitUnionAtom(Vtl::UnionAtomContext *context) = 0;

    virtual std::any visitIntersectAtom(Vtl::IntersectAtomContext *context) = 0;

    virtual std::any visitSetOrSYmDiffAtom(Vtl::SetOrSYmDiffAtomContext *context) = 0;

    virtual std::any visitHierarchyOperators(Vtl::HierarchyOperatorsContext *context) = 0;

    virtual std::any visitValidateDPruleset(Vtl::ValidateDPrulesetContext *context) = 0;

    virtual std::any visitValidateHRruleset(Vtl::ValidateHRrulesetContext *context) = 0;

    virtual std::any visitValidationSimple(Vtl::ValidationSimpleContext *context) = 0;

    virtual std::any visitNvlAtom(Vtl::NvlAtomContext *context) = 0;

    virtual std::any visitNvlAtomComponent(Vtl::NvlAtomComponentContext *context) = 0;

    virtual std::any visitAggrComp(Vtl::AggrCompContext *context) = 0;

    virtual std::any visitCountAggrComp(Vtl::CountAggrCompContext *context) = 0;

    virtual std::any visitAggrDataset(Vtl::AggrDatasetContext *context) = 0;

    virtual std::any visitAnSimpleFunction(Vtl::AnSimpleFunctionContext *context) = 0;

    virtual std::any visitLagOrLeadAn(Vtl::LagOrLeadAnContext *context) = 0;

    virtual std::any visitRatioToReportAn(Vtl::RatioToReportAnContext *context) = 0;

    virtual std::any visitAnSimpleFunctionComponent(Vtl::AnSimpleFunctionComponentContext *context) = 0;

    virtual std::any visitLagOrLeadAnComponent(Vtl::LagOrLeadAnComponentContext *context) = 0;

    virtual std::any visitRankAnComponent(Vtl::RankAnComponentContext *context) = 0;

    virtual std::any visitRatioToReportAnComponent(Vtl::RatioToReportAnComponentContext *context) = 0;

    virtual std::any visitRenameClauseItem(Vtl::RenameClauseItemContext *context) = 0;

    virtual std::any visitAggregateClause(Vtl::AggregateClauseContext *context) = 0;

    virtual std::any visitAggrFunctionClause(Vtl::AggrFunctionClauseContext *context) = 0;

    virtual std::any visitCalcClauseItem(Vtl::CalcClauseItemContext *context) = 0;

    virtual std::any visitSubspaceClauseItem(Vtl::SubspaceClauseItemContext *context) = 0;

    virtual std::any visitSimpleScalar(Vtl::SimpleScalarContext *context) = 0;

    virtual std::any visitScalarWithCast(Vtl::ScalarWithCastContext *context) = 0;

    virtual std::any visitJoinClause(Vtl::JoinClauseContext *context) = 0;

    virtual std::any visitJoinClauseItem(Vtl::JoinClauseItemContext *context) = 0;

    virtual std::any visitUsingClause(Vtl::UsingClauseContext *context) = 0;

    virtual std::any visitNvlJoinClause(Vtl::NvlJoinClauseContext *context) = 0;

    virtual std::any visitJoinBody(Vtl::JoinBodyContext *context) = 0;

    virtual std::any visitJoinApplyClause(Vtl::JoinApplyClauseContext *context) = 0;

    virtual std::any visitPartitionListed(Vtl::PartitionListedContext *context) = 0;

    virtual std::any visitPartitionExceptAll(Vtl::PartitionExceptAllContext *context) = 0;

    virtual std::any visitOrderByClause(Vtl::OrderByClauseContext *context) = 0;

    virtual std::any visitOrderByItem(Vtl::OrderByItemContext *context) = 0;

    virtual std::any visitWindowingClause(Vtl::WindowingClauseContext *context) = 0;

    virtual std::any visitSignedInteger(Vtl::SignedIntegerContext *context) = 0;

    virtual std::any visitSignedNumber(Vtl::SignedNumberContext *context) = 0;

    virtual std::any visitLimitClauseItem(Vtl::LimitClauseItemContext *context) = 0;

    virtual std::any visitGroupByOrExcept(Vtl::GroupByOrExceptContext *context) = 0;

    virtual std::any visitGroupAll(Vtl::GroupAllContext *context) = 0;

    virtual std::any visitHavingClause(Vtl::HavingClauseContext *context) = 0;

    virtual std::any visitParameterItem(Vtl::ParameterItemContext *context) = 0;

    virtual std::any visitOutputParameterType(Vtl::OutputParameterTypeContext *context) = 0;

    virtual std::any visitOutputParameterTypeComponent(Vtl::OutputParameterTypeComponentContext *context) = 0;

    virtual std::any visitInputParameterType(Vtl::InputParameterTypeContext *context) = 0;

    virtual std::any visitRulesetType(Vtl::RulesetTypeContext *context) = 0;

    virtual std::any visitScalarType(Vtl::ScalarTypeContext *context) = 0;

    virtual std::any visitComponentType(Vtl::ComponentTypeContext *context) = 0;

    virtual std::any visitDatasetType(Vtl::DatasetTypeContext *context) = 0;

    virtual std::any visitEvalDatasetType(Vtl::EvalDatasetTypeContext *context) = 0;

    virtual std::any visitScalarSetType(Vtl::ScalarSetTypeContext *context) = 0;

    virtual std::any visitDataPoint(Vtl::DataPointContext *context) = 0;

    virtual std::any visitDataPointVd(Vtl::DataPointVdContext *context) = 0;

    virtual std::any visitDataPointVar(Vtl::DataPointVarContext *context) = 0;

    virtual std::any visitHrRulesetType(Vtl::HrRulesetTypeContext *context) = 0;

    virtual std::any visitHrRulesetVdType(Vtl::HrRulesetVdTypeContext *context) = 0;

    virtual std::any visitHrRulesetVarType(Vtl::HrRulesetVarTypeContext *context) = 0;

    virtual std::any visitValueDomainName(Vtl::ValueDomainNameContext *context) = 0;

    virtual std::any visitRulesetID(Vtl::RulesetIDContext *context) = 0;

    virtual std::any visitRulesetSignature(Vtl::RulesetSignatureContext *context) = 0;

    virtual std::any visitSignature(Vtl::SignatureContext *context) = 0;

    virtual std::any visitRuleClauseDatapoint(Vtl::RuleClauseDatapointContext *context) = 0;

    virtual std::any visitRuleItemDatapoint(Vtl::RuleItemDatapointContext *context) = 0;

    virtual std::any visitRuleClauseHierarchical(Vtl::RuleClauseHierarchicalContext *context) = 0;

    virtual std::any visitRuleItemHierarchical(Vtl::RuleItemHierarchicalContext *context) = 0;

    virtual std::any visitHierRuleSignature(Vtl::HierRuleSignatureContext *context) = 0;

    virtual std::any visitValueDomainSignature(Vtl::ValueDomainSignatureContext *context) = 0;

    virtual std::any visitCodeItemRelation(Vtl::CodeItemRelationContext *context) = 0;

    virtual std::any visitCodeItemRelationClause(Vtl::CodeItemRelationClauseContext *context) = 0;

    virtual std::any visitValueDomainValue(Vtl::ValueDomainValueContext *context) = 0;

    virtual std::any visitConditionConstraint(Vtl::ConditionConstraintContext *context) = 0;

    virtual std::any visitRangeConstraint(Vtl::RangeConstraintContext *context) = 0;

    virtual std::any visitCompConstraint(Vtl::CompConstraintContext *context) = 0;

    virtual std::any visitMultModifier(Vtl::MultModifierContext *context) = 0;

    virtual std::any visitValidationOutput(Vtl::ValidationOutputContext *context) = 0;

    virtual std::any visitValidationMode(Vtl::ValidationModeContext *context) = 0;

    virtual std::any visitConditionClause(Vtl::ConditionClauseContext *context) = 0;

    virtual std::any visitInputMode(Vtl::InputModeContext *context) = 0;

    virtual std::any visitImbalanceExpr(Vtl::ImbalanceExprContext *context) = 0;

    virtual std::any visitInputModeHierarchy(Vtl::InputModeHierarchyContext *context) = 0;

    virtual std::any visitOutputModeHierarchy(Vtl::OutputModeHierarchyContext *context) = 0;

    virtual std::any visitAlias(Vtl::AliasContext *context) = 0;

    virtual std::any visitVarID(Vtl::VarIDContext *context) = 0;

    virtual std::any visitSimpleComponentId(Vtl::SimpleComponentIdContext *context) = 0;

    virtual std::any visitComponentID(Vtl::ComponentIDContext *context) = 0;

    virtual std::any visitLists(Vtl::ListsContext *context) = 0;

    virtual std::any visitErCode(Vtl::ErCodeContext *context) = 0;

    virtual std::any visitErLevel(Vtl::ErLevelContext *context) = 0;

    virtual std::any visitComparisonOperand(Vtl::ComparisonOperandContext *context) = 0;

    virtual std::any visitOptionalExpr(Vtl::OptionalExprContext *context) = 0;

    virtual std::any visitOptionalExprComponent(Vtl::OptionalExprComponentContext *context) = 0;

    virtual std::any visitComponentRole(Vtl::ComponentRoleContext *context) = 0;

    virtual std::any visitViralAttribute(Vtl::ViralAttributeContext *context) = 0;

    virtual std::any visitValueDomainID(Vtl::ValueDomainIDContext *context) = 0;

    virtual std::any visitOperatorID(Vtl::OperatorIDContext *context) = 0;

    virtual std::any visitRoutineName(Vtl::RoutineNameContext *context) = 0;

    virtual std::any visitIntegerLiteral(Vtl::IntegerLiteralContext *context) = 0;

    virtual std::any visitNumberLiteral(Vtl::NumberLiteralContext *context) = 0;

    virtual std::any visitBooleanLiteral(Vtl::BooleanLiteralContext *context) = 0;

    virtual std::any visitStringLiteral(Vtl::StringLiteralContext *context) = 0;

    virtual std::any visitNullLiteral(Vtl::NullLiteralContext *context) = 0;

    virtual std::any visitBasicScalarType(Vtl::BasicScalarTypeContext *context) = 0;

    virtual std::any visitRetainType(Vtl::RetainTypeContext *context) = 0;


};

