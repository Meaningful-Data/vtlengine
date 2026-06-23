
// Generated from Vtl.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"
#include "VtlVisitor.h"


/**
 * This class provides an empty implementation of VtlVisitor, which can be
 * extended to create a visitor which only needs to handle a subset of the available methods.
 */
class  VtlBaseVisitor : public VtlVisitor {
public:

  virtual std::any visitStart(Vtl::StartContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTemporaryAssignment(Vtl::TemporaryAssignmentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPersistAssignment(Vtl::PersistAssignmentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDefineExpression(Vtl::DefineExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitVarIdExpr(Vtl::VarIdExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMembershipExpr(Vtl::MembershipExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitInNotInExpr(Vtl::InNotInExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBooleanExpr(Vtl::BooleanExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitComparisonExpr(Vtl::ComparisonExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUnaryExpr(Vtl::UnaryExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFunctionsExpression(Vtl::FunctionsExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitIfExpr(Vtl::IfExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitClauseExpr(Vtl::ClauseExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCaseExpr(Vtl::CaseExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitArithmeticExpr(Vtl::ArithmeticExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitParenthesisExpr(Vtl::ParenthesisExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitConstantExpr(Vtl::ConstantExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitArithmeticExprOrConcat(Vtl::ArithmeticExprOrConcatContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitArithmeticExprComp(Vtl::ArithmeticExprCompContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitIfExprComp(Vtl::IfExprCompContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitComparisonExprComp(Vtl::ComparisonExprCompContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFunctionsExpressionComp(Vtl::FunctionsExpressionCompContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCompId(Vtl::CompIdContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitConstantExprComp(Vtl::ConstantExprCompContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitArithmeticExprOrConcatComp(Vtl::ArithmeticExprOrConcatCompContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitParenthesisExprComp(Vtl::ParenthesisExprCompContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitInNotInExprComp(Vtl::InNotInExprCompContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUnaryExprComp(Vtl::UnaryExprCompContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCaseExprComp(Vtl::CaseExprCompContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBooleanExprComp(Vtl::BooleanExprCompContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitGenericFunctionsComponents(Vtl::GenericFunctionsComponentsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitStringFunctionsComponents(Vtl::StringFunctionsComponentsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitNumericFunctionsComponents(Vtl::NumericFunctionsComponentsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitComparisonFunctionsComponents(Vtl::ComparisonFunctionsComponentsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTimeFunctionsComponents(Vtl::TimeFunctionsComponentsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitConditionalFunctionsComponents(Vtl::ConditionalFunctionsComponentsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAggregateFunctionsComponents(Vtl::AggregateFunctionsComponentsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAnalyticFunctionsComponents(Vtl::AnalyticFunctionsComponentsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitJoinFunctions(Vtl::JoinFunctionsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitGenericFunctions(Vtl::GenericFunctionsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitStringFunctions(Vtl::StringFunctionsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitNumericFunctions(Vtl::NumericFunctionsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitComparisonFunctions(Vtl::ComparisonFunctionsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTimeFunctions(Vtl::TimeFunctionsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitSetFunctions(Vtl::SetFunctionsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitHierarchyFunctions(Vtl::HierarchyFunctionsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitValidationFunctions(Vtl::ValidationFunctionsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitConditionalFunctions(Vtl::ConditionalFunctionsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAggregateFunctions(Vtl::AggregateFunctionsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAnalyticFunctions(Vtl::AnalyticFunctionsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDatasetClause(Vtl::DatasetClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRenameClause(Vtl::RenameClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAggrClause(Vtl::AggrClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFilterClause(Vtl::FilterClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCalcClause(Vtl::CalcClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitKeepOrDropClause(Vtl::KeepOrDropClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPivotOrUnpivotClause(Vtl::PivotOrUnpivotClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCustomPivotClause(Vtl::CustomPivotClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitSubspaceClause(Vtl::SubspaceClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitInnerJoinExpr(Vtl::InnerJoinExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLeftJoinExpr(Vtl::LeftJoinExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFullJoinExpr(Vtl::FullJoinExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCrossJoinExpr(Vtl::CrossJoinExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDefOperator(Vtl::DefOperatorContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDefDatapointRuleset(Vtl::DefDatapointRulesetContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDefHierarchical(Vtl::DefHierarchicalContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDefViralPropagation(Vtl::DefViralPropagationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitVpSignature(Vtl::VpSignatureContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitVpBody(Vtl::VpBodyContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitEnumeratedVpClause(Vtl::EnumeratedVpClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAggregationVpClause(Vtl::AggregationVpClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDefaultVpClause(Vtl::DefaultVpClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitVpCondition(Vtl::VpConditionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCallDataset(Vtl::CallDatasetContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitEvalAtom(Vtl::EvalAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCastExprDataset(Vtl::CastExprDatasetContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCallComponent(Vtl::CallComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCastExprComponent(Vtl::CastExprComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitEvalAtomComponent(Vtl::EvalAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitParameterComponent(Vtl::ParameterComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitParameter(Vtl::ParameterContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitStringDistanceMethods(Vtl::StringDistanceMethodsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUnaryStringFunction(Vtl::UnaryStringFunctionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitSubstrAtom(Vtl::SubstrAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitReplaceAtom(Vtl::ReplaceAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitInstrAtom(Vtl::InstrAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitStringDistanceAtom(Vtl::StringDistanceAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUnaryStringFunctionComponent(Vtl::UnaryStringFunctionComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitSubstrAtomComponent(Vtl::SubstrAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitReplaceAtomComponent(Vtl::ReplaceAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitInstrAtomComponent(Vtl::InstrAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitStringDistanceAtomComponent(Vtl::StringDistanceAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUnaryNumeric(Vtl::UnaryNumericContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUnaryWithOptionalNumeric(Vtl::UnaryWithOptionalNumericContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBinaryNumeric(Vtl::BinaryNumericContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUnaryNumericComponent(Vtl::UnaryNumericComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUnaryWithOptionalNumericComponent(Vtl::UnaryWithOptionalNumericComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBinaryNumericComponent(Vtl::BinaryNumericComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBetweenAtom(Vtl::BetweenAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCharsetMatchAtom(Vtl::CharsetMatchAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitIsNullAtom(Vtl::IsNullAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitExistInAtom(Vtl::ExistInAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBetweenAtomComponent(Vtl::BetweenAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCharsetMatchAtomComponent(Vtl::CharsetMatchAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitIsNullAtomComponent(Vtl::IsNullAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPeriodAtom(Vtl::PeriodAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFillTimeAtom(Vtl::FillTimeAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFlowAtom(Vtl::FlowAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTimeShiftAtom(Vtl::TimeShiftAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTimeAggAtom(Vtl::TimeAggAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCurrentDateAtom(Vtl::CurrentDateAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDateDiffAtom(Vtl::DateDiffAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDateAddAtom(Vtl::DateAddAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitYearAtom(Vtl::YearAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMonthAtom(Vtl::MonthAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDayOfMonthAtom(Vtl::DayOfMonthAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDayOfYearAtom(Vtl::DayOfYearAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDayToYearAtom(Vtl::DayToYearAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDayToMonthAtom(Vtl::DayToMonthAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitYearTodayAtom(Vtl::YearTodayAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMonthTodayAtom(Vtl::MonthTodayAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPeriodAtomComponent(Vtl::PeriodAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFillTimeAtomComponent(Vtl::FillTimeAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFlowAtomComponent(Vtl::FlowAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTimeShiftAtomComponent(Vtl::TimeShiftAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTimeAggAtomComponent(Vtl::TimeAggAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCurrentDateAtomComponent(Vtl::CurrentDateAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDateDiffAtomComponent(Vtl::DateDiffAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDateAddAtomComponent(Vtl::DateAddAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitYearAtomComponent(Vtl::YearAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMonthAtomComponent(Vtl::MonthAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDayOfMonthAtomComponent(Vtl::DayOfMonthAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDayOfYearAtomComponent(Vtl::DayOfYearAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDayToYearAtomComponent(Vtl::DayToYearAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDayToMonthAtomComponent(Vtl::DayToMonthAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitYearTodayAtomComponent(Vtl::YearTodayAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMonthTodayAtomComponent(Vtl::MonthTodayAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUnionAtom(Vtl::UnionAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitIntersectAtom(Vtl::IntersectAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitSetOrSYmDiffAtom(Vtl::SetOrSYmDiffAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitHierarchyOperators(Vtl::HierarchyOperatorsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitValidateDPruleset(Vtl::ValidateDPrulesetContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitValidateHRruleset(Vtl::ValidateHRrulesetContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitValidationSimple(Vtl::ValidationSimpleContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitNvlAtom(Vtl::NvlAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitNvlAtomComponent(Vtl::NvlAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAggrComp(Vtl::AggrCompContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCountAggrComp(Vtl::CountAggrCompContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAggrDataset(Vtl::AggrDatasetContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAnSimpleFunction(Vtl::AnSimpleFunctionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLagOrLeadAn(Vtl::LagOrLeadAnContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRatioToReportAn(Vtl::RatioToReportAnContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAnSimpleFunctionComponent(Vtl::AnSimpleFunctionComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLagOrLeadAnComponent(Vtl::LagOrLeadAnComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRankAnComponent(Vtl::RankAnComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRatioToReportAnComponent(Vtl::RatioToReportAnComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRenameClauseItem(Vtl::RenameClauseItemContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAggregateClause(Vtl::AggregateClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAggrFunctionClause(Vtl::AggrFunctionClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCalcClauseItem(Vtl::CalcClauseItemContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitSubspaceClauseItem(Vtl::SubspaceClauseItemContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitSimpleScalar(Vtl::SimpleScalarContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitScalarWithCast(Vtl::ScalarWithCastContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitJoinClause(Vtl::JoinClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitJoinClauseItem(Vtl::JoinClauseItemContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUsingClause(Vtl::UsingClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitNvlJoinClause(Vtl::NvlJoinClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitJoinBody(Vtl::JoinBodyContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitJoinApplyClause(Vtl::JoinApplyClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPartitionListed(Vtl::PartitionListedContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPartitionExceptAll(Vtl::PartitionExceptAllContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitOrderByClause(Vtl::OrderByClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitOrderByItem(Vtl::OrderByItemContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitWindowingClause(Vtl::WindowingClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitSignedInteger(Vtl::SignedIntegerContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitSignedNumber(Vtl::SignedNumberContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLimitClauseItem(Vtl::LimitClauseItemContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitGroupByOrExcept(Vtl::GroupByOrExceptContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitGroupAll(Vtl::GroupAllContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitHavingClause(Vtl::HavingClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitParameterItem(Vtl::ParameterItemContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitOutputParameterType(Vtl::OutputParameterTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitOutputParameterTypeComponent(Vtl::OutputParameterTypeComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitInputParameterType(Vtl::InputParameterTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRulesetType(Vtl::RulesetTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitScalarType(Vtl::ScalarTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitComponentType(Vtl::ComponentTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDatasetType(Vtl::DatasetTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitEvalDatasetType(Vtl::EvalDatasetTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitScalarSetType(Vtl::ScalarSetTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDataPoint(Vtl::DataPointContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDataPointVd(Vtl::DataPointVdContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDataPointVar(Vtl::DataPointVarContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitHrRulesetType(Vtl::HrRulesetTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitHrRulesetVdType(Vtl::HrRulesetVdTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitHrRulesetVarType(Vtl::HrRulesetVarTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitValueDomainName(Vtl::ValueDomainNameContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRulesetID(Vtl::RulesetIDContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRulesetSignature(Vtl::RulesetSignatureContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitSignature(Vtl::SignatureContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRuleClauseDatapoint(Vtl::RuleClauseDatapointContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRuleItemDatapoint(Vtl::RuleItemDatapointContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRuleClauseHierarchical(Vtl::RuleClauseHierarchicalContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRuleItemHierarchical(Vtl::RuleItemHierarchicalContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitHierRuleSignature(Vtl::HierRuleSignatureContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitValueDomainSignature(Vtl::ValueDomainSignatureContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCodeItemRelation(Vtl::CodeItemRelationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCodeItemRelationClause(Vtl::CodeItemRelationClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitValueDomainValue(Vtl::ValueDomainValueContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitConditionConstraint(Vtl::ConditionConstraintContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRangeConstraint(Vtl::RangeConstraintContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCompConstraint(Vtl::CompConstraintContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMultModifier(Vtl::MultModifierContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitValidationOutput(Vtl::ValidationOutputContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitValidationMode(Vtl::ValidationModeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitConditionClause(Vtl::ConditionClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitInputMode(Vtl::InputModeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitImbalanceExpr(Vtl::ImbalanceExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitInputModeHierarchy(Vtl::InputModeHierarchyContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitOutputModeHierarchy(Vtl::OutputModeHierarchyContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAlias(Vtl::AliasContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitVarID(Vtl::VarIDContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitSimpleComponentId(Vtl::SimpleComponentIdContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitComponentID(Vtl::ComponentIDContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLists(Vtl::ListsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitErCode(Vtl::ErCodeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitErLevel(Vtl::ErLevelContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitComparisonOperand(Vtl::ComparisonOperandContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitOptionalExpr(Vtl::OptionalExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitOptionalExprComponent(Vtl::OptionalExprComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitComponentRole(Vtl::ComponentRoleContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitViralAttribute(Vtl::ViralAttributeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitValueDomainID(Vtl::ValueDomainIDContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitOperatorID(Vtl::OperatorIDContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRoutineName(Vtl::RoutineNameContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitIntegerLiteral(Vtl::IntegerLiteralContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitNumberLiteral(Vtl::NumberLiteralContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBooleanLiteral(Vtl::BooleanLiteralContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitStringLiteral(Vtl::StringLiteralContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitNullLiteral(Vtl::NullLiteralContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBasicScalarType(Vtl::BasicScalarTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRetainType(Vtl::RetainTypeContext *ctx) override {
    return visitChildren(ctx);
  }


};

