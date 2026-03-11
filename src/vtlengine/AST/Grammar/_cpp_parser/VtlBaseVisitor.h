
// Generated from /home/javier/Programacion/vtlengine/src/vtlengine/AST/Grammar/Vtl.g4 by ANTLR 4.13.1

#pragma once


#include "antlr4-runtime.h"
#include "VtlVisitor.h"


/**
 * This class provides an empty implementation of VtlVisitor, which can be
 * extended to create a visitor which only needs to handle a subset of the available methods.
 */
class  VtlBaseVisitor : public VtlVisitor {
public:

  virtual std::any visitStart(VtlParser::StartContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTemporaryAssignment(VtlParser::TemporaryAssignmentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPersistAssignment(VtlParser::PersistAssignmentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDefineExpression(VtlParser::DefineExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitVarIdExpr(VtlParser::VarIdExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMembershipExpr(VtlParser::MembershipExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitInNotInExpr(VtlParser::InNotInExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBooleanExpr(VtlParser::BooleanExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitComparisonExpr(VtlParser::ComparisonExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUnaryExpr(VtlParser::UnaryExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFunctionsExpression(VtlParser::FunctionsExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitIfExpr(VtlParser::IfExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitClauseExpr(VtlParser::ClauseExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCaseExpr(VtlParser::CaseExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitArithmeticExpr(VtlParser::ArithmeticExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitParenthesisExpr(VtlParser::ParenthesisExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitConstantExpr(VtlParser::ConstantExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitArithmeticExprOrConcat(VtlParser::ArithmeticExprOrConcatContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitArithmeticExprComp(VtlParser::ArithmeticExprCompContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitIfExprComp(VtlParser::IfExprCompContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitComparisonExprComp(VtlParser::ComparisonExprCompContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFunctionsExpressionComp(VtlParser::FunctionsExpressionCompContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCompId(VtlParser::CompIdContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitConstantExprComp(VtlParser::ConstantExprCompContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitArithmeticExprOrConcatComp(VtlParser::ArithmeticExprOrConcatCompContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitParenthesisExprComp(VtlParser::ParenthesisExprCompContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitInNotInExprComp(VtlParser::InNotInExprCompContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUnaryExprComp(VtlParser::UnaryExprCompContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCaseExprComp(VtlParser::CaseExprCompContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBooleanExprComp(VtlParser::BooleanExprCompContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitGenericFunctionsComponents(VtlParser::GenericFunctionsComponentsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitStringFunctionsComponents(VtlParser::StringFunctionsComponentsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitNumericFunctionsComponents(VtlParser::NumericFunctionsComponentsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitComparisonFunctionsComponents(VtlParser::ComparisonFunctionsComponentsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTimeFunctionsComponents(VtlParser::TimeFunctionsComponentsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitConditionalFunctionsComponents(VtlParser::ConditionalFunctionsComponentsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAggregateFunctionsComponents(VtlParser::AggregateFunctionsComponentsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAnalyticFunctionsComponents(VtlParser::AnalyticFunctionsComponentsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitJoinFunctions(VtlParser::JoinFunctionsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitGenericFunctions(VtlParser::GenericFunctionsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitStringFunctions(VtlParser::StringFunctionsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitNumericFunctions(VtlParser::NumericFunctionsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitComparisonFunctions(VtlParser::ComparisonFunctionsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTimeFunctions(VtlParser::TimeFunctionsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitSetFunctions(VtlParser::SetFunctionsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitHierarchyFunctions(VtlParser::HierarchyFunctionsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitValidationFunctions(VtlParser::ValidationFunctionsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitConditionalFunctions(VtlParser::ConditionalFunctionsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAggregateFunctions(VtlParser::AggregateFunctionsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAnalyticFunctions(VtlParser::AnalyticFunctionsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDatasetClause(VtlParser::DatasetClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRenameClause(VtlParser::RenameClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAggrClause(VtlParser::AggrClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFilterClause(VtlParser::FilterClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCalcClause(VtlParser::CalcClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitKeepOrDropClause(VtlParser::KeepOrDropClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPivotOrUnpivotClause(VtlParser::PivotOrUnpivotClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCustomPivotClause(VtlParser::CustomPivotClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitSubspaceClause(VtlParser::SubspaceClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitJoinExpr(VtlParser::JoinExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDefOperator(VtlParser::DefOperatorContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDefDatapointRuleset(VtlParser::DefDatapointRulesetContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDefHierarchical(VtlParser::DefHierarchicalContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCallDataset(VtlParser::CallDatasetContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitEvalAtom(VtlParser::EvalAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCastExprDataset(VtlParser::CastExprDatasetContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCallComponent(VtlParser::CallComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCastExprComponent(VtlParser::CastExprComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitEvalAtomComponent(VtlParser::EvalAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitParameterComponent(VtlParser::ParameterComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitParameter(VtlParser::ParameterContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUnaryStringFunction(VtlParser::UnaryStringFunctionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitSubstrAtom(VtlParser::SubstrAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitReplaceAtom(VtlParser::ReplaceAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitInstrAtom(VtlParser::InstrAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUnaryStringFunctionComponent(VtlParser::UnaryStringFunctionComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitSubstrAtomComponent(VtlParser::SubstrAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitReplaceAtomComponent(VtlParser::ReplaceAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitInstrAtomComponent(VtlParser::InstrAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUnaryNumeric(VtlParser::UnaryNumericContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUnaryWithOptionalNumeric(VtlParser::UnaryWithOptionalNumericContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBinaryNumeric(VtlParser::BinaryNumericContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUnaryNumericComponent(VtlParser::UnaryNumericComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUnaryWithOptionalNumericComponent(VtlParser::UnaryWithOptionalNumericComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBinaryNumericComponent(VtlParser::BinaryNumericComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBetweenAtom(VtlParser::BetweenAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCharsetMatchAtom(VtlParser::CharsetMatchAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitIsNullAtom(VtlParser::IsNullAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitExistInAtom(VtlParser::ExistInAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBetweenAtomComponent(VtlParser::BetweenAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCharsetMatchAtomComponent(VtlParser::CharsetMatchAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitIsNullAtomComponent(VtlParser::IsNullAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPeriodAtom(VtlParser::PeriodAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFillTimeAtom(VtlParser::FillTimeAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFlowAtom(VtlParser::FlowAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTimeShiftAtom(VtlParser::TimeShiftAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTimeAggAtom(VtlParser::TimeAggAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCurrentDateAtom(VtlParser::CurrentDateAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDateDiffAtom(VtlParser::DateDiffAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDateAddAtom(VtlParser::DateAddAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitYearAtom(VtlParser::YearAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMonthAtom(VtlParser::MonthAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDayOfMonthAtom(VtlParser::DayOfMonthAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDayOfYearAtom(VtlParser::DayOfYearAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDayToYearAtom(VtlParser::DayToYearAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDayToMonthAtom(VtlParser::DayToMonthAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitYearTodayAtom(VtlParser::YearTodayAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMonthTodayAtom(VtlParser::MonthTodayAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPeriodAtomComponent(VtlParser::PeriodAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFillTimeAtomComponent(VtlParser::FillTimeAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitFlowAtomComponent(VtlParser::FlowAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTimeShiftAtomComponent(VtlParser::TimeShiftAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitTimeAggAtomComponent(VtlParser::TimeAggAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCurrentDateAtomComponent(VtlParser::CurrentDateAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDateDiffAtomComponent(VtlParser::DateDiffAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDateAddAtomComponent(VtlParser::DateAddAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitYearAtomComponent(VtlParser::YearAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMonthAtomComponent(VtlParser::MonthAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDayOfMonthAtomComponent(VtlParser::DayOfMonthAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDatOfYearAtomComponent(VtlParser::DatOfYearAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDayToYearAtomComponent(VtlParser::DayToYearAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDayToMonthAtomComponent(VtlParser::DayToMonthAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitYearTodayAtomComponent(VtlParser::YearTodayAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMonthTodayAtomComponent(VtlParser::MonthTodayAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitUnionAtom(VtlParser::UnionAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitIntersectAtom(VtlParser::IntersectAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitSetOrSYmDiffAtom(VtlParser::SetOrSYmDiffAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitHierarchyOperators(VtlParser::HierarchyOperatorsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitValidateDPruleset(VtlParser::ValidateDPrulesetContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitValidateHRruleset(VtlParser::ValidateHRrulesetContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitValidationSimple(VtlParser::ValidationSimpleContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitNvlAtom(VtlParser::NvlAtomContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitNvlAtomComponent(VtlParser::NvlAtomComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAggrComp(VtlParser::AggrCompContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCountAggrComp(VtlParser::CountAggrCompContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAggrDataset(VtlParser::AggrDatasetContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAnSimpleFunction(VtlParser::AnSimpleFunctionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLagOrLeadAn(VtlParser::LagOrLeadAnContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRatioToReportAn(VtlParser::RatioToReportAnContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAnSimpleFunctionComponent(VtlParser::AnSimpleFunctionComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLagOrLeadAnComponent(VtlParser::LagOrLeadAnComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRankAnComponent(VtlParser::RankAnComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRatioToReportAnComponent(VtlParser::RatioToReportAnComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRenameClauseItem(VtlParser::RenameClauseItemContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAggregateClause(VtlParser::AggregateClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAggrFunctionClause(VtlParser::AggrFunctionClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCalcClauseItem(VtlParser::CalcClauseItemContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitSubspaceClauseItem(VtlParser::SubspaceClauseItemContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitSimpleScalar(VtlParser::SimpleScalarContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitScalarWithCast(VtlParser::ScalarWithCastContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitJoinClauseWithoutUsing(VtlParser::JoinClauseWithoutUsingContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitJoinClause(VtlParser::JoinClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitJoinClauseItem(VtlParser::JoinClauseItemContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitJoinBody(VtlParser::JoinBodyContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitJoinApplyClause(VtlParser::JoinApplyClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitPartitionByClause(VtlParser::PartitionByClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitOrderByClause(VtlParser::OrderByClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitOrderByItem(VtlParser::OrderByItemContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitWindowingClause(VtlParser::WindowingClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitSignedInteger(VtlParser::SignedIntegerContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitSignedNumber(VtlParser::SignedNumberContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLimitClauseItem(VtlParser::LimitClauseItemContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitGroupByOrExcept(VtlParser::GroupByOrExceptContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitGroupAll(VtlParser::GroupAllContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitHavingClause(VtlParser::HavingClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitParameterItem(VtlParser::ParameterItemContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitOutputParameterType(VtlParser::OutputParameterTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitOutputParameterTypeComponent(VtlParser::OutputParameterTypeComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitInputParameterType(VtlParser::InputParameterTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRulesetType(VtlParser::RulesetTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitScalarType(VtlParser::ScalarTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitComponentType(VtlParser::ComponentTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDatasetType(VtlParser::DatasetTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitEvalDatasetType(VtlParser::EvalDatasetTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitScalarSetType(VtlParser::ScalarSetTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDataPoint(VtlParser::DataPointContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDataPointVd(VtlParser::DataPointVdContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitDataPointVar(VtlParser::DataPointVarContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitHrRulesetType(VtlParser::HrRulesetTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitHrRulesetVdType(VtlParser::HrRulesetVdTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitHrRulesetVarType(VtlParser::HrRulesetVarTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitValueDomainName(VtlParser::ValueDomainNameContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRulesetID(VtlParser::RulesetIDContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRulesetSignature(VtlParser::RulesetSignatureContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitSignature(VtlParser::SignatureContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRuleClauseDatapoint(VtlParser::RuleClauseDatapointContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRuleItemDatapoint(VtlParser::RuleItemDatapointContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRuleClauseHierarchical(VtlParser::RuleClauseHierarchicalContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRuleItemHierarchical(VtlParser::RuleItemHierarchicalContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitHierRuleSignature(VtlParser::HierRuleSignatureContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitValueDomainSignature(VtlParser::ValueDomainSignatureContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCodeItemRelation(VtlParser::CodeItemRelationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCodeItemRelationClause(VtlParser::CodeItemRelationClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitValueDomainValue(VtlParser::ValueDomainValueContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitConditionConstraint(VtlParser::ConditionConstraintContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRangeConstraint(VtlParser::RangeConstraintContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitCompConstraint(VtlParser::CompConstraintContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitMultModifier(VtlParser::MultModifierContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitValidationOutput(VtlParser::ValidationOutputContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitValidationMode(VtlParser::ValidationModeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitConditionClause(VtlParser::ConditionClauseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitInputMode(VtlParser::InputModeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitImbalanceExpr(VtlParser::ImbalanceExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitInputModeHierarchy(VtlParser::InputModeHierarchyContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitOutputModeHierarchy(VtlParser::OutputModeHierarchyContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitAlias(VtlParser::AliasContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitVarID(VtlParser::VarIDContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitSimpleComponentId(VtlParser::SimpleComponentIdContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitComponentID(VtlParser::ComponentIDContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitLists(VtlParser::ListsContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitErCode(VtlParser::ErCodeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitErLevel(VtlParser::ErLevelContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitComparisonOperand(VtlParser::ComparisonOperandContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitOptionalExpr(VtlParser::OptionalExprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitOptionalExprComponent(VtlParser::OptionalExprComponentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitComponentRole(VtlParser::ComponentRoleContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitViralAttribute(VtlParser::ViralAttributeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitValueDomainID(VtlParser::ValueDomainIDContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitOperatorID(VtlParser::OperatorIDContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRoutineName(VtlParser::RoutineNameContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitConstant(VtlParser::ConstantContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitBasicScalarType(VtlParser::BasicScalarTypeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual std::any visitRetainType(VtlParser::RetainTypeContext *ctx) override {
    return visitChildren(ctx);
  }


};

