
// Generated from Vtl.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"
#include "Vtl.h"


/**
 * This interface defines an abstract listener for a parse tree produced by Vtl.
 */
class  VtlListener : public antlr4::tree::ParseTreeListener {
public:

  virtual void enterStart(Vtl::StartContext *ctx) = 0;
  virtual void exitStart(Vtl::StartContext *ctx) = 0;

  virtual void enterTemporaryAssignment(Vtl::TemporaryAssignmentContext *ctx) = 0;
  virtual void exitTemporaryAssignment(Vtl::TemporaryAssignmentContext *ctx) = 0;

  virtual void enterPersistAssignment(Vtl::PersistAssignmentContext *ctx) = 0;
  virtual void exitPersistAssignment(Vtl::PersistAssignmentContext *ctx) = 0;

  virtual void enterDefineExpression(Vtl::DefineExpressionContext *ctx) = 0;
  virtual void exitDefineExpression(Vtl::DefineExpressionContext *ctx) = 0;

  virtual void enterVarIdExpr(Vtl::VarIdExprContext *ctx) = 0;
  virtual void exitVarIdExpr(Vtl::VarIdExprContext *ctx) = 0;

  virtual void enterMembershipExpr(Vtl::MembershipExprContext *ctx) = 0;
  virtual void exitMembershipExpr(Vtl::MembershipExprContext *ctx) = 0;

  virtual void enterInNotInExpr(Vtl::InNotInExprContext *ctx) = 0;
  virtual void exitInNotInExpr(Vtl::InNotInExprContext *ctx) = 0;

  virtual void enterBooleanExpr(Vtl::BooleanExprContext *ctx) = 0;
  virtual void exitBooleanExpr(Vtl::BooleanExprContext *ctx) = 0;

  virtual void enterComparisonExpr(Vtl::ComparisonExprContext *ctx) = 0;
  virtual void exitComparisonExpr(Vtl::ComparisonExprContext *ctx) = 0;

  virtual void enterUnaryExpr(Vtl::UnaryExprContext *ctx) = 0;
  virtual void exitUnaryExpr(Vtl::UnaryExprContext *ctx) = 0;

  virtual void enterFunctionsExpression(Vtl::FunctionsExpressionContext *ctx) = 0;
  virtual void exitFunctionsExpression(Vtl::FunctionsExpressionContext *ctx) = 0;

  virtual void enterIfExpr(Vtl::IfExprContext *ctx) = 0;
  virtual void exitIfExpr(Vtl::IfExprContext *ctx) = 0;

  virtual void enterClauseExpr(Vtl::ClauseExprContext *ctx) = 0;
  virtual void exitClauseExpr(Vtl::ClauseExprContext *ctx) = 0;

  virtual void enterCaseExpr(Vtl::CaseExprContext *ctx) = 0;
  virtual void exitCaseExpr(Vtl::CaseExprContext *ctx) = 0;

  virtual void enterArithmeticExpr(Vtl::ArithmeticExprContext *ctx) = 0;
  virtual void exitArithmeticExpr(Vtl::ArithmeticExprContext *ctx) = 0;

  virtual void enterParenthesisExpr(Vtl::ParenthesisExprContext *ctx) = 0;
  virtual void exitParenthesisExpr(Vtl::ParenthesisExprContext *ctx) = 0;

  virtual void enterConstantExpr(Vtl::ConstantExprContext *ctx) = 0;
  virtual void exitConstantExpr(Vtl::ConstantExprContext *ctx) = 0;

  virtual void enterArithmeticExprOrConcat(Vtl::ArithmeticExprOrConcatContext *ctx) = 0;
  virtual void exitArithmeticExprOrConcat(Vtl::ArithmeticExprOrConcatContext *ctx) = 0;

  virtual void enterArithmeticExprComp(Vtl::ArithmeticExprCompContext *ctx) = 0;
  virtual void exitArithmeticExprComp(Vtl::ArithmeticExprCompContext *ctx) = 0;

  virtual void enterIfExprComp(Vtl::IfExprCompContext *ctx) = 0;
  virtual void exitIfExprComp(Vtl::IfExprCompContext *ctx) = 0;

  virtual void enterComparisonExprComp(Vtl::ComparisonExprCompContext *ctx) = 0;
  virtual void exitComparisonExprComp(Vtl::ComparisonExprCompContext *ctx) = 0;

  virtual void enterFunctionsExpressionComp(Vtl::FunctionsExpressionCompContext *ctx) = 0;
  virtual void exitFunctionsExpressionComp(Vtl::FunctionsExpressionCompContext *ctx) = 0;

  virtual void enterCompId(Vtl::CompIdContext *ctx) = 0;
  virtual void exitCompId(Vtl::CompIdContext *ctx) = 0;

  virtual void enterConstantExprComp(Vtl::ConstantExprCompContext *ctx) = 0;
  virtual void exitConstantExprComp(Vtl::ConstantExprCompContext *ctx) = 0;

  virtual void enterArithmeticExprOrConcatComp(Vtl::ArithmeticExprOrConcatCompContext *ctx) = 0;
  virtual void exitArithmeticExprOrConcatComp(Vtl::ArithmeticExprOrConcatCompContext *ctx) = 0;

  virtual void enterParenthesisExprComp(Vtl::ParenthesisExprCompContext *ctx) = 0;
  virtual void exitParenthesisExprComp(Vtl::ParenthesisExprCompContext *ctx) = 0;

  virtual void enterInNotInExprComp(Vtl::InNotInExprCompContext *ctx) = 0;
  virtual void exitInNotInExprComp(Vtl::InNotInExprCompContext *ctx) = 0;

  virtual void enterUnaryExprComp(Vtl::UnaryExprCompContext *ctx) = 0;
  virtual void exitUnaryExprComp(Vtl::UnaryExprCompContext *ctx) = 0;

  virtual void enterCaseExprComp(Vtl::CaseExprCompContext *ctx) = 0;
  virtual void exitCaseExprComp(Vtl::CaseExprCompContext *ctx) = 0;

  virtual void enterBooleanExprComp(Vtl::BooleanExprCompContext *ctx) = 0;
  virtual void exitBooleanExprComp(Vtl::BooleanExprCompContext *ctx) = 0;

  virtual void enterGenericFunctionsComponents(Vtl::GenericFunctionsComponentsContext *ctx) = 0;
  virtual void exitGenericFunctionsComponents(Vtl::GenericFunctionsComponentsContext *ctx) = 0;

  virtual void enterStringFunctionsComponents(Vtl::StringFunctionsComponentsContext *ctx) = 0;
  virtual void exitStringFunctionsComponents(Vtl::StringFunctionsComponentsContext *ctx) = 0;

  virtual void enterNumericFunctionsComponents(Vtl::NumericFunctionsComponentsContext *ctx) = 0;
  virtual void exitNumericFunctionsComponents(Vtl::NumericFunctionsComponentsContext *ctx) = 0;

  virtual void enterComparisonFunctionsComponents(Vtl::ComparisonFunctionsComponentsContext *ctx) = 0;
  virtual void exitComparisonFunctionsComponents(Vtl::ComparisonFunctionsComponentsContext *ctx) = 0;

  virtual void enterTimeFunctionsComponents(Vtl::TimeFunctionsComponentsContext *ctx) = 0;
  virtual void exitTimeFunctionsComponents(Vtl::TimeFunctionsComponentsContext *ctx) = 0;

  virtual void enterConditionalFunctionsComponents(Vtl::ConditionalFunctionsComponentsContext *ctx) = 0;
  virtual void exitConditionalFunctionsComponents(Vtl::ConditionalFunctionsComponentsContext *ctx) = 0;

  virtual void enterAggregateFunctionsComponents(Vtl::AggregateFunctionsComponentsContext *ctx) = 0;
  virtual void exitAggregateFunctionsComponents(Vtl::AggregateFunctionsComponentsContext *ctx) = 0;

  virtual void enterAnalyticFunctionsComponents(Vtl::AnalyticFunctionsComponentsContext *ctx) = 0;
  virtual void exitAnalyticFunctionsComponents(Vtl::AnalyticFunctionsComponentsContext *ctx) = 0;

  virtual void enterJoinFunctions(Vtl::JoinFunctionsContext *ctx) = 0;
  virtual void exitJoinFunctions(Vtl::JoinFunctionsContext *ctx) = 0;

  virtual void enterGenericFunctions(Vtl::GenericFunctionsContext *ctx) = 0;
  virtual void exitGenericFunctions(Vtl::GenericFunctionsContext *ctx) = 0;

  virtual void enterStringFunctions(Vtl::StringFunctionsContext *ctx) = 0;
  virtual void exitStringFunctions(Vtl::StringFunctionsContext *ctx) = 0;

  virtual void enterNumericFunctions(Vtl::NumericFunctionsContext *ctx) = 0;
  virtual void exitNumericFunctions(Vtl::NumericFunctionsContext *ctx) = 0;

  virtual void enterComparisonFunctions(Vtl::ComparisonFunctionsContext *ctx) = 0;
  virtual void exitComparisonFunctions(Vtl::ComparisonFunctionsContext *ctx) = 0;

  virtual void enterTimeFunctions(Vtl::TimeFunctionsContext *ctx) = 0;
  virtual void exitTimeFunctions(Vtl::TimeFunctionsContext *ctx) = 0;

  virtual void enterSetFunctions(Vtl::SetFunctionsContext *ctx) = 0;
  virtual void exitSetFunctions(Vtl::SetFunctionsContext *ctx) = 0;

  virtual void enterHierarchyFunctions(Vtl::HierarchyFunctionsContext *ctx) = 0;
  virtual void exitHierarchyFunctions(Vtl::HierarchyFunctionsContext *ctx) = 0;

  virtual void enterValidationFunctions(Vtl::ValidationFunctionsContext *ctx) = 0;
  virtual void exitValidationFunctions(Vtl::ValidationFunctionsContext *ctx) = 0;

  virtual void enterConditionalFunctions(Vtl::ConditionalFunctionsContext *ctx) = 0;
  virtual void exitConditionalFunctions(Vtl::ConditionalFunctionsContext *ctx) = 0;

  virtual void enterAggregateFunctions(Vtl::AggregateFunctionsContext *ctx) = 0;
  virtual void exitAggregateFunctions(Vtl::AggregateFunctionsContext *ctx) = 0;

  virtual void enterAnalyticFunctions(Vtl::AnalyticFunctionsContext *ctx) = 0;
  virtual void exitAnalyticFunctions(Vtl::AnalyticFunctionsContext *ctx) = 0;

  virtual void enterDatasetClause(Vtl::DatasetClauseContext *ctx) = 0;
  virtual void exitDatasetClause(Vtl::DatasetClauseContext *ctx) = 0;

  virtual void enterRenameClause(Vtl::RenameClauseContext *ctx) = 0;
  virtual void exitRenameClause(Vtl::RenameClauseContext *ctx) = 0;

  virtual void enterAggrClause(Vtl::AggrClauseContext *ctx) = 0;
  virtual void exitAggrClause(Vtl::AggrClauseContext *ctx) = 0;

  virtual void enterFilterClause(Vtl::FilterClauseContext *ctx) = 0;
  virtual void exitFilterClause(Vtl::FilterClauseContext *ctx) = 0;

  virtual void enterCalcClause(Vtl::CalcClauseContext *ctx) = 0;
  virtual void exitCalcClause(Vtl::CalcClauseContext *ctx) = 0;

  virtual void enterKeepOrDropClause(Vtl::KeepOrDropClauseContext *ctx) = 0;
  virtual void exitKeepOrDropClause(Vtl::KeepOrDropClauseContext *ctx) = 0;

  virtual void enterPivotOrUnpivotClause(Vtl::PivotOrUnpivotClauseContext *ctx) = 0;
  virtual void exitPivotOrUnpivotClause(Vtl::PivotOrUnpivotClauseContext *ctx) = 0;

  virtual void enterCustomPivotClause(Vtl::CustomPivotClauseContext *ctx) = 0;
  virtual void exitCustomPivotClause(Vtl::CustomPivotClauseContext *ctx) = 0;

  virtual void enterSubspaceClause(Vtl::SubspaceClauseContext *ctx) = 0;
  virtual void exitSubspaceClause(Vtl::SubspaceClauseContext *ctx) = 0;

  virtual void enterInnerJoinExpr(Vtl::InnerJoinExprContext *ctx) = 0;
  virtual void exitInnerJoinExpr(Vtl::InnerJoinExprContext *ctx) = 0;

  virtual void enterLeftJoinExpr(Vtl::LeftJoinExprContext *ctx) = 0;
  virtual void exitLeftJoinExpr(Vtl::LeftJoinExprContext *ctx) = 0;

  virtual void enterFullJoinExpr(Vtl::FullJoinExprContext *ctx) = 0;
  virtual void exitFullJoinExpr(Vtl::FullJoinExprContext *ctx) = 0;

  virtual void enterCrossJoinExpr(Vtl::CrossJoinExprContext *ctx) = 0;
  virtual void exitCrossJoinExpr(Vtl::CrossJoinExprContext *ctx) = 0;

  virtual void enterDefOperator(Vtl::DefOperatorContext *ctx) = 0;
  virtual void exitDefOperator(Vtl::DefOperatorContext *ctx) = 0;

  virtual void enterDefDatapointRuleset(Vtl::DefDatapointRulesetContext *ctx) = 0;
  virtual void exitDefDatapointRuleset(Vtl::DefDatapointRulesetContext *ctx) = 0;

  virtual void enterDefHierarchical(Vtl::DefHierarchicalContext *ctx) = 0;
  virtual void exitDefHierarchical(Vtl::DefHierarchicalContext *ctx) = 0;

  virtual void enterDefViralPropagation(Vtl::DefViralPropagationContext *ctx) = 0;
  virtual void exitDefViralPropagation(Vtl::DefViralPropagationContext *ctx) = 0;

  virtual void enterVpSignature(Vtl::VpSignatureContext *ctx) = 0;
  virtual void exitVpSignature(Vtl::VpSignatureContext *ctx) = 0;

  virtual void enterVpBody(Vtl::VpBodyContext *ctx) = 0;
  virtual void exitVpBody(Vtl::VpBodyContext *ctx) = 0;

  virtual void enterEnumeratedVpClause(Vtl::EnumeratedVpClauseContext *ctx) = 0;
  virtual void exitEnumeratedVpClause(Vtl::EnumeratedVpClauseContext *ctx) = 0;

  virtual void enterAggregationVpClause(Vtl::AggregationVpClauseContext *ctx) = 0;
  virtual void exitAggregationVpClause(Vtl::AggregationVpClauseContext *ctx) = 0;

  virtual void enterDefaultVpClause(Vtl::DefaultVpClauseContext *ctx) = 0;
  virtual void exitDefaultVpClause(Vtl::DefaultVpClauseContext *ctx) = 0;

  virtual void enterVpCondition(Vtl::VpConditionContext *ctx) = 0;
  virtual void exitVpCondition(Vtl::VpConditionContext *ctx) = 0;

  virtual void enterCallDataset(Vtl::CallDatasetContext *ctx) = 0;
  virtual void exitCallDataset(Vtl::CallDatasetContext *ctx) = 0;

  virtual void enterEvalAtom(Vtl::EvalAtomContext *ctx) = 0;
  virtual void exitEvalAtom(Vtl::EvalAtomContext *ctx) = 0;

  virtual void enterCastExprDataset(Vtl::CastExprDatasetContext *ctx) = 0;
  virtual void exitCastExprDataset(Vtl::CastExprDatasetContext *ctx) = 0;

  virtual void enterCallComponent(Vtl::CallComponentContext *ctx) = 0;
  virtual void exitCallComponent(Vtl::CallComponentContext *ctx) = 0;

  virtual void enterCastExprComponent(Vtl::CastExprComponentContext *ctx) = 0;
  virtual void exitCastExprComponent(Vtl::CastExprComponentContext *ctx) = 0;

  virtual void enterEvalAtomComponent(Vtl::EvalAtomComponentContext *ctx) = 0;
  virtual void exitEvalAtomComponent(Vtl::EvalAtomComponentContext *ctx) = 0;

  virtual void enterParameterComponent(Vtl::ParameterComponentContext *ctx) = 0;
  virtual void exitParameterComponent(Vtl::ParameterComponentContext *ctx) = 0;

  virtual void enterParameter(Vtl::ParameterContext *ctx) = 0;
  virtual void exitParameter(Vtl::ParameterContext *ctx) = 0;

  virtual void enterStringDistanceMethods(Vtl::StringDistanceMethodsContext *ctx) = 0;
  virtual void exitStringDistanceMethods(Vtl::StringDistanceMethodsContext *ctx) = 0;

  virtual void enterUnaryStringFunction(Vtl::UnaryStringFunctionContext *ctx) = 0;
  virtual void exitUnaryStringFunction(Vtl::UnaryStringFunctionContext *ctx) = 0;

  virtual void enterSubstrAtom(Vtl::SubstrAtomContext *ctx) = 0;
  virtual void exitSubstrAtom(Vtl::SubstrAtomContext *ctx) = 0;

  virtual void enterReplaceAtom(Vtl::ReplaceAtomContext *ctx) = 0;
  virtual void exitReplaceAtom(Vtl::ReplaceAtomContext *ctx) = 0;

  virtual void enterInstrAtom(Vtl::InstrAtomContext *ctx) = 0;
  virtual void exitInstrAtom(Vtl::InstrAtomContext *ctx) = 0;

  virtual void enterStringDistanceAtom(Vtl::StringDistanceAtomContext *ctx) = 0;
  virtual void exitStringDistanceAtom(Vtl::StringDistanceAtomContext *ctx) = 0;

  virtual void enterUnaryStringFunctionComponent(Vtl::UnaryStringFunctionComponentContext *ctx) = 0;
  virtual void exitUnaryStringFunctionComponent(Vtl::UnaryStringFunctionComponentContext *ctx) = 0;

  virtual void enterSubstrAtomComponent(Vtl::SubstrAtomComponentContext *ctx) = 0;
  virtual void exitSubstrAtomComponent(Vtl::SubstrAtomComponentContext *ctx) = 0;

  virtual void enterReplaceAtomComponent(Vtl::ReplaceAtomComponentContext *ctx) = 0;
  virtual void exitReplaceAtomComponent(Vtl::ReplaceAtomComponentContext *ctx) = 0;

  virtual void enterInstrAtomComponent(Vtl::InstrAtomComponentContext *ctx) = 0;
  virtual void exitInstrAtomComponent(Vtl::InstrAtomComponentContext *ctx) = 0;

  virtual void enterStringDistanceAtomComponent(Vtl::StringDistanceAtomComponentContext *ctx) = 0;
  virtual void exitStringDistanceAtomComponent(Vtl::StringDistanceAtomComponentContext *ctx) = 0;

  virtual void enterUnaryNumeric(Vtl::UnaryNumericContext *ctx) = 0;
  virtual void exitUnaryNumeric(Vtl::UnaryNumericContext *ctx) = 0;

  virtual void enterUnaryWithOptionalNumeric(Vtl::UnaryWithOptionalNumericContext *ctx) = 0;
  virtual void exitUnaryWithOptionalNumeric(Vtl::UnaryWithOptionalNumericContext *ctx) = 0;

  virtual void enterBinaryNumeric(Vtl::BinaryNumericContext *ctx) = 0;
  virtual void exitBinaryNumeric(Vtl::BinaryNumericContext *ctx) = 0;

  virtual void enterUnaryNumericComponent(Vtl::UnaryNumericComponentContext *ctx) = 0;
  virtual void exitUnaryNumericComponent(Vtl::UnaryNumericComponentContext *ctx) = 0;

  virtual void enterUnaryWithOptionalNumericComponent(Vtl::UnaryWithOptionalNumericComponentContext *ctx) = 0;
  virtual void exitUnaryWithOptionalNumericComponent(Vtl::UnaryWithOptionalNumericComponentContext *ctx) = 0;

  virtual void enterBinaryNumericComponent(Vtl::BinaryNumericComponentContext *ctx) = 0;
  virtual void exitBinaryNumericComponent(Vtl::BinaryNumericComponentContext *ctx) = 0;

  virtual void enterBetweenAtom(Vtl::BetweenAtomContext *ctx) = 0;
  virtual void exitBetweenAtom(Vtl::BetweenAtomContext *ctx) = 0;

  virtual void enterCharsetMatchAtom(Vtl::CharsetMatchAtomContext *ctx) = 0;
  virtual void exitCharsetMatchAtom(Vtl::CharsetMatchAtomContext *ctx) = 0;

  virtual void enterIsNullAtom(Vtl::IsNullAtomContext *ctx) = 0;
  virtual void exitIsNullAtom(Vtl::IsNullAtomContext *ctx) = 0;

  virtual void enterExistInAtom(Vtl::ExistInAtomContext *ctx) = 0;
  virtual void exitExistInAtom(Vtl::ExistInAtomContext *ctx) = 0;

  virtual void enterBetweenAtomComponent(Vtl::BetweenAtomComponentContext *ctx) = 0;
  virtual void exitBetweenAtomComponent(Vtl::BetweenAtomComponentContext *ctx) = 0;

  virtual void enterCharsetMatchAtomComponent(Vtl::CharsetMatchAtomComponentContext *ctx) = 0;
  virtual void exitCharsetMatchAtomComponent(Vtl::CharsetMatchAtomComponentContext *ctx) = 0;

  virtual void enterIsNullAtomComponent(Vtl::IsNullAtomComponentContext *ctx) = 0;
  virtual void exitIsNullAtomComponent(Vtl::IsNullAtomComponentContext *ctx) = 0;

  virtual void enterPeriodAtom(Vtl::PeriodAtomContext *ctx) = 0;
  virtual void exitPeriodAtom(Vtl::PeriodAtomContext *ctx) = 0;

  virtual void enterFillTimeAtom(Vtl::FillTimeAtomContext *ctx) = 0;
  virtual void exitFillTimeAtom(Vtl::FillTimeAtomContext *ctx) = 0;

  virtual void enterFlowAtom(Vtl::FlowAtomContext *ctx) = 0;
  virtual void exitFlowAtom(Vtl::FlowAtomContext *ctx) = 0;

  virtual void enterTimeShiftAtom(Vtl::TimeShiftAtomContext *ctx) = 0;
  virtual void exitTimeShiftAtom(Vtl::TimeShiftAtomContext *ctx) = 0;

  virtual void enterTimeAggAtom(Vtl::TimeAggAtomContext *ctx) = 0;
  virtual void exitTimeAggAtom(Vtl::TimeAggAtomContext *ctx) = 0;

  virtual void enterCurrentDateAtom(Vtl::CurrentDateAtomContext *ctx) = 0;
  virtual void exitCurrentDateAtom(Vtl::CurrentDateAtomContext *ctx) = 0;

  virtual void enterDateDiffAtom(Vtl::DateDiffAtomContext *ctx) = 0;
  virtual void exitDateDiffAtom(Vtl::DateDiffAtomContext *ctx) = 0;

  virtual void enterDateAddAtom(Vtl::DateAddAtomContext *ctx) = 0;
  virtual void exitDateAddAtom(Vtl::DateAddAtomContext *ctx) = 0;

  virtual void enterYearAtom(Vtl::YearAtomContext *ctx) = 0;
  virtual void exitYearAtom(Vtl::YearAtomContext *ctx) = 0;

  virtual void enterMonthAtom(Vtl::MonthAtomContext *ctx) = 0;
  virtual void exitMonthAtom(Vtl::MonthAtomContext *ctx) = 0;

  virtual void enterDayOfMonthAtom(Vtl::DayOfMonthAtomContext *ctx) = 0;
  virtual void exitDayOfMonthAtom(Vtl::DayOfMonthAtomContext *ctx) = 0;

  virtual void enterDayOfYearAtom(Vtl::DayOfYearAtomContext *ctx) = 0;
  virtual void exitDayOfYearAtom(Vtl::DayOfYearAtomContext *ctx) = 0;

  virtual void enterDayToYearAtom(Vtl::DayToYearAtomContext *ctx) = 0;
  virtual void exitDayToYearAtom(Vtl::DayToYearAtomContext *ctx) = 0;

  virtual void enterDayToMonthAtom(Vtl::DayToMonthAtomContext *ctx) = 0;
  virtual void exitDayToMonthAtom(Vtl::DayToMonthAtomContext *ctx) = 0;

  virtual void enterYearTodayAtom(Vtl::YearTodayAtomContext *ctx) = 0;
  virtual void exitYearTodayAtom(Vtl::YearTodayAtomContext *ctx) = 0;

  virtual void enterMonthTodayAtom(Vtl::MonthTodayAtomContext *ctx) = 0;
  virtual void exitMonthTodayAtom(Vtl::MonthTodayAtomContext *ctx) = 0;

  virtual void enterPeriodAtomComponent(Vtl::PeriodAtomComponentContext *ctx) = 0;
  virtual void exitPeriodAtomComponent(Vtl::PeriodAtomComponentContext *ctx) = 0;

  virtual void enterFillTimeAtomComponent(Vtl::FillTimeAtomComponentContext *ctx) = 0;
  virtual void exitFillTimeAtomComponent(Vtl::FillTimeAtomComponentContext *ctx) = 0;

  virtual void enterFlowAtomComponent(Vtl::FlowAtomComponentContext *ctx) = 0;
  virtual void exitFlowAtomComponent(Vtl::FlowAtomComponentContext *ctx) = 0;

  virtual void enterTimeShiftAtomComponent(Vtl::TimeShiftAtomComponentContext *ctx) = 0;
  virtual void exitTimeShiftAtomComponent(Vtl::TimeShiftAtomComponentContext *ctx) = 0;

  virtual void enterTimeAggAtomComponent(Vtl::TimeAggAtomComponentContext *ctx) = 0;
  virtual void exitTimeAggAtomComponent(Vtl::TimeAggAtomComponentContext *ctx) = 0;

  virtual void enterCurrentDateAtomComponent(Vtl::CurrentDateAtomComponentContext *ctx) = 0;
  virtual void exitCurrentDateAtomComponent(Vtl::CurrentDateAtomComponentContext *ctx) = 0;

  virtual void enterDateDiffAtomComponent(Vtl::DateDiffAtomComponentContext *ctx) = 0;
  virtual void exitDateDiffAtomComponent(Vtl::DateDiffAtomComponentContext *ctx) = 0;

  virtual void enterDateAddAtomComponent(Vtl::DateAddAtomComponentContext *ctx) = 0;
  virtual void exitDateAddAtomComponent(Vtl::DateAddAtomComponentContext *ctx) = 0;

  virtual void enterYearAtomComponent(Vtl::YearAtomComponentContext *ctx) = 0;
  virtual void exitYearAtomComponent(Vtl::YearAtomComponentContext *ctx) = 0;

  virtual void enterMonthAtomComponent(Vtl::MonthAtomComponentContext *ctx) = 0;
  virtual void exitMonthAtomComponent(Vtl::MonthAtomComponentContext *ctx) = 0;

  virtual void enterDayOfMonthAtomComponent(Vtl::DayOfMonthAtomComponentContext *ctx) = 0;
  virtual void exitDayOfMonthAtomComponent(Vtl::DayOfMonthAtomComponentContext *ctx) = 0;

  virtual void enterDayOfYearAtomComponent(Vtl::DayOfYearAtomComponentContext *ctx) = 0;
  virtual void exitDayOfYearAtomComponent(Vtl::DayOfYearAtomComponentContext *ctx) = 0;

  virtual void enterDayToYearAtomComponent(Vtl::DayToYearAtomComponentContext *ctx) = 0;
  virtual void exitDayToYearAtomComponent(Vtl::DayToYearAtomComponentContext *ctx) = 0;

  virtual void enterDayToMonthAtomComponent(Vtl::DayToMonthAtomComponentContext *ctx) = 0;
  virtual void exitDayToMonthAtomComponent(Vtl::DayToMonthAtomComponentContext *ctx) = 0;

  virtual void enterYearTodayAtomComponent(Vtl::YearTodayAtomComponentContext *ctx) = 0;
  virtual void exitYearTodayAtomComponent(Vtl::YearTodayAtomComponentContext *ctx) = 0;

  virtual void enterMonthTodayAtomComponent(Vtl::MonthTodayAtomComponentContext *ctx) = 0;
  virtual void exitMonthTodayAtomComponent(Vtl::MonthTodayAtomComponentContext *ctx) = 0;

  virtual void enterUnionAtom(Vtl::UnionAtomContext *ctx) = 0;
  virtual void exitUnionAtom(Vtl::UnionAtomContext *ctx) = 0;

  virtual void enterIntersectAtom(Vtl::IntersectAtomContext *ctx) = 0;
  virtual void exitIntersectAtom(Vtl::IntersectAtomContext *ctx) = 0;

  virtual void enterSetOrSYmDiffAtom(Vtl::SetOrSYmDiffAtomContext *ctx) = 0;
  virtual void exitSetOrSYmDiffAtom(Vtl::SetOrSYmDiffAtomContext *ctx) = 0;

  virtual void enterHierarchyOperators(Vtl::HierarchyOperatorsContext *ctx) = 0;
  virtual void exitHierarchyOperators(Vtl::HierarchyOperatorsContext *ctx) = 0;

  virtual void enterValidateDPruleset(Vtl::ValidateDPrulesetContext *ctx) = 0;
  virtual void exitValidateDPruleset(Vtl::ValidateDPrulesetContext *ctx) = 0;

  virtual void enterValidateHRruleset(Vtl::ValidateHRrulesetContext *ctx) = 0;
  virtual void exitValidateHRruleset(Vtl::ValidateHRrulesetContext *ctx) = 0;

  virtual void enterValidationSimple(Vtl::ValidationSimpleContext *ctx) = 0;
  virtual void exitValidationSimple(Vtl::ValidationSimpleContext *ctx) = 0;

  virtual void enterNvlAtom(Vtl::NvlAtomContext *ctx) = 0;
  virtual void exitNvlAtom(Vtl::NvlAtomContext *ctx) = 0;

  virtual void enterNvlAtomComponent(Vtl::NvlAtomComponentContext *ctx) = 0;
  virtual void exitNvlAtomComponent(Vtl::NvlAtomComponentContext *ctx) = 0;

  virtual void enterAggrComp(Vtl::AggrCompContext *ctx) = 0;
  virtual void exitAggrComp(Vtl::AggrCompContext *ctx) = 0;

  virtual void enterCountAggrComp(Vtl::CountAggrCompContext *ctx) = 0;
  virtual void exitCountAggrComp(Vtl::CountAggrCompContext *ctx) = 0;

  virtual void enterAggrDataset(Vtl::AggrDatasetContext *ctx) = 0;
  virtual void exitAggrDataset(Vtl::AggrDatasetContext *ctx) = 0;

  virtual void enterAnSimpleFunction(Vtl::AnSimpleFunctionContext *ctx) = 0;
  virtual void exitAnSimpleFunction(Vtl::AnSimpleFunctionContext *ctx) = 0;

  virtual void enterLagOrLeadAn(Vtl::LagOrLeadAnContext *ctx) = 0;
  virtual void exitLagOrLeadAn(Vtl::LagOrLeadAnContext *ctx) = 0;

  virtual void enterRatioToReportAn(Vtl::RatioToReportAnContext *ctx) = 0;
  virtual void exitRatioToReportAn(Vtl::RatioToReportAnContext *ctx) = 0;

  virtual void enterAnSimpleFunctionComponent(Vtl::AnSimpleFunctionComponentContext *ctx) = 0;
  virtual void exitAnSimpleFunctionComponent(Vtl::AnSimpleFunctionComponentContext *ctx) = 0;

  virtual void enterLagOrLeadAnComponent(Vtl::LagOrLeadAnComponentContext *ctx) = 0;
  virtual void exitLagOrLeadAnComponent(Vtl::LagOrLeadAnComponentContext *ctx) = 0;

  virtual void enterRankAnComponent(Vtl::RankAnComponentContext *ctx) = 0;
  virtual void exitRankAnComponent(Vtl::RankAnComponentContext *ctx) = 0;

  virtual void enterRatioToReportAnComponent(Vtl::RatioToReportAnComponentContext *ctx) = 0;
  virtual void exitRatioToReportAnComponent(Vtl::RatioToReportAnComponentContext *ctx) = 0;

  virtual void enterRenameClauseItem(Vtl::RenameClauseItemContext *ctx) = 0;
  virtual void exitRenameClauseItem(Vtl::RenameClauseItemContext *ctx) = 0;

  virtual void enterAggregateClause(Vtl::AggregateClauseContext *ctx) = 0;
  virtual void exitAggregateClause(Vtl::AggregateClauseContext *ctx) = 0;

  virtual void enterAggrFunctionClause(Vtl::AggrFunctionClauseContext *ctx) = 0;
  virtual void exitAggrFunctionClause(Vtl::AggrFunctionClauseContext *ctx) = 0;

  virtual void enterCalcClauseItem(Vtl::CalcClauseItemContext *ctx) = 0;
  virtual void exitCalcClauseItem(Vtl::CalcClauseItemContext *ctx) = 0;

  virtual void enterSubspaceClauseItem(Vtl::SubspaceClauseItemContext *ctx) = 0;
  virtual void exitSubspaceClauseItem(Vtl::SubspaceClauseItemContext *ctx) = 0;

  virtual void enterSimpleScalar(Vtl::SimpleScalarContext *ctx) = 0;
  virtual void exitSimpleScalar(Vtl::SimpleScalarContext *ctx) = 0;

  virtual void enterScalarWithCast(Vtl::ScalarWithCastContext *ctx) = 0;
  virtual void exitScalarWithCast(Vtl::ScalarWithCastContext *ctx) = 0;

  virtual void enterJoinClause(Vtl::JoinClauseContext *ctx) = 0;
  virtual void exitJoinClause(Vtl::JoinClauseContext *ctx) = 0;

  virtual void enterJoinClauseItem(Vtl::JoinClauseItemContext *ctx) = 0;
  virtual void exitJoinClauseItem(Vtl::JoinClauseItemContext *ctx) = 0;

  virtual void enterUsingClause(Vtl::UsingClauseContext *ctx) = 0;
  virtual void exitUsingClause(Vtl::UsingClauseContext *ctx) = 0;

  virtual void enterNvlJoinClause(Vtl::NvlJoinClauseContext *ctx) = 0;
  virtual void exitNvlJoinClause(Vtl::NvlJoinClauseContext *ctx) = 0;

  virtual void enterJoinBody(Vtl::JoinBodyContext *ctx) = 0;
  virtual void exitJoinBody(Vtl::JoinBodyContext *ctx) = 0;

  virtual void enterJoinApplyClause(Vtl::JoinApplyClauseContext *ctx) = 0;
  virtual void exitJoinApplyClause(Vtl::JoinApplyClauseContext *ctx) = 0;

  virtual void enterPartitionListed(Vtl::PartitionListedContext *ctx) = 0;
  virtual void exitPartitionListed(Vtl::PartitionListedContext *ctx) = 0;

  virtual void enterPartitionExceptAll(Vtl::PartitionExceptAllContext *ctx) = 0;
  virtual void exitPartitionExceptAll(Vtl::PartitionExceptAllContext *ctx) = 0;

  virtual void enterOrderByClause(Vtl::OrderByClauseContext *ctx) = 0;
  virtual void exitOrderByClause(Vtl::OrderByClauseContext *ctx) = 0;

  virtual void enterOrderByItem(Vtl::OrderByItemContext *ctx) = 0;
  virtual void exitOrderByItem(Vtl::OrderByItemContext *ctx) = 0;

  virtual void enterWindowingClause(Vtl::WindowingClauseContext *ctx) = 0;
  virtual void exitWindowingClause(Vtl::WindowingClauseContext *ctx) = 0;

  virtual void enterSignedInteger(Vtl::SignedIntegerContext *ctx) = 0;
  virtual void exitSignedInteger(Vtl::SignedIntegerContext *ctx) = 0;

  virtual void enterSignedNumber(Vtl::SignedNumberContext *ctx) = 0;
  virtual void exitSignedNumber(Vtl::SignedNumberContext *ctx) = 0;

  virtual void enterLimitClauseItem(Vtl::LimitClauseItemContext *ctx) = 0;
  virtual void exitLimitClauseItem(Vtl::LimitClauseItemContext *ctx) = 0;

  virtual void enterGroupByOrExcept(Vtl::GroupByOrExceptContext *ctx) = 0;
  virtual void exitGroupByOrExcept(Vtl::GroupByOrExceptContext *ctx) = 0;

  virtual void enterGroupAll(Vtl::GroupAllContext *ctx) = 0;
  virtual void exitGroupAll(Vtl::GroupAllContext *ctx) = 0;

  virtual void enterHavingClause(Vtl::HavingClauseContext *ctx) = 0;
  virtual void exitHavingClause(Vtl::HavingClauseContext *ctx) = 0;

  virtual void enterParameterItem(Vtl::ParameterItemContext *ctx) = 0;
  virtual void exitParameterItem(Vtl::ParameterItemContext *ctx) = 0;

  virtual void enterOutputParameterType(Vtl::OutputParameterTypeContext *ctx) = 0;
  virtual void exitOutputParameterType(Vtl::OutputParameterTypeContext *ctx) = 0;

  virtual void enterOutputParameterTypeComponent(Vtl::OutputParameterTypeComponentContext *ctx) = 0;
  virtual void exitOutputParameterTypeComponent(Vtl::OutputParameterTypeComponentContext *ctx) = 0;

  virtual void enterInputParameterType(Vtl::InputParameterTypeContext *ctx) = 0;
  virtual void exitInputParameterType(Vtl::InputParameterTypeContext *ctx) = 0;

  virtual void enterRulesetType(Vtl::RulesetTypeContext *ctx) = 0;
  virtual void exitRulesetType(Vtl::RulesetTypeContext *ctx) = 0;

  virtual void enterScalarType(Vtl::ScalarTypeContext *ctx) = 0;
  virtual void exitScalarType(Vtl::ScalarTypeContext *ctx) = 0;

  virtual void enterComponentType(Vtl::ComponentTypeContext *ctx) = 0;
  virtual void exitComponentType(Vtl::ComponentTypeContext *ctx) = 0;

  virtual void enterDatasetType(Vtl::DatasetTypeContext *ctx) = 0;
  virtual void exitDatasetType(Vtl::DatasetTypeContext *ctx) = 0;

  virtual void enterEvalDatasetType(Vtl::EvalDatasetTypeContext *ctx) = 0;
  virtual void exitEvalDatasetType(Vtl::EvalDatasetTypeContext *ctx) = 0;

  virtual void enterScalarSetType(Vtl::ScalarSetTypeContext *ctx) = 0;
  virtual void exitScalarSetType(Vtl::ScalarSetTypeContext *ctx) = 0;

  virtual void enterDataPoint(Vtl::DataPointContext *ctx) = 0;
  virtual void exitDataPoint(Vtl::DataPointContext *ctx) = 0;

  virtual void enterDataPointVd(Vtl::DataPointVdContext *ctx) = 0;
  virtual void exitDataPointVd(Vtl::DataPointVdContext *ctx) = 0;

  virtual void enterDataPointVar(Vtl::DataPointVarContext *ctx) = 0;
  virtual void exitDataPointVar(Vtl::DataPointVarContext *ctx) = 0;

  virtual void enterHrRulesetType(Vtl::HrRulesetTypeContext *ctx) = 0;
  virtual void exitHrRulesetType(Vtl::HrRulesetTypeContext *ctx) = 0;

  virtual void enterHrRulesetVdType(Vtl::HrRulesetVdTypeContext *ctx) = 0;
  virtual void exitHrRulesetVdType(Vtl::HrRulesetVdTypeContext *ctx) = 0;

  virtual void enterHrRulesetVarType(Vtl::HrRulesetVarTypeContext *ctx) = 0;
  virtual void exitHrRulesetVarType(Vtl::HrRulesetVarTypeContext *ctx) = 0;

  virtual void enterValueDomainName(Vtl::ValueDomainNameContext *ctx) = 0;
  virtual void exitValueDomainName(Vtl::ValueDomainNameContext *ctx) = 0;

  virtual void enterRulesetID(Vtl::RulesetIDContext *ctx) = 0;
  virtual void exitRulesetID(Vtl::RulesetIDContext *ctx) = 0;

  virtual void enterRulesetSignature(Vtl::RulesetSignatureContext *ctx) = 0;
  virtual void exitRulesetSignature(Vtl::RulesetSignatureContext *ctx) = 0;

  virtual void enterSignature(Vtl::SignatureContext *ctx) = 0;
  virtual void exitSignature(Vtl::SignatureContext *ctx) = 0;

  virtual void enterRuleClauseDatapoint(Vtl::RuleClauseDatapointContext *ctx) = 0;
  virtual void exitRuleClauseDatapoint(Vtl::RuleClauseDatapointContext *ctx) = 0;

  virtual void enterRuleItemDatapoint(Vtl::RuleItemDatapointContext *ctx) = 0;
  virtual void exitRuleItemDatapoint(Vtl::RuleItemDatapointContext *ctx) = 0;

  virtual void enterRuleClauseHierarchical(Vtl::RuleClauseHierarchicalContext *ctx) = 0;
  virtual void exitRuleClauseHierarchical(Vtl::RuleClauseHierarchicalContext *ctx) = 0;

  virtual void enterRuleItemHierarchical(Vtl::RuleItemHierarchicalContext *ctx) = 0;
  virtual void exitRuleItemHierarchical(Vtl::RuleItemHierarchicalContext *ctx) = 0;

  virtual void enterHierRuleSignature(Vtl::HierRuleSignatureContext *ctx) = 0;
  virtual void exitHierRuleSignature(Vtl::HierRuleSignatureContext *ctx) = 0;

  virtual void enterValueDomainSignature(Vtl::ValueDomainSignatureContext *ctx) = 0;
  virtual void exitValueDomainSignature(Vtl::ValueDomainSignatureContext *ctx) = 0;

  virtual void enterCodeItemRelation(Vtl::CodeItemRelationContext *ctx) = 0;
  virtual void exitCodeItemRelation(Vtl::CodeItemRelationContext *ctx) = 0;

  virtual void enterCodeItemRelationClause(Vtl::CodeItemRelationClauseContext *ctx) = 0;
  virtual void exitCodeItemRelationClause(Vtl::CodeItemRelationClauseContext *ctx) = 0;

  virtual void enterValueDomainValue(Vtl::ValueDomainValueContext *ctx) = 0;
  virtual void exitValueDomainValue(Vtl::ValueDomainValueContext *ctx) = 0;

  virtual void enterConditionConstraint(Vtl::ConditionConstraintContext *ctx) = 0;
  virtual void exitConditionConstraint(Vtl::ConditionConstraintContext *ctx) = 0;

  virtual void enterRangeConstraint(Vtl::RangeConstraintContext *ctx) = 0;
  virtual void exitRangeConstraint(Vtl::RangeConstraintContext *ctx) = 0;

  virtual void enterCompConstraint(Vtl::CompConstraintContext *ctx) = 0;
  virtual void exitCompConstraint(Vtl::CompConstraintContext *ctx) = 0;

  virtual void enterMultModifier(Vtl::MultModifierContext *ctx) = 0;
  virtual void exitMultModifier(Vtl::MultModifierContext *ctx) = 0;

  virtual void enterValidationOutput(Vtl::ValidationOutputContext *ctx) = 0;
  virtual void exitValidationOutput(Vtl::ValidationOutputContext *ctx) = 0;

  virtual void enterValidationMode(Vtl::ValidationModeContext *ctx) = 0;
  virtual void exitValidationMode(Vtl::ValidationModeContext *ctx) = 0;

  virtual void enterConditionClause(Vtl::ConditionClauseContext *ctx) = 0;
  virtual void exitConditionClause(Vtl::ConditionClauseContext *ctx) = 0;

  virtual void enterInputMode(Vtl::InputModeContext *ctx) = 0;
  virtual void exitInputMode(Vtl::InputModeContext *ctx) = 0;

  virtual void enterImbalanceExpr(Vtl::ImbalanceExprContext *ctx) = 0;
  virtual void exitImbalanceExpr(Vtl::ImbalanceExprContext *ctx) = 0;

  virtual void enterInputModeHierarchy(Vtl::InputModeHierarchyContext *ctx) = 0;
  virtual void exitInputModeHierarchy(Vtl::InputModeHierarchyContext *ctx) = 0;

  virtual void enterOutputModeHierarchy(Vtl::OutputModeHierarchyContext *ctx) = 0;
  virtual void exitOutputModeHierarchy(Vtl::OutputModeHierarchyContext *ctx) = 0;

  virtual void enterAlias(Vtl::AliasContext *ctx) = 0;
  virtual void exitAlias(Vtl::AliasContext *ctx) = 0;

  virtual void enterVarID(Vtl::VarIDContext *ctx) = 0;
  virtual void exitVarID(Vtl::VarIDContext *ctx) = 0;

  virtual void enterSimpleComponentId(Vtl::SimpleComponentIdContext *ctx) = 0;
  virtual void exitSimpleComponentId(Vtl::SimpleComponentIdContext *ctx) = 0;

  virtual void enterComponentID(Vtl::ComponentIDContext *ctx) = 0;
  virtual void exitComponentID(Vtl::ComponentIDContext *ctx) = 0;

  virtual void enterLists(Vtl::ListsContext *ctx) = 0;
  virtual void exitLists(Vtl::ListsContext *ctx) = 0;

  virtual void enterErCode(Vtl::ErCodeContext *ctx) = 0;
  virtual void exitErCode(Vtl::ErCodeContext *ctx) = 0;

  virtual void enterErLevel(Vtl::ErLevelContext *ctx) = 0;
  virtual void exitErLevel(Vtl::ErLevelContext *ctx) = 0;

  virtual void enterComparisonOperand(Vtl::ComparisonOperandContext *ctx) = 0;
  virtual void exitComparisonOperand(Vtl::ComparisonOperandContext *ctx) = 0;

  virtual void enterOptionalExpr(Vtl::OptionalExprContext *ctx) = 0;
  virtual void exitOptionalExpr(Vtl::OptionalExprContext *ctx) = 0;

  virtual void enterOptionalExprComponent(Vtl::OptionalExprComponentContext *ctx) = 0;
  virtual void exitOptionalExprComponent(Vtl::OptionalExprComponentContext *ctx) = 0;

  virtual void enterComponentRole(Vtl::ComponentRoleContext *ctx) = 0;
  virtual void exitComponentRole(Vtl::ComponentRoleContext *ctx) = 0;

  virtual void enterViralAttribute(Vtl::ViralAttributeContext *ctx) = 0;
  virtual void exitViralAttribute(Vtl::ViralAttributeContext *ctx) = 0;

  virtual void enterValueDomainID(Vtl::ValueDomainIDContext *ctx) = 0;
  virtual void exitValueDomainID(Vtl::ValueDomainIDContext *ctx) = 0;

  virtual void enterOperatorID(Vtl::OperatorIDContext *ctx) = 0;
  virtual void exitOperatorID(Vtl::OperatorIDContext *ctx) = 0;

  virtual void enterRoutineName(Vtl::RoutineNameContext *ctx) = 0;
  virtual void exitRoutineName(Vtl::RoutineNameContext *ctx) = 0;

  virtual void enterIntegerLiteral(Vtl::IntegerLiteralContext *ctx) = 0;
  virtual void exitIntegerLiteral(Vtl::IntegerLiteralContext *ctx) = 0;

  virtual void enterNumberLiteral(Vtl::NumberLiteralContext *ctx) = 0;
  virtual void exitNumberLiteral(Vtl::NumberLiteralContext *ctx) = 0;

  virtual void enterBooleanLiteral(Vtl::BooleanLiteralContext *ctx) = 0;
  virtual void exitBooleanLiteral(Vtl::BooleanLiteralContext *ctx) = 0;

  virtual void enterStringLiteral(Vtl::StringLiteralContext *ctx) = 0;
  virtual void exitStringLiteral(Vtl::StringLiteralContext *ctx) = 0;

  virtual void enterNullLiteral(Vtl::NullLiteralContext *ctx) = 0;
  virtual void exitNullLiteral(Vtl::NullLiteralContext *ctx) = 0;

  virtual void enterBasicScalarType(Vtl::BasicScalarTypeContext *ctx) = 0;
  virtual void exitBasicScalarType(Vtl::BasicScalarTypeContext *ctx) = 0;

  virtual void enterRetainType(Vtl::RetainTypeContext *ctx) = 0;
  virtual void exitRetainType(Vtl::RetainTypeContext *ctx) = 0;


};

