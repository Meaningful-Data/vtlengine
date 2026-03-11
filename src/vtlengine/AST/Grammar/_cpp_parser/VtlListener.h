
// Generated from /home/javier/Programacion/vtlengine/src/vtlengine/AST/Grammar/Vtl.g4 by ANTLR 4.13.1

#pragma once


#include "antlr4-runtime.h"
#include "VtlParser.h"


/**
 * This interface defines an abstract listener for a parse tree produced by VtlParser.
 */
class  VtlListener : public antlr4::tree::ParseTreeListener {
public:

  virtual void enterStart(VtlParser::StartContext *ctx) = 0;
  virtual void exitStart(VtlParser::StartContext *ctx) = 0;

  virtual void enterTemporaryAssignment(VtlParser::TemporaryAssignmentContext *ctx) = 0;
  virtual void exitTemporaryAssignment(VtlParser::TemporaryAssignmentContext *ctx) = 0;

  virtual void enterPersistAssignment(VtlParser::PersistAssignmentContext *ctx) = 0;
  virtual void exitPersistAssignment(VtlParser::PersistAssignmentContext *ctx) = 0;

  virtual void enterDefineExpression(VtlParser::DefineExpressionContext *ctx) = 0;
  virtual void exitDefineExpression(VtlParser::DefineExpressionContext *ctx) = 0;

  virtual void enterVarIdExpr(VtlParser::VarIdExprContext *ctx) = 0;
  virtual void exitVarIdExpr(VtlParser::VarIdExprContext *ctx) = 0;

  virtual void enterMembershipExpr(VtlParser::MembershipExprContext *ctx) = 0;
  virtual void exitMembershipExpr(VtlParser::MembershipExprContext *ctx) = 0;

  virtual void enterInNotInExpr(VtlParser::InNotInExprContext *ctx) = 0;
  virtual void exitInNotInExpr(VtlParser::InNotInExprContext *ctx) = 0;

  virtual void enterBooleanExpr(VtlParser::BooleanExprContext *ctx) = 0;
  virtual void exitBooleanExpr(VtlParser::BooleanExprContext *ctx) = 0;

  virtual void enterComparisonExpr(VtlParser::ComparisonExprContext *ctx) = 0;
  virtual void exitComparisonExpr(VtlParser::ComparisonExprContext *ctx) = 0;

  virtual void enterUnaryExpr(VtlParser::UnaryExprContext *ctx) = 0;
  virtual void exitUnaryExpr(VtlParser::UnaryExprContext *ctx) = 0;

  virtual void enterFunctionsExpression(VtlParser::FunctionsExpressionContext *ctx) = 0;
  virtual void exitFunctionsExpression(VtlParser::FunctionsExpressionContext *ctx) = 0;

  virtual void enterIfExpr(VtlParser::IfExprContext *ctx) = 0;
  virtual void exitIfExpr(VtlParser::IfExprContext *ctx) = 0;

  virtual void enterClauseExpr(VtlParser::ClauseExprContext *ctx) = 0;
  virtual void exitClauseExpr(VtlParser::ClauseExprContext *ctx) = 0;

  virtual void enterCaseExpr(VtlParser::CaseExprContext *ctx) = 0;
  virtual void exitCaseExpr(VtlParser::CaseExprContext *ctx) = 0;

  virtual void enterArithmeticExpr(VtlParser::ArithmeticExprContext *ctx) = 0;
  virtual void exitArithmeticExpr(VtlParser::ArithmeticExprContext *ctx) = 0;

  virtual void enterParenthesisExpr(VtlParser::ParenthesisExprContext *ctx) = 0;
  virtual void exitParenthesisExpr(VtlParser::ParenthesisExprContext *ctx) = 0;

  virtual void enterConstantExpr(VtlParser::ConstantExprContext *ctx) = 0;
  virtual void exitConstantExpr(VtlParser::ConstantExprContext *ctx) = 0;

  virtual void enterArithmeticExprOrConcat(VtlParser::ArithmeticExprOrConcatContext *ctx) = 0;
  virtual void exitArithmeticExprOrConcat(VtlParser::ArithmeticExprOrConcatContext *ctx) = 0;

  virtual void enterArithmeticExprComp(VtlParser::ArithmeticExprCompContext *ctx) = 0;
  virtual void exitArithmeticExprComp(VtlParser::ArithmeticExprCompContext *ctx) = 0;

  virtual void enterIfExprComp(VtlParser::IfExprCompContext *ctx) = 0;
  virtual void exitIfExprComp(VtlParser::IfExprCompContext *ctx) = 0;

  virtual void enterComparisonExprComp(VtlParser::ComparisonExprCompContext *ctx) = 0;
  virtual void exitComparisonExprComp(VtlParser::ComparisonExprCompContext *ctx) = 0;

  virtual void enterFunctionsExpressionComp(VtlParser::FunctionsExpressionCompContext *ctx) = 0;
  virtual void exitFunctionsExpressionComp(VtlParser::FunctionsExpressionCompContext *ctx) = 0;

  virtual void enterCompId(VtlParser::CompIdContext *ctx) = 0;
  virtual void exitCompId(VtlParser::CompIdContext *ctx) = 0;

  virtual void enterConstantExprComp(VtlParser::ConstantExprCompContext *ctx) = 0;
  virtual void exitConstantExprComp(VtlParser::ConstantExprCompContext *ctx) = 0;

  virtual void enterArithmeticExprOrConcatComp(VtlParser::ArithmeticExprOrConcatCompContext *ctx) = 0;
  virtual void exitArithmeticExprOrConcatComp(VtlParser::ArithmeticExprOrConcatCompContext *ctx) = 0;

  virtual void enterParenthesisExprComp(VtlParser::ParenthesisExprCompContext *ctx) = 0;
  virtual void exitParenthesisExprComp(VtlParser::ParenthesisExprCompContext *ctx) = 0;

  virtual void enterInNotInExprComp(VtlParser::InNotInExprCompContext *ctx) = 0;
  virtual void exitInNotInExprComp(VtlParser::InNotInExprCompContext *ctx) = 0;

  virtual void enterUnaryExprComp(VtlParser::UnaryExprCompContext *ctx) = 0;
  virtual void exitUnaryExprComp(VtlParser::UnaryExprCompContext *ctx) = 0;

  virtual void enterCaseExprComp(VtlParser::CaseExprCompContext *ctx) = 0;
  virtual void exitCaseExprComp(VtlParser::CaseExprCompContext *ctx) = 0;

  virtual void enterBooleanExprComp(VtlParser::BooleanExprCompContext *ctx) = 0;
  virtual void exitBooleanExprComp(VtlParser::BooleanExprCompContext *ctx) = 0;

  virtual void enterGenericFunctionsComponents(VtlParser::GenericFunctionsComponentsContext *ctx) = 0;
  virtual void exitGenericFunctionsComponents(VtlParser::GenericFunctionsComponentsContext *ctx) = 0;

  virtual void enterStringFunctionsComponents(VtlParser::StringFunctionsComponentsContext *ctx) = 0;
  virtual void exitStringFunctionsComponents(VtlParser::StringFunctionsComponentsContext *ctx) = 0;

  virtual void enterNumericFunctionsComponents(VtlParser::NumericFunctionsComponentsContext *ctx) = 0;
  virtual void exitNumericFunctionsComponents(VtlParser::NumericFunctionsComponentsContext *ctx) = 0;

  virtual void enterComparisonFunctionsComponents(VtlParser::ComparisonFunctionsComponentsContext *ctx) = 0;
  virtual void exitComparisonFunctionsComponents(VtlParser::ComparisonFunctionsComponentsContext *ctx) = 0;

  virtual void enterTimeFunctionsComponents(VtlParser::TimeFunctionsComponentsContext *ctx) = 0;
  virtual void exitTimeFunctionsComponents(VtlParser::TimeFunctionsComponentsContext *ctx) = 0;

  virtual void enterConditionalFunctionsComponents(VtlParser::ConditionalFunctionsComponentsContext *ctx) = 0;
  virtual void exitConditionalFunctionsComponents(VtlParser::ConditionalFunctionsComponentsContext *ctx) = 0;

  virtual void enterAggregateFunctionsComponents(VtlParser::AggregateFunctionsComponentsContext *ctx) = 0;
  virtual void exitAggregateFunctionsComponents(VtlParser::AggregateFunctionsComponentsContext *ctx) = 0;

  virtual void enterAnalyticFunctionsComponents(VtlParser::AnalyticFunctionsComponentsContext *ctx) = 0;
  virtual void exitAnalyticFunctionsComponents(VtlParser::AnalyticFunctionsComponentsContext *ctx) = 0;

  virtual void enterJoinFunctions(VtlParser::JoinFunctionsContext *ctx) = 0;
  virtual void exitJoinFunctions(VtlParser::JoinFunctionsContext *ctx) = 0;

  virtual void enterGenericFunctions(VtlParser::GenericFunctionsContext *ctx) = 0;
  virtual void exitGenericFunctions(VtlParser::GenericFunctionsContext *ctx) = 0;

  virtual void enterStringFunctions(VtlParser::StringFunctionsContext *ctx) = 0;
  virtual void exitStringFunctions(VtlParser::StringFunctionsContext *ctx) = 0;

  virtual void enterNumericFunctions(VtlParser::NumericFunctionsContext *ctx) = 0;
  virtual void exitNumericFunctions(VtlParser::NumericFunctionsContext *ctx) = 0;

  virtual void enterComparisonFunctions(VtlParser::ComparisonFunctionsContext *ctx) = 0;
  virtual void exitComparisonFunctions(VtlParser::ComparisonFunctionsContext *ctx) = 0;

  virtual void enterTimeFunctions(VtlParser::TimeFunctionsContext *ctx) = 0;
  virtual void exitTimeFunctions(VtlParser::TimeFunctionsContext *ctx) = 0;

  virtual void enterSetFunctions(VtlParser::SetFunctionsContext *ctx) = 0;
  virtual void exitSetFunctions(VtlParser::SetFunctionsContext *ctx) = 0;

  virtual void enterHierarchyFunctions(VtlParser::HierarchyFunctionsContext *ctx) = 0;
  virtual void exitHierarchyFunctions(VtlParser::HierarchyFunctionsContext *ctx) = 0;

  virtual void enterValidationFunctions(VtlParser::ValidationFunctionsContext *ctx) = 0;
  virtual void exitValidationFunctions(VtlParser::ValidationFunctionsContext *ctx) = 0;

  virtual void enterConditionalFunctions(VtlParser::ConditionalFunctionsContext *ctx) = 0;
  virtual void exitConditionalFunctions(VtlParser::ConditionalFunctionsContext *ctx) = 0;

  virtual void enterAggregateFunctions(VtlParser::AggregateFunctionsContext *ctx) = 0;
  virtual void exitAggregateFunctions(VtlParser::AggregateFunctionsContext *ctx) = 0;

  virtual void enterAnalyticFunctions(VtlParser::AnalyticFunctionsContext *ctx) = 0;
  virtual void exitAnalyticFunctions(VtlParser::AnalyticFunctionsContext *ctx) = 0;

  virtual void enterDatasetClause(VtlParser::DatasetClauseContext *ctx) = 0;
  virtual void exitDatasetClause(VtlParser::DatasetClauseContext *ctx) = 0;

  virtual void enterRenameClause(VtlParser::RenameClauseContext *ctx) = 0;
  virtual void exitRenameClause(VtlParser::RenameClauseContext *ctx) = 0;

  virtual void enterAggrClause(VtlParser::AggrClauseContext *ctx) = 0;
  virtual void exitAggrClause(VtlParser::AggrClauseContext *ctx) = 0;

  virtual void enterFilterClause(VtlParser::FilterClauseContext *ctx) = 0;
  virtual void exitFilterClause(VtlParser::FilterClauseContext *ctx) = 0;

  virtual void enterCalcClause(VtlParser::CalcClauseContext *ctx) = 0;
  virtual void exitCalcClause(VtlParser::CalcClauseContext *ctx) = 0;

  virtual void enterKeepOrDropClause(VtlParser::KeepOrDropClauseContext *ctx) = 0;
  virtual void exitKeepOrDropClause(VtlParser::KeepOrDropClauseContext *ctx) = 0;

  virtual void enterPivotOrUnpivotClause(VtlParser::PivotOrUnpivotClauseContext *ctx) = 0;
  virtual void exitPivotOrUnpivotClause(VtlParser::PivotOrUnpivotClauseContext *ctx) = 0;

  virtual void enterCustomPivotClause(VtlParser::CustomPivotClauseContext *ctx) = 0;
  virtual void exitCustomPivotClause(VtlParser::CustomPivotClauseContext *ctx) = 0;

  virtual void enterSubspaceClause(VtlParser::SubspaceClauseContext *ctx) = 0;
  virtual void exitSubspaceClause(VtlParser::SubspaceClauseContext *ctx) = 0;

  virtual void enterJoinExpr(VtlParser::JoinExprContext *ctx) = 0;
  virtual void exitJoinExpr(VtlParser::JoinExprContext *ctx) = 0;

  virtual void enterDefOperator(VtlParser::DefOperatorContext *ctx) = 0;
  virtual void exitDefOperator(VtlParser::DefOperatorContext *ctx) = 0;

  virtual void enterDefDatapointRuleset(VtlParser::DefDatapointRulesetContext *ctx) = 0;
  virtual void exitDefDatapointRuleset(VtlParser::DefDatapointRulesetContext *ctx) = 0;

  virtual void enterDefHierarchical(VtlParser::DefHierarchicalContext *ctx) = 0;
  virtual void exitDefHierarchical(VtlParser::DefHierarchicalContext *ctx) = 0;

  virtual void enterCallDataset(VtlParser::CallDatasetContext *ctx) = 0;
  virtual void exitCallDataset(VtlParser::CallDatasetContext *ctx) = 0;

  virtual void enterEvalAtom(VtlParser::EvalAtomContext *ctx) = 0;
  virtual void exitEvalAtom(VtlParser::EvalAtomContext *ctx) = 0;

  virtual void enterCastExprDataset(VtlParser::CastExprDatasetContext *ctx) = 0;
  virtual void exitCastExprDataset(VtlParser::CastExprDatasetContext *ctx) = 0;

  virtual void enterCallComponent(VtlParser::CallComponentContext *ctx) = 0;
  virtual void exitCallComponent(VtlParser::CallComponentContext *ctx) = 0;

  virtual void enterCastExprComponent(VtlParser::CastExprComponentContext *ctx) = 0;
  virtual void exitCastExprComponent(VtlParser::CastExprComponentContext *ctx) = 0;

  virtual void enterEvalAtomComponent(VtlParser::EvalAtomComponentContext *ctx) = 0;
  virtual void exitEvalAtomComponent(VtlParser::EvalAtomComponentContext *ctx) = 0;

  virtual void enterParameterComponent(VtlParser::ParameterComponentContext *ctx) = 0;
  virtual void exitParameterComponent(VtlParser::ParameterComponentContext *ctx) = 0;

  virtual void enterParameter(VtlParser::ParameterContext *ctx) = 0;
  virtual void exitParameter(VtlParser::ParameterContext *ctx) = 0;

  virtual void enterUnaryStringFunction(VtlParser::UnaryStringFunctionContext *ctx) = 0;
  virtual void exitUnaryStringFunction(VtlParser::UnaryStringFunctionContext *ctx) = 0;

  virtual void enterSubstrAtom(VtlParser::SubstrAtomContext *ctx) = 0;
  virtual void exitSubstrAtom(VtlParser::SubstrAtomContext *ctx) = 0;

  virtual void enterReplaceAtom(VtlParser::ReplaceAtomContext *ctx) = 0;
  virtual void exitReplaceAtom(VtlParser::ReplaceAtomContext *ctx) = 0;

  virtual void enterInstrAtom(VtlParser::InstrAtomContext *ctx) = 0;
  virtual void exitInstrAtom(VtlParser::InstrAtomContext *ctx) = 0;

  virtual void enterUnaryStringFunctionComponent(VtlParser::UnaryStringFunctionComponentContext *ctx) = 0;
  virtual void exitUnaryStringFunctionComponent(VtlParser::UnaryStringFunctionComponentContext *ctx) = 0;

  virtual void enterSubstrAtomComponent(VtlParser::SubstrAtomComponentContext *ctx) = 0;
  virtual void exitSubstrAtomComponent(VtlParser::SubstrAtomComponentContext *ctx) = 0;

  virtual void enterReplaceAtomComponent(VtlParser::ReplaceAtomComponentContext *ctx) = 0;
  virtual void exitReplaceAtomComponent(VtlParser::ReplaceAtomComponentContext *ctx) = 0;

  virtual void enterInstrAtomComponent(VtlParser::InstrAtomComponentContext *ctx) = 0;
  virtual void exitInstrAtomComponent(VtlParser::InstrAtomComponentContext *ctx) = 0;

  virtual void enterUnaryNumeric(VtlParser::UnaryNumericContext *ctx) = 0;
  virtual void exitUnaryNumeric(VtlParser::UnaryNumericContext *ctx) = 0;

  virtual void enterUnaryWithOptionalNumeric(VtlParser::UnaryWithOptionalNumericContext *ctx) = 0;
  virtual void exitUnaryWithOptionalNumeric(VtlParser::UnaryWithOptionalNumericContext *ctx) = 0;

  virtual void enterBinaryNumeric(VtlParser::BinaryNumericContext *ctx) = 0;
  virtual void exitBinaryNumeric(VtlParser::BinaryNumericContext *ctx) = 0;

  virtual void enterUnaryNumericComponent(VtlParser::UnaryNumericComponentContext *ctx) = 0;
  virtual void exitUnaryNumericComponent(VtlParser::UnaryNumericComponentContext *ctx) = 0;

  virtual void enterUnaryWithOptionalNumericComponent(VtlParser::UnaryWithOptionalNumericComponentContext *ctx) = 0;
  virtual void exitUnaryWithOptionalNumericComponent(VtlParser::UnaryWithOptionalNumericComponentContext *ctx) = 0;

  virtual void enterBinaryNumericComponent(VtlParser::BinaryNumericComponentContext *ctx) = 0;
  virtual void exitBinaryNumericComponent(VtlParser::BinaryNumericComponentContext *ctx) = 0;

  virtual void enterBetweenAtom(VtlParser::BetweenAtomContext *ctx) = 0;
  virtual void exitBetweenAtom(VtlParser::BetweenAtomContext *ctx) = 0;

  virtual void enterCharsetMatchAtom(VtlParser::CharsetMatchAtomContext *ctx) = 0;
  virtual void exitCharsetMatchAtom(VtlParser::CharsetMatchAtomContext *ctx) = 0;

  virtual void enterIsNullAtom(VtlParser::IsNullAtomContext *ctx) = 0;
  virtual void exitIsNullAtom(VtlParser::IsNullAtomContext *ctx) = 0;

  virtual void enterExistInAtom(VtlParser::ExistInAtomContext *ctx) = 0;
  virtual void exitExistInAtom(VtlParser::ExistInAtomContext *ctx) = 0;

  virtual void enterBetweenAtomComponent(VtlParser::BetweenAtomComponentContext *ctx) = 0;
  virtual void exitBetweenAtomComponent(VtlParser::BetweenAtomComponentContext *ctx) = 0;

  virtual void enterCharsetMatchAtomComponent(VtlParser::CharsetMatchAtomComponentContext *ctx) = 0;
  virtual void exitCharsetMatchAtomComponent(VtlParser::CharsetMatchAtomComponentContext *ctx) = 0;

  virtual void enterIsNullAtomComponent(VtlParser::IsNullAtomComponentContext *ctx) = 0;
  virtual void exitIsNullAtomComponent(VtlParser::IsNullAtomComponentContext *ctx) = 0;

  virtual void enterPeriodAtom(VtlParser::PeriodAtomContext *ctx) = 0;
  virtual void exitPeriodAtom(VtlParser::PeriodAtomContext *ctx) = 0;

  virtual void enterFillTimeAtom(VtlParser::FillTimeAtomContext *ctx) = 0;
  virtual void exitFillTimeAtom(VtlParser::FillTimeAtomContext *ctx) = 0;

  virtual void enterFlowAtom(VtlParser::FlowAtomContext *ctx) = 0;
  virtual void exitFlowAtom(VtlParser::FlowAtomContext *ctx) = 0;

  virtual void enterTimeShiftAtom(VtlParser::TimeShiftAtomContext *ctx) = 0;
  virtual void exitTimeShiftAtom(VtlParser::TimeShiftAtomContext *ctx) = 0;

  virtual void enterTimeAggAtom(VtlParser::TimeAggAtomContext *ctx) = 0;
  virtual void exitTimeAggAtom(VtlParser::TimeAggAtomContext *ctx) = 0;

  virtual void enterCurrentDateAtom(VtlParser::CurrentDateAtomContext *ctx) = 0;
  virtual void exitCurrentDateAtom(VtlParser::CurrentDateAtomContext *ctx) = 0;

  virtual void enterDateDiffAtom(VtlParser::DateDiffAtomContext *ctx) = 0;
  virtual void exitDateDiffAtom(VtlParser::DateDiffAtomContext *ctx) = 0;

  virtual void enterDateAddAtom(VtlParser::DateAddAtomContext *ctx) = 0;
  virtual void exitDateAddAtom(VtlParser::DateAddAtomContext *ctx) = 0;

  virtual void enterYearAtom(VtlParser::YearAtomContext *ctx) = 0;
  virtual void exitYearAtom(VtlParser::YearAtomContext *ctx) = 0;

  virtual void enterMonthAtom(VtlParser::MonthAtomContext *ctx) = 0;
  virtual void exitMonthAtom(VtlParser::MonthAtomContext *ctx) = 0;

  virtual void enterDayOfMonthAtom(VtlParser::DayOfMonthAtomContext *ctx) = 0;
  virtual void exitDayOfMonthAtom(VtlParser::DayOfMonthAtomContext *ctx) = 0;

  virtual void enterDayOfYearAtom(VtlParser::DayOfYearAtomContext *ctx) = 0;
  virtual void exitDayOfYearAtom(VtlParser::DayOfYearAtomContext *ctx) = 0;

  virtual void enterDayToYearAtom(VtlParser::DayToYearAtomContext *ctx) = 0;
  virtual void exitDayToYearAtom(VtlParser::DayToYearAtomContext *ctx) = 0;

  virtual void enterDayToMonthAtom(VtlParser::DayToMonthAtomContext *ctx) = 0;
  virtual void exitDayToMonthAtom(VtlParser::DayToMonthAtomContext *ctx) = 0;

  virtual void enterYearTodayAtom(VtlParser::YearTodayAtomContext *ctx) = 0;
  virtual void exitYearTodayAtom(VtlParser::YearTodayAtomContext *ctx) = 0;

  virtual void enterMonthTodayAtom(VtlParser::MonthTodayAtomContext *ctx) = 0;
  virtual void exitMonthTodayAtom(VtlParser::MonthTodayAtomContext *ctx) = 0;

  virtual void enterPeriodAtomComponent(VtlParser::PeriodAtomComponentContext *ctx) = 0;
  virtual void exitPeriodAtomComponent(VtlParser::PeriodAtomComponentContext *ctx) = 0;

  virtual void enterFillTimeAtomComponent(VtlParser::FillTimeAtomComponentContext *ctx) = 0;
  virtual void exitFillTimeAtomComponent(VtlParser::FillTimeAtomComponentContext *ctx) = 0;

  virtual void enterFlowAtomComponent(VtlParser::FlowAtomComponentContext *ctx) = 0;
  virtual void exitFlowAtomComponent(VtlParser::FlowAtomComponentContext *ctx) = 0;

  virtual void enterTimeShiftAtomComponent(VtlParser::TimeShiftAtomComponentContext *ctx) = 0;
  virtual void exitTimeShiftAtomComponent(VtlParser::TimeShiftAtomComponentContext *ctx) = 0;

  virtual void enterTimeAggAtomComponent(VtlParser::TimeAggAtomComponentContext *ctx) = 0;
  virtual void exitTimeAggAtomComponent(VtlParser::TimeAggAtomComponentContext *ctx) = 0;

  virtual void enterCurrentDateAtomComponent(VtlParser::CurrentDateAtomComponentContext *ctx) = 0;
  virtual void exitCurrentDateAtomComponent(VtlParser::CurrentDateAtomComponentContext *ctx) = 0;

  virtual void enterDateDiffAtomComponent(VtlParser::DateDiffAtomComponentContext *ctx) = 0;
  virtual void exitDateDiffAtomComponent(VtlParser::DateDiffAtomComponentContext *ctx) = 0;

  virtual void enterDateAddAtomComponent(VtlParser::DateAddAtomComponentContext *ctx) = 0;
  virtual void exitDateAddAtomComponent(VtlParser::DateAddAtomComponentContext *ctx) = 0;

  virtual void enterYearAtomComponent(VtlParser::YearAtomComponentContext *ctx) = 0;
  virtual void exitYearAtomComponent(VtlParser::YearAtomComponentContext *ctx) = 0;

  virtual void enterMonthAtomComponent(VtlParser::MonthAtomComponentContext *ctx) = 0;
  virtual void exitMonthAtomComponent(VtlParser::MonthAtomComponentContext *ctx) = 0;

  virtual void enterDayOfMonthAtomComponent(VtlParser::DayOfMonthAtomComponentContext *ctx) = 0;
  virtual void exitDayOfMonthAtomComponent(VtlParser::DayOfMonthAtomComponentContext *ctx) = 0;

  virtual void enterDatOfYearAtomComponent(VtlParser::DatOfYearAtomComponentContext *ctx) = 0;
  virtual void exitDatOfYearAtomComponent(VtlParser::DatOfYearAtomComponentContext *ctx) = 0;

  virtual void enterDayToYearAtomComponent(VtlParser::DayToYearAtomComponentContext *ctx) = 0;
  virtual void exitDayToYearAtomComponent(VtlParser::DayToYearAtomComponentContext *ctx) = 0;

  virtual void enterDayToMonthAtomComponent(VtlParser::DayToMonthAtomComponentContext *ctx) = 0;
  virtual void exitDayToMonthAtomComponent(VtlParser::DayToMonthAtomComponentContext *ctx) = 0;

  virtual void enterYearTodayAtomComponent(VtlParser::YearTodayAtomComponentContext *ctx) = 0;
  virtual void exitYearTodayAtomComponent(VtlParser::YearTodayAtomComponentContext *ctx) = 0;

  virtual void enterMonthTodayAtomComponent(VtlParser::MonthTodayAtomComponentContext *ctx) = 0;
  virtual void exitMonthTodayAtomComponent(VtlParser::MonthTodayAtomComponentContext *ctx) = 0;

  virtual void enterUnionAtom(VtlParser::UnionAtomContext *ctx) = 0;
  virtual void exitUnionAtom(VtlParser::UnionAtomContext *ctx) = 0;

  virtual void enterIntersectAtom(VtlParser::IntersectAtomContext *ctx) = 0;
  virtual void exitIntersectAtom(VtlParser::IntersectAtomContext *ctx) = 0;

  virtual void enterSetOrSYmDiffAtom(VtlParser::SetOrSYmDiffAtomContext *ctx) = 0;
  virtual void exitSetOrSYmDiffAtom(VtlParser::SetOrSYmDiffAtomContext *ctx) = 0;

  virtual void enterHierarchyOperators(VtlParser::HierarchyOperatorsContext *ctx) = 0;
  virtual void exitHierarchyOperators(VtlParser::HierarchyOperatorsContext *ctx) = 0;

  virtual void enterValidateDPruleset(VtlParser::ValidateDPrulesetContext *ctx) = 0;
  virtual void exitValidateDPruleset(VtlParser::ValidateDPrulesetContext *ctx) = 0;

  virtual void enterValidateHRruleset(VtlParser::ValidateHRrulesetContext *ctx) = 0;
  virtual void exitValidateHRruleset(VtlParser::ValidateHRrulesetContext *ctx) = 0;

  virtual void enterValidationSimple(VtlParser::ValidationSimpleContext *ctx) = 0;
  virtual void exitValidationSimple(VtlParser::ValidationSimpleContext *ctx) = 0;

  virtual void enterNvlAtom(VtlParser::NvlAtomContext *ctx) = 0;
  virtual void exitNvlAtom(VtlParser::NvlAtomContext *ctx) = 0;

  virtual void enterNvlAtomComponent(VtlParser::NvlAtomComponentContext *ctx) = 0;
  virtual void exitNvlAtomComponent(VtlParser::NvlAtomComponentContext *ctx) = 0;

  virtual void enterAggrComp(VtlParser::AggrCompContext *ctx) = 0;
  virtual void exitAggrComp(VtlParser::AggrCompContext *ctx) = 0;

  virtual void enterCountAggrComp(VtlParser::CountAggrCompContext *ctx) = 0;
  virtual void exitCountAggrComp(VtlParser::CountAggrCompContext *ctx) = 0;

  virtual void enterAggrDataset(VtlParser::AggrDatasetContext *ctx) = 0;
  virtual void exitAggrDataset(VtlParser::AggrDatasetContext *ctx) = 0;

  virtual void enterAnSimpleFunction(VtlParser::AnSimpleFunctionContext *ctx) = 0;
  virtual void exitAnSimpleFunction(VtlParser::AnSimpleFunctionContext *ctx) = 0;

  virtual void enterLagOrLeadAn(VtlParser::LagOrLeadAnContext *ctx) = 0;
  virtual void exitLagOrLeadAn(VtlParser::LagOrLeadAnContext *ctx) = 0;

  virtual void enterRatioToReportAn(VtlParser::RatioToReportAnContext *ctx) = 0;
  virtual void exitRatioToReportAn(VtlParser::RatioToReportAnContext *ctx) = 0;

  virtual void enterAnSimpleFunctionComponent(VtlParser::AnSimpleFunctionComponentContext *ctx) = 0;
  virtual void exitAnSimpleFunctionComponent(VtlParser::AnSimpleFunctionComponentContext *ctx) = 0;

  virtual void enterLagOrLeadAnComponent(VtlParser::LagOrLeadAnComponentContext *ctx) = 0;
  virtual void exitLagOrLeadAnComponent(VtlParser::LagOrLeadAnComponentContext *ctx) = 0;

  virtual void enterRankAnComponent(VtlParser::RankAnComponentContext *ctx) = 0;
  virtual void exitRankAnComponent(VtlParser::RankAnComponentContext *ctx) = 0;

  virtual void enterRatioToReportAnComponent(VtlParser::RatioToReportAnComponentContext *ctx) = 0;
  virtual void exitRatioToReportAnComponent(VtlParser::RatioToReportAnComponentContext *ctx) = 0;

  virtual void enterRenameClauseItem(VtlParser::RenameClauseItemContext *ctx) = 0;
  virtual void exitRenameClauseItem(VtlParser::RenameClauseItemContext *ctx) = 0;

  virtual void enterAggregateClause(VtlParser::AggregateClauseContext *ctx) = 0;
  virtual void exitAggregateClause(VtlParser::AggregateClauseContext *ctx) = 0;

  virtual void enterAggrFunctionClause(VtlParser::AggrFunctionClauseContext *ctx) = 0;
  virtual void exitAggrFunctionClause(VtlParser::AggrFunctionClauseContext *ctx) = 0;

  virtual void enterCalcClauseItem(VtlParser::CalcClauseItemContext *ctx) = 0;
  virtual void exitCalcClauseItem(VtlParser::CalcClauseItemContext *ctx) = 0;

  virtual void enterSubspaceClauseItem(VtlParser::SubspaceClauseItemContext *ctx) = 0;
  virtual void exitSubspaceClauseItem(VtlParser::SubspaceClauseItemContext *ctx) = 0;

  virtual void enterSimpleScalar(VtlParser::SimpleScalarContext *ctx) = 0;
  virtual void exitSimpleScalar(VtlParser::SimpleScalarContext *ctx) = 0;

  virtual void enterScalarWithCast(VtlParser::ScalarWithCastContext *ctx) = 0;
  virtual void exitScalarWithCast(VtlParser::ScalarWithCastContext *ctx) = 0;

  virtual void enterJoinClauseWithoutUsing(VtlParser::JoinClauseWithoutUsingContext *ctx) = 0;
  virtual void exitJoinClauseWithoutUsing(VtlParser::JoinClauseWithoutUsingContext *ctx) = 0;

  virtual void enterJoinClause(VtlParser::JoinClauseContext *ctx) = 0;
  virtual void exitJoinClause(VtlParser::JoinClauseContext *ctx) = 0;

  virtual void enterJoinClauseItem(VtlParser::JoinClauseItemContext *ctx) = 0;
  virtual void exitJoinClauseItem(VtlParser::JoinClauseItemContext *ctx) = 0;

  virtual void enterJoinBody(VtlParser::JoinBodyContext *ctx) = 0;
  virtual void exitJoinBody(VtlParser::JoinBodyContext *ctx) = 0;

  virtual void enterJoinApplyClause(VtlParser::JoinApplyClauseContext *ctx) = 0;
  virtual void exitJoinApplyClause(VtlParser::JoinApplyClauseContext *ctx) = 0;

  virtual void enterPartitionByClause(VtlParser::PartitionByClauseContext *ctx) = 0;
  virtual void exitPartitionByClause(VtlParser::PartitionByClauseContext *ctx) = 0;

  virtual void enterOrderByClause(VtlParser::OrderByClauseContext *ctx) = 0;
  virtual void exitOrderByClause(VtlParser::OrderByClauseContext *ctx) = 0;

  virtual void enterOrderByItem(VtlParser::OrderByItemContext *ctx) = 0;
  virtual void exitOrderByItem(VtlParser::OrderByItemContext *ctx) = 0;

  virtual void enterWindowingClause(VtlParser::WindowingClauseContext *ctx) = 0;
  virtual void exitWindowingClause(VtlParser::WindowingClauseContext *ctx) = 0;

  virtual void enterSignedInteger(VtlParser::SignedIntegerContext *ctx) = 0;
  virtual void exitSignedInteger(VtlParser::SignedIntegerContext *ctx) = 0;

  virtual void enterSignedNumber(VtlParser::SignedNumberContext *ctx) = 0;
  virtual void exitSignedNumber(VtlParser::SignedNumberContext *ctx) = 0;

  virtual void enterLimitClauseItem(VtlParser::LimitClauseItemContext *ctx) = 0;
  virtual void exitLimitClauseItem(VtlParser::LimitClauseItemContext *ctx) = 0;

  virtual void enterGroupByOrExcept(VtlParser::GroupByOrExceptContext *ctx) = 0;
  virtual void exitGroupByOrExcept(VtlParser::GroupByOrExceptContext *ctx) = 0;

  virtual void enterGroupAll(VtlParser::GroupAllContext *ctx) = 0;
  virtual void exitGroupAll(VtlParser::GroupAllContext *ctx) = 0;

  virtual void enterHavingClause(VtlParser::HavingClauseContext *ctx) = 0;
  virtual void exitHavingClause(VtlParser::HavingClauseContext *ctx) = 0;

  virtual void enterParameterItem(VtlParser::ParameterItemContext *ctx) = 0;
  virtual void exitParameterItem(VtlParser::ParameterItemContext *ctx) = 0;

  virtual void enterOutputParameterType(VtlParser::OutputParameterTypeContext *ctx) = 0;
  virtual void exitOutputParameterType(VtlParser::OutputParameterTypeContext *ctx) = 0;

  virtual void enterOutputParameterTypeComponent(VtlParser::OutputParameterTypeComponentContext *ctx) = 0;
  virtual void exitOutputParameterTypeComponent(VtlParser::OutputParameterTypeComponentContext *ctx) = 0;

  virtual void enterInputParameterType(VtlParser::InputParameterTypeContext *ctx) = 0;
  virtual void exitInputParameterType(VtlParser::InputParameterTypeContext *ctx) = 0;

  virtual void enterRulesetType(VtlParser::RulesetTypeContext *ctx) = 0;
  virtual void exitRulesetType(VtlParser::RulesetTypeContext *ctx) = 0;

  virtual void enterScalarType(VtlParser::ScalarTypeContext *ctx) = 0;
  virtual void exitScalarType(VtlParser::ScalarTypeContext *ctx) = 0;

  virtual void enterComponentType(VtlParser::ComponentTypeContext *ctx) = 0;
  virtual void exitComponentType(VtlParser::ComponentTypeContext *ctx) = 0;

  virtual void enterDatasetType(VtlParser::DatasetTypeContext *ctx) = 0;
  virtual void exitDatasetType(VtlParser::DatasetTypeContext *ctx) = 0;

  virtual void enterEvalDatasetType(VtlParser::EvalDatasetTypeContext *ctx) = 0;
  virtual void exitEvalDatasetType(VtlParser::EvalDatasetTypeContext *ctx) = 0;

  virtual void enterScalarSetType(VtlParser::ScalarSetTypeContext *ctx) = 0;
  virtual void exitScalarSetType(VtlParser::ScalarSetTypeContext *ctx) = 0;

  virtual void enterDataPoint(VtlParser::DataPointContext *ctx) = 0;
  virtual void exitDataPoint(VtlParser::DataPointContext *ctx) = 0;

  virtual void enterDataPointVd(VtlParser::DataPointVdContext *ctx) = 0;
  virtual void exitDataPointVd(VtlParser::DataPointVdContext *ctx) = 0;

  virtual void enterDataPointVar(VtlParser::DataPointVarContext *ctx) = 0;
  virtual void exitDataPointVar(VtlParser::DataPointVarContext *ctx) = 0;

  virtual void enterHrRulesetType(VtlParser::HrRulesetTypeContext *ctx) = 0;
  virtual void exitHrRulesetType(VtlParser::HrRulesetTypeContext *ctx) = 0;

  virtual void enterHrRulesetVdType(VtlParser::HrRulesetVdTypeContext *ctx) = 0;
  virtual void exitHrRulesetVdType(VtlParser::HrRulesetVdTypeContext *ctx) = 0;

  virtual void enterHrRulesetVarType(VtlParser::HrRulesetVarTypeContext *ctx) = 0;
  virtual void exitHrRulesetVarType(VtlParser::HrRulesetVarTypeContext *ctx) = 0;

  virtual void enterValueDomainName(VtlParser::ValueDomainNameContext *ctx) = 0;
  virtual void exitValueDomainName(VtlParser::ValueDomainNameContext *ctx) = 0;

  virtual void enterRulesetID(VtlParser::RulesetIDContext *ctx) = 0;
  virtual void exitRulesetID(VtlParser::RulesetIDContext *ctx) = 0;

  virtual void enterRulesetSignature(VtlParser::RulesetSignatureContext *ctx) = 0;
  virtual void exitRulesetSignature(VtlParser::RulesetSignatureContext *ctx) = 0;

  virtual void enterSignature(VtlParser::SignatureContext *ctx) = 0;
  virtual void exitSignature(VtlParser::SignatureContext *ctx) = 0;

  virtual void enterRuleClauseDatapoint(VtlParser::RuleClauseDatapointContext *ctx) = 0;
  virtual void exitRuleClauseDatapoint(VtlParser::RuleClauseDatapointContext *ctx) = 0;

  virtual void enterRuleItemDatapoint(VtlParser::RuleItemDatapointContext *ctx) = 0;
  virtual void exitRuleItemDatapoint(VtlParser::RuleItemDatapointContext *ctx) = 0;

  virtual void enterRuleClauseHierarchical(VtlParser::RuleClauseHierarchicalContext *ctx) = 0;
  virtual void exitRuleClauseHierarchical(VtlParser::RuleClauseHierarchicalContext *ctx) = 0;

  virtual void enterRuleItemHierarchical(VtlParser::RuleItemHierarchicalContext *ctx) = 0;
  virtual void exitRuleItemHierarchical(VtlParser::RuleItemHierarchicalContext *ctx) = 0;

  virtual void enterHierRuleSignature(VtlParser::HierRuleSignatureContext *ctx) = 0;
  virtual void exitHierRuleSignature(VtlParser::HierRuleSignatureContext *ctx) = 0;

  virtual void enterValueDomainSignature(VtlParser::ValueDomainSignatureContext *ctx) = 0;
  virtual void exitValueDomainSignature(VtlParser::ValueDomainSignatureContext *ctx) = 0;

  virtual void enterCodeItemRelation(VtlParser::CodeItemRelationContext *ctx) = 0;
  virtual void exitCodeItemRelation(VtlParser::CodeItemRelationContext *ctx) = 0;

  virtual void enterCodeItemRelationClause(VtlParser::CodeItemRelationClauseContext *ctx) = 0;
  virtual void exitCodeItemRelationClause(VtlParser::CodeItemRelationClauseContext *ctx) = 0;

  virtual void enterValueDomainValue(VtlParser::ValueDomainValueContext *ctx) = 0;
  virtual void exitValueDomainValue(VtlParser::ValueDomainValueContext *ctx) = 0;

  virtual void enterConditionConstraint(VtlParser::ConditionConstraintContext *ctx) = 0;
  virtual void exitConditionConstraint(VtlParser::ConditionConstraintContext *ctx) = 0;

  virtual void enterRangeConstraint(VtlParser::RangeConstraintContext *ctx) = 0;
  virtual void exitRangeConstraint(VtlParser::RangeConstraintContext *ctx) = 0;

  virtual void enterCompConstraint(VtlParser::CompConstraintContext *ctx) = 0;
  virtual void exitCompConstraint(VtlParser::CompConstraintContext *ctx) = 0;

  virtual void enterMultModifier(VtlParser::MultModifierContext *ctx) = 0;
  virtual void exitMultModifier(VtlParser::MultModifierContext *ctx) = 0;

  virtual void enterValidationOutput(VtlParser::ValidationOutputContext *ctx) = 0;
  virtual void exitValidationOutput(VtlParser::ValidationOutputContext *ctx) = 0;

  virtual void enterValidationMode(VtlParser::ValidationModeContext *ctx) = 0;
  virtual void exitValidationMode(VtlParser::ValidationModeContext *ctx) = 0;

  virtual void enterConditionClause(VtlParser::ConditionClauseContext *ctx) = 0;
  virtual void exitConditionClause(VtlParser::ConditionClauseContext *ctx) = 0;

  virtual void enterInputMode(VtlParser::InputModeContext *ctx) = 0;
  virtual void exitInputMode(VtlParser::InputModeContext *ctx) = 0;

  virtual void enterImbalanceExpr(VtlParser::ImbalanceExprContext *ctx) = 0;
  virtual void exitImbalanceExpr(VtlParser::ImbalanceExprContext *ctx) = 0;

  virtual void enterInputModeHierarchy(VtlParser::InputModeHierarchyContext *ctx) = 0;
  virtual void exitInputModeHierarchy(VtlParser::InputModeHierarchyContext *ctx) = 0;

  virtual void enterOutputModeHierarchy(VtlParser::OutputModeHierarchyContext *ctx) = 0;
  virtual void exitOutputModeHierarchy(VtlParser::OutputModeHierarchyContext *ctx) = 0;

  virtual void enterAlias(VtlParser::AliasContext *ctx) = 0;
  virtual void exitAlias(VtlParser::AliasContext *ctx) = 0;

  virtual void enterVarID(VtlParser::VarIDContext *ctx) = 0;
  virtual void exitVarID(VtlParser::VarIDContext *ctx) = 0;

  virtual void enterSimpleComponentId(VtlParser::SimpleComponentIdContext *ctx) = 0;
  virtual void exitSimpleComponentId(VtlParser::SimpleComponentIdContext *ctx) = 0;

  virtual void enterComponentID(VtlParser::ComponentIDContext *ctx) = 0;
  virtual void exitComponentID(VtlParser::ComponentIDContext *ctx) = 0;

  virtual void enterLists(VtlParser::ListsContext *ctx) = 0;
  virtual void exitLists(VtlParser::ListsContext *ctx) = 0;

  virtual void enterErCode(VtlParser::ErCodeContext *ctx) = 0;
  virtual void exitErCode(VtlParser::ErCodeContext *ctx) = 0;

  virtual void enterErLevel(VtlParser::ErLevelContext *ctx) = 0;
  virtual void exitErLevel(VtlParser::ErLevelContext *ctx) = 0;

  virtual void enterComparisonOperand(VtlParser::ComparisonOperandContext *ctx) = 0;
  virtual void exitComparisonOperand(VtlParser::ComparisonOperandContext *ctx) = 0;

  virtual void enterOptionalExpr(VtlParser::OptionalExprContext *ctx) = 0;
  virtual void exitOptionalExpr(VtlParser::OptionalExprContext *ctx) = 0;

  virtual void enterOptionalExprComponent(VtlParser::OptionalExprComponentContext *ctx) = 0;
  virtual void exitOptionalExprComponent(VtlParser::OptionalExprComponentContext *ctx) = 0;

  virtual void enterComponentRole(VtlParser::ComponentRoleContext *ctx) = 0;
  virtual void exitComponentRole(VtlParser::ComponentRoleContext *ctx) = 0;

  virtual void enterViralAttribute(VtlParser::ViralAttributeContext *ctx) = 0;
  virtual void exitViralAttribute(VtlParser::ViralAttributeContext *ctx) = 0;

  virtual void enterValueDomainID(VtlParser::ValueDomainIDContext *ctx) = 0;
  virtual void exitValueDomainID(VtlParser::ValueDomainIDContext *ctx) = 0;

  virtual void enterOperatorID(VtlParser::OperatorIDContext *ctx) = 0;
  virtual void exitOperatorID(VtlParser::OperatorIDContext *ctx) = 0;

  virtual void enterRoutineName(VtlParser::RoutineNameContext *ctx) = 0;
  virtual void exitRoutineName(VtlParser::RoutineNameContext *ctx) = 0;

  virtual void enterConstant(VtlParser::ConstantContext *ctx) = 0;
  virtual void exitConstant(VtlParser::ConstantContext *ctx) = 0;

  virtual void enterBasicScalarType(VtlParser::BasicScalarTypeContext *ctx) = 0;
  virtual void exitBasicScalarType(VtlParser::BasicScalarTypeContext *ctx) = 0;

  virtual void enterRetainType(VtlParser::RetainTypeContext *ctx) = 0;
  virtual void exitRetainType(VtlParser::RetainTypeContext *ctx) = 0;


};

