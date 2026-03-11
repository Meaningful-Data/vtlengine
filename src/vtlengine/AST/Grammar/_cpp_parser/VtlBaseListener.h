
// Generated from /home/javier/Programacion/vtlengine/src/vtlengine/AST/Grammar/Vtl.g4 by ANTLR 4.13.1

#pragma once


#include "antlr4-runtime.h"
#include "VtlListener.h"


/**
 * This class provides an empty implementation of VtlListener,
 * which can be extended to create a listener which only needs to handle a subset
 * of the available methods.
 */
class  VtlBaseListener : public VtlListener {
public:

  virtual void enterStart(VtlParser::StartContext * /*ctx*/) override { }
  virtual void exitStart(VtlParser::StartContext * /*ctx*/) override { }

  virtual void enterTemporaryAssignment(VtlParser::TemporaryAssignmentContext * /*ctx*/) override { }
  virtual void exitTemporaryAssignment(VtlParser::TemporaryAssignmentContext * /*ctx*/) override { }

  virtual void enterPersistAssignment(VtlParser::PersistAssignmentContext * /*ctx*/) override { }
  virtual void exitPersistAssignment(VtlParser::PersistAssignmentContext * /*ctx*/) override { }

  virtual void enterDefineExpression(VtlParser::DefineExpressionContext * /*ctx*/) override { }
  virtual void exitDefineExpression(VtlParser::DefineExpressionContext * /*ctx*/) override { }

  virtual void enterVarIdExpr(VtlParser::VarIdExprContext * /*ctx*/) override { }
  virtual void exitVarIdExpr(VtlParser::VarIdExprContext * /*ctx*/) override { }

  virtual void enterMembershipExpr(VtlParser::MembershipExprContext * /*ctx*/) override { }
  virtual void exitMembershipExpr(VtlParser::MembershipExprContext * /*ctx*/) override { }

  virtual void enterInNotInExpr(VtlParser::InNotInExprContext * /*ctx*/) override { }
  virtual void exitInNotInExpr(VtlParser::InNotInExprContext * /*ctx*/) override { }

  virtual void enterBooleanExpr(VtlParser::BooleanExprContext * /*ctx*/) override { }
  virtual void exitBooleanExpr(VtlParser::BooleanExprContext * /*ctx*/) override { }

  virtual void enterComparisonExpr(VtlParser::ComparisonExprContext * /*ctx*/) override { }
  virtual void exitComparisonExpr(VtlParser::ComparisonExprContext * /*ctx*/) override { }

  virtual void enterUnaryExpr(VtlParser::UnaryExprContext * /*ctx*/) override { }
  virtual void exitUnaryExpr(VtlParser::UnaryExprContext * /*ctx*/) override { }

  virtual void enterFunctionsExpression(VtlParser::FunctionsExpressionContext * /*ctx*/) override { }
  virtual void exitFunctionsExpression(VtlParser::FunctionsExpressionContext * /*ctx*/) override { }

  virtual void enterIfExpr(VtlParser::IfExprContext * /*ctx*/) override { }
  virtual void exitIfExpr(VtlParser::IfExprContext * /*ctx*/) override { }

  virtual void enterClauseExpr(VtlParser::ClauseExprContext * /*ctx*/) override { }
  virtual void exitClauseExpr(VtlParser::ClauseExprContext * /*ctx*/) override { }

  virtual void enterCaseExpr(VtlParser::CaseExprContext * /*ctx*/) override { }
  virtual void exitCaseExpr(VtlParser::CaseExprContext * /*ctx*/) override { }

  virtual void enterArithmeticExpr(VtlParser::ArithmeticExprContext * /*ctx*/) override { }
  virtual void exitArithmeticExpr(VtlParser::ArithmeticExprContext * /*ctx*/) override { }

  virtual void enterParenthesisExpr(VtlParser::ParenthesisExprContext * /*ctx*/) override { }
  virtual void exitParenthesisExpr(VtlParser::ParenthesisExprContext * /*ctx*/) override { }

  virtual void enterConstantExpr(VtlParser::ConstantExprContext * /*ctx*/) override { }
  virtual void exitConstantExpr(VtlParser::ConstantExprContext * /*ctx*/) override { }

  virtual void enterArithmeticExprOrConcat(VtlParser::ArithmeticExprOrConcatContext * /*ctx*/) override { }
  virtual void exitArithmeticExprOrConcat(VtlParser::ArithmeticExprOrConcatContext * /*ctx*/) override { }

  virtual void enterArithmeticExprComp(VtlParser::ArithmeticExprCompContext * /*ctx*/) override { }
  virtual void exitArithmeticExprComp(VtlParser::ArithmeticExprCompContext * /*ctx*/) override { }

  virtual void enterIfExprComp(VtlParser::IfExprCompContext * /*ctx*/) override { }
  virtual void exitIfExprComp(VtlParser::IfExprCompContext * /*ctx*/) override { }

  virtual void enterComparisonExprComp(VtlParser::ComparisonExprCompContext * /*ctx*/) override { }
  virtual void exitComparisonExprComp(VtlParser::ComparisonExprCompContext * /*ctx*/) override { }

  virtual void enterFunctionsExpressionComp(VtlParser::FunctionsExpressionCompContext * /*ctx*/) override { }
  virtual void exitFunctionsExpressionComp(VtlParser::FunctionsExpressionCompContext * /*ctx*/) override { }

  virtual void enterCompId(VtlParser::CompIdContext * /*ctx*/) override { }
  virtual void exitCompId(VtlParser::CompIdContext * /*ctx*/) override { }

  virtual void enterConstantExprComp(VtlParser::ConstantExprCompContext * /*ctx*/) override { }
  virtual void exitConstantExprComp(VtlParser::ConstantExprCompContext * /*ctx*/) override { }

  virtual void enterArithmeticExprOrConcatComp(VtlParser::ArithmeticExprOrConcatCompContext * /*ctx*/) override { }
  virtual void exitArithmeticExprOrConcatComp(VtlParser::ArithmeticExprOrConcatCompContext * /*ctx*/) override { }

  virtual void enterParenthesisExprComp(VtlParser::ParenthesisExprCompContext * /*ctx*/) override { }
  virtual void exitParenthesisExprComp(VtlParser::ParenthesisExprCompContext * /*ctx*/) override { }

  virtual void enterInNotInExprComp(VtlParser::InNotInExprCompContext * /*ctx*/) override { }
  virtual void exitInNotInExprComp(VtlParser::InNotInExprCompContext * /*ctx*/) override { }

  virtual void enterUnaryExprComp(VtlParser::UnaryExprCompContext * /*ctx*/) override { }
  virtual void exitUnaryExprComp(VtlParser::UnaryExprCompContext * /*ctx*/) override { }

  virtual void enterCaseExprComp(VtlParser::CaseExprCompContext * /*ctx*/) override { }
  virtual void exitCaseExprComp(VtlParser::CaseExprCompContext * /*ctx*/) override { }

  virtual void enterBooleanExprComp(VtlParser::BooleanExprCompContext * /*ctx*/) override { }
  virtual void exitBooleanExprComp(VtlParser::BooleanExprCompContext * /*ctx*/) override { }

  virtual void enterGenericFunctionsComponents(VtlParser::GenericFunctionsComponentsContext * /*ctx*/) override { }
  virtual void exitGenericFunctionsComponents(VtlParser::GenericFunctionsComponentsContext * /*ctx*/) override { }

  virtual void enterStringFunctionsComponents(VtlParser::StringFunctionsComponentsContext * /*ctx*/) override { }
  virtual void exitStringFunctionsComponents(VtlParser::StringFunctionsComponentsContext * /*ctx*/) override { }

  virtual void enterNumericFunctionsComponents(VtlParser::NumericFunctionsComponentsContext * /*ctx*/) override { }
  virtual void exitNumericFunctionsComponents(VtlParser::NumericFunctionsComponentsContext * /*ctx*/) override { }

  virtual void enterComparisonFunctionsComponents(VtlParser::ComparisonFunctionsComponentsContext * /*ctx*/) override { }
  virtual void exitComparisonFunctionsComponents(VtlParser::ComparisonFunctionsComponentsContext * /*ctx*/) override { }

  virtual void enterTimeFunctionsComponents(VtlParser::TimeFunctionsComponentsContext * /*ctx*/) override { }
  virtual void exitTimeFunctionsComponents(VtlParser::TimeFunctionsComponentsContext * /*ctx*/) override { }

  virtual void enterConditionalFunctionsComponents(VtlParser::ConditionalFunctionsComponentsContext * /*ctx*/) override { }
  virtual void exitConditionalFunctionsComponents(VtlParser::ConditionalFunctionsComponentsContext * /*ctx*/) override { }

  virtual void enterAggregateFunctionsComponents(VtlParser::AggregateFunctionsComponentsContext * /*ctx*/) override { }
  virtual void exitAggregateFunctionsComponents(VtlParser::AggregateFunctionsComponentsContext * /*ctx*/) override { }

  virtual void enterAnalyticFunctionsComponents(VtlParser::AnalyticFunctionsComponentsContext * /*ctx*/) override { }
  virtual void exitAnalyticFunctionsComponents(VtlParser::AnalyticFunctionsComponentsContext * /*ctx*/) override { }

  virtual void enterJoinFunctions(VtlParser::JoinFunctionsContext * /*ctx*/) override { }
  virtual void exitJoinFunctions(VtlParser::JoinFunctionsContext * /*ctx*/) override { }

  virtual void enterGenericFunctions(VtlParser::GenericFunctionsContext * /*ctx*/) override { }
  virtual void exitGenericFunctions(VtlParser::GenericFunctionsContext * /*ctx*/) override { }

  virtual void enterStringFunctions(VtlParser::StringFunctionsContext * /*ctx*/) override { }
  virtual void exitStringFunctions(VtlParser::StringFunctionsContext * /*ctx*/) override { }

  virtual void enterNumericFunctions(VtlParser::NumericFunctionsContext * /*ctx*/) override { }
  virtual void exitNumericFunctions(VtlParser::NumericFunctionsContext * /*ctx*/) override { }

  virtual void enterComparisonFunctions(VtlParser::ComparisonFunctionsContext * /*ctx*/) override { }
  virtual void exitComparisonFunctions(VtlParser::ComparisonFunctionsContext * /*ctx*/) override { }

  virtual void enterTimeFunctions(VtlParser::TimeFunctionsContext * /*ctx*/) override { }
  virtual void exitTimeFunctions(VtlParser::TimeFunctionsContext * /*ctx*/) override { }

  virtual void enterSetFunctions(VtlParser::SetFunctionsContext * /*ctx*/) override { }
  virtual void exitSetFunctions(VtlParser::SetFunctionsContext * /*ctx*/) override { }

  virtual void enterHierarchyFunctions(VtlParser::HierarchyFunctionsContext * /*ctx*/) override { }
  virtual void exitHierarchyFunctions(VtlParser::HierarchyFunctionsContext * /*ctx*/) override { }

  virtual void enterValidationFunctions(VtlParser::ValidationFunctionsContext * /*ctx*/) override { }
  virtual void exitValidationFunctions(VtlParser::ValidationFunctionsContext * /*ctx*/) override { }

  virtual void enterConditionalFunctions(VtlParser::ConditionalFunctionsContext * /*ctx*/) override { }
  virtual void exitConditionalFunctions(VtlParser::ConditionalFunctionsContext * /*ctx*/) override { }

  virtual void enterAggregateFunctions(VtlParser::AggregateFunctionsContext * /*ctx*/) override { }
  virtual void exitAggregateFunctions(VtlParser::AggregateFunctionsContext * /*ctx*/) override { }

  virtual void enterAnalyticFunctions(VtlParser::AnalyticFunctionsContext * /*ctx*/) override { }
  virtual void exitAnalyticFunctions(VtlParser::AnalyticFunctionsContext * /*ctx*/) override { }

  virtual void enterDatasetClause(VtlParser::DatasetClauseContext * /*ctx*/) override { }
  virtual void exitDatasetClause(VtlParser::DatasetClauseContext * /*ctx*/) override { }

  virtual void enterRenameClause(VtlParser::RenameClauseContext * /*ctx*/) override { }
  virtual void exitRenameClause(VtlParser::RenameClauseContext * /*ctx*/) override { }

  virtual void enterAggrClause(VtlParser::AggrClauseContext * /*ctx*/) override { }
  virtual void exitAggrClause(VtlParser::AggrClauseContext * /*ctx*/) override { }

  virtual void enterFilterClause(VtlParser::FilterClauseContext * /*ctx*/) override { }
  virtual void exitFilterClause(VtlParser::FilterClauseContext * /*ctx*/) override { }

  virtual void enterCalcClause(VtlParser::CalcClauseContext * /*ctx*/) override { }
  virtual void exitCalcClause(VtlParser::CalcClauseContext * /*ctx*/) override { }

  virtual void enterKeepOrDropClause(VtlParser::KeepOrDropClauseContext * /*ctx*/) override { }
  virtual void exitKeepOrDropClause(VtlParser::KeepOrDropClauseContext * /*ctx*/) override { }

  virtual void enterPivotOrUnpivotClause(VtlParser::PivotOrUnpivotClauseContext * /*ctx*/) override { }
  virtual void exitPivotOrUnpivotClause(VtlParser::PivotOrUnpivotClauseContext * /*ctx*/) override { }

  virtual void enterCustomPivotClause(VtlParser::CustomPivotClauseContext * /*ctx*/) override { }
  virtual void exitCustomPivotClause(VtlParser::CustomPivotClauseContext * /*ctx*/) override { }

  virtual void enterSubspaceClause(VtlParser::SubspaceClauseContext * /*ctx*/) override { }
  virtual void exitSubspaceClause(VtlParser::SubspaceClauseContext * /*ctx*/) override { }

  virtual void enterJoinExpr(VtlParser::JoinExprContext * /*ctx*/) override { }
  virtual void exitJoinExpr(VtlParser::JoinExprContext * /*ctx*/) override { }

  virtual void enterDefOperator(VtlParser::DefOperatorContext * /*ctx*/) override { }
  virtual void exitDefOperator(VtlParser::DefOperatorContext * /*ctx*/) override { }

  virtual void enterDefDatapointRuleset(VtlParser::DefDatapointRulesetContext * /*ctx*/) override { }
  virtual void exitDefDatapointRuleset(VtlParser::DefDatapointRulesetContext * /*ctx*/) override { }

  virtual void enterDefHierarchical(VtlParser::DefHierarchicalContext * /*ctx*/) override { }
  virtual void exitDefHierarchical(VtlParser::DefHierarchicalContext * /*ctx*/) override { }

  virtual void enterCallDataset(VtlParser::CallDatasetContext * /*ctx*/) override { }
  virtual void exitCallDataset(VtlParser::CallDatasetContext * /*ctx*/) override { }

  virtual void enterEvalAtom(VtlParser::EvalAtomContext * /*ctx*/) override { }
  virtual void exitEvalAtom(VtlParser::EvalAtomContext * /*ctx*/) override { }

  virtual void enterCastExprDataset(VtlParser::CastExprDatasetContext * /*ctx*/) override { }
  virtual void exitCastExprDataset(VtlParser::CastExprDatasetContext * /*ctx*/) override { }

  virtual void enterCallComponent(VtlParser::CallComponentContext * /*ctx*/) override { }
  virtual void exitCallComponent(VtlParser::CallComponentContext * /*ctx*/) override { }

  virtual void enterCastExprComponent(VtlParser::CastExprComponentContext * /*ctx*/) override { }
  virtual void exitCastExprComponent(VtlParser::CastExprComponentContext * /*ctx*/) override { }

  virtual void enterEvalAtomComponent(VtlParser::EvalAtomComponentContext * /*ctx*/) override { }
  virtual void exitEvalAtomComponent(VtlParser::EvalAtomComponentContext * /*ctx*/) override { }

  virtual void enterParameterComponent(VtlParser::ParameterComponentContext * /*ctx*/) override { }
  virtual void exitParameterComponent(VtlParser::ParameterComponentContext * /*ctx*/) override { }

  virtual void enterParameter(VtlParser::ParameterContext * /*ctx*/) override { }
  virtual void exitParameter(VtlParser::ParameterContext * /*ctx*/) override { }

  virtual void enterUnaryStringFunction(VtlParser::UnaryStringFunctionContext * /*ctx*/) override { }
  virtual void exitUnaryStringFunction(VtlParser::UnaryStringFunctionContext * /*ctx*/) override { }

  virtual void enterSubstrAtom(VtlParser::SubstrAtomContext * /*ctx*/) override { }
  virtual void exitSubstrAtom(VtlParser::SubstrAtomContext * /*ctx*/) override { }

  virtual void enterReplaceAtom(VtlParser::ReplaceAtomContext * /*ctx*/) override { }
  virtual void exitReplaceAtom(VtlParser::ReplaceAtomContext * /*ctx*/) override { }

  virtual void enterInstrAtom(VtlParser::InstrAtomContext * /*ctx*/) override { }
  virtual void exitInstrAtom(VtlParser::InstrAtomContext * /*ctx*/) override { }

  virtual void enterUnaryStringFunctionComponent(VtlParser::UnaryStringFunctionComponentContext * /*ctx*/) override { }
  virtual void exitUnaryStringFunctionComponent(VtlParser::UnaryStringFunctionComponentContext * /*ctx*/) override { }

  virtual void enterSubstrAtomComponent(VtlParser::SubstrAtomComponentContext * /*ctx*/) override { }
  virtual void exitSubstrAtomComponent(VtlParser::SubstrAtomComponentContext * /*ctx*/) override { }

  virtual void enterReplaceAtomComponent(VtlParser::ReplaceAtomComponentContext * /*ctx*/) override { }
  virtual void exitReplaceAtomComponent(VtlParser::ReplaceAtomComponentContext * /*ctx*/) override { }

  virtual void enterInstrAtomComponent(VtlParser::InstrAtomComponentContext * /*ctx*/) override { }
  virtual void exitInstrAtomComponent(VtlParser::InstrAtomComponentContext * /*ctx*/) override { }

  virtual void enterUnaryNumeric(VtlParser::UnaryNumericContext * /*ctx*/) override { }
  virtual void exitUnaryNumeric(VtlParser::UnaryNumericContext * /*ctx*/) override { }

  virtual void enterUnaryWithOptionalNumeric(VtlParser::UnaryWithOptionalNumericContext * /*ctx*/) override { }
  virtual void exitUnaryWithOptionalNumeric(VtlParser::UnaryWithOptionalNumericContext * /*ctx*/) override { }

  virtual void enterBinaryNumeric(VtlParser::BinaryNumericContext * /*ctx*/) override { }
  virtual void exitBinaryNumeric(VtlParser::BinaryNumericContext * /*ctx*/) override { }

  virtual void enterUnaryNumericComponent(VtlParser::UnaryNumericComponentContext * /*ctx*/) override { }
  virtual void exitUnaryNumericComponent(VtlParser::UnaryNumericComponentContext * /*ctx*/) override { }

  virtual void enterUnaryWithOptionalNumericComponent(VtlParser::UnaryWithOptionalNumericComponentContext * /*ctx*/) override { }
  virtual void exitUnaryWithOptionalNumericComponent(VtlParser::UnaryWithOptionalNumericComponentContext * /*ctx*/) override { }

  virtual void enterBinaryNumericComponent(VtlParser::BinaryNumericComponentContext * /*ctx*/) override { }
  virtual void exitBinaryNumericComponent(VtlParser::BinaryNumericComponentContext * /*ctx*/) override { }

  virtual void enterBetweenAtom(VtlParser::BetweenAtomContext * /*ctx*/) override { }
  virtual void exitBetweenAtom(VtlParser::BetweenAtomContext * /*ctx*/) override { }

  virtual void enterCharsetMatchAtom(VtlParser::CharsetMatchAtomContext * /*ctx*/) override { }
  virtual void exitCharsetMatchAtom(VtlParser::CharsetMatchAtomContext * /*ctx*/) override { }

  virtual void enterIsNullAtom(VtlParser::IsNullAtomContext * /*ctx*/) override { }
  virtual void exitIsNullAtom(VtlParser::IsNullAtomContext * /*ctx*/) override { }

  virtual void enterExistInAtom(VtlParser::ExistInAtomContext * /*ctx*/) override { }
  virtual void exitExistInAtom(VtlParser::ExistInAtomContext * /*ctx*/) override { }

  virtual void enterBetweenAtomComponent(VtlParser::BetweenAtomComponentContext * /*ctx*/) override { }
  virtual void exitBetweenAtomComponent(VtlParser::BetweenAtomComponentContext * /*ctx*/) override { }

  virtual void enterCharsetMatchAtomComponent(VtlParser::CharsetMatchAtomComponentContext * /*ctx*/) override { }
  virtual void exitCharsetMatchAtomComponent(VtlParser::CharsetMatchAtomComponentContext * /*ctx*/) override { }

  virtual void enterIsNullAtomComponent(VtlParser::IsNullAtomComponentContext * /*ctx*/) override { }
  virtual void exitIsNullAtomComponent(VtlParser::IsNullAtomComponentContext * /*ctx*/) override { }

  virtual void enterPeriodAtom(VtlParser::PeriodAtomContext * /*ctx*/) override { }
  virtual void exitPeriodAtom(VtlParser::PeriodAtomContext * /*ctx*/) override { }

  virtual void enterFillTimeAtom(VtlParser::FillTimeAtomContext * /*ctx*/) override { }
  virtual void exitFillTimeAtom(VtlParser::FillTimeAtomContext * /*ctx*/) override { }

  virtual void enterFlowAtom(VtlParser::FlowAtomContext * /*ctx*/) override { }
  virtual void exitFlowAtom(VtlParser::FlowAtomContext * /*ctx*/) override { }

  virtual void enterTimeShiftAtom(VtlParser::TimeShiftAtomContext * /*ctx*/) override { }
  virtual void exitTimeShiftAtom(VtlParser::TimeShiftAtomContext * /*ctx*/) override { }

  virtual void enterTimeAggAtom(VtlParser::TimeAggAtomContext * /*ctx*/) override { }
  virtual void exitTimeAggAtom(VtlParser::TimeAggAtomContext * /*ctx*/) override { }

  virtual void enterCurrentDateAtom(VtlParser::CurrentDateAtomContext * /*ctx*/) override { }
  virtual void exitCurrentDateAtom(VtlParser::CurrentDateAtomContext * /*ctx*/) override { }

  virtual void enterDateDiffAtom(VtlParser::DateDiffAtomContext * /*ctx*/) override { }
  virtual void exitDateDiffAtom(VtlParser::DateDiffAtomContext * /*ctx*/) override { }

  virtual void enterDateAddAtom(VtlParser::DateAddAtomContext * /*ctx*/) override { }
  virtual void exitDateAddAtom(VtlParser::DateAddAtomContext * /*ctx*/) override { }

  virtual void enterYearAtom(VtlParser::YearAtomContext * /*ctx*/) override { }
  virtual void exitYearAtom(VtlParser::YearAtomContext * /*ctx*/) override { }

  virtual void enterMonthAtom(VtlParser::MonthAtomContext * /*ctx*/) override { }
  virtual void exitMonthAtom(VtlParser::MonthAtomContext * /*ctx*/) override { }

  virtual void enterDayOfMonthAtom(VtlParser::DayOfMonthAtomContext * /*ctx*/) override { }
  virtual void exitDayOfMonthAtom(VtlParser::DayOfMonthAtomContext * /*ctx*/) override { }

  virtual void enterDayOfYearAtom(VtlParser::DayOfYearAtomContext * /*ctx*/) override { }
  virtual void exitDayOfYearAtom(VtlParser::DayOfYearAtomContext * /*ctx*/) override { }

  virtual void enterDayToYearAtom(VtlParser::DayToYearAtomContext * /*ctx*/) override { }
  virtual void exitDayToYearAtom(VtlParser::DayToYearAtomContext * /*ctx*/) override { }

  virtual void enterDayToMonthAtom(VtlParser::DayToMonthAtomContext * /*ctx*/) override { }
  virtual void exitDayToMonthAtom(VtlParser::DayToMonthAtomContext * /*ctx*/) override { }

  virtual void enterYearTodayAtom(VtlParser::YearTodayAtomContext * /*ctx*/) override { }
  virtual void exitYearTodayAtom(VtlParser::YearTodayAtomContext * /*ctx*/) override { }

  virtual void enterMonthTodayAtom(VtlParser::MonthTodayAtomContext * /*ctx*/) override { }
  virtual void exitMonthTodayAtom(VtlParser::MonthTodayAtomContext * /*ctx*/) override { }

  virtual void enterPeriodAtomComponent(VtlParser::PeriodAtomComponentContext * /*ctx*/) override { }
  virtual void exitPeriodAtomComponent(VtlParser::PeriodAtomComponentContext * /*ctx*/) override { }

  virtual void enterFillTimeAtomComponent(VtlParser::FillTimeAtomComponentContext * /*ctx*/) override { }
  virtual void exitFillTimeAtomComponent(VtlParser::FillTimeAtomComponentContext * /*ctx*/) override { }

  virtual void enterFlowAtomComponent(VtlParser::FlowAtomComponentContext * /*ctx*/) override { }
  virtual void exitFlowAtomComponent(VtlParser::FlowAtomComponentContext * /*ctx*/) override { }

  virtual void enterTimeShiftAtomComponent(VtlParser::TimeShiftAtomComponentContext * /*ctx*/) override { }
  virtual void exitTimeShiftAtomComponent(VtlParser::TimeShiftAtomComponentContext * /*ctx*/) override { }

  virtual void enterTimeAggAtomComponent(VtlParser::TimeAggAtomComponentContext * /*ctx*/) override { }
  virtual void exitTimeAggAtomComponent(VtlParser::TimeAggAtomComponentContext * /*ctx*/) override { }

  virtual void enterCurrentDateAtomComponent(VtlParser::CurrentDateAtomComponentContext * /*ctx*/) override { }
  virtual void exitCurrentDateAtomComponent(VtlParser::CurrentDateAtomComponentContext * /*ctx*/) override { }

  virtual void enterDateDiffAtomComponent(VtlParser::DateDiffAtomComponentContext * /*ctx*/) override { }
  virtual void exitDateDiffAtomComponent(VtlParser::DateDiffAtomComponentContext * /*ctx*/) override { }

  virtual void enterDateAddAtomComponent(VtlParser::DateAddAtomComponentContext * /*ctx*/) override { }
  virtual void exitDateAddAtomComponent(VtlParser::DateAddAtomComponentContext * /*ctx*/) override { }

  virtual void enterYearAtomComponent(VtlParser::YearAtomComponentContext * /*ctx*/) override { }
  virtual void exitYearAtomComponent(VtlParser::YearAtomComponentContext * /*ctx*/) override { }

  virtual void enterMonthAtomComponent(VtlParser::MonthAtomComponentContext * /*ctx*/) override { }
  virtual void exitMonthAtomComponent(VtlParser::MonthAtomComponentContext * /*ctx*/) override { }

  virtual void enterDayOfMonthAtomComponent(VtlParser::DayOfMonthAtomComponentContext * /*ctx*/) override { }
  virtual void exitDayOfMonthAtomComponent(VtlParser::DayOfMonthAtomComponentContext * /*ctx*/) override { }

  virtual void enterDatOfYearAtomComponent(VtlParser::DatOfYearAtomComponentContext * /*ctx*/) override { }
  virtual void exitDatOfYearAtomComponent(VtlParser::DatOfYearAtomComponentContext * /*ctx*/) override { }

  virtual void enterDayToYearAtomComponent(VtlParser::DayToYearAtomComponentContext * /*ctx*/) override { }
  virtual void exitDayToYearAtomComponent(VtlParser::DayToYearAtomComponentContext * /*ctx*/) override { }

  virtual void enterDayToMonthAtomComponent(VtlParser::DayToMonthAtomComponentContext * /*ctx*/) override { }
  virtual void exitDayToMonthAtomComponent(VtlParser::DayToMonthAtomComponentContext * /*ctx*/) override { }

  virtual void enterYearTodayAtomComponent(VtlParser::YearTodayAtomComponentContext * /*ctx*/) override { }
  virtual void exitYearTodayAtomComponent(VtlParser::YearTodayAtomComponentContext * /*ctx*/) override { }

  virtual void enterMonthTodayAtomComponent(VtlParser::MonthTodayAtomComponentContext * /*ctx*/) override { }
  virtual void exitMonthTodayAtomComponent(VtlParser::MonthTodayAtomComponentContext * /*ctx*/) override { }

  virtual void enterUnionAtom(VtlParser::UnionAtomContext * /*ctx*/) override { }
  virtual void exitUnionAtom(VtlParser::UnionAtomContext * /*ctx*/) override { }

  virtual void enterIntersectAtom(VtlParser::IntersectAtomContext * /*ctx*/) override { }
  virtual void exitIntersectAtom(VtlParser::IntersectAtomContext * /*ctx*/) override { }

  virtual void enterSetOrSYmDiffAtom(VtlParser::SetOrSYmDiffAtomContext * /*ctx*/) override { }
  virtual void exitSetOrSYmDiffAtom(VtlParser::SetOrSYmDiffAtomContext * /*ctx*/) override { }

  virtual void enterHierarchyOperators(VtlParser::HierarchyOperatorsContext * /*ctx*/) override { }
  virtual void exitHierarchyOperators(VtlParser::HierarchyOperatorsContext * /*ctx*/) override { }

  virtual void enterValidateDPruleset(VtlParser::ValidateDPrulesetContext * /*ctx*/) override { }
  virtual void exitValidateDPruleset(VtlParser::ValidateDPrulesetContext * /*ctx*/) override { }

  virtual void enterValidateHRruleset(VtlParser::ValidateHRrulesetContext * /*ctx*/) override { }
  virtual void exitValidateHRruleset(VtlParser::ValidateHRrulesetContext * /*ctx*/) override { }

  virtual void enterValidationSimple(VtlParser::ValidationSimpleContext * /*ctx*/) override { }
  virtual void exitValidationSimple(VtlParser::ValidationSimpleContext * /*ctx*/) override { }

  virtual void enterNvlAtom(VtlParser::NvlAtomContext * /*ctx*/) override { }
  virtual void exitNvlAtom(VtlParser::NvlAtomContext * /*ctx*/) override { }

  virtual void enterNvlAtomComponent(VtlParser::NvlAtomComponentContext * /*ctx*/) override { }
  virtual void exitNvlAtomComponent(VtlParser::NvlAtomComponentContext * /*ctx*/) override { }

  virtual void enterAggrComp(VtlParser::AggrCompContext * /*ctx*/) override { }
  virtual void exitAggrComp(VtlParser::AggrCompContext * /*ctx*/) override { }

  virtual void enterCountAggrComp(VtlParser::CountAggrCompContext * /*ctx*/) override { }
  virtual void exitCountAggrComp(VtlParser::CountAggrCompContext * /*ctx*/) override { }

  virtual void enterAggrDataset(VtlParser::AggrDatasetContext * /*ctx*/) override { }
  virtual void exitAggrDataset(VtlParser::AggrDatasetContext * /*ctx*/) override { }

  virtual void enterAnSimpleFunction(VtlParser::AnSimpleFunctionContext * /*ctx*/) override { }
  virtual void exitAnSimpleFunction(VtlParser::AnSimpleFunctionContext * /*ctx*/) override { }

  virtual void enterLagOrLeadAn(VtlParser::LagOrLeadAnContext * /*ctx*/) override { }
  virtual void exitLagOrLeadAn(VtlParser::LagOrLeadAnContext * /*ctx*/) override { }

  virtual void enterRatioToReportAn(VtlParser::RatioToReportAnContext * /*ctx*/) override { }
  virtual void exitRatioToReportAn(VtlParser::RatioToReportAnContext * /*ctx*/) override { }

  virtual void enterAnSimpleFunctionComponent(VtlParser::AnSimpleFunctionComponentContext * /*ctx*/) override { }
  virtual void exitAnSimpleFunctionComponent(VtlParser::AnSimpleFunctionComponentContext * /*ctx*/) override { }

  virtual void enterLagOrLeadAnComponent(VtlParser::LagOrLeadAnComponentContext * /*ctx*/) override { }
  virtual void exitLagOrLeadAnComponent(VtlParser::LagOrLeadAnComponentContext * /*ctx*/) override { }

  virtual void enterRankAnComponent(VtlParser::RankAnComponentContext * /*ctx*/) override { }
  virtual void exitRankAnComponent(VtlParser::RankAnComponentContext * /*ctx*/) override { }

  virtual void enterRatioToReportAnComponent(VtlParser::RatioToReportAnComponentContext * /*ctx*/) override { }
  virtual void exitRatioToReportAnComponent(VtlParser::RatioToReportAnComponentContext * /*ctx*/) override { }

  virtual void enterRenameClauseItem(VtlParser::RenameClauseItemContext * /*ctx*/) override { }
  virtual void exitRenameClauseItem(VtlParser::RenameClauseItemContext * /*ctx*/) override { }

  virtual void enterAggregateClause(VtlParser::AggregateClauseContext * /*ctx*/) override { }
  virtual void exitAggregateClause(VtlParser::AggregateClauseContext * /*ctx*/) override { }

  virtual void enterAggrFunctionClause(VtlParser::AggrFunctionClauseContext * /*ctx*/) override { }
  virtual void exitAggrFunctionClause(VtlParser::AggrFunctionClauseContext * /*ctx*/) override { }

  virtual void enterCalcClauseItem(VtlParser::CalcClauseItemContext * /*ctx*/) override { }
  virtual void exitCalcClauseItem(VtlParser::CalcClauseItemContext * /*ctx*/) override { }

  virtual void enterSubspaceClauseItem(VtlParser::SubspaceClauseItemContext * /*ctx*/) override { }
  virtual void exitSubspaceClauseItem(VtlParser::SubspaceClauseItemContext * /*ctx*/) override { }

  virtual void enterSimpleScalar(VtlParser::SimpleScalarContext * /*ctx*/) override { }
  virtual void exitSimpleScalar(VtlParser::SimpleScalarContext * /*ctx*/) override { }

  virtual void enterScalarWithCast(VtlParser::ScalarWithCastContext * /*ctx*/) override { }
  virtual void exitScalarWithCast(VtlParser::ScalarWithCastContext * /*ctx*/) override { }

  virtual void enterJoinClauseWithoutUsing(VtlParser::JoinClauseWithoutUsingContext * /*ctx*/) override { }
  virtual void exitJoinClauseWithoutUsing(VtlParser::JoinClauseWithoutUsingContext * /*ctx*/) override { }

  virtual void enterJoinClause(VtlParser::JoinClauseContext * /*ctx*/) override { }
  virtual void exitJoinClause(VtlParser::JoinClauseContext * /*ctx*/) override { }

  virtual void enterJoinClauseItem(VtlParser::JoinClauseItemContext * /*ctx*/) override { }
  virtual void exitJoinClauseItem(VtlParser::JoinClauseItemContext * /*ctx*/) override { }

  virtual void enterJoinBody(VtlParser::JoinBodyContext * /*ctx*/) override { }
  virtual void exitJoinBody(VtlParser::JoinBodyContext * /*ctx*/) override { }

  virtual void enterJoinApplyClause(VtlParser::JoinApplyClauseContext * /*ctx*/) override { }
  virtual void exitJoinApplyClause(VtlParser::JoinApplyClauseContext * /*ctx*/) override { }

  virtual void enterPartitionByClause(VtlParser::PartitionByClauseContext * /*ctx*/) override { }
  virtual void exitPartitionByClause(VtlParser::PartitionByClauseContext * /*ctx*/) override { }

  virtual void enterOrderByClause(VtlParser::OrderByClauseContext * /*ctx*/) override { }
  virtual void exitOrderByClause(VtlParser::OrderByClauseContext * /*ctx*/) override { }

  virtual void enterOrderByItem(VtlParser::OrderByItemContext * /*ctx*/) override { }
  virtual void exitOrderByItem(VtlParser::OrderByItemContext * /*ctx*/) override { }

  virtual void enterWindowingClause(VtlParser::WindowingClauseContext * /*ctx*/) override { }
  virtual void exitWindowingClause(VtlParser::WindowingClauseContext * /*ctx*/) override { }

  virtual void enterSignedInteger(VtlParser::SignedIntegerContext * /*ctx*/) override { }
  virtual void exitSignedInteger(VtlParser::SignedIntegerContext * /*ctx*/) override { }

  virtual void enterSignedNumber(VtlParser::SignedNumberContext * /*ctx*/) override { }
  virtual void exitSignedNumber(VtlParser::SignedNumberContext * /*ctx*/) override { }

  virtual void enterLimitClauseItem(VtlParser::LimitClauseItemContext * /*ctx*/) override { }
  virtual void exitLimitClauseItem(VtlParser::LimitClauseItemContext * /*ctx*/) override { }

  virtual void enterGroupByOrExcept(VtlParser::GroupByOrExceptContext * /*ctx*/) override { }
  virtual void exitGroupByOrExcept(VtlParser::GroupByOrExceptContext * /*ctx*/) override { }

  virtual void enterGroupAll(VtlParser::GroupAllContext * /*ctx*/) override { }
  virtual void exitGroupAll(VtlParser::GroupAllContext * /*ctx*/) override { }

  virtual void enterHavingClause(VtlParser::HavingClauseContext * /*ctx*/) override { }
  virtual void exitHavingClause(VtlParser::HavingClauseContext * /*ctx*/) override { }

  virtual void enterParameterItem(VtlParser::ParameterItemContext * /*ctx*/) override { }
  virtual void exitParameterItem(VtlParser::ParameterItemContext * /*ctx*/) override { }

  virtual void enterOutputParameterType(VtlParser::OutputParameterTypeContext * /*ctx*/) override { }
  virtual void exitOutputParameterType(VtlParser::OutputParameterTypeContext * /*ctx*/) override { }

  virtual void enterOutputParameterTypeComponent(VtlParser::OutputParameterTypeComponentContext * /*ctx*/) override { }
  virtual void exitOutputParameterTypeComponent(VtlParser::OutputParameterTypeComponentContext * /*ctx*/) override { }

  virtual void enterInputParameterType(VtlParser::InputParameterTypeContext * /*ctx*/) override { }
  virtual void exitInputParameterType(VtlParser::InputParameterTypeContext * /*ctx*/) override { }

  virtual void enterRulesetType(VtlParser::RulesetTypeContext * /*ctx*/) override { }
  virtual void exitRulesetType(VtlParser::RulesetTypeContext * /*ctx*/) override { }

  virtual void enterScalarType(VtlParser::ScalarTypeContext * /*ctx*/) override { }
  virtual void exitScalarType(VtlParser::ScalarTypeContext * /*ctx*/) override { }

  virtual void enterComponentType(VtlParser::ComponentTypeContext * /*ctx*/) override { }
  virtual void exitComponentType(VtlParser::ComponentTypeContext * /*ctx*/) override { }

  virtual void enterDatasetType(VtlParser::DatasetTypeContext * /*ctx*/) override { }
  virtual void exitDatasetType(VtlParser::DatasetTypeContext * /*ctx*/) override { }

  virtual void enterEvalDatasetType(VtlParser::EvalDatasetTypeContext * /*ctx*/) override { }
  virtual void exitEvalDatasetType(VtlParser::EvalDatasetTypeContext * /*ctx*/) override { }

  virtual void enterScalarSetType(VtlParser::ScalarSetTypeContext * /*ctx*/) override { }
  virtual void exitScalarSetType(VtlParser::ScalarSetTypeContext * /*ctx*/) override { }

  virtual void enterDataPoint(VtlParser::DataPointContext * /*ctx*/) override { }
  virtual void exitDataPoint(VtlParser::DataPointContext * /*ctx*/) override { }

  virtual void enterDataPointVd(VtlParser::DataPointVdContext * /*ctx*/) override { }
  virtual void exitDataPointVd(VtlParser::DataPointVdContext * /*ctx*/) override { }

  virtual void enterDataPointVar(VtlParser::DataPointVarContext * /*ctx*/) override { }
  virtual void exitDataPointVar(VtlParser::DataPointVarContext * /*ctx*/) override { }

  virtual void enterHrRulesetType(VtlParser::HrRulesetTypeContext * /*ctx*/) override { }
  virtual void exitHrRulesetType(VtlParser::HrRulesetTypeContext * /*ctx*/) override { }

  virtual void enterHrRulesetVdType(VtlParser::HrRulesetVdTypeContext * /*ctx*/) override { }
  virtual void exitHrRulesetVdType(VtlParser::HrRulesetVdTypeContext * /*ctx*/) override { }

  virtual void enterHrRulesetVarType(VtlParser::HrRulesetVarTypeContext * /*ctx*/) override { }
  virtual void exitHrRulesetVarType(VtlParser::HrRulesetVarTypeContext * /*ctx*/) override { }

  virtual void enterValueDomainName(VtlParser::ValueDomainNameContext * /*ctx*/) override { }
  virtual void exitValueDomainName(VtlParser::ValueDomainNameContext * /*ctx*/) override { }

  virtual void enterRulesetID(VtlParser::RulesetIDContext * /*ctx*/) override { }
  virtual void exitRulesetID(VtlParser::RulesetIDContext * /*ctx*/) override { }

  virtual void enterRulesetSignature(VtlParser::RulesetSignatureContext * /*ctx*/) override { }
  virtual void exitRulesetSignature(VtlParser::RulesetSignatureContext * /*ctx*/) override { }

  virtual void enterSignature(VtlParser::SignatureContext * /*ctx*/) override { }
  virtual void exitSignature(VtlParser::SignatureContext * /*ctx*/) override { }

  virtual void enterRuleClauseDatapoint(VtlParser::RuleClauseDatapointContext * /*ctx*/) override { }
  virtual void exitRuleClauseDatapoint(VtlParser::RuleClauseDatapointContext * /*ctx*/) override { }

  virtual void enterRuleItemDatapoint(VtlParser::RuleItemDatapointContext * /*ctx*/) override { }
  virtual void exitRuleItemDatapoint(VtlParser::RuleItemDatapointContext * /*ctx*/) override { }

  virtual void enterRuleClauseHierarchical(VtlParser::RuleClauseHierarchicalContext * /*ctx*/) override { }
  virtual void exitRuleClauseHierarchical(VtlParser::RuleClauseHierarchicalContext * /*ctx*/) override { }

  virtual void enterRuleItemHierarchical(VtlParser::RuleItemHierarchicalContext * /*ctx*/) override { }
  virtual void exitRuleItemHierarchical(VtlParser::RuleItemHierarchicalContext * /*ctx*/) override { }

  virtual void enterHierRuleSignature(VtlParser::HierRuleSignatureContext * /*ctx*/) override { }
  virtual void exitHierRuleSignature(VtlParser::HierRuleSignatureContext * /*ctx*/) override { }

  virtual void enterValueDomainSignature(VtlParser::ValueDomainSignatureContext * /*ctx*/) override { }
  virtual void exitValueDomainSignature(VtlParser::ValueDomainSignatureContext * /*ctx*/) override { }

  virtual void enterCodeItemRelation(VtlParser::CodeItemRelationContext * /*ctx*/) override { }
  virtual void exitCodeItemRelation(VtlParser::CodeItemRelationContext * /*ctx*/) override { }

  virtual void enterCodeItemRelationClause(VtlParser::CodeItemRelationClauseContext * /*ctx*/) override { }
  virtual void exitCodeItemRelationClause(VtlParser::CodeItemRelationClauseContext * /*ctx*/) override { }

  virtual void enterValueDomainValue(VtlParser::ValueDomainValueContext * /*ctx*/) override { }
  virtual void exitValueDomainValue(VtlParser::ValueDomainValueContext * /*ctx*/) override { }

  virtual void enterConditionConstraint(VtlParser::ConditionConstraintContext * /*ctx*/) override { }
  virtual void exitConditionConstraint(VtlParser::ConditionConstraintContext * /*ctx*/) override { }

  virtual void enterRangeConstraint(VtlParser::RangeConstraintContext * /*ctx*/) override { }
  virtual void exitRangeConstraint(VtlParser::RangeConstraintContext * /*ctx*/) override { }

  virtual void enterCompConstraint(VtlParser::CompConstraintContext * /*ctx*/) override { }
  virtual void exitCompConstraint(VtlParser::CompConstraintContext * /*ctx*/) override { }

  virtual void enterMultModifier(VtlParser::MultModifierContext * /*ctx*/) override { }
  virtual void exitMultModifier(VtlParser::MultModifierContext * /*ctx*/) override { }

  virtual void enterValidationOutput(VtlParser::ValidationOutputContext * /*ctx*/) override { }
  virtual void exitValidationOutput(VtlParser::ValidationOutputContext * /*ctx*/) override { }

  virtual void enterValidationMode(VtlParser::ValidationModeContext * /*ctx*/) override { }
  virtual void exitValidationMode(VtlParser::ValidationModeContext * /*ctx*/) override { }

  virtual void enterConditionClause(VtlParser::ConditionClauseContext * /*ctx*/) override { }
  virtual void exitConditionClause(VtlParser::ConditionClauseContext * /*ctx*/) override { }

  virtual void enterInputMode(VtlParser::InputModeContext * /*ctx*/) override { }
  virtual void exitInputMode(VtlParser::InputModeContext * /*ctx*/) override { }

  virtual void enterImbalanceExpr(VtlParser::ImbalanceExprContext * /*ctx*/) override { }
  virtual void exitImbalanceExpr(VtlParser::ImbalanceExprContext * /*ctx*/) override { }

  virtual void enterInputModeHierarchy(VtlParser::InputModeHierarchyContext * /*ctx*/) override { }
  virtual void exitInputModeHierarchy(VtlParser::InputModeHierarchyContext * /*ctx*/) override { }

  virtual void enterOutputModeHierarchy(VtlParser::OutputModeHierarchyContext * /*ctx*/) override { }
  virtual void exitOutputModeHierarchy(VtlParser::OutputModeHierarchyContext * /*ctx*/) override { }

  virtual void enterAlias(VtlParser::AliasContext * /*ctx*/) override { }
  virtual void exitAlias(VtlParser::AliasContext * /*ctx*/) override { }

  virtual void enterVarID(VtlParser::VarIDContext * /*ctx*/) override { }
  virtual void exitVarID(VtlParser::VarIDContext * /*ctx*/) override { }

  virtual void enterSimpleComponentId(VtlParser::SimpleComponentIdContext * /*ctx*/) override { }
  virtual void exitSimpleComponentId(VtlParser::SimpleComponentIdContext * /*ctx*/) override { }

  virtual void enterComponentID(VtlParser::ComponentIDContext * /*ctx*/) override { }
  virtual void exitComponentID(VtlParser::ComponentIDContext * /*ctx*/) override { }

  virtual void enterLists(VtlParser::ListsContext * /*ctx*/) override { }
  virtual void exitLists(VtlParser::ListsContext * /*ctx*/) override { }

  virtual void enterErCode(VtlParser::ErCodeContext * /*ctx*/) override { }
  virtual void exitErCode(VtlParser::ErCodeContext * /*ctx*/) override { }

  virtual void enterErLevel(VtlParser::ErLevelContext * /*ctx*/) override { }
  virtual void exitErLevel(VtlParser::ErLevelContext * /*ctx*/) override { }

  virtual void enterComparisonOperand(VtlParser::ComparisonOperandContext * /*ctx*/) override { }
  virtual void exitComparisonOperand(VtlParser::ComparisonOperandContext * /*ctx*/) override { }

  virtual void enterOptionalExpr(VtlParser::OptionalExprContext * /*ctx*/) override { }
  virtual void exitOptionalExpr(VtlParser::OptionalExprContext * /*ctx*/) override { }

  virtual void enterOptionalExprComponent(VtlParser::OptionalExprComponentContext * /*ctx*/) override { }
  virtual void exitOptionalExprComponent(VtlParser::OptionalExprComponentContext * /*ctx*/) override { }

  virtual void enterComponentRole(VtlParser::ComponentRoleContext * /*ctx*/) override { }
  virtual void exitComponentRole(VtlParser::ComponentRoleContext * /*ctx*/) override { }

  virtual void enterViralAttribute(VtlParser::ViralAttributeContext * /*ctx*/) override { }
  virtual void exitViralAttribute(VtlParser::ViralAttributeContext * /*ctx*/) override { }

  virtual void enterValueDomainID(VtlParser::ValueDomainIDContext * /*ctx*/) override { }
  virtual void exitValueDomainID(VtlParser::ValueDomainIDContext * /*ctx*/) override { }

  virtual void enterOperatorID(VtlParser::OperatorIDContext * /*ctx*/) override { }
  virtual void exitOperatorID(VtlParser::OperatorIDContext * /*ctx*/) override { }

  virtual void enterRoutineName(VtlParser::RoutineNameContext * /*ctx*/) override { }
  virtual void exitRoutineName(VtlParser::RoutineNameContext * /*ctx*/) override { }

  virtual void enterConstant(VtlParser::ConstantContext * /*ctx*/) override { }
  virtual void exitConstant(VtlParser::ConstantContext * /*ctx*/) override { }

  virtual void enterBasicScalarType(VtlParser::BasicScalarTypeContext * /*ctx*/) override { }
  virtual void exitBasicScalarType(VtlParser::BasicScalarTypeContext * /*ctx*/) override { }

  virtual void enterRetainType(VtlParser::RetainTypeContext * /*ctx*/) override { }
  virtual void exitRetainType(VtlParser::RetainTypeContext * /*ctx*/) override { }


  virtual void enterEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void exitEveryRule(antlr4::ParserRuleContext * /*ctx*/) override { }
  virtual void visitTerminal(antlr4::tree::TerminalNode * /*node*/) override { }
  virtual void visitErrorNode(antlr4::tree::ErrorNode * /*node*/) override { }

};

