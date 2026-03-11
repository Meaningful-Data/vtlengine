
// Generated from /home/javier/Programacion/vtlengine/src/vtlengine/AST/Grammar/Vtl.g4 by ANTLR 4.13.1

#pragma once


#include "antlr4-runtime.h"
#include "VtlParser.h"



/**
 * This class defines an abstract visitor for a parse tree
 * produced by VtlParser.
 */
class  VtlVisitor : public antlr4::tree::AbstractParseTreeVisitor {
public:

  /**
   * Visit parse trees produced by VtlParser.
   */
    virtual std::any visitStart(VtlParser::StartContext *context) = 0;

    virtual std::any visitTemporaryAssignment(VtlParser::TemporaryAssignmentContext *context) = 0;

    virtual std::any visitPersistAssignment(VtlParser::PersistAssignmentContext *context) = 0;

    virtual std::any visitDefineExpression(VtlParser::DefineExpressionContext *context) = 0;

    virtual std::any visitVarIdExpr(VtlParser::VarIdExprContext *context) = 0;

    virtual std::any visitMembershipExpr(VtlParser::MembershipExprContext *context) = 0;

    virtual std::any visitInNotInExpr(VtlParser::InNotInExprContext *context) = 0;

    virtual std::any visitBooleanExpr(VtlParser::BooleanExprContext *context) = 0;

    virtual std::any visitComparisonExpr(VtlParser::ComparisonExprContext *context) = 0;

    virtual std::any visitUnaryExpr(VtlParser::UnaryExprContext *context) = 0;

    virtual std::any visitFunctionsExpression(VtlParser::FunctionsExpressionContext *context) = 0;

    virtual std::any visitIfExpr(VtlParser::IfExprContext *context) = 0;

    virtual std::any visitClauseExpr(VtlParser::ClauseExprContext *context) = 0;

    virtual std::any visitCaseExpr(VtlParser::CaseExprContext *context) = 0;

    virtual std::any visitArithmeticExpr(VtlParser::ArithmeticExprContext *context) = 0;

    virtual std::any visitParenthesisExpr(VtlParser::ParenthesisExprContext *context) = 0;

    virtual std::any visitConstantExpr(VtlParser::ConstantExprContext *context) = 0;

    virtual std::any visitArithmeticExprOrConcat(VtlParser::ArithmeticExprOrConcatContext *context) = 0;

    virtual std::any visitArithmeticExprComp(VtlParser::ArithmeticExprCompContext *context) = 0;

    virtual std::any visitIfExprComp(VtlParser::IfExprCompContext *context) = 0;

    virtual std::any visitComparisonExprComp(VtlParser::ComparisonExprCompContext *context) = 0;

    virtual std::any visitFunctionsExpressionComp(VtlParser::FunctionsExpressionCompContext *context) = 0;

    virtual std::any visitCompId(VtlParser::CompIdContext *context) = 0;

    virtual std::any visitConstantExprComp(VtlParser::ConstantExprCompContext *context) = 0;

    virtual std::any visitArithmeticExprOrConcatComp(VtlParser::ArithmeticExprOrConcatCompContext *context) = 0;

    virtual std::any visitParenthesisExprComp(VtlParser::ParenthesisExprCompContext *context) = 0;

    virtual std::any visitInNotInExprComp(VtlParser::InNotInExprCompContext *context) = 0;

    virtual std::any visitUnaryExprComp(VtlParser::UnaryExprCompContext *context) = 0;

    virtual std::any visitCaseExprComp(VtlParser::CaseExprCompContext *context) = 0;

    virtual std::any visitBooleanExprComp(VtlParser::BooleanExprCompContext *context) = 0;

    virtual std::any visitGenericFunctionsComponents(VtlParser::GenericFunctionsComponentsContext *context) = 0;

    virtual std::any visitStringFunctionsComponents(VtlParser::StringFunctionsComponentsContext *context) = 0;

    virtual std::any visitNumericFunctionsComponents(VtlParser::NumericFunctionsComponentsContext *context) = 0;

    virtual std::any visitComparisonFunctionsComponents(VtlParser::ComparisonFunctionsComponentsContext *context) = 0;

    virtual std::any visitTimeFunctionsComponents(VtlParser::TimeFunctionsComponentsContext *context) = 0;

    virtual std::any visitConditionalFunctionsComponents(VtlParser::ConditionalFunctionsComponentsContext *context) = 0;

    virtual std::any visitAggregateFunctionsComponents(VtlParser::AggregateFunctionsComponentsContext *context) = 0;

    virtual std::any visitAnalyticFunctionsComponents(VtlParser::AnalyticFunctionsComponentsContext *context) = 0;

    virtual std::any visitJoinFunctions(VtlParser::JoinFunctionsContext *context) = 0;

    virtual std::any visitGenericFunctions(VtlParser::GenericFunctionsContext *context) = 0;

    virtual std::any visitStringFunctions(VtlParser::StringFunctionsContext *context) = 0;

    virtual std::any visitNumericFunctions(VtlParser::NumericFunctionsContext *context) = 0;

    virtual std::any visitComparisonFunctions(VtlParser::ComparisonFunctionsContext *context) = 0;

    virtual std::any visitTimeFunctions(VtlParser::TimeFunctionsContext *context) = 0;

    virtual std::any visitSetFunctions(VtlParser::SetFunctionsContext *context) = 0;

    virtual std::any visitHierarchyFunctions(VtlParser::HierarchyFunctionsContext *context) = 0;

    virtual std::any visitValidationFunctions(VtlParser::ValidationFunctionsContext *context) = 0;

    virtual std::any visitConditionalFunctions(VtlParser::ConditionalFunctionsContext *context) = 0;

    virtual std::any visitAggregateFunctions(VtlParser::AggregateFunctionsContext *context) = 0;

    virtual std::any visitAnalyticFunctions(VtlParser::AnalyticFunctionsContext *context) = 0;

    virtual std::any visitDatasetClause(VtlParser::DatasetClauseContext *context) = 0;

    virtual std::any visitRenameClause(VtlParser::RenameClauseContext *context) = 0;

    virtual std::any visitAggrClause(VtlParser::AggrClauseContext *context) = 0;

    virtual std::any visitFilterClause(VtlParser::FilterClauseContext *context) = 0;

    virtual std::any visitCalcClause(VtlParser::CalcClauseContext *context) = 0;

    virtual std::any visitKeepOrDropClause(VtlParser::KeepOrDropClauseContext *context) = 0;

    virtual std::any visitPivotOrUnpivotClause(VtlParser::PivotOrUnpivotClauseContext *context) = 0;

    virtual std::any visitCustomPivotClause(VtlParser::CustomPivotClauseContext *context) = 0;

    virtual std::any visitSubspaceClause(VtlParser::SubspaceClauseContext *context) = 0;

    virtual std::any visitJoinExpr(VtlParser::JoinExprContext *context) = 0;

    virtual std::any visitDefOperator(VtlParser::DefOperatorContext *context) = 0;

    virtual std::any visitDefDatapointRuleset(VtlParser::DefDatapointRulesetContext *context) = 0;

    virtual std::any visitDefHierarchical(VtlParser::DefHierarchicalContext *context) = 0;

    virtual std::any visitCallDataset(VtlParser::CallDatasetContext *context) = 0;

    virtual std::any visitEvalAtom(VtlParser::EvalAtomContext *context) = 0;

    virtual std::any visitCastExprDataset(VtlParser::CastExprDatasetContext *context) = 0;

    virtual std::any visitCallComponent(VtlParser::CallComponentContext *context) = 0;

    virtual std::any visitCastExprComponent(VtlParser::CastExprComponentContext *context) = 0;

    virtual std::any visitEvalAtomComponent(VtlParser::EvalAtomComponentContext *context) = 0;

    virtual std::any visitParameterComponent(VtlParser::ParameterComponentContext *context) = 0;

    virtual std::any visitParameter(VtlParser::ParameterContext *context) = 0;

    virtual std::any visitUnaryStringFunction(VtlParser::UnaryStringFunctionContext *context) = 0;

    virtual std::any visitSubstrAtom(VtlParser::SubstrAtomContext *context) = 0;

    virtual std::any visitReplaceAtom(VtlParser::ReplaceAtomContext *context) = 0;

    virtual std::any visitInstrAtom(VtlParser::InstrAtomContext *context) = 0;

    virtual std::any visitUnaryStringFunctionComponent(VtlParser::UnaryStringFunctionComponentContext *context) = 0;

    virtual std::any visitSubstrAtomComponent(VtlParser::SubstrAtomComponentContext *context) = 0;

    virtual std::any visitReplaceAtomComponent(VtlParser::ReplaceAtomComponentContext *context) = 0;

    virtual std::any visitInstrAtomComponent(VtlParser::InstrAtomComponentContext *context) = 0;

    virtual std::any visitUnaryNumeric(VtlParser::UnaryNumericContext *context) = 0;

    virtual std::any visitUnaryWithOptionalNumeric(VtlParser::UnaryWithOptionalNumericContext *context) = 0;

    virtual std::any visitBinaryNumeric(VtlParser::BinaryNumericContext *context) = 0;

    virtual std::any visitUnaryNumericComponent(VtlParser::UnaryNumericComponentContext *context) = 0;

    virtual std::any visitUnaryWithOptionalNumericComponent(VtlParser::UnaryWithOptionalNumericComponentContext *context) = 0;

    virtual std::any visitBinaryNumericComponent(VtlParser::BinaryNumericComponentContext *context) = 0;

    virtual std::any visitBetweenAtom(VtlParser::BetweenAtomContext *context) = 0;

    virtual std::any visitCharsetMatchAtom(VtlParser::CharsetMatchAtomContext *context) = 0;

    virtual std::any visitIsNullAtom(VtlParser::IsNullAtomContext *context) = 0;

    virtual std::any visitExistInAtom(VtlParser::ExistInAtomContext *context) = 0;

    virtual std::any visitBetweenAtomComponent(VtlParser::BetweenAtomComponentContext *context) = 0;

    virtual std::any visitCharsetMatchAtomComponent(VtlParser::CharsetMatchAtomComponentContext *context) = 0;

    virtual std::any visitIsNullAtomComponent(VtlParser::IsNullAtomComponentContext *context) = 0;

    virtual std::any visitPeriodAtom(VtlParser::PeriodAtomContext *context) = 0;

    virtual std::any visitFillTimeAtom(VtlParser::FillTimeAtomContext *context) = 0;

    virtual std::any visitFlowAtom(VtlParser::FlowAtomContext *context) = 0;

    virtual std::any visitTimeShiftAtom(VtlParser::TimeShiftAtomContext *context) = 0;

    virtual std::any visitTimeAggAtom(VtlParser::TimeAggAtomContext *context) = 0;

    virtual std::any visitCurrentDateAtom(VtlParser::CurrentDateAtomContext *context) = 0;

    virtual std::any visitDateDiffAtom(VtlParser::DateDiffAtomContext *context) = 0;

    virtual std::any visitDateAddAtom(VtlParser::DateAddAtomContext *context) = 0;

    virtual std::any visitYearAtom(VtlParser::YearAtomContext *context) = 0;

    virtual std::any visitMonthAtom(VtlParser::MonthAtomContext *context) = 0;

    virtual std::any visitDayOfMonthAtom(VtlParser::DayOfMonthAtomContext *context) = 0;

    virtual std::any visitDayOfYearAtom(VtlParser::DayOfYearAtomContext *context) = 0;

    virtual std::any visitDayToYearAtom(VtlParser::DayToYearAtomContext *context) = 0;

    virtual std::any visitDayToMonthAtom(VtlParser::DayToMonthAtomContext *context) = 0;

    virtual std::any visitYearTodayAtom(VtlParser::YearTodayAtomContext *context) = 0;

    virtual std::any visitMonthTodayAtom(VtlParser::MonthTodayAtomContext *context) = 0;

    virtual std::any visitPeriodAtomComponent(VtlParser::PeriodAtomComponentContext *context) = 0;

    virtual std::any visitFillTimeAtomComponent(VtlParser::FillTimeAtomComponentContext *context) = 0;

    virtual std::any visitFlowAtomComponent(VtlParser::FlowAtomComponentContext *context) = 0;

    virtual std::any visitTimeShiftAtomComponent(VtlParser::TimeShiftAtomComponentContext *context) = 0;

    virtual std::any visitTimeAggAtomComponent(VtlParser::TimeAggAtomComponentContext *context) = 0;

    virtual std::any visitCurrentDateAtomComponent(VtlParser::CurrentDateAtomComponentContext *context) = 0;

    virtual std::any visitDateDiffAtomComponent(VtlParser::DateDiffAtomComponentContext *context) = 0;

    virtual std::any visitDateAddAtomComponent(VtlParser::DateAddAtomComponentContext *context) = 0;

    virtual std::any visitYearAtomComponent(VtlParser::YearAtomComponentContext *context) = 0;

    virtual std::any visitMonthAtomComponent(VtlParser::MonthAtomComponentContext *context) = 0;

    virtual std::any visitDayOfMonthAtomComponent(VtlParser::DayOfMonthAtomComponentContext *context) = 0;

    virtual std::any visitDatOfYearAtomComponent(VtlParser::DatOfYearAtomComponentContext *context) = 0;

    virtual std::any visitDayToYearAtomComponent(VtlParser::DayToYearAtomComponentContext *context) = 0;

    virtual std::any visitDayToMonthAtomComponent(VtlParser::DayToMonthAtomComponentContext *context) = 0;

    virtual std::any visitYearTodayAtomComponent(VtlParser::YearTodayAtomComponentContext *context) = 0;

    virtual std::any visitMonthTodayAtomComponent(VtlParser::MonthTodayAtomComponentContext *context) = 0;

    virtual std::any visitUnionAtom(VtlParser::UnionAtomContext *context) = 0;

    virtual std::any visitIntersectAtom(VtlParser::IntersectAtomContext *context) = 0;

    virtual std::any visitSetOrSYmDiffAtom(VtlParser::SetOrSYmDiffAtomContext *context) = 0;

    virtual std::any visitHierarchyOperators(VtlParser::HierarchyOperatorsContext *context) = 0;

    virtual std::any visitValidateDPruleset(VtlParser::ValidateDPrulesetContext *context) = 0;

    virtual std::any visitValidateHRruleset(VtlParser::ValidateHRrulesetContext *context) = 0;

    virtual std::any visitValidationSimple(VtlParser::ValidationSimpleContext *context) = 0;

    virtual std::any visitNvlAtom(VtlParser::NvlAtomContext *context) = 0;

    virtual std::any visitNvlAtomComponent(VtlParser::NvlAtomComponentContext *context) = 0;

    virtual std::any visitAggrComp(VtlParser::AggrCompContext *context) = 0;

    virtual std::any visitCountAggrComp(VtlParser::CountAggrCompContext *context) = 0;

    virtual std::any visitAggrDataset(VtlParser::AggrDatasetContext *context) = 0;

    virtual std::any visitAnSimpleFunction(VtlParser::AnSimpleFunctionContext *context) = 0;

    virtual std::any visitLagOrLeadAn(VtlParser::LagOrLeadAnContext *context) = 0;

    virtual std::any visitRatioToReportAn(VtlParser::RatioToReportAnContext *context) = 0;

    virtual std::any visitAnSimpleFunctionComponent(VtlParser::AnSimpleFunctionComponentContext *context) = 0;

    virtual std::any visitLagOrLeadAnComponent(VtlParser::LagOrLeadAnComponentContext *context) = 0;

    virtual std::any visitRankAnComponent(VtlParser::RankAnComponentContext *context) = 0;

    virtual std::any visitRatioToReportAnComponent(VtlParser::RatioToReportAnComponentContext *context) = 0;

    virtual std::any visitRenameClauseItem(VtlParser::RenameClauseItemContext *context) = 0;

    virtual std::any visitAggregateClause(VtlParser::AggregateClauseContext *context) = 0;

    virtual std::any visitAggrFunctionClause(VtlParser::AggrFunctionClauseContext *context) = 0;

    virtual std::any visitCalcClauseItem(VtlParser::CalcClauseItemContext *context) = 0;

    virtual std::any visitSubspaceClauseItem(VtlParser::SubspaceClauseItemContext *context) = 0;

    virtual std::any visitSimpleScalar(VtlParser::SimpleScalarContext *context) = 0;

    virtual std::any visitScalarWithCast(VtlParser::ScalarWithCastContext *context) = 0;

    virtual std::any visitJoinClauseWithoutUsing(VtlParser::JoinClauseWithoutUsingContext *context) = 0;

    virtual std::any visitJoinClause(VtlParser::JoinClauseContext *context) = 0;

    virtual std::any visitJoinClauseItem(VtlParser::JoinClauseItemContext *context) = 0;

    virtual std::any visitJoinBody(VtlParser::JoinBodyContext *context) = 0;

    virtual std::any visitJoinApplyClause(VtlParser::JoinApplyClauseContext *context) = 0;

    virtual std::any visitPartitionByClause(VtlParser::PartitionByClauseContext *context) = 0;

    virtual std::any visitOrderByClause(VtlParser::OrderByClauseContext *context) = 0;

    virtual std::any visitOrderByItem(VtlParser::OrderByItemContext *context) = 0;

    virtual std::any visitWindowingClause(VtlParser::WindowingClauseContext *context) = 0;

    virtual std::any visitSignedInteger(VtlParser::SignedIntegerContext *context) = 0;

    virtual std::any visitSignedNumber(VtlParser::SignedNumberContext *context) = 0;

    virtual std::any visitLimitClauseItem(VtlParser::LimitClauseItemContext *context) = 0;

    virtual std::any visitGroupByOrExcept(VtlParser::GroupByOrExceptContext *context) = 0;

    virtual std::any visitGroupAll(VtlParser::GroupAllContext *context) = 0;

    virtual std::any visitHavingClause(VtlParser::HavingClauseContext *context) = 0;

    virtual std::any visitParameterItem(VtlParser::ParameterItemContext *context) = 0;

    virtual std::any visitOutputParameterType(VtlParser::OutputParameterTypeContext *context) = 0;

    virtual std::any visitOutputParameterTypeComponent(VtlParser::OutputParameterTypeComponentContext *context) = 0;

    virtual std::any visitInputParameterType(VtlParser::InputParameterTypeContext *context) = 0;

    virtual std::any visitRulesetType(VtlParser::RulesetTypeContext *context) = 0;

    virtual std::any visitScalarType(VtlParser::ScalarTypeContext *context) = 0;

    virtual std::any visitComponentType(VtlParser::ComponentTypeContext *context) = 0;

    virtual std::any visitDatasetType(VtlParser::DatasetTypeContext *context) = 0;

    virtual std::any visitEvalDatasetType(VtlParser::EvalDatasetTypeContext *context) = 0;

    virtual std::any visitScalarSetType(VtlParser::ScalarSetTypeContext *context) = 0;

    virtual std::any visitDataPoint(VtlParser::DataPointContext *context) = 0;

    virtual std::any visitDataPointVd(VtlParser::DataPointVdContext *context) = 0;

    virtual std::any visitDataPointVar(VtlParser::DataPointVarContext *context) = 0;

    virtual std::any visitHrRulesetType(VtlParser::HrRulesetTypeContext *context) = 0;

    virtual std::any visitHrRulesetVdType(VtlParser::HrRulesetVdTypeContext *context) = 0;

    virtual std::any visitHrRulesetVarType(VtlParser::HrRulesetVarTypeContext *context) = 0;

    virtual std::any visitValueDomainName(VtlParser::ValueDomainNameContext *context) = 0;

    virtual std::any visitRulesetID(VtlParser::RulesetIDContext *context) = 0;

    virtual std::any visitRulesetSignature(VtlParser::RulesetSignatureContext *context) = 0;

    virtual std::any visitSignature(VtlParser::SignatureContext *context) = 0;

    virtual std::any visitRuleClauseDatapoint(VtlParser::RuleClauseDatapointContext *context) = 0;

    virtual std::any visitRuleItemDatapoint(VtlParser::RuleItemDatapointContext *context) = 0;

    virtual std::any visitRuleClauseHierarchical(VtlParser::RuleClauseHierarchicalContext *context) = 0;

    virtual std::any visitRuleItemHierarchical(VtlParser::RuleItemHierarchicalContext *context) = 0;

    virtual std::any visitHierRuleSignature(VtlParser::HierRuleSignatureContext *context) = 0;

    virtual std::any visitValueDomainSignature(VtlParser::ValueDomainSignatureContext *context) = 0;

    virtual std::any visitCodeItemRelation(VtlParser::CodeItemRelationContext *context) = 0;

    virtual std::any visitCodeItemRelationClause(VtlParser::CodeItemRelationClauseContext *context) = 0;

    virtual std::any visitValueDomainValue(VtlParser::ValueDomainValueContext *context) = 0;

    virtual std::any visitConditionConstraint(VtlParser::ConditionConstraintContext *context) = 0;

    virtual std::any visitRangeConstraint(VtlParser::RangeConstraintContext *context) = 0;

    virtual std::any visitCompConstraint(VtlParser::CompConstraintContext *context) = 0;

    virtual std::any visitMultModifier(VtlParser::MultModifierContext *context) = 0;

    virtual std::any visitValidationOutput(VtlParser::ValidationOutputContext *context) = 0;

    virtual std::any visitValidationMode(VtlParser::ValidationModeContext *context) = 0;

    virtual std::any visitConditionClause(VtlParser::ConditionClauseContext *context) = 0;

    virtual std::any visitInputMode(VtlParser::InputModeContext *context) = 0;

    virtual std::any visitImbalanceExpr(VtlParser::ImbalanceExprContext *context) = 0;

    virtual std::any visitInputModeHierarchy(VtlParser::InputModeHierarchyContext *context) = 0;

    virtual std::any visitOutputModeHierarchy(VtlParser::OutputModeHierarchyContext *context) = 0;

    virtual std::any visitAlias(VtlParser::AliasContext *context) = 0;

    virtual std::any visitVarID(VtlParser::VarIDContext *context) = 0;

    virtual std::any visitSimpleComponentId(VtlParser::SimpleComponentIdContext *context) = 0;

    virtual std::any visitComponentID(VtlParser::ComponentIDContext *context) = 0;

    virtual std::any visitLists(VtlParser::ListsContext *context) = 0;

    virtual std::any visitErCode(VtlParser::ErCodeContext *context) = 0;

    virtual std::any visitErLevel(VtlParser::ErLevelContext *context) = 0;

    virtual std::any visitComparisonOperand(VtlParser::ComparisonOperandContext *context) = 0;

    virtual std::any visitOptionalExpr(VtlParser::OptionalExprContext *context) = 0;

    virtual std::any visitOptionalExprComponent(VtlParser::OptionalExprComponentContext *context) = 0;

    virtual std::any visitComponentRole(VtlParser::ComponentRoleContext *context) = 0;

    virtual std::any visitViralAttribute(VtlParser::ViralAttributeContext *context) = 0;

    virtual std::any visitValueDomainID(VtlParser::ValueDomainIDContext *context) = 0;

    virtual std::any visitOperatorID(VtlParser::OperatorIDContext *context) = 0;

    virtual std::any visitRoutineName(VtlParser::RoutineNameContext *context) = 0;

    virtual std::any visitConstant(VtlParser::ConstantContext *context) = 0;

    virtual std::any visitBasicScalarType(VtlParser::BasicScalarTypeContext *context) = 0;

    virtual std::any visitRetainType(VtlParser::RetainTypeContext *context) = 0;


};

