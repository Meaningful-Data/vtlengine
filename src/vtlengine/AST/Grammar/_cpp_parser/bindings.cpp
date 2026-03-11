/**
 * vtl_cpp_parser - pybind11 bindings for ANTLR4 C++ VTL parser
 *
 * Exposes two lightweight wrapper classes (ParseNode, TerminalNode) plus
 * a parse() function that lexes + parses VTL text in C++ and returns
 * a wrapped parse tree to Python.
 *
 * Uses LAZY wrapping: ParseNode holds a raw pointer to the C++ parse tree
 * context. Children are only wrapped into Python objects when .children is
 * accessed. This avoids creating ~611K Python objects eagerly.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antlr4-runtime.h"
#include "VtlLexer.h"
#include "VtlParser.h"

#include "ast_builder.h"

#include <memory>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

// ============================================================
// Type → (rule_index, alt_index) mapping
// ============================================================
std::unordered_map<std::type_index, std::pair<int, int>> g_type_map;

void init_type_map() {
    if (!g_type_map.empty()) return;

    // Rules without labeled alternatives: alt_index = -1
    g_type_map[typeid(VtlParser::StartContext)] = {VtlParser::RuleStart, -1};
    g_type_map[typeid(VtlParser::DatasetClauseContext)] = {VtlParser::RuleDatasetClause, -1};
    g_type_map[typeid(VtlParser::RenameClauseContext)] = {VtlParser::RuleRenameClause, -1};
    g_type_map[typeid(VtlParser::AggrClauseContext)] = {VtlParser::RuleAggrClause, -1};
    g_type_map[typeid(VtlParser::FilterClauseContext)] = {VtlParser::RuleFilterClause, -1};
    g_type_map[typeid(VtlParser::CalcClauseContext)] = {VtlParser::RuleCalcClause, -1};
    g_type_map[typeid(VtlParser::KeepOrDropClauseContext)] = {VtlParser::RuleKeepOrDropClause, -1};
    g_type_map[typeid(VtlParser::PivotOrUnpivotClauseContext)] = {VtlParser::RulePivotOrUnpivotClause, -1};
    g_type_map[typeid(VtlParser::CustomPivotClauseContext)] = {VtlParser::RuleCustomPivotClause, -1};
    g_type_map[typeid(VtlParser::SubspaceClauseContext)] = {VtlParser::RuleSubspaceClause, -1};
    g_type_map[typeid(VtlParser::RenameClauseItemContext)] = {VtlParser::RuleRenameClauseItem, -1};
    g_type_map[typeid(VtlParser::AggregateClauseContext)] = {VtlParser::RuleAggregateClause, -1};
    g_type_map[typeid(VtlParser::AggrFunctionClauseContext)] = {VtlParser::RuleAggrFunctionClause, -1};
    g_type_map[typeid(VtlParser::CalcClauseItemContext)] = {VtlParser::RuleCalcClauseItem, -1};
    g_type_map[typeid(VtlParser::SubspaceClauseItemContext)] = {VtlParser::RuleSubspaceClauseItem, -1};
    g_type_map[typeid(VtlParser::JoinClauseWithoutUsingContext)] = {VtlParser::RuleJoinClauseWithoutUsing, -1};
    g_type_map[typeid(VtlParser::JoinClauseContext)] = {VtlParser::RuleJoinClause, -1};
    g_type_map[typeid(VtlParser::JoinClauseItemContext)] = {VtlParser::RuleJoinClauseItem, -1};
    g_type_map[typeid(VtlParser::JoinBodyContext)] = {VtlParser::RuleJoinBody, -1};
    g_type_map[typeid(VtlParser::JoinApplyClauseContext)] = {VtlParser::RuleJoinApplyClause, -1};
    g_type_map[typeid(VtlParser::PartitionByClauseContext)] = {VtlParser::RulePartitionByClause, -1};
    g_type_map[typeid(VtlParser::OrderByClauseContext)] = {VtlParser::RuleOrderByClause, -1};
    g_type_map[typeid(VtlParser::OrderByItemContext)] = {VtlParser::RuleOrderByItem, -1};
    g_type_map[typeid(VtlParser::WindowingClauseContext)] = {VtlParser::RuleWindowingClause, -1};
    g_type_map[typeid(VtlParser::SignedIntegerContext)] = {VtlParser::RuleSignedInteger, -1};
    g_type_map[typeid(VtlParser::SignedNumberContext)] = {VtlParser::RuleSignedNumber, -1};
    g_type_map[typeid(VtlParser::LimitClauseItemContext)] = {VtlParser::RuleLimitClauseItem, -1};
    g_type_map[typeid(VtlParser::HavingClauseContext)] = {VtlParser::RuleHavingClause, -1};
    g_type_map[typeid(VtlParser::ParameterItemContext)] = {VtlParser::RuleParameterItem, -1};
    g_type_map[typeid(VtlParser::OutputParameterTypeContext)] = {VtlParser::RuleOutputParameterType, -1};
    g_type_map[typeid(VtlParser::OutputParameterTypeComponentContext)] = {VtlParser::RuleOutputParameterTypeComponent, -1};
    g_type_map[typeid(VtlParser::InputParameterTypeContext)] = {VtlParser::RuleInputParameterType, -1};
    g_type_map[typeid(VtlParser::RulesetTypeContext)] = {VtlParser::RuleRulesetType, -1};
    g_type_map[typeid(VtlParser::ScalarTypeContext)] = {VtlParser::RuleScalarType, -1};
    g_type_map[typeid(VtlParser::ComponentTypeContext)] = {VtlParser::RuleComponentType, -1};
    g_type_map[typeid(VtlParser::DatasetTypeContext)] = {VtlParser::RuleDatasetType, -1};
    g_type_map[typeid(VtlParser::EvalDatasetTypeContext)] = {VtlParser::RuleEvalDatasetType, -1};
    g_type_map[typeid(VtlParser::ScalarSetTypeContext)] = {VtlParser::RuleScalarSetType, -1};
    g_type_map[typeid(VtlParser::ValueDomainNameContext)] = {VtlParser::RuleValueDomainName, -1};
    g_type_map[typeid(VtlParser::RulesetIDContext)] = {VtlParser::RuleRulesetID, -1};
    g_type_map[typeid(VtlParser::RulesetSignatureContext)] = {VtlParser::RuleRulesetSignature, -1};
    g_type_map[typeid(VtlParser::SignatureContext)] = {VtlParser::RuleSignature, -1};
    g_type_map[typeid(VtlParser::RuleClauseDatapointContext)] = {VtlParser::RuleRuleClauseDatapoint, -1};
    g_type_map[typeid(VtlParser::RuleItemDatapointContext)] = {VtlParser::RuleRuleItemDatapoint, -1};
    g_type_map[typeid(VtlParser::RuleClauseHierarchicalContext)] = {VtlParser::RuleRuleClauseHierarchical, -1};
    g_type_map[typeid(VtlParser::RuleItemHierarchicalContext)] = {VtlParser::RuleRuleItemHierarchical, -1};
    g_type_map[typeid(VtlParser::HierRuleSignatureContext)] = {VtlParser::RuleHierRuleSignature, -1};
    g_type_map[typeid(VtlParser::ValueDomainSignatureContext)] = {VtlParser::RuleValueDomainSignature, -1};
    g_type_map[typeid(VtlParser::CodeItemRelationContext)] = {VtlParser::RuleCodeItemRelation, -1};
    g_type_map[typeid(VtlParser::CodeItemRelationClauseContext)] = {VtlParser::RuleCodeItemRelationClause, -1};
    g_type_map[typeid(VtlParser::ValueDomainValueContext)] = {VtlParser::RuleValueDomainValue, -1};
    g_type_map[typeid(VtlParser::CompConstraintContext)] = {VtlParser::RuleCompConstraint, -1};
    g_type_map[typeid(VtlParser::MultModifierContext)] = {VtlParser::RuleMultModifier, -1};
    g_type_map[typeid(VtlParser::ValidationOutputContext)] = {VtlParser::RuleValidationOutput, -1};
    g_type_map[typeid(VtlParser::ValidationModeContext)] = {VtlParser::RuleValidationMode, -1};
    g_type_map[typeid(VtlParser::ConditionClauseContext)] = {VtlParser::RuleConditionClause, -1};
    g_type_map[typeid(VtlParser::InputModeContext)] = {VtlParser::RuleInputMode, -1};
    g_type_map[typeid(VtlParser::ImbalanceExprContext)] = {VtlParser::RuleImbalanceExpr, -1};
    g_type_map[typeid(VtlParser::InputModeHierarchyContext)] = {VtlParser::RuleInputModeHierarchy, -1};
    g_type_map[typeid(VtlParser::OutputModeHierarchyContext)] = {VtlParser::RuleOutputModeHierarchy, -1};
    g_type_map[typeid(VtlParser::AliasContext)] = {VtlParser::RuleAlias, -1};
    g_type_map[typeid(VtlParser::VarIDContext)] = {VtlParser::RuleVarID, -1};
    g_type_map[typeid(VtlParser::SimpleComponentIdContext)] = {VtlParser::RuleSimpleComponentId, -1};
    g_type_map[typeid(VtlParser::ComponentIDContext)] = {VtlParser::RuleComponentID, -1};
    g_type_map[typeid(VtlParser::ListsContext)] = {VtlParser::RuleLists, -1};
    g_type_map[typeid(VtlParser::ErCodeContext)] = {VtlParser::RuleErCode, -1};
    g_type_map[typeid(VtlParser::ErLevelContext)] = {VtlParser::RuleErLevel, -1};
    g_type_map[typeid(VtlParser::ComparisonOperandContext)] = {VtlParser::RuleComparisonOperand, -1};
    g_type_map[typeid(VtlParser::OptionalExprContext)] = {VtlParser::RuleOptionalExpr, -1};
    g_type_map[typeid(VtlParser::OptionalExprComponentContext)] = {VtlParser::RuleOptionalExprComponent, -1};
    g_type_map[typeid(VtlParser::ComponentRoleContext)] = {VtlParser::RuleComponentRole, -1};
    g_type_map[typeid(VtlParser::ViralAttributeContext)] = {VtlParser::RuleViralAttribute, -1};
    g_type_map[typeid(VtlParser::ValueDomainIDContext)] = {VtlParser::RuleValueDomainID, -1};
    g_type_map[typeid(VtlParser::OperatorIDContext)] = {VtlParser::RuleOperatorID, -1};
    g_type_map[typeid(VtlParser::RoutineNameContext)] = {VtlParser::RuleRoutineName, -1};
    g_type_map[typeid(VtlParser::ConstantContext)] = {VtlParser::RuleConstant, -1};
    g_type_map[typeid(VtlParser::BasicScalarTypeContext)] = {VtlParser::RuleBasicScalarType, -1};
    g_type_map[typeid(VtlParser::RetainTypeContext)] = {VtlParser::RuleRetainType, -1};
    g_type_map[typeid(VtlParser::ParameterComponentContext)] = {VtlParser::RuleParameterComponent, -1};
    g_type_map[typeid(VtlParser::ParameterContext)] = {VtlParser::RuleParameter, -1};
    g_type_map[typeid(VtlParser::HierarchyOperatorsContext)] = {VtlParser::RuleHierarchyOperators, -1};

    // Statement alternatives
    g_type_map[typeid(VtlParser::TemporaryAssignmentContext)] = {VtlParser::RuleStatement, 0};
    g_type_map[typeid(VtlParser::PersistAssignmentContext)] = {VtlParser::RuleStatement, 1};
    g_type_map[typeid(VtlParser::DefineExpressionContext)] = {VtlParser::RuleStatement, 2};

    // Expr alternatives
    g_type_map[typeid(VtlParser::ParenthesisExprContext)] = {VtlParser::RuleExpr, 0};
    g_type_map[typeid(VtlParser::FunctionsExpressionContext)] = {VtlParser::RuleExpr, 1};
    g_type_map[typeid(VtlParser::ClauseExprContext)] = {VtlParser::RuleExpr, 2};
    g_type_map[typeid(VtlParser::MembershipExprContext)] = {VtlParser::RuleExpr, 3};
    g_type_map[typeid(VtlParser::UnaryExprContext)] = {VtlParser::RuleExpr, 4};
    g_type_map[typeid(VtlParser::ArithmeticExprContext)] = {VtlParser::RuleExpr, 5};
    g_type_map[typeid(VtlParser::ArithmeticExprOrConcatContext)] = {VtlParser::RuleExpr, 6};
    g_type_map[typeid(VtlParser::ComparisonExprContext)] = {VtlParser::RuleExpr, 7};
    g_type_map[typeid(VtlParser::InNotInExprContext)] = {VtlParser::RuleExpr, 8};
    g_type_map[typeid(VtlParser::BooleanExprContext)] = {VtlParser::RuleExpr, 9};
    g_type_map[typeid(VtlParser::IfExprContext)] = {VtlParser::RuleExpr, 10};
    g_type_map[typeid(VtlParser::CaseExprContext)] = {VtlParser::RuleExpr, 11};
    g_type_map[typeid(VtlParser::ConstantExprContext)] = {VtlParser::RuleExpr, 12};
    g_type_map[typeid(VtlParser::VarIdExprContext)] = {VtlParser::RuleExpr, 13};

    // ExprComponent alternatives
    g_type_map[typeid(VtlParser::ParenthesisExprCompContext)] = {VtlParser::RuleExprComponent, 0};
    g_type_map[typeid(VtlParser::FunctionsExpressionCompContext)] = {VtlParser::RuleExprComponent, 1};
    g_type_map[typeid(VtlParser::UnaryExprCompContext)] = {VtlParser::RuleExprComponent, 2};
    g_type_map[typeid(VtlParser::ArithmeticExprCompContext)] = {VtlParser::RuleExprComponent, 3};
    g_type_map[typeid(VtlParser::ArithmeticExprOrConcatCompContext)] = {VtlParser::RuleExprComponent, 4};
    g_type_map[typeid(VtlParser::ComparisonExprCompContext)] = {VtlParser::RuleExprComponent, 5};
    g_type_map[typeid(VtlParser::InNotInExprCompContext)] = {VtlParser::RuleExprComponent, 6};
    g_type_map[typeid(VtlParser::BooleanExprCompContext)] = {VtlParser::RuleExprComponent, 7};
    g_type_map[typeid(VtlParser::IfExprCompContext)] = {VtlParser::RuleExprComponent, 8};
    g_type_map[typeid(VtlParser::CaseExprCompContext)] = {VtlParser::RuleExprComponent, 9};
    g_type_map[typeid(VtlParser::ConstantExprCompContext)] = {VtlParser::RuleExprComponent, 10};
    g_type_map[typeid(VtlParser::CompIdContext)] = {VtlParser::RuleExprComponent, 11};

    // FunctionsComponents alternatives
    g_type_map[typeid(VtlParser::GenericFunctionsComponentsContext)] = {VtlParser::RuleFunctionsComponents, 0};
    g_type_map[typeid(VtlParser::StringFunctionsComponentsContext)] = {VtlParser::RuleFunctionsComponents, 1};
    g_type_map[typeid(VtlParser::NumericFunctionsComponentsContext)] = {VtlParser::RuleFunctionsComponents, 2};
    g_type_map[typeid(VtlParser::ComparisonFunctionsComponentsContext)] = {VtlParser::RuleFunctionsComponents, 3};
    g_type_map[typeid(VtlParser::TimeFunctionsComponentsContext)] = {VtlParser::RuleFunctionsComponents, 4};
    g_type_map[typeid(VtlParser::ConditionalFunctionsComponentsContext)] = {VtlParser::RuleFunctionsComponents, 5};
    g_type_map[typeid(VtlParser::AggregateFunctionsComponentsContext)] = {VtlParser::RuleFunctionsComponents, 6};
    g_type_map[typeid(VtlParser::AnalyticFunctionsComponentsContext)] = {VtlParser::RuleFunctionsComponents, 7};

    // Functions alternatives
    g_type_map[typeid(VtlParser::JoinFunctionsContext)] = {VtlParser::RuleFunctions, 0};
    g_type_map[typeid(VtlParser::GenericFunctionsContext)] = {VtlParser::RuleFunctions, 1};
    g_type_map[typeid(VtlParser::StringFunctionsContext)] = {VtlParser::RuleFunctions, 2};
    g_type_map[typeid(VtlParser::NumericFunctionsContext)] = {VtlParser::RuleFunctions, 3};
    g_type_map[typeid(VtlParser::ComparisonFunctionsContext)] = {VtlParser::RuleFunctions, 4};
    g_type_map[typeid(VtlParser::TimeFunctionsContext)] = {VtlParser::RuleFunctions, 5};
    g_type_map[typeid(VtlParser::SetFunctionsContext)] = {VtlParser::RuleFunctions, 6};
    g_type_map[typeid(VtlParser::HierarchyFunctionsContext)] = {VtlParser::RuleFunctions, 7};
    g_type_map[typeid(VtlParser::ValidationFunctionsContext)] = {VtlParser::RuleFunctions, 8};
    g_type_map[typeid(VtlParser::ConditionalFunctionsContext)] = {VtlParser::RuleFunctions, 9};
    g_type_map[typeid(VtlParser::AggregateFunctionsContext)] = {VtlParser::RuleFunctions, 10};
    g_type_map[typeid(VtlParser::AnalyticFunctionsContext)] = {VtlParser::RuleFunctions, 11};

    // JoinOperators alternatives
    g_type_map[typeid(VtlParser::JoinExprContext)] = {VtlParser::RuleJoinOperators, 0};

    // DefOperators alternatives
    g_type_map[typeid(VtlParser::DefOperatorContext)] = {VtlParser::RuleDefOperators, 0};
    g_type_map[typeid(VtlParser::DefDatapointRulesetContext)] = {VtlParser::RuleDefOperators, 1};
    g_type_map[typeid(VtlParser::DefHierarchicalContext)] = {VtlParser::RuleDefOperators, 2};

    // GenericOperators alternatives
    g_type_map[typeid(VtlParser::CallDatasetContext)] = {VtlParser::RuleGenericOperators, 0};
    g_type_map[typeid(VtlParser::EvalAtomContext)] = {VtlParser::RuleGenericOperators, 1};
    g_type_map[typeid(VtlParser::CastExprDatasetContext)] = {VtlParser::RuleGenericOperators, 2};

    // GenericOperatorsComponent alternatives
    g_type_map[typeid(VtlParser::CallComponentContext)] = {VtlParser::RuleGenericOperatorsComponent, 0};
    g_type_map[typeid(VtlParser::CastExprComponentContext)] = {VtlParser::RuleGenericOperatorsComponent, 1};
    g_type_map[typeid(VtlParser::EvalAtomComponentContext)] = {VtlParser::RuleGenericOperatorsComponent, 2};

    // StringOperators alternatives
    g_type_map[typeid(VtlParser::UnaryStringFunctionContext)] = {VtlParser::RuleStringOperators, 0};
    g_type_map[typeid(VtlParser::SubstrAtomContext)] = {VtlParser::RuleStringOperators, 1};
    g_type_map[typeid(VtlParser::ReplaceAtomContext)] = {VtlParser::RuleStringOperators, 2};
    g_type_map[typeid(VtlParser::InstrAtomContext)] = {VtlParser::RuleStringOperators, 3};

    // StringOperatorsComponent alternatives
    g_type_map[typeid(VtlParser::UnaryStringFunctionComponentContext)] = {VtlParser::RuleStringOperatorsComponent, 0};
    g_type_map[typeid(VtlParser::SubstrAtomComponentContext)] = {VtlParser::RuleStringOperatorsComponent, 1};
    g_type_map[typeid(VtlParser::ReplaceAtomComponentContext)] = {VtlParser::RuleStringOperatorsComponent, 2};
    g_type_map[typeid(VtlParser::InstrAtomComponentContext)] = {VtlParser::RuleStringOperatorsComponent, 3};

    // NumericOperators alternatives
    g_type_map[typeid(VtlParser::UnaryNumericContext)] = {VtlParser::RuleNumericOperators, 0};
    g_type_map[typeid(VtlParser::UnaryWithOptionalNumericContext)] = {VtlParser::RuleNumericOperators, 1};
    g_type_map[typeid(VtlParser::BinaryNumericContext)] = {VtlParser::RuleNumericOperators, 2};

    // NumericOperatorsComponent alternatives
    g_type_map[typeid(VtlParser::UnaryNumericComponentContext)] = {VtlParser::RuleNumericOperatorsComponent, 0};
    g_type_map[typeid(VtlParser::UnaryWithOptionalNumericComponentContext)] = {VtlParser::RuleNumericOperatorsComponent, 1};
    g_type_map[typeid(VtlParser::BinaryNumericComponentContext)] = {VtlParser::RuleNumericOperatorsComponent, 2};

    // ComparisonOperators alternatives
    g_type_map[typeid(VtlParser::BetweenAtomContext)] = {VtlParser::RuleComparisonOperators, 0};
    g_type_map[typeid(VtlParser::CharsetMatchAtomContext)] = {VtlParser::RuleComparisonOperators, 1};
    g_type_map[typeid(VtlParser::IsNullAtomContext)] = {VtlParser::RuleComparisonOperators, 2};
    g_type_map[typeid(VtlParser::ExistInAtomContext)] = {VtlParser::RuleComparisonOperators, 3};

    // ComparisonOperatorsComponent alternatives
    g_type_map[typeid(VtlParser::BetweenAtomComponentContext)] = {VtlParser::RuleComparisonOperatorsComponent, 0};
    g_type_map[typeid(VtlParser::CharsetMatchAtomComponentContext)] = {VtlParser::RuleComparisonOperatorsComponent, 1};
    g_type_map[typeid(VtlParser::IsNullAtomComponentContext)] = {VtlParser::RuleComparisonOperatorsComponent, 2};

    // TimeOperators alternatives
    g_type_map[typeid(VtlParser::PeriodAtomContext)] = {VtlParser::RuleTimeOperators, 0};
    g_type_map[typeid(VtlParser::FillTimeAtomContext)] = {VtlParser::RuleTimeOperators, 1};
    g_type_map[typeid(VtlParser::FlowAtomContext)] = {VtlParser::RuleTimeOperators, 2};
    g_type_map[typeid(VtlParser::TimeShiftAtomContext)] = {VtlParser::RuleTimeOperators, 3};
    g_type_map[typeid(VtlParser::TimeAggAtomContext)] = {VtlParser::RuleTimeOperators, 4};
    g_type_map[typeid(VtlParser::CurrentDateAtomContext)] = {VtlParser::RuleTimeOperators, 5};
    g_type_map[typeid(VtlParser::DateDiffAtomContext)] = {VtlParser::RuleTimeOperators, 6};
    g_type_map[typeid(VtlParser::DateAddAtomContext)] = {VtlParser::RuleTimeOperators, 7};
    g_type_map[typeid(VtlParser::YearAtomContext)] = {VtlParser::RuleTimeOperators, 8};
    g_type_map[typeid(VtlParser::MonthAtomContext)] = {VtlParser::RuleTimeOperators, 9};
    g_type_map[typeid(VtlParser::DayOfMonthAtomContext)] = {VtlParser::RuleTimeOperators, 10};
    g_type_map[typeid(VtlParser::DayOfYearAtomContext)] = {VtlParser::RuleTimeOperators, 11};
    g_type_map[typeid(VtlParser::DayToYearAtomContext)] = {VtlParser::RuleTimeOperators, 12};
    g_type_map[typeid(VtlParser::DayToMonthAtomContext)] = {VtlParser::RuleTimeOperators, 13};
    g_type_map[typeid(VtlParser::YearTodayAtomContext)] = {VtlParser::RuleTimeOperators, 14};
    g_type_map[typeid(VtlParser::MonthTodayAtomContext)] = {VtlParser::RuleTimeOperators, 15};

    // TimeOperatorsComponent alternatives
    g_type_map[typeid(VtlParser::PeriodAtomComponentContext)] = {VtlParser::RuleTimeOperatorsComponent, 0};
    g_type_map[typeid(VtlParser::FillTimeAtomComponentContext)] = {VtlParser::RuleTimeOperatorsComponent, 1};
    g_type_map[typeid(VtlParser::FlowAtomComponentContext)] = {VtlParser::RuleTimeOperatorsComponent, 2};
    g_type_map[typeid(VtlParser::TimeShiftAtomComponentContext)] = {VtlParser::RuleTimeOperatorsComponent, 3};
    g_type_map[typeid(VtlParser::TimeAggAtomComponentContext)] = {VtlParser::RuleTimeOperatorsComponent, 4};
    g_type_map[typeid(VtlParser::CurrentDateAtomComponentContext)] = {VtlParser::RuleTimeOperatorsComponent, 5};
    g_type_map[typeid(VtlParser::DateDiffAtomComponentContext)] = {VtlParser::RuleTimeOperatorsComponent, 6};
    g_type_map[typeid(VtlParser::DateAddAtomComponentContext)] = {VtlParser::RuleTimeOperatorsComponent, 7};
    g_type_map[typeid(VtlParser::YearAtomComponentContext)] = {VtlParser::RuleTimeOperatorsComponent, 8};
    g_type_map[typeid(VtlParser::MonthAtomComponentContext)] = {VtlParser::RuleTimeOperatorsComponent, 9};
    g_type_map[typeid(VtlParser::DayOfMonthAtomComponentContext)] = {VtlParser::RuleTimeOperatorsComponent, 10};
    g_type_map[typeid(VtlParser::DatOfYearAtomComponentContext)] = {VtlParser::RuleTimeOperatorsComponent, 11};
    g_type_map[typeid(VtlParser::DayToYearAtomComponentContext)] = {VtlParser::RuleTimeOperatorsComponent, 12};
    g_type_map[typeid(VtlParser::DayToMonthAtomComponentContext)] = {VtlParser::RuleTimeOperatorsComponent, 13};
    g_type_map[typeid(VtlParser::YearTodayAtomComponentContext)] = {VtlParser::RuleTimeOperatorsComponent, 14};
    g_type_map[typeid(VtlParser::MonthTodayAtomComponentContext)] = {VtlParser::RuleTimeOperatorsComponent, 15};

    // SetOperators alternatives
    g_type_map[typeid(VtlParser::UnionAtomContext)] = {VtlParser::RuleSetOperators, 0};
    g_type_map[typeid(VtlParser::IntersectAtomContext)] = {VtlParser::RuleSetOperators, 1};
    g_type_map[typeid(VtlParser::SetOrSYmDiffAtomContext)] = {VtlParser::RuleSetOperators, 2};

    // ValidationOperators alternatives
    g_type_map[typeid(VtlParser::ValidateDPrulesetContext)] = {VtlParser::RuleValidationOperators, 0};
    g_type_map[typeid(VtlParser::ValidateHRrulesetContext)] = {VtlParser::RuleValidationOperators, 1};
    g_type_map[typeid(VtlParser::ValidationSimpleContext)] = {VtlParser::RuleValidationOperators, 2};

    // ConditionalOperators alternatives
    g_type_map[typeid(VtlParser::NvlAtomContext)] = {VtlParser::RuleConditionalOperators, 0};

    // ConditionalOperatorsComponent alternatives
    g_type_map[typeid(VtlParser::NvlAtomComponentContext)] = {VtlParser::RuleConditionalOperatorsComponent, 0};

    // AggrOperators alternatives
    g_type_map[typeid(VtlParser::AggrCompContext)] = {VtlParser::RuleAggrOperators, 0};
    g_type_map[typeid(VtlParser::CountAggrCompContext)] = {VtlParser::RuleAggrOperators, 1};

    // AggrOperatorsGrouping alternatives
    g_type_map[typeid(VtlParser::AggrDatasetContext)] = {VtlParser::RuleAggrOperatorsGrouping, 0};

    // AnFunction alternatives
    g_type_map[typeid(VtlParser::AnSimpleFunctionContext)] = {VtlParser::RuleAnFunction, 0};
    g_type_map[typeid(VtlParser::LagOrLeadAnContext)] = {VtlParser::RuleAnFunction, 1};
    g_type_map[typeid(VtlParser::RatioToReportAnContext)] = {VtlParser::RuleAnFunction, 2};

    // AnFunctionComponent alternatives
    g_type_map[typeid(VtlParser::AnSimpleFunctionComponentContext)] = {VtlParser::RuleAnFunctionComponent, 0};
    g_type_map[typeid(VtlParser::LagOrLeadAnComponentContext)] = {VtlParser::RuleAnFunctionComponent, 1};
    g_type_map[typeid(VtlParser::RankAnComponentContext)] = {VtlParser::RuleAnFunctionComponent, 2};
    g_type_map[typeid(VtlParser::RatioToReportAnComponentContext)] = {VtlParser::RuleAnFunctionComponent, 3};

    // ScalarItem alternatives
    g_type_map[typeid(VtlParser::SimpleScalarContext)] = {VtlParser::RuleScalarItem, 0};
    g_type_map[typeid(VtlParser::ScalarWithCastContext)] = {VtlParser::RuleScalarItem, 1};

    // GroupingClause alternatives
    g_type_map[typeid(VtlParser::GroupByOrExceptContext)] = {VtlParser::RuleGroupingClause, 0};
    g_type_map[typeid(VtlParser::GroupAllContext)] = {VtlParser::RuleGroupingClause, 1};

    // DpRuleset alternatives
    g_type_map[typeid(VtlParser::DataPointContext)] = {VtlParser::RuleDpRuleset, 0};
    g_type_map[typeid(VtlParser::DataPointVdContext)] = {VtlParser::RuleDpRuleset, 1};
    g_type_map[typeid(VtlParser::DataPointVarContext)] = {VtlParser::RuleDpRuleset, 2};

    // HrRuleset alternatives
    g_type_map[typeid(VtlParser::HrRulesetTypeContext)] = {VtlParser::RuleHrRuleset, 0};
    g_type_map[typeid(VtlParser::HrRulesetVdTypeContext)] = {VtlParser::RuleHrRuleset, 1};
    g_type_map[typeid(VtlParser::HrRulesetVarTypeContext)] = {VtlParser::RuleHrRuleset, 2};

    // ScalarTypeConstraint alternatives
    g_type_map[typeid(VtlParser::ConditionConstraintContext)] = {VtlParser::RuleScalarTypeConstraint, 0};
    g_type_map[typeid(VtlParser::RangeConstraintContext)] = {VtlParser::RuleScalarTypeConstraint, 1};
}


// ============================================================
// Module state: holds parser objects alive for the last parse
// ============================================================
struct ParserState {
    std::unique_ptr<antlr4::ANTLRInputStream> input;
    std::unique_ptr<VtlLexer> lexer;
    std::unique_ptr<antlr4::CommonTokenStream> tokens;
    std::unique_ptr<VtlParser> parser;
    std::string input_text;

    struct CommentInfo {
        int type;
        std::string text;
        int line;
        int column;
    };
    std::vector<CommentInfo> comments;
};

static ParserState g_state;


// ============================================================
// Forward declarations for lazy wrapper classes
// ============================================================
class LazyParseNode;
class LazyTerminalNode;

// Helper to wrap a single ANTLR tree node into a py::object
static py::object wrap_node(antlr4::tree::ParseTree* node);


// ============================================================
// LazyTerminalNode: wraps an antlr4::tree::TerminalNode*
// Properties are computed eagerly (terminals are cheap - just 4 fields)
// ============================================================
class LazyTerminalNode {
public:
    int symbol_type;
    std::string text;
    int line;
    int column;

    LazyTerminalNode(antlr4::tree::TerminalNode* t) {
        auto* tok = t->getSymbol();
        symbol_type = static_cast<int>(tok->getType());
        text = tok->getText();
        line = static_cast<int>(tok->getLine());
        column = static_cast<int>(tok->getCharPositionInLine());
    }
};


// ============================================================
// LazyParseNode: wraps an antlr4::ParserRuleContext*
// Children list is built lazily on first .children access
// ============================================================
class LazyParseNode {
public:
    antlr4::ParserRuleContext* ctx;  // raw pointer - kept alive by g_state
    int rule_index;
    int alt_index;

    // Cached properties (computed lazily)
    mutable bool children_built = false;
    mutable py::list children_cache;

    LazyParseNode(antlr4::ParserRuleContext* c) : ctx(c) {
        auto it = g_type_map.find(typeid(*c));
        if (it != g_type_map.end()) {
            rule_index = it->second.first;
            alt_index = it->second.second;
        } else {
            rule_index = static_cast<int>(c->getRuleIndex());
            alt_index = -1;
        }
    }

    py::list get_children() const {
        if (!children_built) {
            children_built = true;
            children_cache = py::list();
            for (auto* child : ctx->children) {
                children_cache.append(wrap_node(child));
            }
        }
        return children_cache;
    }

    int get_start_line() const {
        return ctx->start ? static_cast<int>(ctx->start->getLine()) : 0;
    }

    int get_start_column() const {
        return ctx->start ? static_cast<int>(ctx->start->getCharPositionInLine()) : 0;
    }

    int get_stop_line() const {
        return ctx->stop ? static_cast<int>(ctx->stop->getLine()) : 0;
    }

    int get_stop_column() const {
        return ctx->stop ? static_cast<int>(ctx->stop->getCharPositionInLine()) : 0;
    }

    std::string get_stop_text() const {
        return ctx->stop ? ctx->stop->getText() : "";
    }

    std::string getText() const {
        return ctx->getText();
    }

    py::tuple ctx_id() const {
        return py::make_tuple(rule_index, alt_index);
    }
};


// ============================================================
// wrap_node: creates LazyParseNode or LazyTerminalNode
// ============================================================
static py::object wrap_node(antlr4::tree::ParseTree* node) {
    if (auto* terminal = dynamic_cast<antlr4::tree::TerminalNode*>(node)) {
        return py::cast(LazyTerminalNode(terminal));
    }
    auto* ctx = dynamic_cast<antlr4::ParserRuleContext*>(node);
    if (!ctx) {
        // Shouldn't happen
        auto tn = LazyTerminalNode(nullptr);
        tn.symbol_type = -1;
        tn.text = node->getText();
        tn.line = 0;
        tn.column = 0;
        return py::cast(tn);
    }
    return py::cast(LazyParseNode(ctx));
}


// ============================================================
// parse() function
// ============================================================
static py::object do_parse(const std::string& text) {
    init_type_map();

    g_state.input_text = text;
    g_state.comments.clear();

    g_state.input = std::make_unique<antlr4::ANTLRInputStream>(text);
    g_state.lexer = std::make_unique<VtlLexer>(g_state.input.get());
    g_state.tokens = std::make_unique<antlr4::CommonTokenStream>(g_state.lexer.get());
    g_state.parser = std::make_unique<VtlParser>(g_state.tokens.get());

    // SLL mode for speed
    g_state.parser->getInterpreter<antlr4::atn::ParserATNSimulator>()
        ->setPredictionMode(antlr4::atn::PredictionMode::SLL);

    // Remove default error listeners for cleaner output
    g_state.parser->removeErrorListeners();

    // Parse
    auto* tree = g_state.parser->start();

    // Collect comment tokens
    g_state.tokens->fill();
    for (auto* tok : g_state.tokens->getTokens()) {
        int type = static_cast<int>(tok->getType());
        if (type == VtlParser::ML_COMMENT || type == VtlParser::SL_COMMENT) {
            g_state.comments.push_back({
                type,
                tok->getText(),
                static_cast<int>(tok->getLine()),
                static_cast<int>(tok->getCharPositionInLine())
            });
        }
    }

    // Return lazy root node — children are NOT wrapped yet
    return py::cast(LazyParseNode(tree));
}

static std::string get_input_text() {
    return g_state.input_text;
}

static py::list get_comments() {
    py::list result;
    for (auto& c : g_state.comments) {
        py::dict d;
        d["type"] = c.type;
        d["text"] = c.text;
        d["line"] = c.line;
        d["column"] = c.column;
        result.append(d);
    }
    return result;
}


// ============================================================
// pybind11 module definition
// ============================================================
PYBIND11_MODULE(vtl_cpp_parser, m) {
    m.doc() = "C++ ANTLR4 VTL parser with pybind11 bindings (lazy wrapping)";

    py::class_<LazyTerminalNode>(m, "TerminalNode")
        .def_readonly("symbol_type", &LazyTerminalNode::symbol_type)
        .def_readonly("text", &LazyTerminalNode::text)
        .def_readonly("line", &LazyTerminalNode::line)
        .def_readonly("column", &LazyTerminalNode::column)
        .def_property_readonly("is_terminal", [](const LazyTerminalNode&) { return true; });

    py::class_<LazyParseNode>(m, "ParseNode")
        .def_readonly("rule_index", &LazyParseNode::rule_index)
        .def_readonly("alt_index", &LazyParseNode::alt_index)
        .def_property_readonly("children", &LazyParseNode::get_children)
        .def_property_readonly("start_line", &LazyParseNode::get_start_line)
        .def_property_readonly("start_column", &LazyParseNode::get_start_column)
        .def_property_readonly("stop_line", &LazyParseNode::get_stop_line)
        .def_property_readonly("stop_column", &LazyParseNode::get_stop_column)
        .def_property_readonly("stop_text", &LazyParseNode::get_stop_text)
        .def_property_readonly("is_terminal", [](const LazyParseNode&) { return false; })
        .def_property_readonly("text", &LazyParseNode::getText)
        .def_property_readonly("ctx_id", &LazyParseNode::ctx_id);

    m.def("parse", &do_parse, py::arg("text"),
          "Parse VTL text and return the parse tree root node");
    m.def("get_input_text", &get_input_text,
          "Get the input text from the last parse() call");
    m.def("get_comments", &get_comments,
          "Get comment tokens from the last parse() call");

    // Initialize AST builder lazily on first use
    m.def("init_ast_builder", []() { ASTBuilder::init(); },
          "Initialize the C++ AST builder (cached Python class refs)");

    // Release cached Python refs (call before interpreter shutdown to avoid segfault)
    m.def("cleanup_ast_builder", []() { ASTBuilder::cleanup(); },
          "Release cached Python class refs to prevent segfault at shutdown");

    // AST builder: walks the full parse tree and returns Python AST
    m.def("build_ast", [](py::object parse_node) -> py::object {
        ASTBuilder::init();
        auto& pn = parse_node.cast<LazyParseNode&>();
        return ASTBuilder::build_ast(pn.ctx);
    }, py::arg("root"), "Build a complete Python AST from a C++ parse tree root node");

    // Comment token type (used by ASTComment.py)
    m.attr("ML_COMMENT") = static_cast<int>(VtlParser::ML_COMMENT);

}
