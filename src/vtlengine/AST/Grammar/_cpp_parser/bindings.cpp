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

    // AST builder: walks the full parse tree and returns Python AST
    m.def("build_ast", [](py::object parse_node) -> py::object {
        ASTBuilder::init();
        auto& pn = parse_node.cast<LazyParseNode&>();
        return ASTBuilder::build_ast(pn.ctx);
    }, py::arg("root"), "Build a complete Python AST from a C++ parse tree root node");

    // Expose individual terminal visitors for incremental adoption
    m.def("visit_constant", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitConstant(pn.ctx);
    }, "Visit a Constant rule node");

    m.def("visit_var_id", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitVarID(pn.ctx);
    }, "Visit a VarID rule node");

    m.def("visit_var_id_expr", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitVarIdExpr(pn.ctx);
    }, "Visit a VarIdExpr rule node");

    m.def("visit_simple_component_id", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitSimpleComponentId(pn.ctx);
    }, "Visit a SimpleComponentId rule node");

    m.def("visit_component_id", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitComponentID(pn.ctx);
    }, "Visit a ComponentID rule node");

    m.def("visit_operator_id", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitOperatorID(pn.ctx);
    }, "Visit an OperatorID rule node");

    m.def("visit_value_domain_id", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitValueDomainID(pn.ctx);
    }, "Visit a ValueDomainID rule node");

    m.def("visit_ruleset_id", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitRulesetID(pn.ctx);
    }, "Visit a RulesetID rule node");

    m.def("visit_value_domain_value", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitValueDomainValue(pn.ctx);
    }, "Visit a ValueDomainValue rule node");

    m.def("visit_routine_name", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitRoutineName(pn.ctx);
    }, "Visit a RoutineName rule node");

    m.def("visit_basic_scalar_type", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitBasicScalarType(pn.ctx);
    }, "Visit a BasicScalarType rule node");

    m.def("visit_component_role", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitComponentRole(pn.ctx);
    }, "Visit a ComponentRole rule node");

    m.def("visit_lists", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitLists(pn.ctx);
    }, "Visit a Lists rule node");

    m.def("visit_comp_constraint", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitCompConstraint(pn.ctx);
    }, "Visit a CompConstraint rule node");

    m.def("visit_simple_scalar", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitSimpleScalar(pn.ctx);
    }, "Visit a SimpleScalar rule node");

    m.def("visit_scalar_type", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitScalarType(pn.ctx);
    }, "Visit a ScalarType rule node");

    m.def("visit_dataset_type", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitDatasetType(pn.ctx);
    }, "Visit a DatasetType rule node");

    m.def("visit_component_type", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitComponentType(pn.ctx);
    }, "Visit a ComponentType rule node");

    m.def("visit_input_parameter_type", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitInputParameterType(pn.ctx);
    }, "Visit an InputParameterType rule node");

    m.def("visit_output_parameter_type", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitOutputParameterType(pn.ctx);
    }, "Visit an OutputParameterType rule node");

    m.def("visit_output_parameter_type_component", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitOutputParameterTypeComponent(pn.ctx);
    }, "Visit an OutputParameterTypeComponent rule node");

    m.def("visit_scalar_item", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitScalarItem(pn.ctx);
    }, "Visit a ScalarItem rule node");

    m.def("visit_scalar_with_cast", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitScalarWithCast(pn.ctx);
    }, "Visit a ScalarWithCast rule node");

    m.def("visit_retain_type", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitRetainType(pn.ctx);
    }, "Visit a RetainType rule node");

    m.def("visit_eval_dataset_type", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitEvalDatasetType(pn.ctx);
    }, "Visit an EvalDatasetType rule node");

    m.def("visit_alias", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitAlias(pn.ctx);
    }, "Visit an Alias rule node");

    m.def("visit_signed_integer", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitSignedInteger(pn.ctx);
    }, "Visit a SignedInteger rule node");

    m.def("visit_signed_number", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitSignedNumber(pn.ctx);
    }, "Visit a SignedNumber rule node");

    m.def("visit_comparison_operand", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitComparisonOperand(pn.ctx);
    }, "Visit a ComparisonOperand rule node");

    m.def("visit_er_code", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitErCode(pn.ctx);
    }, "Visit an ErCode rule node");

    m.def("visit_er_level", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitErLevel(pn.ctx);
    }, "Visit an ErLevel rule node");

    m.def("visit_signature", [](py::object node, const std::string& kind) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitSignature(pn.ctx, kind);
    }, py::arg("node"), py::arg("kind") = "ComponentID",
    "Visit a Signature rule node");

    m.def("visit_condition_clause", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitConditionClause(pn.ctx);
    }, "Visit a ConditionClause rule node");

    m.def("visit_validation_mode", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitValidationMode(pn.ctx);
    }, "Visit a ValidationMode rule node");

    m.def("visit_validation_output", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitValidationOutput(pn.ctx);
    }, "Visit a ValidationOutput rule node");

    m.def("visit_input_mode", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitInputMode(pn.ctx);
    }, "Visit an InputMode rule node");

    m.def("visit_input_mode_hierarchy", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitInputModeHierarchy(pn.ctx);
    }, "Visit an InputModeHierarchy rule node");

    m.def("visit_output_mode_hierarchy", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitOutputModeHierarchy(pn.ctx);
    }, "Visit an OutputModeHierarchy rule node");

    m.def("visit_partition_by_clause", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitPartitionByClause(pn.ctx);
    }, "Visit a PartitionByClause rule node");

    m.def("visit_order_by_clause", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitOrderByClause(pn.ctx);
    }, "Visit an OrderByClause rule node");

    m.def("visit_order_by_item", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitOrderByItem(pn.ctx);
    }, "Visit an OrderByItem rule node");

    m.def("visit_windowing_clause", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitWindowingClause(pn.ctx);
    }, "Visit a WindowingClause rule node");

    // ---- Phase 2: ExprComponents bindings ----

    m.def("visit_expr_component", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitExprComponent(pn.ctx);
    }, "Visit an ExprComponent rule node");

    m.def("visit_optional_expr_component", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitOptionalExprComponent(pn.ctx);
    }, "Visit an OptionalExprComponent rule node");

    m.def("visit_functions_components", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitFunctionsComponents(pn.ctx);
    }, "Visit a FunctionsComponents rule node");

    m.def("visit_call_component", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitCallComponent(pn.ctx);
    }, "Visit a CallComponent rule node");

    m.def("visit_eval_atom_component", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitEvalAtomComponent(pn.ctx);
    }, "Visit an EvalAtomComponent rule node");

    m.def("visit_cast_expr_component", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitCastExprComponent(pn.ctx);
    }, "Visit a CastExprComponent rule node");

    m.def("visit_parameter_component", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitParameterComponent(pn.ctx);
    }, "Visit a ParameterComponent rule node");

    m.def("visit_substr_atom_component", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitSubstrAtomComponent(pn.ctx);
    }, "Visit a SubstrAtomComponent rule node");

    m.def("visit_replace_atom_component", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitReplaceAtomComponent(pn.ctx);
    }, "Visit a ReplaceAtomComponent rule node");

    m.def("visit_instr_atom_component", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitInstrAtomComponent(pn.ctx);
    }, "Visit an InstrAtomComponent rule node");

    m.def("visit_time_agg_atom_component", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitTimeAggAtomComponent(pn.ctx);
    }, "Visit a TimeAggAtomComponent rule node");

    m.def("visit_aggr_comp", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitAggrComp(pn.ctx);
    }, "Visit an AggrComp rule node");

    m.def("visit_count_aggr_comp", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitCountAggrComp(pn.ctx);
    }, "Visit a CountAggrComp rule node");

    m.def("visit_an_simple_function_component", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitAnSimpleFunctionComponent(pn.ctx);
    }, "Visit an AnSimpleFunctionComponent rule node");

    m.def("visit_lag_or_lead_an_component", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitLagOrLeadAnComponent(pn.ctx);
    }, "Visit a LagOrLeadAnComponent rule node");

    m.def("visit_rank_an_component", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitRankAnComponent(pn.ctx);
    }, "Visit a RankAnComponent rule node");

    m.def("visit_ratio_to_report_an_component", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitRatioToReportAnComponent(pn.ctx);
    }, "Visit a RatioToReportAnComponent rule node");

    // Token type constants
    m.attr("LPAREN") = static_cast<int>(VtlParser::LPAREN);
    m.attr("RPAREN") = static_cast<int>(VtlParser::RPAREN);
    m.attr("QLPAREN") = static_cast<int>(VtlParser::QLPAREN);
    m.attr("QRPAREN") = static_cast<int>(VtlParser::QRPAREN);
    m.attr("GLPAREN") = static_cast<int>(VtlParser::GLPAREN);
    m.attr("GRPAREN") = static_cast<int>(VtlParser::GRPAREN);
    m.attr("EQ") = static_cast<int>(VtlParser::EQ);
    m.attr("LT") = static_cast<int>(VtlParser::LT);
    m.attr("MT") = static_cast<int>(VtlParser::MT);
    m.attr("ME") = static_cast<int>(VtlParser::ME);
    m.attr("NEQ") = static_cast<int>(VtlParser::NEQ);
    m.attr("LE") = static_cast<int>(VtlParser::LE);
    m.attr("PLUS") = static_cast<int>(VtlParser::PLUS);
    m.attr("MINUS") = static_cast<int>(VtlParser::MINUS);
    m.attr("MUL") = static_cast<int>(VtlParser::MUL);
    m.attr("DIV") = static_cast<int>(VtlParser::DIV);
    m.attr("COMMA") = static_cast<int>(VtlParser::COMMA);
    m.attr("POINTER") = static_cast<int>(VtlParser::POINTER);
    m.attr("COLON") = static_cast<int>(VtlParser::COLON);
    m.attr("ASSIGN") = static_cast<int>(VtlParser::ASSIGN);
    m.attr("MEMBERSHIP") = static_cast<int>(VtlParser::MEMBERSHIP);
    m.attr("EVAL") = static_cast<int>(VtlParser::EVAL);
    m.attr("IF") = static_cast<int>(VtlParser::IF);
    m.attr("CASE") = static_cast<int>(VtlParser::CASE);
    m.attr("THEN") = static_cast<int>(VtlParser::THEN);
    m.attr("ELSE") = static_cast<int>(VtlParser::ELSE);
    m.attr("USING") = static_cast<int>(VtlParser::USING);
    m.attr("WITH") = static_cast<int>(VtlParser::WITH);
    m.attr("CURRENT_DATE") = static_cast<int>(VtlParser::CURRENT_DATE);
    m.attr("ON") = static_cast<int>(VtlParser::ON);
    m.attr("DROP") = static_cast<int>(VtlParser::DROP);
    m.attr("KEEP") = static_cast<int>(VtlParser::KEEP);
    m.attr("CALC") = static_cast<int>(VtlParser::CALC);
    m.attr("RENAME") = static_cast<int>(VtlParser::RENAME);
    m.attr("AS") = static_cast<int>(VtlParser::AS);
    m.attr("AND") = static_cast<int>(VtlParser::AND);
    m.attr("OR") = static_cast<int>(VtlParser::OR);
    m.attr("XOR") = static_cast<int>(VtlParser::XOR);
    m.attr("NOT") = static_cast<int>(VtlParser::NOT);
    m.attr("BETWEEN") = static_cast<int>(VtlParser::BETWEEN);
    m.attr("IN") = static_cast<int>(VtlParser::IN);
    m.attr("NOT_IN") = static_cast<int>(VtlParser::NOT_IN);
    m.attr("NULL_CONSTANT") = static_cast<int>(VtlParser::NULL_CONSTANT);
    m.attr("ISNULL") = static_cast<int>(VtlParser::ISNULL);
    m.attr("UNION") = static_cast<int>(VtlParser::UNION);
    m.attr("INTERSECT") = static_cast<int>(VtlParser::INTERSECT);
    m.attr("CHECK") = static_cast<int>(VtlParser::CHECK);
    m.attr("EXISTS_IN") = static_cast<int>(VtlParser::EXISTS_IN);
    m.attr("TO") = static_cast<int>(VtlParser::TO);
    m.attr("IMBALANCE") = static_cast<int>(VtlParser::IMBALANCE);
    m.attr("ERRORCODE") = static_cast<int>(VtlParser::ERRORCODE);
    m.attr("ALL") = static_cast<int>(VtlParser::ALL);
    m.attr("AGGREGATE") = static_cast<int>(VtlParser::AGGREGATE);
    m.attr("ERRORLEVEL") = static_cast<int>(VtlParser::ERRORLEVEL);
    m.attr("ORDER") = static_cast<int>(VtlParser::ORDER);
    m.attr("BY") = static_cast<int>(VtlParser::BY);
    m.attr("RANK") = static_cast<int>(VtlParser::RANK);
    m.attr("ASC") = static_cast<int>(VtlParser::ASC);
    m.attr("DESC") = static_cast<int>(VtlParser::DESC);
    m.attr("MIN") = static_cast<int>(VtlParser::MIN);
    m.attr("MAX") = static_cast<int>(VtlParser::MAX);
    m.attr("FIRST") = static_cast<int>(VtlParser::FIRST);
    m.attr("LAST") = static_cast<int>(VtlParser::LAST);
    m.attr("ABS") = static_cast<int>(VtlParser::ABS);
    m.attr("LN") = static_cast<int>(VtlParser::LN);
    m.attr("LOG") = static_cast<int>(VtlParser::LOG);
    m.attr("TRUNC") = static_cast<int>(VtlParser::TRUNC);
    m.attr("ROUND") = static_cast<int>(VtlParser::ROUND);
    m.attr("POWER") = static_cast<int>(VtlParser::POWER);
    m.attr("MOD") = static_cast<int>(VtlParser::MOD);
    m.attr("LEN") = static_cast<int>(VtlParser::LEN);
    m.attr("CONCAT") = static_cast<int>(VtlParser::CONCAT);
    m.attr("TRIM") = static_cast<int>(VtlParser::TRIM);
    m.attr("UCASE") = static_cast<int>(VtlParser::UCASE);
    m.attr("LCASE") = static_cast<int>(VtlParser::LCASE);
    m.attr("SUBSTR") = static_cast<int>(VtlParser::SUBSTR);
    m.attr("SUM") = static_cast<int>(VtlParser::SUM);
    m.attr("AVG") = static_cast<int>(VtlParser::AVG);
    m.attr("MEDIAN") = static_cast<int>(VtlParser::MEDIAN);
    m.attr("COUNT") = static_cast<int>(VtlParser::COUNT);
    m.attr("DIMENSION") = static_cast<int>(VtlParser::DIMENSION);
    m.attr("MEASURE") = static_cast<int>(VtlParser::MEASURE);
    m.attr("ATTRIBUTE") = static_cast<int>(VtlParser::ATTRIBUTE);
    m.attr("FILTER") = static_cast<int>(VtlParser::FILTER);
    m.attr("EXP") = static_cast<int>(VtlParser::EXP);
    m.attr("VIRAL") = static_cast<int>(VtlParser::VIRAL);
    m.attr("CHARSET_MATCH") = static_cast<int>(VtlParser::CHARSET_MATCH);
    m.attr("NVL") = static_cast<int>(VtlParser::NVL);
    m.attr("HIERARCHY") = static_cast<int>(VtlParser::HIERARCHY);
    m.attr("OPTIONAL") = static_cast<int>(VtlParser::OPTIONAL);
    m.attr("INVALID") = static_cast<int>(VtlParser::INVALID);
    m.attr("VALUE_DOMAIN") = static_cast<int>(VtlParser::VALUE_DOMAIN);
    m.attr("VARIABLE") = static_cast<int>(VtlParser::VARIABLE);
    m.attr("DATA") = static_cast<int>(VtlParser::DATA);
    m.attr("DATASET") = static_cast<int>(VtlParser::DATASET);
    m.attr("OPERATOR") = static_cast<int>(VtlParser::OPERATOR);
    m.attr("DEFINE") = static_cast<int>(VtlParser::DEFINE);
    m.attr("PUT_SYMBOL") = static_cast<int>(VtlParser::PUT_SYMBOL);
    m.attr("DATAPOINT") = static_cast<int>(VtlParser::DATAPOINT);
    m.attr("HIERARCHICAL") = static_cast<int>(VtlParser::HIERARCHICAL);
    m.attr("RULESET") = static_cast<int>(VtlParser::RULESET);
    m.attr("RULE") = static_cast<int>(VtlParser::RULE);
    m.attr("END") = static_cast<int>(VtlParser::END);
    m.attr("LTRIM") = static_cast<int>(VtlParser::LTRIM);
    m.attr("RTRIM") = static_cast<int>(VtlParser::RTRIM);
    m.attr("INSTR") = static_cast<int>(VtlParser::INSTR);
    m.attr("REPLACE") = static_cast<int>(VtlParser::REPLACE);
    m.attr("CEIL") = static_cast<int>(VtlParser::CEIL);
    m.attr("FLOOR") = static_cast<int>(VtlParser::FLOOR);
    m.attr("SQRT") = static_cast<int>(VtlParser::SQRT);
    m.attr("SETDIFF") = static_cast<int>(VtlParser::SETDIFF);
    m.attr("STDDEV_POP") = static_cast<int>(VtlParser::STDDEV_POP);
    m.attr("STDDEV_SAMP") = static_cast<int>(VtlParser::STDDEV_SAMP);
    m.attr("VAR_POP") = static_cast<int>(VtlParser::VAR_POP);
    m.attr("VAR_SAMP") = static_cast<int>(VtlParser::VAR_SAMP);
    m.attr("GROUP") = static_cast<int>(VtlParser::GROUP);
    m.attr("EXCEPT") = static_cast<int>(VtlParser::EXCEPT);
    m.attr("HAVING") = static_cast<int>(VtlParser::HAVING);
    m.attr("FIRST_VALUE") = static_cast<int>(VtlParser::FIRST_VALUE);
    m.attr("LAST_VALUE") = static_cast<int>(VtlParser::LAST_VALUE);
    m.attr("LAG") = static_cast<int>(VtlParser::LAG);
    m.attr("LEAD") = static_cast<int>(VtlParser::LEAD);
    m.attr("RATIO_TO_REPORT") = static_cast<int>(VtlParser::RATIO_TO_REPORT);
    m.attr("OVER") = static_cast<int>(VtlParser::OVER);
    m.attr("PRECEDING") = static_cast<int>(VtlParser::PRECEDING);
    m.attr("FOLLOWING") = static_cast<int>(VtlParser::FOLLOWING);
    m.attr("UNBOUNDED") = static_cast<int>(VtlParser::UNBOUNDED);
    m.attr("PARTITION") = static_cast<int>(VtlParser::PARTITION);
    m.attr("RANGE") = static_cast<int>(VtlParser::RANGE);
    m.attr("CURRENT") = static_cast<int>(VtlParser::CURRENT);
    m.attr("FILL_TIME_SERIES") = static_cast<int>(VtlParser::FILL_TIME_SERIES);
    m.attr("FLOW_TO_STOCK") = static_cast<int>(VtlParser::FLOW_TO_STOCK);
    m.attr("STOCK_TO_FLOW") = static_cast<int>(VtlParser::STOCK_TO_FLOW);
    m.attr("TIMESHIFT") = static_cast<int>(VtlParser::TIMESHIFT);
    m.attr("CONDITION") = static_cast<int>(VtlParser::CONDITION);
    m.attr("BOOLEAN") = static_cast<int>(VtlParser::BOOLEAN);
    m.attr("DATE") = static_cast<int>(VtlParser::DATE);
    m.attr("TIME_PERIOD") = static_cast<int>(VtlParser::TIME_PERIOD);
    m.attr("NUMBER") = static_cast<int>(VtlParser::NUMBER);
    m.attr("STRING") = static_cast<int>(VtlParser::STRING);
    m.attr("TIME") = static_cast<int>(VtlParser::TIME);
    m.attr("INTEGER") = static_cast<int>(VtlParser::INTEGER);
    m.attr("IS") = static_cast<int>(VtlParser::IS);
    m.attr("WHEN") = static_cast<int>(VtlParser::WHEN);
    m.attr("POINTS") = static_cast<int>(VtlParser::POINTS);
    m.attr("POINT") = static_cast<int>(VtlParser::POINT);
    m.attr("INNER_JOIN") = static_cast<int>(VtlParser::INNER_JOIN);
    m.attr("LEFT_JOIN") = static_cast<int>(VtlParser::LEFT_JOIN);
    m.attr("CROSS_JOIN") = static_cast<int>(VtlParser::CROSS_JOIN);
    m.attr("FULL_JOIN") = static_cast<int>(VtlParser::FULL_JOIN);
    m.attr("RETURNS") = static_cast<int>(VtlParser::RETURNS);
    m.attr("PIVOT") = static_cast<int>(VtlParser::PIVOT);
    m.attr("UNPIVOT") = static_cast<int>(VtlParser::UNPIVOT);
    m.attr("SUBSPACE") = static_cast<int>(VtlParser::SUBSPACE);
    m.attr("APPLY") = static_cast<int>(VtlParser::APPLY);
    m.attr("PERIOD_INDICATOR") = static_cast<int>(VtlParser::PERIOD_INDICATOR);
    m.attr("SINGLE") = static_cast<int>(VtlParser::SINGLE);
    m.attr("DURATION") = static_cast<int>(VtlParser::DURATION);
    m.attr("TIME_AGG") = static_cast<int>(VtlParser::TIME_AGG);
    m.attr("CAST") = static_cast<int>(VtlParser::CAST);
    m.attr("RULE_PRIORITY") = static_cast<int>(VtlParser::RULE_PRIORITY);
    m.attr("DATASET_PRIORITY") = static_cast<int>(VtlParser::DATASET_PRIORITY);
    m.attr("DEFAULT") = static_cast<int>(VtlParser::DEFAULT);
    m.attr("CHECK_DATAPOINT") = static_cast<int>(VtlParser::CHECK_DATAPOINT);
    m.attr("CHECK_HIERARCHY") = static_cast<int>(VtlParser::CHECK_HIERARCHY);
    m.attr("COMPUTED") = static_cast<int>(VtlParser::COMPUTED);
    m.attr("NON_NULL") = static_cast<int>(VtlParser::NON_NULL);
    m.attr("NON_ZERO") = static_cast<int>(VtlParser::NON_ZERO);
    m.attr("PARTIAL_NULL") = static_cast<int>(VtlParser::PARTIAL_NULL);
    m.attr("PARTIAL_ZERO") = static_cast<int>(VtlParser::PARTIAL_ZERO);
    m.attr("ALWAYS_NULL") = static_cast<int>(VtlParser::ALWAYS_NULL);
    m.attr("ALWAYS_ZERO") = static_cast<int>(VtlParser::ALWAYS_ZERO);
    m.attr("COMPONENTS") = static_cast<int>(VtlParser::COMPONENTS);
    m.attr("ALL_MEASURES") = static_cast<int>(VtlParser::ALL_MEASURES);
    m.attr("SCALAR") = static_cast<int>(VtlParser::SCALAR);
    m.attr("COMPONENT") = static_cast<int>(VtlParser::COMPONENT);
    m.attr("DATAPOINT_ON_VD") = static_cast<int>(VtlParser::DATAPOINT_ON_VD);
    m.attr("DATAPOINT_ON_VAR") = static_cast<int>(VtlParser::DATAPOINT_ON_VAR);
    m.attr("HIERARCHICAL_ON_VD") = static_cast<int>(VtlParser::HIERARCHICAL_ON_VD);
    m.attr("HIERARCHICAL_ON_VAR") = static_cast<int>(VtlParser::HIERARCHICAL_ON_VAR);
    m.attr("SET") = static_cast<int>(VtlParser::SET);
    m.attr("LANGUAGE") = static_cast<int>(VtlParser::LANGUAGE);
    m.attr("INTEGER_CONSTANT") = static_cast<int>(VtlParser::INTEGER_CONSTANT);
    m.attr("NUMBER_CONSTANT") = static_cast<int>(VtlParser::NUMBER_CONSTANT);
    m.attr("BOOLEAN_CONSTANT") = static_cast<int>(VtlParser::BOOLEAN_CONSTANT);
    m.attr("STRING_CONSTANT") = static_cast<int>(VtlParser::STRING_CONSTANT);
    m.attr("IDENTIFIER") = static_cast<int>(VtlParser::IDENTIFIER);
    m.attr("WS") = static_cast<int>(VtlParser::WS);
    m.attr("EOL") = static_cast<int>(VtlParser::EOL);
    m.attr("ML_COMMENT") = static_cast<int>(VtlParser::ML_COMMENT);
    m.attr("SL_COMMENT") = static_cast<int>(VtlParser::SL_COMMENT);
    m.attr("TOKEN_EOF") = static_cast<int>(antlr4::Token::EOF);

    // Rule index constants
    m.attr("RULE_START") = static_cast<int>(VtlParser::RuleStart);
    m.attr("RULE_STATEMENT") = static_cast<int>(VtlParser::RuleStatement);
    m.attr("RULE_EXPR") = static_cast<int>(VtlParser::RuleExpr);
    m.attr("RULE_EXPR_COMPONENT") = static_cast<int>(VtlParser::RuleExprComponent);
    m.attr("RULE_FUNCTIONS_COMPONENTS") = static_cast<int>(VtlParser::RuleFunctionsComponents);
    m.attr("RULE_FUNCTIONS") = static_cast<int>(VtlParser::RuleFunctions);
    m.attr("RULE_DATASET_CLAUSE") = static_cast<int>(VtlParser::RuleDatasetClause);
    m.attr("RULE_RENAME_CLAUSE") = static_cast<int>(VtlParser::RuleRenameClause);
    m.attr("RULE_AGGR_CLAUSE") = static_cast<int>(VtlParser::RuleAggrClause);
    m.attr("RULE_FILTER_CLAUSE") = static_cast<int>(VtlParser::RuleFilterClause);
    m.attr("RULE_CALC_CLAUSE") = static_cast<int>(VtlParser::RuleCalcClause);
    m.attr("RULE_KEEP_OR_DROP_CLAUSE") = static_cast<int>(VtlParser::RuleKeepOrDropClause);
    m.attr("RULE_PIVOT_OR_UNPIVOT_CLAUSE") = static_cast<int>(VtlParser::RulePivotOrUnpivotClause);
    m.attr("RULE_SUBSPACE_CLAUSE") = static_cast<int>(VtlParser::RuleSubspaceClause);
    m.attr("RULE_JOIN_OPERATORS") = static_cast<int>(VtlParser::RuleJoinOperators);
    m.attr("RULE_DEF_OPERATORS") = static_cast<int>(VtlParser::RuleDefOperators);
    m.attr("RULE_GENERIC_OPERATORS") = static_cast<int>(VtlParser::RuleGenericOperators);
    m.attr("RULE_GENERIC_OPERATORS_COMPONENT") = static_cast<int>(VtlParser::RuleGenericOperatorsComponent);
    m.attr("RULE_STRING_OPERATORS") = static_cast<int>(VtlParser::RuleStringOperators);
    m.attr("RULE_STRING_OPERATORS_COMPONENT") = static_cast<int>(VtlParser::RuleStringOperatorsComponent);
    m.attr("RULE_NUMERIC_OPERATORS") = static_cast<int>(VtlParser::RuleNumericOperators);
    m.attr("RULE_NUMERIC_OPERATORS_COMPONENT") = static_cast<int>(VtlParser::RuleNumericOperatorsComponent);
    m.attr("RULE_COMPARISON_OPERATORS") = static_cast<int>(VtlParser::RuleComparisonOperators);
    m.attr("RULE_COMPARISON_OPERATORS_COMPONENT") = static_cast<int>(VtlParser::RuleComparisonOperatorsComponent);
    m.attr("RULE_TIME_OPERATORS") = static_cast<int>(VtlParser::RuleTimeOperators);
    m.attr("RULE_TIME_OPERATORS_COMPONENT") = static_cast<int>(VtlParser::RuleTimeOperatorsComponent);
    m.attr("RULE_SET_OPERATORS") = static_cast<int>(VtlParser::RuleSetOperators);
    m.attr("RULE_HIERARCHY_OPERATORS") = static_cast<int>(VtlParser::RuleHierarchyOperators);
    m.attr("RULE_VALIDATION_OPERATORS") = static_cast<int>(VtlParser::RuleValidationOperators);
    m.attr("RULE_CONDITIONAL_OPERATORS") = static_cast<int>(VtlParser::RuleConditionalOperators);
    m.attr("RULE_CONDITIONAL_OPERATORS_COMPONENT") = static_cast<int>(VtlParser::RuleConditionalOperatorsComponent);
    m.attr("RULE_AGGR_OPERATORS") = static_cast<int>(VtlParser::RuleAggrOperators);
    m.attr("RULE_AGGR_OPERATORS_GROUPING") = static_cast<int>(VtlParser::RuleAggrOperatorsGrouping);
    m.attr("RULE_AN_FUNCTION") = static_cast<int>(VtlParser::RuleAnFunction);
    m.attr("RULE_AN_FUNCTION_COMPONENT") = static_cast<int>(VtlParser::RuleAnFunctionComponent);
    m.attr("RULE_SCALAR_ITEM") = static_cast<int>(VtlParser::RuleScalarItem);
    m.attr("RULE_JOIN_CLAUSE") = static_cast<int>(VtlParser::RuleJoinClause);
    m.attr("RULE_JOIN_CLAUSE_ITEM") = static_cast<int>(VtlParser::RuleJoinClauseItem);
    m.attr("RULE_JOIN_BODY") = static_cast<int>(VtlParser::RuleJoinBody);
    m.attr("RULE_PARTITION_BY_CLAUSE") = static_cast<int>(VtlParser::RulePartitionByClause);
    m.attr("RULE_ORDER_BY_CLAUSE") = static_cast<int>(VtlParser::RuleOrderByClause);
    m.attr("RULE_ORDER_BY_ITEM") = static_cast<int>(VtlParser::RuleOrderByItem);
    m.attr("RULE_WINDOWING_CLAUSE") = static_cast<int>(VtlParser::RuleWindowingClause);
    m.attr("RULE_SIGNED_INTEGER") = static_cast<int>(VtlParser::RuleSignedInteger);
    m.attr("RULE_SIGNED_NUMBER") = static_cast<int>(VtlParser::RuleSignedNumber);
    m.attr("RULE_LIMIT_CLAUSE_ITEM") = static_cast<int>(VtlParser::RuleLimitClauseItem);
    m.attr("RULE_GROUPING_CLAUSE") = static_cast<int>(VtlParser::RuleGroupingClause);
    m.attr("RULE_HAVING_CLAUSE") = static_cast<int>(VtlParser::RuleHavingClause);
    m.attr("RULE_PARAMETER_ITEM") = static_cast<int>(VtlParser::RuleParameterItem);
    m.attr("RULE_OUTPUT_PARAMETER_TYPE") = static_cast<int>(VtlParser::RuleOutputParameterType);
    m.attr("RULE_INPUT_PARAMETER_TYPE") = static_cast<int>(VtlParser::RuleInputParameterType);
    m.attr("RULE_RULESET_TYPE") = static_cast<int>(VtlParser::RuleRulesetType);
    m.attr("RULE_SCALAR_TYPE") = static_cast<int>(VtlParser::RuleScalarType);
    m.attr("RULE_COMPONENT_TYPE") = static_cast<int>(VtlParser::RuleComponentType);
    m.attr("RULE_DATASET_TYPE") = static_cast<int>(VtlParser::RuleDatasetType);
    m.attr("RULE_SCALAR_SET_TYPE") = static_cast<int>(VtlParser::RuleScalarSetType);
    m.attr("RULE_DP_RULESET") = static_cast<int>(VtlParser::RuleDpRuleset);
    m.attr("RULE_HR_RULESET") = static_cast<int>(VtlParser::RuleHrRuleset);
    m.attr("RULE_VALUE_DOMAIN_NAME") = static_cast<int>(VtlParser::RuleValueDomainName);
    m.attr("RULE_RULESET_ID") = static_cast<int>(VtlParser::RuleRulesetID);
    m.attr("RULE_RULESET_SIGNATURE") = static_cast<int>(VtlParser::RuleRulesetSignature);
    m.attr("RULE_SIGNATURE") = static_cast<int>(VtlParser::RuleSignature);
    m.attr("RULE_RULE_ITEM_DATAPOINT") = static_cast<int>(VtlParser::RuleRuleItemDatapoint);
    m.attr("RULE_RULE_ITEM_HIERARCHICAL") = static_cast<int>(VtlParser::RuleRuleItemHierarchical);
    m.attr("RULE_HIER_RULE_SIGNATURE") = static_cast<int>(VtlParser::RuleHierRuleSignature);
    m.attr("RULE_CODE_ITEM_RELATION") = static_cast<int>(VtlParser::RuleCodeItemRelation);
    m.attr("RULE_CODE_ITEM_RELATION_CLAUSE") = static_cast<int>(VtlParser::RuleCodeItemRelationClause);
    m.attr("RULE_VALUE_DOMAIN_VALUE") = static_cast<int>(VtlParser::RuleValueDomainValue);
    m.attr("RULE_SCALAR_TYPE_CONSTRAINT") = static_cast<int>(VtlParser::RuleScalarTypeConstraint);
    m.attr("RULE_COMP_CONSTRAINT") = static_cast<int>(VtlParser::RuleCompConstraint);
    m.attr("RULE_VALIDATION_OUTPUT") = static_cast<int>(VtlParser::RuleValidationOutput);
    m.attr("RULE_VALIDATION_MODE") = static_cast<int>(VtlParser::RuleValidationMode);
    m.attr("RULE_CONDITION_CLAUSE") = static_cast<int>(VtlParser::RuleConditionClause);
    m.attr("RULE_INPUT_MODE") = static_cast<int>(VtlParser::RuleInputMode);
    m.attr("RULE_INPUT_MODE_HIERARCHY") = static_cast<int>(VtlParser::RuleInputModeHierarchy);
    m.attr("RULE_OUTPUT_MODE_HIERARCHY") = static_cast<int>(VtlParser::RuleOutputModeHierarchy);
    m.attr("RULE_VAR_ID") = static_cast<int>(VtlParser::RuleVarID);
    m.attr("RULE_SIMPLE_COMPONENT_ID") = static_cast<int>(VtlParser::RuleSimpleComponentId);
    m.attr("RULE_COMPONENT_ID") = static_cast<int>(VtlParser::RuleComponentID);
    m.attr("RULE_COMPARISON_OPERAND") = static_cast<int>(VtlParser::RuleComparisonOperand);
    m.attr("RULE_OPTIONAL_EXPR") = static_cast<int>(VtlParser::RuleOptionalExpr);
    m.attr("RULE_COMPONENT_ROLE") = static_cast<int>(VtlParser::RuleComponentRole);
    m.attr("RULE_VIRAL_ATTRIBUTE") = static_cast<int>(VtlParser::RuleViralAttribute);
    m.attr("RULE_VALUE_DOMAIN_ID") = static_cast<int>(VtlParser::RuleValueDomainID);
    m.attr("RULE_OPERATOR_ID") = static_cast<int>(VtlParser::RuleOperatorID);
    m.attr("RULE_CONSTANT") = static_cast<int>(VtlParser::RuleConstant);
    m.attr("RULE_BASIC_SCALAR_TYPE") = static_cast<int>(VtlParser::RuleBasicScalarType);
    m.attr("RULE_RETAIN_TYPE") = static_cast<int>(VtlParser::RuleRetainType);
    m.attr("RULE_ER_CODE") = static_cast<int>(VtlParser::RuleErCode);
    m.attr("RULE_ER_LEVEL") = static_cast<int>(VtlParser::RuleErLevel);
    m.attr("RULE_ALIAS") = static_cast<int>(VtlParser::RuleAlias);
    m.attr("RULE_LISTS") = static_cast<int>(VtlParser::RuleLists);

    // ---- Phase 3: Expr bindings ----

    m.def("visit_expr", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitExpr(pn.ctx);
    }, "Visit an Expr rule node");

    m.def("visit_optional_expr", [](py::object node) {
        ASTBuilder::init();
        auto& pn = node.cast<LazyParseNode&>();
        return ASTBuilder::visitOptionalExpr(pn.ctx);
    }, "Visit an OptionalExpr rule node");

}
