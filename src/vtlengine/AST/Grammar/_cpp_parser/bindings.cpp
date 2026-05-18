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
#include "VtlTokens.h"
#include "Vtl.h"

#include <memory>
#include <optional>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

// ============================================================
// Type → (rule_index, alt_index) mapping
// ============================================================
static std::unordered_map<std::type_index, std::pair<int, int>> g_type_map;

static void init_type_map() {
    if (!g_type_map.empty()) return;

    // Rules without labeled alternatives: alt_index = -1
    g_type_map[typeid(Vtl::StartContext)] = {Vtl::RuleStart, -1};
    g_type_map[typeid(Vtl::DatasetClauseContext)] = {Vtl::RuleDatasetClause, -1};
    g_type_map[typeid(Vtl::RenameClauseContext)] = {Vtl::RuleRenameClause, -1};
    g_type_map[typeid(Vtl::AggrClauseContext)] = {Vtl::RuleAggrClause, -1};
    g_type_map[typeid(Vtl::FilterClauseContext)] = {Vtl::RuleFilterClause, -1};
    g_type_map[typeid(Vtl::CalcClauseContext)] = {Vtl::RuleCalcClause, -1};
    g_type_map[typeid(Vtl::KeepOrDropClauseContext)] = {Vtl::RuleKeepOrDropClause, -1};
    g_type_map[typeid(Vtl::PivotOrUnpivotClauseContext)] = {Vtl::RulePivotOrUnpivotClause, -1};
    g_type_map[typeid(Vtl::CustomPivotClauseContext)] = {Vtl::RuleCustomPivotClause, -1};
    g_type_map[typeid(Vtl::SubspaceClauseContext)] = {Vtl::RuleSubspaceClause, -1};
    g_type_map[typeid(Vtl::RenameClauseItemContext)] = {Vtl::RuleRenameClauseItem, -1};
    g_type_map[typeid(Vtl::AggregateClauseContext)] = {Vtl::RuleAggregateClause, -1};
    g_type_map[typeid(Vtl::AggrFunctionClauseContext)] = {Vtl::RuleAggrFunctionClause, -1};
    g_type_map[typeid(Vtl::CalcClauseItemContext)] = {Vtl::RuleCalcClauseItem, -1};
    g_type_map[typeid(Vtl::SubspaceClauseItemContext)] = {Vtl::RuleSubspaceClauseItem, -1};
    g_type_map[typeid(Vtl::JoinClauseContext)] = {Vtl::RuleJoinClause, -1};
    g_type_map[typeid(Vtl::JoinClauseItemContext)] = {Vtl::RuleJoinClauseItem, -1};
    g_type_map[typeid(Vtl::UsingClauseContext)] = {Vtl::RuleUsingClause, -1};
    g_type_map[typeid(Vtl::NvlJoinClauseContext)] = {Vtl::RuleNvlJoinClause, -1};
    g_type_map[typeid(Vtl::JoinBodyContext)] = {Vtl::RuleJoinBody, -1};
    g_type_map[typeid(Vtl::JoinApplyClauseContext)] = {Vtl::RuleJoinApplyClause, -1};
    g_type_map[typeid(Vtl::StringDistanceMethodsContext)] = {Vtl::RuleStringDistanceMethods, -1};
    g_type_map[typeid(Vtl::OrderByClauseContext)] = {Vtl::RuleOrderByClause, -1};
    g_type_map[typeid(Vtl::OrderByItemContext)] = {Vtl::RuleOrderByItem, -1};
    g_type_map[typeid(Vtl::WindowingClauseContext)] = {Vtl::RuleWindowingClause, -1};
    g_type_map[typeid(Vtl::SignedIntegerContext)] = {Vtl::RuleSignedInteger, -1};
    g_type_map[typeid(Vtl::SignedNumberContext)] = {Vtl::RuleSignedNumber, -1};
    g_type_map[typeid(Vtl::LimitClauseItemContext)] = {Vtl::RuleLimitClauseItem, -1};
    g_type_map[typeid(Vtl::HavingClauseContext)] = {Vtl::RuleHavingClause, -1};
    g_type_map[typeid(Vtl::ParameterItemContext)] = {Vtl::RuleParameterItem, -1};
    g_type_map[typeid(Vtl::OutputParameterTypeContext)] = {Vtl::RuleOutputParameterType, -1};
    g_type_map[typeid(Vtl::OutputParameterTypeComponentContext)] = {Vtl::RuleOutputParameterTypeComponent, -1};
    g_type_map[typeid(Vtl::InputParameterTypeContext)] = {Vtl::RuleInputParameterType, -1};
    g_type_map[typeid(Vtl::RulesetTypeContext)] = {Vtl::RuleRulesetType, -1};
    g_type_map[typeid(Vtl::ScalarTypeContext)] = {Vtl::RuleScalarType, -1};
    g_type_map[typeid(Vtl::ComponentTypeContext)] = {Vtl::RuleComponentType, -1};
    g_type_map[typeid(Vtl::DatasetTypeContext)] = {Vtl::RuleDatasetType, -1};
    g_type_map[typeid(Vtl::EvalDatasetTypeContext)] = {Vtl::RuleEvalDatasetType, -1};
    g_type_map[typeid(Vtl::ScalarSetTypeContext)] = {Vtl::RuleScalarSetType, -1};
    g_type_map[typeid(Vtl::ValueDomainNameContext)] = {Vtl::RuleValueDomainName, -1};
    g_type_map[typeid(Vtl::RulesetIDContext)] = {Vtl::RuleRulesetID, -1};
    g_type_map[typeid(Vtl::RulesetSignatureContext)] = {Vtl::RuleRulesetSignature, -1};
    g_type_map[typeid(Vtl::SignatureContext)] = {Vtl::RuleSignature, -1};
    g_type_map[typeid(Vtl::RuleClauseDatapointContext)] = {Vtl::RuleRuleClauseDatapoint, -1};
    g_type_map[typeid(Vtl::RuleItemDatapointContext)] = {Vtl::RuleRuleItemDatapoint, -1};
    g_type_map[typeid(Vtl::RuleClauseHierarchicalContext)] = {Vtl::RuleRuleClauseHierarchical, -1};
    g_type_map[typeid(Vtl::RuleItemHierarchicalContext)] = {Vtl::RuleRuleItemHierarchical, -1};
    g_type_map[typeid(Vtl::HierRuleSignatureContext)] = {Vtl::RuleHierRuleSignature, -1};
    g_type_map[typeid(Vtl::ValueDomainSignatureContext)] = {Vtl::RuleValueDomainSignature, -1};
    g_type_map[typeid(Vtl::CodeItemRelationContext)] = {Vtl::RuleCodeItemRelation, -1};
    g_type_map[typeid(Vtl::CodeItemRelationClauseContext)] = {Vtl::RuleCodeItemRelationClause, -1};
    g_type_map[typeid(Vtl::ValueDomainValueContext)] = {Vtl::RuleValueDomainValue, -1};
    g_type_map[typeid(Vtl::CompConstraintContext)] = {Vtl::RuleCompConstraint, -1};
    g_type_map[typeid(Vtl::MultModifierContext)] = {Vtl::RuleMultModifier, -1};
    g_type_map[typeid(Vtl::ValidationOutputContext)] = {Vtl::RuleValidationOutput, -1};
    g_type_map[typeid(Vtl::ValidationModeContext)] = {Vtl::RuleValidationMode, -1};
    g_type_map[typeid(Vtl::ConditionClauseContext)] = {Vtl::RuleConditionClause, -1};
    g_type_map[typeid(Vtl::InputModeContext)] = {Vtl::RuleInputMode, -1};
    g_type_map[typeid(Vtl::ImbalanceExprContext)] = {Vtl::RuleImbalanceExpr, -1};
    g_type_map[typeid(Vtl::InputModeHierarchyContext)] = {Vtl::RuleInputModeHierarchy, -1};
    g_type_map[typeid(Vtl::OutputModeHierarchyContext)] = {Vtl::RuleOutputModeHierarchy, -1};
    g_type_map[typeid(Vtl::AliasContext)] = {Vtl::RuleAlias, -1};
    g_type_map[typeid(Vtl::VarIDContext)] = {Vtl::RuleVarID, -1};
    g_type_map[typeid(Vtl::SimpleComponentIdContext)] = {Vtl::RuleSimpleComponentId, -1};
    g_type_map[typeid(Vtl::ComponentIDContext)] = {Vtl::RuleComponentID, -1};
    g_type_map[typeid(Vtl::ListsContext)] = {Vtl::RuleLists, -1};
    g_type_map[typeid(Vtl::ErCodeContext)] = {Vtl::RuleErCode, -1};
    g_type_map[typeid(Vtl::ErLevelContext)] = {Vtl::RuleErLevel, -1};
    g_type_map[typeid(Vtl::ComparisonOperandContext)] = {Vtl::RuleComparisonOperand, -1};
    g_type_map[typeid(Vtl::OptionalExprContext)] = {Vtl::RuleOptionalExpr, -1};
    g_type_map[typeid(Vtl::OptionalExprComponentContext)] = {Vtl::RuleOptionalExprComponent, -1};
    g_type_map[typeid(Vtl::ComponentRoleContext)] = {Vtl::RuleComponentRole, -1};
    g_type_map[typeid(Vtl::ViralAttributeContext)] = {Vtl::RuleViralAttribute, -1};
    g_type_map[typeid(Vtl::ValueDomainIDContext)] = {Vtl::RuleValueDomainID, -1};
    g_type_map[typeid(Vtl::OperatorIDContext)] = {Vtl::RuleOperatorID, -1};
    g_type_map[typeid(Vtl::RoutineNameContext)] = {Vtl::RuleRoutineName, -1};
    g_type_map[typeid(Vtl::BasicScalarTypeContext)] = {Vtl::RuleBasicScalarType, -1};
    g_type_map[typeid(Vtl::RetainTypeContext)] = {Vtl::RuleRetainType, -1};
    g_type_map[typeid(Vtl::ParameterComponentContext)] = {Vtl::RuleParameterComponent, -1};
    g_type_map[typeid(Vtl::ParameterContext)] = {Vtl::RuleParameter, -1};
    g_type_map[typeid(Vtl::HierarchyOperatorsContext)] = {Vtl::RuleHierarchyOperators, -1};

    // Statement alternatives
    g_type_map[typeid(Vtl::TemporaryAssignmentContext)] = {Vtl::RuleStatement, 0};
    g_type_map[typeid(Vtl::PersistAssignmentContext)] = {Vtl::RuleStatement, 1};
    g_type_map[typeid(Vtl::DefineExpressionContext)] = {Vtl::RuleStatement, 2};

    // Expr alternatives
    g_type_map[typeid(Vtl::ParenthesisExprContext)] = {Vtl::RuleExpr, 0};
    g_type_map[typeid(Vtl::FunctionsExpressionContext)] = {Vtl::RuleExpr, 1};
    g_type_map[typeid(Vtl::ClauseExprContext)] = {Vtl::RuleExpr, 2};
    g_type_map[typeid(Vtl::MembershipExprContext)] = {Vtl::RuleExpr, 3};
    g_type_map[typeid(Vtl::UnaryExprContext)] = {Vtl::RuleExpr, 4};
    g_type_map[typeid(Vtl::ArithmeticExprContext)] = {Vtl::RuleExpr, 5};
    g_type_map[typeid(Vtl::ArithmeticExprOrConcatContext)] = {Vtl::RuleExpr, 6};
    g_type_map[typeid(Vtl::ComparisonExprContext)] = {Vtl::RuleExpr, 7};
    g_type_map[typeid(Vtl::InNotInExprContext)] = {Vtl::RuleExpr, 8};
    g_type_map[typeid(Vtl::BooleanExprContext)] = {Vtl::RuleExpr, 9};
    g_type_map[typeid(Vtl::IfExprContext)] = {Vtl::RuleExpr, 10};
    g_type_map[typeid(Vtl::CaseExprContext)] = {Vtl::RuleExpr, 11};
    g_type_map[typeid(Vtl::ConstantExprContext)] = {Vtl::RuleExpr, 12};
    g_type_map[typeid(Vtl::VarIdExprContext)] = {Vtl::RuleExpr, 13};

    // ExprComponent alternatives
    g_type_map[typeid(Vtl::ParenthesisExprCompContext)] = {Vtl::RuleExprComponent, 0};
    g_type_map[typeid(Vtl::FunctionsExpressionCompContext)] = {Vtl::RuleExprComponent, 1};
    g_type_map[typeid(Vtl::UnaryExprCompContext)] = {Vtl::RuleExprComponent, 2};
    g_type_map[typeid(Vtl::ArithmeticExprCompContext)] = {Vtl::RuleExprComponent, 3};
    g_type_map[typeid(Vtl::ArithmeticExprOrConcatCompContext)] = {Vtl::RuleExprComponent, 4};
    g_type_map[typeid(Vtl::ComparisonExprCompContext)] = {Vtl::RuleExprComponent, 5};
    g_type_map[typeid(Vtl::InNotInExprCompContext)] = {Vtl::RuleExprComponent, 6};
    g_type_map[typeid(Vtl::BooleanExprCompContext)] = {Vtl::RuleExprComponent, 7};
    g_type_map[typeid(Vtl::IfExprCompContext)] = {Vtl::RuleExprComponent, 8};
    g_type_map[typeid(Vtl::CaseExprCompContext)] = {Vtl::RuleExprComponent, 9};
    g_type_map[typeid(Vtl::ConstantExprCompContext)] = {Vtl::RuleExprComponent, 10};
    g_type_map[typeid(Vtl::CompIdContext)] = {Vtl::RuleExprComponent, 11};

    // FunctionsComponents alternatives
    g_type_map[typeid(Vtl::GenericFunctionsComponentsContext)] = {Vtl::RuleFunctionsComponents, 0};
    g_type_map[typeid(Vtl::StringFunctionsComponentsContext)] = {Vtl::RuleFunctionsComponents, 1};
    g_type_map[typeid(Vtl::NumericFunctionsComponentsContext)] = {Vtl::RuleFunctionsComponents, 2};
    g_type_map[typeid(Vtl::ComparisonFunctionsComponentsContext)] = {Vtl::RuleFunctionsComponents, 3};
    g_type_map[typeid(Vtl::TimeFunctionsComponentsContext)] = {Vtl::RuleFunctionsComponents, 4};
    g_type_map[typeid(Vtl::ConditionalFunctionsComponentsContext)] = {Vtl::RuleFunctionsComponents, 5};
    g_type_map[typeid(Vtl::AggregateFunctionsComponentsContext)] = {Vtl::RuleFunctionsComponents, 6};
    g_type_map[typeid(Vtl::AnalyticFunctionsComponentsContext)] = {Vtl::RuleFunctionsComponents, 7};

    // Functions alternatives
    g_type_map[typeid(Vtl::JoinFunctionsContext)] = {Vtl::RuleFunctions, 0};
    g_type_map[typeid(Vtl::GenericFunctionsContext)] = {Vtl::RuleFunctions, 1};
    g_type_map[typeid(Vtl::StringFunctionsContext)] = {Vtl::RuleFunctions, 2};
    g_type_map[typeid(Vtl::NumericFunctionsContext)] = {Vtl::RuleFunctions, 3};
    g_type_map[typeid(Vtl::ComparisonFunctionsContext)] = {Vtl::RuleFunctions, 4};
    g_type_map[typeid(Vtl::TimeFunctionsContext)] = {Vtl::RuleFunctions, 5};
    g_type_map[typeid(Vtl::SetFunctionsContext)] = {Vtl::RuleFunctions, 6};
    g_type_map[typeid(Vtl::HierarchyFunctionsContext)] = {Vtl::RuleFunctions, 7};
    g_type_map[typeid(Vtl::ValidationFunctionsContext)] = {Vtl::RuleFunctions, 8};
    g_type_map[typeid(Vtl::ConditionalFunctionsContext)] = {Vtl::RuleFunctions, 9};
    g_type_map[typeid(Vtl::AggregateFunctionsContext)] = {Vtl::RuleFunctions, 10};
    g_type_map[typeid(Vtl::AnalyticFunctionsContext)] = {Vtl::RuleFunctions, 11};

    // JoinOperators alternatives
    g_type_map[typeid(Vtl::InnerJoinExprContext)] = {Vtl::RuleJoinOperators, 0};
    g_type_map[typeid(Vtl::LeftJoinExprContext)] = {Vtl::RuleJoinOperators, 1};
    g_type_map[typeid(Vtl::FullJoinExprContext)] = {Vtl::RuleJoinOperators, 2};
    g_type_map[typeid(Vtl::CrossJoinExprContext)] = {Vtl::RuleJoinOperators, 3};

    // DefOperators alternatives
    g_type_map[typeid(Vtl::DefOperatorContext)] = {Vtl::RuleDefOperators, 0};
    g_type_map[typeid(Vtl::DefDatapointRulesetContext)] = {Vtl::RuleDefOperators, 1};
    g_type_map[typeid(Vtl::DefHierarchicalContext)] = {Vtl::RuleDefOperators, 2};
    g_type_map[typeid(Vtl::DefViralPropagationContext)] = {Vtl::RuleDefOperators, 3};

    // Viral propagation rules without alternatives
    g_type_map[typeid(Vtl::VpSignatureContext)] = {Vtl::RuleVpSignature, -1};
    g_type_map[typeid(Vtl::VpBodyContext)] = {Vtl::RuleVpBody, -1};
    g_type_map[typeid(Vtl::VpConditionContext)] = {Vtl::RuleVpCondition, -1};

    // VpClause alternatives
    g_type_map[typeid(Vtl::EnumeratedVpClauseContext)] = {Vtl::RuleVpClause, 0};
    g_type_map[typeid(Vtl::AggregationVpClauseContext)] = {Vtl::RuleVpClause, 1};
    g_type_map[typeid(Vtl::DefaultVpClauseContext)] = {Vtl::RuleVpClause, 2};

    // GenericOperators alternatives
    g_type_map[typeid(Vtl::CallDatasetContext)] = {Vtl::RuleGenericOperators, 0};
    g_type_map[typeid(Vtl::EvalAtomContext)] = {Vtl::RuleGenericOperators, 1};
    g_type_map[typeid(Vtl::CastExprDatasetContext)] = {Vtl::RuleGenericOperators, 2};

    // GenericOperatorsComponent alternatives
    g_type_map[typeid(Vtl::CallComponentContext)] = {Vtl::RuleGenericOperatorsComponent, 0};
    g_type_map[typeid(Vtl::CastExprComponentContext)] = {Vtl::RuleGenericOperatorsComponent, 1};
    g_type_map[typeid(Vtl::EvalAtomComponentContext)] = {Vtl::RuleGenericOperatorsComponent, 2};

    // StringOperators alternatives
    g_type_map[typeid(Vtl::UnaryStringFunctionContext)] = {Vtl::RuleStringOperators, 0};
    g_type_map[typeid(Vtl::SubstrAtomContext)] = {Vtl::RuleStringOperators, 1};
    g_type_map[typeid(Vtl::ReplaceAtomContext)] = {Vtl::RuleStringOperators, 2};
    g_type_map[typeid(Vtl::InstrAtomContext)] = {Vtl::RuleStringOperators, 3};
    g_type_map[typeid(Vtl::StringDistanceAtomContext)] = {Vtl::RuleStringOperators, 4};

    // StringOperatorsComponent alternatives
    g_type_map[typeid(Vtl::UnaryStringFunctionComponentContext)] = {Vtl::RuleStringOperatorsComponent, 0};
    g_type_map[typeid(Vtl::SubstrAtomComponentContext)] = {Vtl::RuleStringOperatorsComponent, 1};
    g_type_map[typeid(Vtl::ReplaceAtomComponentContext)] = {Vtl::RuleStringOperatorsComponent, 2};
    g_type_map[typeid(Vtl::InstrAtomComponentContext)] = {Vtl::RuleStringOperatorsComponent, 3};
    g_type_map[typeid(Vtl::StringDistanceAtomComponentContext)] = {Vtl::RuleStringOperatorsComponent, 4};

    // NumericOperators alternatives
    g_type_map[typeid(Vtl::UnaryNumericContext)] = {Vtl::RuleNumericOperators, 0};
    g_type_map[typeid(Vtl::UnaryWithOptionalNumericContext)] = {Vtl::RuleNumericOperators, 1};
    g_type_map[typeid(Vtl::BinaryNumericContext)] = {Vtl::RuleNumericOperators, 2};

    // NumericOperatorsComponent alternatives
    g_type_map[typeid(Vtl::UnaryNumericComponentContext)] = {Vtl::RuleNumericOperatorsComponent, 0};
    g_type_map[typeid(Vtl::UnaryWithOptionalNumericComponentContext)] = {Vtl::RuleNumericOperatorsComponent, 1};
    g_type_map[typeid(Vtl::BinaryNumericComponentContext)] = {Vtl::RuleNumericOperatorsComponent, 2};

    // ComparisonOperators alternatives
    g_type_map[typeid(Vtl::BetweenAtomContext)] = {Vtl::RuleComparisonOperators, 0};
    g_type_map[typeid(Vtl::CharsetMatchAtomContext)] = {Vtl::RuleComparisonOperators, 1};
    g_type_map[typeid(Vtl::IsNullAtomContext)] = {Vtl::RuleComparisonOperators, 2};
    g_type_map[typeid(Vtl::ExistInAtomContext)] = {Vtl::RuleComparisonOperators, 3};

    // ComparisonOperatorsComponent alternatives
    g_type_map[typeid(Vtl::BetweenAtomComponentContext)] = {Vtl::RuleComparisonOperatorsComponent, 0};
    g_type_map[typeid(Vtl::CharsetMatchAtomComponentContext)] = {Vtl::RuleComparisonOperatorsComponent, 1};
    g_type_map[typeid(Vtl::IsNullAtomComponentContext)] = {Vtl::RuleComparisonOperatorsComponent, 2};

    // TimeOperators alternatives
    g_type_map[typeid(Vtl::PeriodAtomContext)] = {Vtl::RuleTimeOperators, 0};
    g_type_map[typeid(Vtl::FillTimeAtomContext)] = {Vtl::RuleTimeOperators, 1};
    g_type_map[typeid(Vtl::FlowAtomContext)] = {Vtl::RuleTimeOperators, 2};
    g_type_map[typeid(Vtl::TimeShiftAtomContext)] = {Vtl::RuleTimeOperators, 3};
    g_type_map[typeid(Vtl::TimeAggAtomContext)] = {Vtl::RuleTimeOperators, 4};
    g_type_map[typeid(Vtl::CurrentDateAtomContext)] = {Vtl::RuleTimeOperators, 5};
    g_type_map[typeid(Vtl::DateDiffAtomContext)] = {Vtl::RuleTimeOperators, 6};
    g_type_map[typeid(Vtl::DateAddAtomContext)] = {Vtl::RuleTimeOperators, 7};
    g_type_map[typeid(Vtl::YearAtomContext)] = {Vtl::RuleTimeOperators, 8};
    g_type_map[typeid(Vtl::MonthAtomContext)] = {Vtl::RuleTimeOperators, 9};
    g_type_map[typeid(Vtl::DayOfMonthAtomContext)] = {Vtl::RuleTimeOperators, 10};
    g_type_map[typeid(Vtl::DayOfYearAtomContext)] = {Vtl::RuleTimeOperators, 11};
    g_type_map[typeid(Vtl::DayToYearAtomContext)] = {Vtl::RuleTimeOperators, 12};
    g_type_map[typeid(Vtl::DayToMonthAtomContext)] = {Vtl::RuleTimeOperators, 13};
    g_type_map[typeid(Vtl::YearTodayAtomContext)] = {Vtl::RuleTimeOperators, 14};
    g_type_map[typeid(Vtl::MonthTodayAtomContext)] = {Vtl::RuleTimeOperators, 15};

    // TimeOperatorsComponent alternatives
    g_type_map[typeid(Vtl::PeriodAtomComponentContext)] = {Vtl::RuleTimeOperatorsComponent, 0};
    g_type_map[typeid(Vtl::FillTimeAtomComponentContext)] = {Vtl::RuleTimeOperatorsComponent, 1};
    g_type_map[typeid(Vtl::FlowAtomComponentContext)] = {Vtl::RuleTimeOperatorsComponent, 2};
    g_type_map[typeid(Vtl::TimeShiftAtomComponentContext)] = {Vtl::RuleTimeOperatorsComponent, 3};
    g_type_map[typeid(Vtl::TimeAggAtomComponentContext)] = {Vtl::RuleTimeOperatorsComponent, 4};
    g_type_map[typeid(Vtl::CurrentDateAtomComponentContext)] = {Vtl::RuleTimeOperatorsComponent, 5};
    g_type_map[typeid(Vtl::DateDiffAtomComponentContext)] = {Vtl::RuleTimeOperatorsComponent, 6};
    g_type_map[typeid(Vtl::DateAddAtomComponentContext)] = {Vtl::RuleTimeOperatorsComponent, 7};
    g_type_map[typeid(Vtl::YearAtomComponentContext)] = {Vtl::RuleTimeOperatorsComponent, 8};
    g_type_map[typeid(Vtl::MonthAtomComponentContext)] = {Vtl::RuleTimeOperatorsComponent, 9};
    g_type_map[typeid(Vtl::DayOfMonthAtomComponentContext)] = {Vtl::RuleTimeOperatorsComponent, 10};
    g_type_map[typeid(Vtl::DayOfYearAtomComponentContext)] = {Vtl::RuleTimeOperatorsComponent, 11};
    g_type_map[typeid(Vtl::DayToYearAtomComponentContext)] = {Vtl::RuleTimeOperatorsComponent, 12};
    g_type_map[typeid(Vtl::DayToMonthAtomComponentContext)] = {Vtl::RuleTimeOperatorsComponent, 13};
    g_type_map[typeid(Vtl::YearTodayAtomComponentContext)] = {Vtl::RuleTimeOperatorsComponent, 14};
    g_type_map[typeid(Vtl::MonthTodayAtomComponentContext)] = {Vtl::RuleTimeOperatorsComponent, 15};

    // SetOperators alternatives
    g_type_map[typeid(Vtl::UnionAtomContext)] = {Vtl::RuleSetOperators, 0};
    g_type_map[typeid(Vtl::IntersectAtomContext)] = {Vtl::RuleSetOperators, 1};
    g_type_map[typeid(Vtl::SetOrSYmDiffAtomContext)] = {Vtl::RuleSetOperators, 2};

    // ValidationOperators alternatives
    g_type_map[typeid(Vtl::ValidateDPrulesetContext)] = {Vtl::RuleValidationOperators, 0};
    g_type_map[typeid(Vtl::ValidateHRrulesetContext)] = {Vtl::RuleValidationOperators, 1};
    g_type_map[typeid(Vtl::ValidationSimpleContext)] = {Vtl::RuleValidationOperators, 2};

    // ConditionalOperators alternatives
    g_type_map[typeid(Vtl::NvlAtomContext)] = {Vtl::RuleConditionalOperators, 0};

    // ConditionalOperatorsComponent alternatives
    g_type_map[typeid(Vtl::NvlAtomComponentContext)] = {Vtl::RuleConditionalOperatorsComponent, 0};

    // AggrOperators alternatives
    g_type_map[typeid(Vtl::AggrCompContext)] = {Vtl::RuleAggrOperators, 0};
    g_type_map[typeid(Vtl::CountAggrCompContext)] = {Vtl::RuleAggrOperators, 1};

    // AggrOperatorsGrouping alternatives
    g_type_map[typeid(Vtl::AggrDatasetContext)] = {Vtl::RuleAggrOperatorsGrouping, 0};

    // AnFunction alternatives
    g_type_map[typeid(Vtl::AnSimpleFunctionContext)] = {Vtl::RuleAnFunction, 0};
    g_type_map[typeid(Vtl::LagOrLeadAnContext)] = {Vtl::RuleAnFunction, 1};
    g_type_map[typeid(Vtl::RatioToReportAnContext)] = {Vtl::RuleAnFunction, 2};

    // AnFunctionComponent alternatives
    g_type_map[typeid(Vtl::AnSimpleFunctionComponentContext)] = {Vtl::RuleAnFunctionComponent, 0};
    g_type_map[typeid(Vtl::LagOrLeadAnComponentContext)] = {Vtl::RuleAnFunctionComponent, 1};
    g_type_map[typeid(Vtl::RankAnComponentContext)] = {Vtl::RuleAnFunctionComponent, 2};
    g_type_map[typeid(Vtl::RatioToReportAnComponentContext)] = {Vtl::RuleAnFunctionComponent, 3};

    // ScalarItem alternatives
    g_type_map[typeid(Vtl::SimpleScalarContext)] = {Vtl::RuleScalarItem, 0};
    g_type_map[typeid(Vtl::ScalarWithCastContext)] = {Vtl::RuleScalarItem, 1};

    // GroupingClause alternatives
    g_type_map[typeid(Vtl::GroupByOrExceptContext)] = {Vtl::RuleGroupingClause, 0};
    g_type_map[typeid(Vtl::GroupAllContext)] = {Vtl::RuleGroupingClause, 1};

    // DpRuleset alternatives
    g_type_map[typeid(Vtl::DataPointContext)] = {Vtl::RuleDpRuleset, 0};
    g_type_map[typeid(Vtl::DataPointVdContext)] = {Vtl::RuleDpRuleset, 1};
    g_type_map[typeid(Vtl::DataPointVarContext)] = {Vtl::RuleDpRuleset, 2};

    // HrRuleset alternatives
    g_type_map[typeid(Vtl::HrRulesetTypeContext)] = {Vtl::RuleHrRuleset, 0};
    g_type_map[typeid(Vtl::HrRulesetVdTypeContext)] = {Vtl::RuleHrRuleset, 1};
    g_type_map[typeid(Vtl::HrRulesetVarTypeContext)] = {Vtl::RuleHrRuleset, 2};

    // ScalarTypeConstraint alternatives
    g_type_map[typeid(Vtl::ConditionConstraintContext)] = {Vtl::RuleScalarTypeConstraint, 0};
    g_type_map[typeid(Vtl::RangeConstraintContext)] = {Vtl::RuleScalarTypeConstraint, 1};

    // PartitionByClause alternatives
    g_type_map[typeid(Vtl::PartitionListedContext)] = {Vtl::RulePartitionByClause, 0};
    g_type_map[typeid(Vtl::PartitionExceptAllContext)] = {Vtl::RulePartitionByClause, 1};

    // Constant alternatives
    g_type_map[typeid(Vtl::IntegerLiteralContext)] = {Vtl::RuleConstant, 0};
    g_type_map[typeid(Vtl::NumberLiteralContext)] = {Vtl::RuleConstant, 1};
    g_type_map[typeid(Vtl::BooleanLiteralContext)] = {Vtl::RuleConstant, 2};
    g_type_map[typeid(Vtl::StringLiteralContext)] = {Vtl::RuleConstant, 3};
    g_type_map[typeid(Vtl::NullLiteralContext)] = {Vtl::RuleConstant, 4};
}


// ============================================================
// Module state: holds parser objects alive for the last parse
// ============================================================
struct ParserState {
    std::unique_ptr<antlr4::ANTLRInputStream> input;
    std::unique_ptr<VtlTokens> lexer;
    std::unique_ptr<antlr4::CommonTokenStream> tokens;
    std::unique_ptr<Vtl> parser;
    std::string input_text;

    struct CommentInfo {
        int type;
        std::string text;
        int line;
        int column;
    };
    std::vector<CommentInfo> comments;

    struct SyntaxErrorInfo {
        int line;
        int column;
        std::string message;
        std::string offending_text;
        std::string source_line;
        int underline_length;
    };
    std::optional<SyntaxErrorInfo> syntax_error;
};

static ParserState g_state;


// ============================================================
// Helper: extract one line (1-based) from the input buffer, with
// each \t expanded to TAB_WIDTH spaces. Adjusts the caller-supplied
// column so the caret stays aligned after tab expansion.
// ============================================================
static constexpr int TAB_WIDTH = 4;

static std::string extract_source_line_expanded(int line_1based, int& column_in_out) {
    const std::string& src = g_state.input_text;
    if (line_1based < 1) return "";

    // Find start of the requested line.
    size_t start = 0;
    int current_line = 1;
    while (current_line < line_1based && start < src.size()) {
        if (src[start] == '\n') {
            ++current_line;
        }
        ++start;
    }
    if (current_line != line_1based) return "";

    // Walk to end of the requested line, expanding tabs and tracking the
    // caller's column index (1-based, counted in original-source columns).
    std::string out;
    int orig_col = 1;        // 1-based original column index walking the source
    int target_col = column_in_out;  // 1-based original column we want to remap
    int remapped = target_col;       // 1-based output column after tab expansion
    for (size_t i = start; i < src.size() && src[i] != '\n'; ++i) {
        char c = src[i];
        if (orig_col == target_col) {
            remapped = static_cast<int>(out.size()) + 1;
        }
        if (c == '\t') {
            out.append(TAB_WIDTH, ' ');
        } else if (c != '\r') {
            out.push_back(c);
        }
        ++orig_col;
    }
    // Caret past end of line: snap to end.
    if (target_col > orig_col) {
        remapped = static_cast<int>(out.size()) + 1;
    }
    column_in_out = remapped;
    return out;
}


// ============================================================
// CollectingErrorListener: captures the first syntax error instead
// of printing to stderr. Subsequent errors from ANTLR's error
// recovery are usually noise, so we ignore them.
// ============================================================
class CollectingErrorListener : public antlr4::BaseErrorListener {
public:
    void syntaxError(antlr4::Recognizer* /*recognizer*/,
                     antlr4::Token* offendingSymbol,
                     size_t line,
                     size_t charPositionInLine,
                     const std::string& msg,
                     std::exception_ptr /*e*/) override {
        if (g_state.syntax_error.has_value()) return;

        std::string text;
        int underline_length = 1;
        if (offendingSymbol) {
            text = offendingSymbol->getText();
            size_t start_idx = offendingSymbol->getStartIndex();
            size_t stop_idx = offendingSymbol->getStopIndex();
            if (stop_idx != static_cast<size_t>(-1) && stop_idx >= start_idx) {
                underline_length = static_cast<int>(stop_idx - start_idx + 1);
            }
        }

        // ANTLR uses 0-based columns; we use 1-based externally. The helper
        // rewrites the column to account for tab expansion.
        int column_1based = static_cast<int>(charPositionInLine) + 1;
        std::string src_line = extract_source_line_expanded(
            static_cast<int>(line), column_1based);

        g_state.syntax_error = ParserState::SyntaxErrorInfo{
            static_cast<int>(line),
            column_1based - 1,    // store 0-based for back-compat with Python; API adds 1.
            msg,
            text,
            src_line,
            underline_length
        };
    }
};

static CollectingErrorListener g_collecting_listener;


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
    g_state.syntax_error.reset();

    g_state.input = std::make_unique<antlr4::ANTLRInputStream>(text);
    g_state.lexer = std::make_unique<VtlTokens>(g_state.input.get());
    g_state.tokens = std::make_unique<antlr4::CommonTokenStream>(g_state.lexer.get());
    g_state.parser = std::make_unique<Vtl>(g_state.tokens.get());

    // SLL mode for speed
    g_state.parser->getInterpreter<antlr4::atn::ParserATNSimulator>()
        ->setPredictionMode(antlr4::atn::PredictionMode::SLL);

    // Replace default error listeners (which print to stderr) with the
    // collecting listener so Python can surface a clean error.
    g_state.lexer->removeErrorListeners();
    g_state.lexer->addErrorListener(&g_collecting_listener);
    g_state.parser->removeErrorListeners();
    g_state.parser->addErrorListener(&g_collecting_listener);

    // Parse
    auto* tree = g_state.parser->start();

    // Collect comment tokens
    g_state.tokens->fill();
    for (auto* tok : g_state.tokens->getTokens()) {
        int type = static_cast<int>(tok->getType());
        if (type == Vtl::ML_COMMENT || type == Vtl::SL_COMMENT) {
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

static py::object get_syntax_error() {
    if (!g_state.syntax_error.has_value()) {
        return py::none();
    }
    auto& e = g_state.syntax_error.value();
    py::dict d;
    d["line"] = e.line;
    d["column"] = e.column;
    d["message"] = e.message;
    d["offending_text"] = e.offending_text;
    d["source_line"] = e.source_line;
    d["underline_length"] = e.underline_length;
    return d;
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
    m.def("get_syntax_error", &get_syntax_error,
          "Get the first syntax error from the last parse() call, or None if there were none");

    // Token type constants
    m.attr("LPAREN") = static_cast<int>(Vtl::LPAREN);
    m.attr("RPAREN") = static_cast<int>(Vtl::RPAREN);
    m.attr("QLPAREN") = static_cast<int>(Vtl::QLPAREN);
    m.attr("QRPAREN") = static_cast<int>(Vtl::QRPAREN);
    m.attr("GLPAREN") = static_cast<int>(Vtl::GLPAREN);
    m.attr("GRPAREN") = static_cast<int>(Vtl::GRPAREN);
    m.attr("EQ") = static_cast<int>(Vtl::EQ);
    m.attr("LT") = static_cast<int>(Vtl::LT);
    m.attr("MT") = static_cast<int>(Vtl::MT);
    m.attr("ME") = static_cast<int>(Vtl::ME);
    m.attr("NEQ") = static_cast<int>(Vtl::NEQ);
    m.attr("LE") = static_cast<int>(Vtl::LE);
    m.attr("PLUS") = static_cast<int>(Vtl::PLUS);
    m.attr("MINUS") = static_cast<int>(Vtl::MINUS);
    m.attr("MUL") = static_cast<int>(Vtl::MUL);
    m.attr("DIV") = static_cast<int>(Vtl::DIV);
    m.attr("COMMA") = static_cast<int>(Vtl::COMMA);
    m.attr("POINTER") = static_cast<int>(Vtl::POINTER);
    m.attr("COLON") = static_cast<int>(Vtl::COLON);
    m.attr("ASSIGN") = static_cast<int>(Vtl::ASSIGN);
    m.attr("MEMBERSHIP") = static_cast<int>(Vtl::MEMBERSHIP);
    m.attr("EVAL") = static_cast<int>(Vtl::EVAL);
    m.attr("IF") = static_cast<int>(Vtl::IF);
    m.attr("CASE") = static_cast<int>(Vtl::CASE);
    m.attr("THEN") = static_cast<int>(Vtl::THEN);
    m.attr("ELSE") = static_cast<int>(Vtl::ELSE);
    m.attr("USING") = static_cast<int>(Vtl::USING);
    m.attr("WITH") = static_cast<int>(Vtl::WITH);
    m.attr("CURRENT_DATE") = static_cast<int>(Vtl::CURRENT_DATE);
    m.attr("ON") = static_cast<int>(Vtl::ON);
    m.attr("DROP") = static_cast<int>(Vtl::DROP);
    m.attr("KEEP") = static_cast<int>(Vtl::KEEP);
    m.attr("CALC") = static_cast<int>(Vtl::CALC);
    m.attr("RENAME") = static_cast<int>(Vtl::RENAME);
    m.attr("AS") = static_cast<int>(Vtl::AS);
    m.attr("AND") = static_cast<int>(Vtl::AND);
    m.attr("OR") = static_cast<int>(Vtl::OR);
    m.attr("XOR") = static_cast<int>(Vtl::XOR);
    m.attr("NOT") = static_cast<int>(Vtl::NOT);
    m.attr("BETWEEN") = static_cast<int>(Vtl::BETWEEN);
    m.attr("IN") = static_cast<int>(Vtl::IN);
    m.attr("NOT_IN") = static_cast<int>(Vtl::NOT_IN);
    m.attr("NULL_CONSTANT") = static_cast<int>(Vtl::NULL_CONSTANT);
    m.attr("ISNULL") = static_cast<int>(Vtl::ISNULL);
    m.attr("UNION") = static_cast<int>(Vtl::UNION);
    m.attr("INTERSECT") = static_cast<int>(Vtl::INTERSECT);
    m.attr("CHECK") = static_cast<int>(Vtl::CHECK);
    m.attr("EXISTS_IN") = static_cast<int>(Vtl::EXISTS_IN);
    m.attr("TO") = static_cast<int>(Vtl::TO);
    m.attr("IMBALANCE") = static_cast<int>(Vtl::IMBALANCE);
    m.attr("ERRORCODE") = static_cast<int>(Vtl::ERRORCODE);
    m.attr("ALL") = static_cast<int>(Vtl::ALL);
    m.attr("AGGREGATE") = static_cast<int>(Vtl::AGGREGATE);
    m.attr("ERRORLEVEL") = static_cast<int>(Vtl::ERRORLEVEL);
    m.attr("ORDER") = static_cast<int>(Vtl::ORDER);
    m.attr("BY") = static_cast<int>(Vtl::BY);
    m.attr("RANK") = static_cast<int>(Vtl::RANK);
    m.attr("ASC") = static_cast<int>(Vtl::ASC);
    m.attr("DESC") = static_cast<int>(Vtl::DESC);
    m.attr("MIN") = static_cast<int>(Vtl::MIN);
    m.attr("MAX") = static_cast<int>(Vtl::MAX);
    m.attr("FIRST") = static_cast<int>(Vtl::FIRST);
    m.attr("LAST") = static_cast<int>(Vtl::LAST);
    m.attr("ABS") = static_cast<int>(Vtl::ABS);
    m.attr("LN") = static_cast<int>(Vtl::LN);
    m.attr("LOG") = static_cast<int>(Vtl::LOG);
    m.attr("TRUNC") = static_cast<int>(Vtl::TRUNC);
    m.attr("ROUND") = static_cast<int>(Vtl::ROUND);
    m.attr("POWER") = static_cast<int>(Vtl::POWER);
    m.attr("MOD") = static_cast<int>(Vtl::MOD);
    m.attr("LEN") = static_cast<int>(Vtl::LEN);
    m.attr("CONCAT") = static_cast<int>(Vtl::CONCAT);
    m.attr("TRIM") = static_cast<int>(Vtl::TRIM);
    m.attr("UCASE") = static_cast<int>(Vtl::UCASE);
    m.attr("LCASE") = static_cast<int>(Vtl::LCASE);
    m.attr("SUBSTR") = static_cast<int>(Vtl::SUBSTR);
    m.attr("SUM") = static_cast<int>(Vtl::SUM);
    m.attr("AVG") = static_cast<int>(Vtl::AVG);
    m.attr("MEDIAN") = static_cast<int>(Vtl::MEDIAN);
    m.attr("COUNT") = static_cast<int>(Vtl::COUNT);
    m.attr("DIMENSION") = static_cast<int>(Vtl::DIMENSION);
    m.attr("MEASURE") = static_cast<int>(Vtl::MEASURE);
    m.attr("ATTRIBUTE") = static_cast<int>(Vtl::ATTRIBUTE);
    m.attr("FILTER") = static_cast<int>(Vtl::FILTER);
    m.attr("EXP") = static_cast<int>(Vtl::EXP);
    m.attr("VIRAL") = static_cast<int>(Vtl::VIRAL);
    m.attr("PROPAGATION") = static_cast<int>(Vtl::PROPAGATION);
    m.attr("CHARSET_MATCH") = static_cast<int>(Vtl::CHARSET_MATCH);
    m.attr("NVL") = static_cast<int>(Vtl::NVL);
    m.attr("HIERARCHY") = static_cast<int>(Vtl::HIERARCHY);
    m.attr("OPTIONAL") = static_cast<int>(Vtl::OPTIONAL);
    m.attr("INVALID") = static_cast<int>(Vtl::INVALID);
    m.attr("VALUE_DOMAIN") = static_cast<int>(Vtl::VALUE_DOMAIN);
    m.attr("VARIABLE") = static_cast<int>(Vtl::VARIABLE);
    m.attr("DATA") = static_cast<int>(Vtl::DATA);
    m.attr("DATASET") = static_cast<int>(Vtl::DATASET);
    m.attr("OPERATOR") = static_cast<int>(Vtl::OPERATOR);
    m.attr("DEFINE") = static_cast<int>(Vtl::DEFINE);
    m.attr("PUT_SYMBOL") = static_cast<int>(Vtl::PUT_SYMBOL);
    m.attr("DATAPOINT") = static_cast<int>(Vtl::DATAPOINT);
    m.attr("HIERARCHICAL") = static_cast<int>(Vtl::HIERARCHICAL);
    m.attr("RULESET") = static_cast<int>(Vtl::RULESET);
    m.attr("RULE") = static_cast<int>(Vtl::RULE);
    m.attr("END") = static_cast<int>(Vtl::END);
    m.attr("LTRIM") = static_cast<int>(Vtl::LTRIM);
    m.attr("RTRIM") = static_cast<int>(Vtl::RTRIM);
    m.attr("INSTR") = static_cast<int>(Vtl::INSTR);
    m.attr("REPLACE") = static_cast<int>(Vtl::REPLACE);
    m.attr("CEIL") = static_cast<int>(Vtl::CEIL);
    m.attr("FLOOR") = static_cast<int>(Vtl::FLOOR);
    m.attr("SQRT") = static_cast<int>(Vtl::SQRT);
    m.attr("SETDIFF") = static_cast<int>(Vtl::SETDIFF);
    m.attr("STDDEV_POP") = static_cast<int>(Vtl::STDDEV_POP);
    m.attr("STDDEV_SAMP") = static_cast<int>(Vtl::STDDEV_SAMP);
    m.attr("VAR_POP") = static_cast<int>(Vtl::VAR_POP);
    m.attr("VAR_SAMP") = static_cast<int>(Vtl::VAR_SAMP);
    m.attr("GROUP") = static_cast<int>(Vtl::GROUP);
    m.attr("EXCEPT") = static_cast<int>(Vtl::EXCEPT);
    m.attr("HAVING") = static_cast<int>(Vtl::HAVING);
    m.attr("FIRST_VALUE") = static_cast<int>(Vtl::FIRST_VALUE);
    m.attr("LAST_VALUE") = static_cast<int>(Vtl::LAST_VALUE);
    m.attr("LAG") = static_cast<int>(Vtl::LAG);
    m.attr("LEAD") = static_cast<int>(Vtl::LEAD);
    m.attr("RATIO_TO_REPORT") = static_cast<int>(Vtl::RATIO_TO_REPORT);
    m.attr("OVER") = static_cast<int>(Vtl::OVER);
    m.attr("PRECEDING") = static_cast<int>(Vtl::PRECEDING);
    m.attr("FOLLOWING") = static_cast<int>(Vtl::FOLLOWING);
    m.attr("UNBOUNDED") = static_cast<int>(Vtl::UNBOUNDED);
    m.attr("PARTITION") = static_cast<int>(Vtl::PARTITION);
    m.attr("RANGE") = static_cast<int>(Vtl::RANGE);
    m.attr("CURRENT") = static_cast<int>(Vtl::CURRENT);
    m.attr("FILL_TIME_SERIES") = static_cast<int>(Vtl::FILL_TIME_SERIES);
    m.attr("FLOW_TO_STOCK") = static_cast<int>(Vtl::FLOW_TO_STOCK);
    m.attr("STOCK_TO_FLOW") = static_cast<int>(Vtl::STOCK_TO_FLOW);
    m.attr("TIMESHIFT") = static_cast<int>(Vtl::TIMESHIFT);
    m.attr("CONDITION") = static_cast<int>(Vtl::CONDITION);
    m.attr("BOOLEAN") = static_cast<int>(Vtl::BOOLEAN);
    m.attr("DATE") = static_cast<int>(Vtl::DATE);
    m.attr("TIME_PERIOD") = static_cast<int>(Vtl::TIME_PERIOD);
    m.attr("NUMBER") = static_cast<int>(Vtl::NUMBER);
    m.attr("STRING") = static_cast<int>(Vtl::STRING);
    m.attr("TIME") = static_cast<int>(Vtl::TIME);
    m.attr("INTEGER") = static_cast<int>(Vtl::INTEGER);
    m.attr("IS") = static_cast<int>(Vtl::IS);
    m.attr("WHEN") = static_cast<int>(Vtl::WHEN);
    m.attr("POINTS") = static_cast<int>(Vtl::POINTS);
    m.attr("POINT") = static_cast<int>(Vtl::POINT);
    m.attr("INNER_JOIN") = static_cast<int>(Vtl::INNER_JOIN);
    m.attr("LEFT_JOIN") = static_cast<int>(Vtl::LEFT_JOIN);
    m.attr("CROSS_JOIN") = static_cast<int>(Vtl::CROSS_JOIN);
    m.attr("FULL_JOIN") = static_cast<int>(Vtl::FULL_JOIN);
    m.attr("RETURNS") = static_cast<int>(Vtl::RETURNS);
    m.attr("PIVOT") = static_cast<int>(Vtl::PIVOT);
    m.attr("UNPIVOT") = static_cast<int>(Vtl::UNPIVOT);
    m.attr("SUBSPACE") = static_cast<int>(Vtl::SUBSPACE);
    m.attr("APPLY") = static_cast<int>(Vtl::APPLY);
    m.attr("PERIOD_INDICATOR") = static_cast<int>(Vtl::PERIOD_INDICATOR);
    m.attr("SINGLE") = static_cast<int>(Vtl::SINGLE);
    m.attr("DURATION") = static_cast<int>(Vtl::DURATION);
    m.attr("TIME_AGG") = static_cast<int>(Vtl::TIME_AGG);
    m.attr("CAST") = static_cast<int>(Vtl::CAST);
    m.attr("RULE_PRIORITY") = static_cast<int>(Vtl::RULE_PRIORITY);
    m.attr("DATASET_PRIORITY") = static_cast<int>(Vtl::DATASET_PRIORITY);
    m.attr("DEFAULT") = static_cast<int>(Vtl::DEFAULT);
    m.attr("CHECK_DATAPOINT") = static_cast<int>(Vtl::CHECK_DATAPOINT);
    m.attr("CHECK_HIERARCHY") = static_cast<int>(Vtl::CHECK_HIERARCHY);
    m.attr("COMPUTED") = static_cast<int>(Vtl::COMPUTED);
    m.attr("NON_NULL") = static_cast<int>(Vtl::NON_NULL);
    m.attr("NON_ZERO") = static_cast<int>(Vtl::NON_ZERO);
    m.attr("PARTIAL_NULL") = static_cast<int>(Vtl::PARTIAL_NULL);
    m.attr("PARTIAL_ZERO") = static_cast<int>(Vtl::PARTIAL_ZERO);
    m.attr("ALWAYS_NULL") = static_cast<int>(Vtl::ALWAYS_NULL);
    m.attr("ALWAYS_ZERO") = static_cast<int>(Vtl::ALWAYS_ZERO);
    m.attr("COMPONENTS") = static_cast<int>(Vtl::COMPONENTS);
    m.attr("ALL_MEASURES") = static_cast<int>(Vtl::ALL_MEASURES);
    m.attr("SCALAR") = static_cast<int>(Vtl::SCALAR);
    m.attr("COMPONENT") = static_cast<int>(Vtl::COMPONENT);
    m.attr("DATAPOINT_ON_VD") = static_cast<int>(Vtl::DATAPOINT_ON_VD);
    m.attr("DATAPOINT_ON_VAR") = static_cast<int>(Vtl::DATAPOINT_ON_VAR);
    m.attr("HIERARCHICAL_ON_VD") = static_cast<int>(Vtl::HIERARCHICAL_ON_VD);
    m.attr("HIERARCHICAL_ON_VAR") = static_cast<int>(Vtl::HIERARCHICAL_ON_VAR);
    m.attr("SET") = static_cast<int>(Vtl::SET);
    m.attr("LANGUAGE") = static_cast<int>(Vtl::LANGUAGE);
    m.attr("INTEGER_CONSTANT") = static_cast<int>(Vtl::INTEGER_CONSTANT);
    m.attr("NUMBER_CONSTANT") = static_cast<int>(Vtl::NUMBER_CONSTANT);
    m.attr("BOOLEAN_CONSTANT") = static_cast<int>(Vtl::BOOLEAN_CONSTANT);
    m.attr("STRING_CONSTANT") = static_cast<int>(Vtl::STRING_CONSTANT);
    m.attr("IDENTIFIER") = static_cast<int>(Vtl::IDENTIFIER);
    m.attr("WS") = static_cast<int>(Vtl::WS);
    m.attr("EOL") = static_cast<int>(Vtl::EOL);
    m.attr("ML_COMMENT") = static_cast<int>(Vtl::ML_COMMENT);
    m.attr("SL_COMMENT") = static_cast<int>(Vtl::SL_COMMENT);
    m.attr("TOKEN_EOF") = static_cast<int>(antlr4::Token::EOF);

    // Rule index constants
    m.attr("RULE_START") = static_cast<int>(Vtl::RuleStart);
    m.attr("RULE_STATEMENT") = static_cast<int>(Vtl::RuleStatement);
    m.attr("RULE_EXPR") = static_cast<int>(Vtl::RuleExpr);
    m.attr("RULE_EXPR_COMPONENT") = static_cast<int>(Vtl::RuleExprComponent);
    m.attr("RULE_FUNCTIONS_COMPONENTS") = static_cast<int>(Vtl::RuleFunctionsComponents);
    m.attr("RULE_FUNCTIONS") = static_cast<int>(Vtl::RuleFunctions);
    m.attr("RULE_DATASET_CLAUSE") = static_cast<int>(Vtl::RuleDatasetClause);
    m.attr("RULE_RENAME_CLAUSE") = static_cast<int>(Vtl::RuleRenameClause);
    m.attr("RULE_AGGR_CLAUSE") = static_cast<int>(Vtl::RuleAggrClause);
    m.attr("RULE_FILTER_CLAUSE") = static_cast<int>(Vtl::RuleFilterClause);
    m.attr("RULE_CALC_CLAUSE") = static_cast<int>(Vtl::RuleCalcClause);
    m.attr("RULE_KEEP_OR_DROP_CLAUSE") = static_cast<int>(Vtl::RuleKeepOrDropClause);
    m.attr("RULE_PIVOT_OR_UNPIVOT_CLAUSE") = static_cast<int>(Vtl::RulePivotOrUnpivotClause);
    m.attr("RULE_SUBSPACE_CLAUSE") = static_cast<int>(Vtl::RuleSubspaceClause);
    m.attr("RULE_JOIN_OPERATORS") = static_cast<int>(Vtl::RuleJoinOperators);
    m.attr("RULE_DEF_OPERATORS") = static_cast<int>(Vtl::RuleDefOperators);
    m.attr("RULE_GENERIC_OPERATORS") = static_cast<int>(Vtl::RuleGenericOperators);
    m.attr("RULE_GENERIC_OPERATORS_COMPONENT") = static_cast<int>(Vtl::RuleGenericOperatorsComponent);
    m.attr("RULE_STRING_OPERATORS") = static_cast<int>(Vtl::RuleStringOperators);
    m.attr("RULE_STRING_OPERATORS_COMPONENT") = static_cast<int>(Vtl::RuleStringOperatorsComponent);
    m.attr("RULE_NUMERIC_OPERATORS") = static_cast<int>(Vtl::RuleNumericOperators);
    m.attr("RULE_NUMERIC_OPERATORS_COMPONENT") = static_cast<int>(Vtl::RuleNumericOperatorsComponent);
    m.attr("RULE_COMPARISON_OPERATORS") = static_cast<int>(Vtl::RuleComparisonOperators);
    m.attr("RULE_COMPARISON_OPERATORS_COMPONENT") = static_cast<int>(Vtl::RuleComparisonOperatorsComponent);
    m.attr("RULE_TIME_OPERATORS") = static_cast<int>(Vtl::RuleTimeOperators);
    m.attr("RULE_TIME_OPERATORS_COMPONENT") = static_cast<int>(Vtl::RuleTimeOperatorsComponent);
    m.attr("RULE_SET_OPERATORS") = static_cast<int>(Vtl::RuleSetOperators);
    m.attr("RULE_HIERARCHY_OPERATORS") = static_cast<int>(Vtl::RuleHierarchyOperators);
    m.attr("RULE_VALIDATION_OPERATORS") = static_cast<int>(Vtl::RuleValidationOperators);
    m.attr("RULE_CONDITIONAL_OPERATORS") = static_cast<int>(Vtl::RuleConditionalOperators);
    m.attr("RULE_CONDITIONAL_OPERATORS_COMPONENT") = static_cast<int>(Vtl::RuleConditionalOperatorsComponent);
    m.attr("RULE_AGGR_OPERATORS") = static_cast<int>(Vtl::RuleAggrOperators);
    m.attr("RULE_AGGR_OPERATORS_GROUPING") = static_cast<int>(Vtl::RuleAggrOperatorsGrouping);
    m.attr("RULE_AN_FUNCTION") = static_cast<int>(Vtl::RuleAnFunction);
    m.attr("RULE_AN_FUNCTION_COMPONENT") = static_cast<int>(Vtl::RuleAnFunctionComponent);
    m.attr("RULE_SCALAR_ITEM") = static_cast<int>(Vtl::RuleScalarItem);
    m.attr("RULE_JOIN_CLAUSE") = static_cast<int>(Vtl::RuleJoinClause);
    m.attr("RULE_JOIN_CLAUSE_ITEM") = static_cast<int>(Vtl::RuleJoinClauseItem);
    m.attr("RULE_JOIN_BODY") = static_cast<int>(Vtl::RuleJoinBody);
    m.attr("RULE_PARTITION_BY_CLAUSE") = static_cast<int>(Vtl::RulePartitionByClause);
    m.attr("RULE_ORDER_BY_CLAUSE") = static_cast<int>(Vtl::RuleOrderByClause);
    m.attr("RULE_ORDER_BY_ITEM") = static_cast<int>(Vtl::RuleOrderByItem);
    m.attr("RULE_WINDOWING_CLAUSE") = static_cast<int>(Vtl::RuleWindowingClause);
    m.attr("RULE_SIGNED_INTEGER") = static_cast<int>(Vtl::RuleSignedInteger);
    m.attr("RULE_SIGNED_NUMBER") = static_cast<int>(Vtl::RuleSignedNumber);
    m.attr("RULE_LIMIT_CLAUSE_ITEM") = static_cast<int>(Vtl::RuleLimitClauseItem);
    m.attr("RULE_GROUPING_CLAUSE") = static_cast<int>(Vtl::RuleGroupingClause);
    m.attr("RULE_HAVING_CLAUSE") = static_cast<int>(Vtl::RuleHavingClause);
    m.attr("RULE_PARAMETER_ITEM") = static_cast<int>(Vtl::RuleParameterItem);
    m.attr("RULE_OUTPUT_PARAMETER_TYPE") = static_cast<int>(Vtl::RuleOutputParameterType);
    m.attr("RULE_INPUT_PARAMETER_TYPE") = static_cast<int>(Vtl::RuleInputParameterType);
    m.attr("RULE_RULESET_TYPE") = static_cast<int>(Vtl::RuleRulesetType);
    m.attr("RULE_SCALAR_TYPE") = static_cast<int>(Vtl::RuleScalarType);
    m.attr("RULE_COMPONENT_TYPE") = static_cast<int>(Vtl::RuleComponentType);
    m.attr("RULE_DATASET_TYPE") = static_cast<int>(Vtl::RuleDatasetType);
    m.attr("RULE_SCALAR_SET_TYPE") = static_cast<int>(Vtl::RuleScalarSetType);
    m.attr("RULE_DP_RULESET") = static_cast<int>(Vtl::RuleDpRuleset);
    m.attr("RULE_HR_RULESET") = static_cast<int>(Vtl::RuleHrRuleset);
    m.attr("RULE_VALUE_DOMAIN_NAME") = static_cast<int>(Vtl::RuleValueDomainName);
    m.attr("RULE_RULESET_ID") = static_cast<int>(Vtl::RuleRulesetID);
    m.attr("RULE_RULESET_SIGNATURE") = static_cast<int>(Vtl::RuleRulesetSignature);
    m.attr("RULE_SIGNATURE") = static_cast<int>(Vtl::RuleSignature);
    m.attr("RULE_RULE_ITEM_DATAPOINT") = static_cast<int>(Vtl::RuleRuleItemDatapoint);
    m.attr("RULE_RULE_ITEM_HIERARCHICAL") = static_cast<int>(Vtl::RuleRuleItemHierarchical);
    m.attr("RULE_HIER_RULE_SIGNATURE") = static_cast<int>(Vtl::RuleHierRuleSignature);
    m.attr("RULE_CODE_ITEM_RELATION") = static_cast<int>(Vtl::RuleCodeItemRelation);
    m.attr("RULE_CODE_ITEM_RELATION_CLAUSE") = static_cast<int>(Vtl::RuleCodeItemRelationClause);
    m.attr("RULE_VALUE_DOMAIN_VALUE") = static_cast<int>(Vtl::RuleValueDomainValue);
    m.attr("RULE_SCALAR_TYPE_CONSTRAINT") = static_cast<int>(Vtl::RuleScalarTypeConstraint);
    m.attr("RULE_COMP_CONSTRAINT") = static_cast<int>(Vtl::RuleCompConstraint);
    m.attr("RULE_VALIDATION_OUTPUT") = static_cast<int>(Vtl::RuleValidationOutput);
    m.attr("RULE_VALIDATION_MODE") = static_cast<int>(Vtl::RuleValidationMode);
    m.attr("RULE_CONDITION_CLAUSE") = static_cast<int>(Vtl::RuleConditionClause);
    m.attr("RULE_INPUT_MODE") = static_cast<int>(Vtl::RuleInputMode);
    m.attr("RULE_INPUT_MODE_HIERARCHY") = static_cast<int>(Vtl::RuleInputModeHierarchy);
    m.attr("RULE_OUTPUT_MODE_HIERARCHY") = static_cast<int>(Vtl::RuleOutputModeHierarchy);
    m.attr("RULE_VAR_ID") = static_cast<int>(Vtl::RuleVarID);
    m.attr("RULE_SIMPLE_COMPONENT_ID") = static_cast<int>(Vtl::RuleSimpleComponentId);
    m.attr("RULE_COMPONENT_ID") = static_cast<int>(Vtl::RuleComponentID);
    m.attr("RULE_COMPARISON_OPERAND") = static_cast<int>(Vtl::RuleComparisonOperand);
    m.attr("RULE_OPTIONAL_EXPR") = static_cast<int>(Vtl::RuleOptionalExpr);
    m.attr("RULE_COMPONENT_ROLE") = static_cast<int>(Vtl::RuleComponentRole);
    m.attr("RULE_VIRAL_ATTRIBUTE") = static_cast<int>(Vtl::RuleViralAttribute);
    m.attr("RULE_VALUE_DOMAIN_ID") = static_cast<int>(Vtl::RuleValueDomainID);
    m.attr("RULE_OPERATOR_ID") = static_cast<int>(Vtl::RuleOperatorID);
    m.attr("RULE_CONSTANT") = static_cast<int>(Vtl::RuleConstant);
    m.attr("RULE_BASIC_SCALAR_TYPE") = static_cast<int>(Vtl::RuleBasicScalarType);
    m.attr("RULE_RETAIN_TYPE") = static_cast<int>(Vtl::RuleRetainType);
    m.attr("RULE_ER_CODE") = static_cast<int>(Vtl::RuleErCode);
    m.attr("RULE_ER_LEVEL") = static_cast<int>(Vtl::RuleErLevel);
    m.attr("RULE_ALIAS") = static_cast<int>(Vtl::RuleAlias);
    m.attr("RULE_LISTS") = static_cast<int>(Vtl::RuleLists);
}
