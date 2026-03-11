/**
 * ast_builder.h - C++ AST builder for VTL parse trees
 *
 * Walks the ANTLR C++ parse tree directly and constructs Python AST
 * dataclass instances using pybind11, eliminating the wrapping overhead
 * of the Python-side visitor pattern.
 */

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "antlr4-runtime.h"
#include "VtlParser.h"

#include <string>
#include <typeindex>
#include <unordered_map>
#include <utility>

namespace py = pybind11;

// Defined in bindings.cpp
extern std::unordered_map<std::type_index, std::pair<int, int>> g_type_map;
extern void init_type_map();

namespace ASTBuilder {

/**
 * Initialize cached Python class references. Must be called once
 * (typically at module init time) before any build_ast calls.
 */
void init();

/**
 * Build a Python AST from an ANTLR C++ parse tree.
 * Currently only Terminal-level visitor methods are implemented (Phase 1).
 * Returns a py::object representing the visited AST node.
 */
py::object build_ast(antlr4::ParserRuleContext* root);

// ---- Terminal visitor methods (Phase 1) ----

py::object visitConstant(antlr4::ParserRuleContext* ctx);
py::object visitVarID(antlr4::ParserRuleContext* ctx);
py::object visitVarIdExpr(antlr4::ParserRuleContext* ctx);
py::object visitSimpleComponentId(antlr4::ParserRuleContext* ctx);
py::object visitComponentID(antlr4::ParserRuleContext* ctx);
py::object visitOperatorID(antlr4::ParserRuleContext* ctx);
py::object visitValueDomainID(antlr4::ParserRuleContext* ctx);
py::object visitRulesetID(antlr4::ParserRuleContext* ctx);
py::object visitValueDomainName(antlr4::ParserRuleContext* ctx);
py::object visitValueDomainValue(antlr4::ParserRuleContext* ctx);
py::object visitRoutineName(antlr4::ParserRuleContext* ctx);
py::object visitBasicScalarType(antlr4::ParserRuleContext* ctx);
py::object visitComponentRole(antlr4::ParserRuleContext* ctx);
py::object visitViralAttribute(antlr4::ParserRuleContext* ctx);
py::object visitLists(antlr4::ParserRuleContext* ctx);
py::object visitMultModifier(antlr4::ParserRuleContext* ctx);
py::object visitCompConstraint(antlr4::ParserRuleContext* ctx);
py::object visitSimpleScalar(antlr4::ParserRuleContext* ctx);
py::object visitScalarType(antlr4::ParserRuleContext* ctx);
py::object visitDatasetType(antlr4::ParserRuleContext* ctx);
py::object visitRulesetType(antlr4::ParserRuleContext* ctx);
py::object visitComponentType(antlr4::ParserRuleContext* ctx);
py::object visitInputParameterType(antlr4::ParserRuleContext* ctx);
py::object visitOutputParameterType(antlr4::ParserRuleContext* ctx);
py::object visitOutputParameterTypeComponent(antlr4::ParserRuleContext* ctx);
py::object visitScalarItem(antlr4::ParserRuleContext* ctx);
py::object visitScalarWithCast(antlr4::ParserRuleContext* ctx);
py::object visitScalarSetType(antlr4::ParserRuleContext* ctx);
py::object visitRetainType(antlr4::ParserRuleContext* ctx);
py::object visitEvalDatasetType(antlr4::ParserRuleContext* ctx);
py::object visitAlias(antlr4::ParserRuleContext* ctx);
py::object visitSignedInteger(antlr4::ParserRuleContext* ctx);
py::object visitSignedNumber(antlr4::ParserRuleContext* ctx);
py::object visitComparisonOperand(antlr4::ParserRuleContext* ctx);
py::object visitErCode(antlr4::ParserRuleContext* ctx);
py::object visitErLevel(antlr4::ParserRuleContext* ctx);
py::object visitSignature(antlr4::ParserRuleContext* ctx, const std::string& kind = "ComponentID");
py::object visitConditionClause(antlr4::ParserRuleContext* ctx);
py::object visitValidationMode(antlr4::ParserRuleContext* ctx);
py::object visitValidationOutput(antlr4::ParserRuleContext* ctx);
py::object visitInputMode(antlr4::ParserRuleContext* ctx);
py::object visitInputModeHierarchy(antlr4::ParserRuleContext* ctx);
py::object visitOutputModeHierarchy(antlr4::ParserRuleContext* ctx);
py::object visitPartitionByClause(antlr4::ParserRuleContext* ctx);
py::object visitOrderByClause(antlr4::ParserRuleContext* ctx);
py::object visitWindowingClause(antlr4::ParserRuleContext* ctx);
py::object visitOrderByItem(antlr4::ParserRuleContext* ctx);
std::pair<py::object, std::string> visitLimitClauseItem(antlr4::ParserRuleContext* ctx);

// ---- ExprComponents visitor methods (Phase 2) ----

py::object visitExprComponent(antlr4::ParserRuleContext* ctx);
py::object visitOptionalExprComponent(antlr4::ParserRuleContext* ctx);
py::object visitFunctionsComponents(antlr4::ParserRuleContext* ctx);
py::object visitGenericFunctionsComponents(antlr4::ParserRuleContext* ctx);
py::object visitCallComponent(antlr4::ParserRuleContext* ctx);
py::object visitEvalAtomComponent(antlr4::ParserRuleContext* ctx);
py::object visitCastExprComponent(antlr4::ParserRuleContext* ctx);
py::object visitParameterComponent(antlr4::ParserRuleContext* ctx);
py::object visitStringFunctionsComponents(antlr4::ParserRuleContext* ctx);
py::object visitUnaryStringFunctionComponent(antlr4::ParserRuleContext* ctx);
py::object visitSubstrAtomComponent(antlr4::ParserRuleContext* ctx);
py::object visitReplaceAtomComponent(antlr4::ParserRuleContext* ctx);
py::object visitInstrAtomComponent(antlr4::ParserRuleContext* ctx);
py::object visitNumericFunctionsComponents(antlr4::ParserRuleContext* ctx);
py::object visitUnaryNumericComponent(antlr4::ParserRuleContext* ctx);
py::object visitUnaryWithOptionalNumericComponent(antlr4::ParserRuleContext* ctx);
py::object visitBinaryNumericComponent(antlr4::ParserRuleContext* ctx);
py::object visitTimeFunctionsComponents(antlr4::ParserRuleContext* ctx);
py::object visitTimeUnaryAtomComponent(antlr4::ParserRuleContext* ctx);
py::object visitTimeShiftAtomComponent(antlr4::ParserRuleContext* ctx);
py::object visitFillTimeAtomComponent(antlr4::ParserRuleContext* ctx);
py::object visitTimeAggAtomComponent(antlr4::ParserRuleContext* ctx);
py::object visitCurrentDateAtomComponent(antlr4::ParserRuleContext* ctx);
py::object visitDateDiffAtomComponent(antlr4::ParserRuleContext* ctx);
py::object visitDateAddAtomComponentContext(antlr4::ParserRuleContext* ctx);
py::object visitConditionalFunctionsComponents(antlr4::ParserRuleContext* ctx);
py::object visitNvlAtomComponent(antlr4::ParserRuleContext* ctx);
py::object visitComparisonFunctionsComponents(antlr4::ParserRuleContext* ctx);
py::object visitBetweenAtomComponent(antlr4::ParserRuleContext* ctx);
py::object visitCharsetMatchAtomComponent(antlr4::ParserRuleContext* ctx);
py::object visitIsNullAtomComponent(antlr4::ParserRuleContext* ctx);
py::object visitAggregateFunctionsComponents(antlr4::ParserRuleContext* ctx);
py::object visitAggrComp(antlr4::ParserRuleContext* ctx);
py::object visitCountAggrComp(antlr4::ParserRuleContext* ctx);
py::object visitAnalyticFunctionsComponents(antlr4::ParserRuleContext* ctx);
py::object visitAnSimpleFunctionComponent(antlr4::ParserRuleContext* ctx);
py::object visitLagOrLeadAnComponent(antlr4::ParserRuleContext* ctx);
py::object visitRankAnComponent(antlr4::ParserRuleContext* ctx);
py::object visitRatioToReportAnComponent(antlr4::ParserRuleContext* ctx);

// ---- Forward declaration for Phase 3 (Expr) ----
py::object visitExpr(antlr4::ParserRuleContext* ctx);

} // namespace ASTBuilder
