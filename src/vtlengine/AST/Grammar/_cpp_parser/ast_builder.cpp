/**
 * ast_builder.cpp - C++ AST builder implementation
 *
 * Phase 0: Infrastructure (cached Python refs, helpers)
 * Phase 1: Full port of all Terminals.py visitor methods
 */

#include "ast_builder.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

// ============================================================
// Cached Python class references (Phase 0)
// ============================================================

// AST node classes
static py::object py_Start, py_Assignment, py_PersistentAssignment, py_VarID, py_UnaryOp, py_BinOp;
static py::object py_MulOp, py_ParamOp, py_UDOCall, py_JoinOp, py_Constant, py_ParamConstant;
static py::object py_Identifier, py_ID, py_Collection, py_Windowing, py_OrderBy, py_Analytic;
static py::object py_RegularAggregation, py_RenameNode, py_Aggregation, py_TimeAggregation;
static py::object py_If, py_CaseObj, py_Case, py_Validation, py_ComponentType_AST, py_ASTScalarType;
static py::object py_DatasetType_AST, py_Types, py_Argument, py_Operator, py_DefIdentifier;
static py::object py_DPRIdentifier, py_HRBinOp, py_HRUnOp, py_HRule, py_DPRule;
static py::object py_HRuleset, py_HROperation, py_DPValidation, py_DPRuleset;
static py::object py_EvalOp, py_NoOp, py_ParFunction, py_Comment;

// ScalarType classes
static py::object py_String, py_Integer, py_Number, py_Boolean, py_Date;
static py::object py_TimePeriod, py_Duration, py_TimeInterval;

// Model classes
static py::object py_Role, py_Component, py_Dataset, py_Scalar;

// Exception classes
static py::object py_SemanticError;

// Initialized flag
static bool g_initialized = false;

// ============================================================
// Helper functions
// ============================================================

static std::string remove_escaped_chars(const std::string& text) {
    std::string result;
    result.reserve(text.size());
    for (char c : text) {
        if (c != '\'') {
            result.push_back(c);
        }
    }
    return result;
}

static py::dict extract_token_info(antlr4::ParserRuleContext* ctx) {
    py::dict d;
    d["line_start"] = ctx->start ? (int)ctx->start->getLine() : 0;
    d["column_start"] = ctx->start ? (int)ctx->start->getCharPositionInLine() : 0;
    d["line_stop"] = ctx->stop ? (int)ctx->stop->getLine() : 0;
    std::string stop_text = ctx->stop ? ctx->stop->getText() : "";
    if (stop_text == "<EOF>") stop_text = "";
    d["column_stop"] = ctx->stop ? (int)ctx->stop->getCharPositionInLine() + (int)stop_text.size() : 0;
    return d;
}

static py::dict extract_token_info_terminal(antlr4::tree::TerminalNode* t) {
    auto* tok = t->getSymbol();
    py::dict d;
    int col = (int)tok->getCharPositionInLine();
    int line = (int)tok->getLine();
    std::string text = tok->getText();
    d["line_start"] = line;
    d["column_start"] = col;
    d["line_stop"] = line;
    d["column_stop"] = col + (int)text.size();
    return d;
}

static std::pair<int, int> get_ctx_id(antlr4::ParserRuleContext* ctx) {
    auto it = g_type_map.find(typeid(*ctx));
    if (it != g_type_map.end()) return it->second;
    return {(int)ctx->getRuleIndex(), -1};
}

static bool is_terminal(antlr4::tree::ParseTree* node) {
    return dynamic_cast<antlr4::tree::TerminalNode*>(node) != nullptr;
}

static antlr4::tree::TerminalNode* as_terminal(antlr4::tree::ParseTree* node) {
    return dynamic_cast<antlr4::tree::TerminalNode*>(node);
}

static antlr4::ParserRuleContext* as_rule(antlr4::tree::ParseTree* node) {
    return dynamic_cast<antlr4::ParserRuleContext*>(node);
}

static int terminal_type(antlr4::tree::ParseTree* node) {
    auto* t = as_terminal(node);
    if (!t) return -1;
    return (int)t->getSymbol()->getType();
}

static std::string terminal_text(antlr4::tree::ParseTree* node) {
    auto* t = as_terminal(node);
    if (!t) return "";
    return t->getSymbol()->getText();
}

static std::string node_text(antlr4::tree::ParseTree* node) {
    return node->getText();
}

// Helper to call a Python class with keyword arguments from a dict
// py_cls(**kwargs)
static py::object call_with_kwargs(py::object& cls, py::dict kwargs) {
    py::tuple args = py::make_tuple();
    return cls(*args, **kwargs);
}

// Helper: create Windowing node
static py::object create_windowing(const std::string& win_mode,
                                   py::object val0, py::object val1,
                                   const std::string& mode0, const std::string& mode1,
                                   py::dict& token_info) {
    // Normalize values: -1 -> "unbounded", 0 -> "current row"
    auto normalize = [](py::object v) -> py::object {
        if (py::isinstance<py::int_>(v)) {
            int iv = v.cast<int>();
            if (iv == -1) return py::str("unbounded");
            if (iv == 0) return py::str("current row");
        }
        return v;
    };

    py::object start = normalize(val0);
    py::object stop = normalize(val1);

    py::dict kwargs;
    kwargs["type_"] = win_mode;
    kwargs["start"] = start;
    kwargs["stop"] = stop;
    kwargs["start_mode"] = mode0;
    kwargs["stop_mode"] = mode1;
    // Merge token_info
    for (auto item : token_info) {
        kwargs[item.first] = item.second;
    }

    return call_with_kwargs(py_Windowing, kwargs);
}

// ============================================================
// ASTBuilder implementation
// ============================================================

namespace ASTBuilder {

void init() {
    if (g_initialized) return;

    init_type_map();

    auto ast_mod = py::module_::import("vtlengine.AST");
    py_Start = ast_mod.attr("Start");
    py_Assignment = ast_mod.attr("Assignment");
    py_PersistentAssignment = ast_mod.attr("PersistentAssignment");
    py_VarID = ast_mod.attr("VarID");
    py_UnaryOp = ast_mod.attr("UnaryOp");
    py_BinOp = ast_mod.attr("BinOp");
    py_MulOp = ast_mod.attr("MulOp");
    py_ParamOp = ast_mod.attr("ParamOp");
    py_UDOCall = ast_mod.attr("UDOCall");
    py_JoinOp = ast_mod.attr("JoinOp");
    py_Constant = ast_mod.attr("Constant");
    py_ParamConstant = ast_mod.attr("ParamConstant");
    py_Identifier = ast_mod.attr("Identifier");
    py_ID = ast_mod.attr("ID");
    py_Collection = ast_mod.attr("Collection");
    py_Windowing = ast_mod.attr("Windowing");
    py_OrderBy = ast_mod.attr("OrderBy");
    py_Analytic = ast_mod.attr("Analytic");
    py_RegularAggregation = ast_mod.attr("RegularAggregation");
    py_RenameNode = ast_mod.attr("RenameNode");
    py_Aggregation = ast_mod.attr("Aggregation");
    py_TimeAggregation = ast_mod.attr("TimeAggregation");
    py_If = ast_mod.attr("If");
    py_CaseObj = ast_mod.attr("CaseObj");
    py_Case = ast_mod.attr("Case");
    py_DPRIdentifier = ast_mod.attr("DPRIdentifier");
    py_Comment = ast_mod.attr("Comment");
    py_EvalOp = ast_mod.attr("EvalOp");
    py_DefIdentifier = ast_mod.attr("DefIdentifier");

    auto dt_mod = py::module_::import("vtlengine.DataTypes");
    py_String = dt_mod.attr("String");
    py_Integer = dt_mod.attr("Integer");
    py_Number = dt_mod.attr("Number");
    py_Boolean = dt_mod.attr("Boolean");
    py_Date = dt_mod.attr("Date");
    py_TimePeriod = dt_mod.attr("TimePeriod");
    py_Duration = dt_mod.attr("Duration");
    py_TimeInterval = dt_mod.attr("TimeInterval");

    auto model_mod = py::module_::import("vtlengine.Model");
    py_Role = model_mod.attr("Role");
    py_Component = model_mod.attr("Component");
    py_Dataset = model_mod.attr("Dataset");
    py_Scalar = model_mod.attr("Scalar");

    auto exc_mod = py::module_::import("vtlengine.Exceptions");
    py_SemanticError = exc_mod.attr("SemanticError");

    g_initialized = true;
}

// ============================================================
// build_ast - entry point (placeholder for full tree walk)
// ============================================================
py::object build_ast(antlr4::ParserRuleContext* root) {
    if (!g_initialized) init();
    // Phase 1 only provides terminal visitors.
    // Full tree walk will be added in Phase 2+.
    // For now, return None to indicate not yet implemented.
    return py::none();
}

// ============================================================
// Terminal visitors (Phase 1) - Full implementation
// ============================================================

py::object visitConstant(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    auto* child = children[0];
    auto ti = extract_token_info(ctx);

    auto* rule_child = as_rule(child);
    if (rule_child) {
        auto cid = get_ctx_id(rule_child);
        if (cid.first == VtlParser::RuleSignedInteger && cid.second == -1) {
            py::dict kwargs;
            kwargs["type_"] = "INTEGER_CONSTANT";
            kwargs["value"] = visitSignedInteger(rule_child);
            for (auto item : ti) kwargs[item.first] = item.second;
            return call_with_kwargs(py_Constant, kwargs);
        }
        if (cid.first == VtlParser::RuleSignedNumber && cid.second == -1) {
            py::dict kwargs;
            kwargs["type_"] = "FLOAT_CONSTANT";
            kwargs["value"] = visitSignedNumber(rule_child);
            for (auto item : ti) kwargs[item.first] = item.second;
            return call_with_kwargs(py_Constant, kwargs);
        }
    }

    // Terminal child
    auto* term = as_terminal(child);
    if (!term) throw std::runtime_error("visitConstant: unexpected child type");

    int sym = (int)term->getSymbol()->getType();

    if (sym == VtlParser::BOOLEAN_CONSTANT) {
        std::string text = term->getText();
        py::dict kwargs;
        kwargs["type_"] = "BOOLEAN_CONSTANT";
        if (text == "true") {
            kwargs["value"] = py::bool_(true);
        } else if (text == "false") {
            kwargs["value"] = py::bool_(false);
        } else {
            throw std::runtime_error("visitConstant: unexpected BOOLEAN_CONSTANT value");
        }
        for (auto item : ti) kwargs[item.first] = item.second;
        return call_with_kwargs(py_Constant, kwargs);
    }

    if (sym == VtlParser::STRING_CONSTANT) {
        std::string text = term->getText();
        // Remove surrounding quotes
        std::string value = text.substr(1, text.size() - 2);
        py::dict kwargs;
        kwargs["type_"] = "STRING_CONSTANT";
        kwargs["value"] = value;
        for (auto item : ti) kwargs[item.first] = item.second;
        return call_with_kwargs(py_Constant, kwargs);
    }

    if (sym == VtlParser::NULL_CONSTANT) {
        py::dict kwargs;
        kwargs["type_"] = "NULL_CONSTANT";
        kwargs["value"] = py::none();
        for (auto item : ti) kwargs[item.first] = item.second;
        return call_with_kwargs(py_Constant, kwargs);
    }

    throw std::runtime_error("visitConstant: unexpected terminal type");
}

py::object visitVarID(antlr4::ParserRuleContext* ctx) {
    auto* token = as_terminal(ctx->children[0]);
    if (!token) throw std::runtime_error("visitVarID: expected terminal child");
    std::string text = remove_escaped_chars(token->getText());
    auto ti = extract_token_info_terminal(token);

    py::dict kwargs;
    kwargs["value"] = text;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_VarID, kwargs);
}

py::object visitVarIdExpr(antlr4::ParserRuleContext* ctx) {
    auto* child = ctx->children[0];
    auto* rule_child = as_rule(child);

    if (rule_child) {
        auto cid = get_ctx_id(rule_child);
        if (cid.first == VtlParser::RuleVarID && cid.second == -1) {
            return visitVarID(rule_child);
        }
    }

    // Terminal child (keyword used as identifier)
    auto* token = as_terminal(child);
    if (!token) throw std::runtime_error("visitVarIdExpr: expected terminal child");
    std::string text = remove_escaped_chars(token->getText());
    auto ti = extract_token_info_terminal(token);

    py::dict kwargs;
    kwargs["value"] = text;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_VarID, kwargs);
}

py::object visitSimpleComponentId(antlr4::ParserRuleContext* ctx) {
    auto* token = as_terminal(ctx->children[0]);
    if (!token) throw std::runtime_error("visitSimpleComponentId: expected terminal child");
    std::string text = remove_escaped_chars(token->getText());
    auto ti = extract_token_info(ctx);

    py::dict kwargs;
    kwargs["value"] = text;
    kwargs["kind"] = "ComponentID";
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_Identifier, kwargs);
}

py::object visitComponentID(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    size_t n = children.size();

    if (n == 1) {
        std::string component_name = node_text(children[0]);
        if (component_name.size() >= 2 &&
            component_name.front() == '\'' && component_name.back() == '\'') {
            component_name = component_name.substr(1, component_name.size() - 2);
        }

        py::dict ti;
        auto* term = as_terminal(children[0]);
        if (term) {
            ti = extract_token_info_terminal(term);
        } else {
            auto* rule = as_rule(children[0]);
            if (rule) ti = extract_token_info(rule);
        }

        py::dict kwargs;
        kwargs["value"] = component_name;
        kwargs["kind"] = "ComponentID";
        for (auto item : ti) kwargs[item.first] = item.second;
        return call_with_kwargs(py_Identifier, kwargs);
    }

    // 3 children: datasetID # componentID
    std::string component_name = node_text(children[2]);
    if (component_name.size() >= 2 &&
        component_name.front() == '\'' && component_name.back() == '\'') {
        component_name = component_name.substr(1, component_name.size() - 2);
    }

    std::string op_text = node_text(children[1]);
    std::string dataset_text = node_text(children[0]);

    // Build token_info for left (child 0)
    py::dict ti_left;
    auto* t0 = as_terminal(children[0]);
    if (t0) {
        ti_left = extract_token_info_terminal(t0);
    } else {
        auto* r0 = as_rule(children[0]);
        if (r0) ti_left = extract_token_info(r0);
    }

    // Build token_info for right (child 1 - the operator '#')
    py::dict ti_right;
    auto* t1 = as_terminal(children[1]);
    if (t1) {
        ti_right = extract_token_info_terminal(t1);
    } else {
        auto* r1 = as_rule(children[1]);
        if (r1) ti_right = extract_token_info(r1);
    }

    auto ti_ctx = extract_token_info(ctx);

    // Build left Identifier
    py::dict kwargs_left;
    kwargs_left["value"] = dataset_text;
    kwargs_left["kind"] = "DatasetID";
    for (auto item : ti_left) kwargs_left[item.first] = item.second;
    py::object left = call_with_kwargs(py_Identifier, kwargs_left);

    // Build right Identifier
    py::dict kwargs_right;
    kwargs_right["value"] = component_name;
    kwargs_right["kind"] = "ComponentID";
    for (auto item : ti_right) kwargs_right[item.first] = item.second;
    py::object right = call_with_kwargs(py_Identifier, kwargs_right);

    // Build BinOp
    py::dict kwargs;
    kwargs["left"] = left;
    kwargs["op"] = op_text;
    kwargs["right"] = right;
    for (auto item : ti_ctx) kwargs[item.first] = item.second;
    return call_with_kwargs(py_BinOp, kwargs);
}

py::object visitOperatorID(antlr4::ParserRuleContext* ctx) {
    return py::str(node_text(ctx->children[0]));
}

py::object visitValueDomainID(antlr4::ParserRuleContext* ctx) {
    std::string name = node_text(ctx->children[0]);
    auto ti = extract_token_info(ctx);

    py::dict kwargs;
    kwargs["name"] = name;
    kwargs["children"] = py::list();
    kwargs["kind"] = "ValueDomain";
    kwargs["type"] = "";
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_Collection, kwargs);
}

py::object visitRulesetID(antlr4::ParserRuleContext* ctx) {
    return py::str(node_text(ctx->children[0]));
}

py::object visitValueDomainName(antlr4::ParserRuleContext* ctx) {
    std::string text = node_text(ctx->children[0]);
    throw std::runtime_error(
        "Value Domain '" + text + "' not available for cast operator or scalar type "
        "representation or rulesets."
    );
}

py::object visitValueDomainValue(antlr4::ParserRuleContext* ctx) {
    auto* child = ctx->children[0];
    auto* rule_child = as_rule(child);
    if (rule_child) {
        auto cid = get_ctx_id(rule_child);
        if ((cid.first == VtlParser::RuleSignedInteger && cid.second == -1) ||
            (cid.first == VtlParser::RuleSignedNumber && cid.second == -1)) {
            return py::str(node_text(child));
        }
    }
    return py::str(remove_escaped_chars(node_text(child)));
}

py::object visitRoutineName(antlr4::ParserRuleContext* ctx) {
    return py::str(node_text(ctx->children[0]));
}

py::object visitBasicScalarType(antlr4::ParserRuleContext* ctx) {
    auto* child = as_terminal(ctx->children[0]);
    if (!child) throw std::runtime_error("visitBasicScalarType: expected terminal child");
    int sym = (int)child->getSymbol()->getType();

    if (sym == VtlParser::STRING) return py_String;
    if (sym == VtlParser::INTEGER) return py_Integer;
    if (sym == VtlParser::NUMBER) return py_Number;
    if (sym == VtlParser::BOOLEAN) return py_Boolean;
    if (sym == VtlParser::DATE) return py_Date;
    if (sym == VtlParser::TIME_PERIOD) return py_TimePeriod;
    if (sym == VtlParser::DURATION) return py_Duration;
    if (sym == VtlParser::SCALAR) return py::str("Scalar");
    if (sym == VtlParser::TIME) return py_TimeInterval;

    throw std::runtime_error("visitBasicScalarType: unexpected token type");
}

py::object visitComponentRole(antlr4::ParserRuleContext* ctx) {
    auto* child = ctx->children[0];
    auto* rule_child = as_rule(child);

    if (rule_child) {
        auto cid = get_ctx_id(rule_child);
        if (cid.first == VtlParser::RuleViralAttribute && cid.second == -1) {
            return visitViralAttribute(rule_child);
        }
    }

    std::string text = node_text(child);
    if (text == "component") {
        return py::none();
    }
    // Capitalize first letter
    std::string capitalized = text;
    if (!capitalized.empty()) {
        capitalized[0] = (char)std::toupper((unsigned char)capitalized[0]);
        for (size_t i = 1; i < capitalized.size(); i++) {
            capitalized[i] = (char)std::tolower((unsigned char)capitalized[i]);
        }
    }
    return py_Role(py::str(capitalized));
}

py::object visitViralAttribute(antlr4::ParserRuleContext* /*ctx*/) {
    throw std::runtime_error("NotImplementedError: visitViralAttribute");
}

py::object visitLists(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    py::list scalar_nodes;

    // Collect SIMPLE_SCALAR children first, then SCALAR_WITH_CAST
    std::vector<antlr4::ParserRuleContext*> simple_scalars;
    std::vector<antlr4::ParserRuleContext*> cast_scalars;

    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        auto cid = get_ctx_id(rule_child);
        if (cid.first == VtlParser::RuleScalarItem && cid.second == 0) {
            // SIMPLE_SCALAR
            simple_scalars.push_back(rule_child);
        } else if (cid.first == VtlParser::RuleScalarItem && cid.second == 1) {
            // SCALAR_WITH_CAST
            cast_scalars.push_back(rule_child);
        }
    }

    for (auto* s : simple_scalars) {
        scalar_nodes.append(visitSimpleScalar(s));
    }
    for (auto* s : cast_scalars) {
        scalar_nodes.append(visitScalarWithCast(s));
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["name"] = "List";
    kwargs["type"] = "Lists";
    kwargs["children"] = scalar_nodes;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_Collection, kwargs);
}

py::object visitMultModifier(antlr4::ParserRuleContext* /*ctx*/) {
    return py::none();
}

py::object visitCompConstraint(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    std::vector<py::object> component_nodes;
    std::vector<std::string> component_names;
    bool has_mult = false;

    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        auto cid = get_ctx_id(rule_child);

        if (cid.first == VtlParser::RuleComponentType && cid.second == -1) {
            component_nodes.push_back(visitComponentType(rule_child));
        } else if (cid.first == VtlParser::RuleComponentID && cid.second == -1) {
            py::object comp_id = visitComponentID(rule_child);
            component_names.push_back(comp_id.attr("value").cast<std::string>());
        } else if (cid.first == VtlParser::RuleMultModifier && cid.second == -1) {
            has_mult = true;
        }
    }

    if (has_mult) {
        throw std::runtime_error("NotImplementedError: multModifier in compConstraint");
    }

    // Set the name on the component
    if (!component_nodes.empty() && !component_names.empty()) {
        component_nodes[0].attr("name") = component_names[0];
    }

    return component_nodes[0];
}

py::object visitSimpleScalar(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    auto* child = children[0];
    auto* rule_child = as_rule(child);

    if (rule_child) {
        auto cid = get_ctx_id(rule_child);
        if (cid.first == VtlParser::RuleConstant && cid.second == -1) {
            return visitConstant(rule_child);
        }
    }
    throw std::runtime_error("NotImplementedError: visitSimpleScalar unexpected child");
}

py::object visitScalarType(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    // Find the type child (basicScalarType or valueDomainName or constraint)
    antlr4::ParserRuleContext* scalartype = nullptr;
    bool has_constraint = false;
    bool has_not = false;
    bool has_null = false;

    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (rule_child) {
            auto cid = get_ctx_id(rule_child);
            if (cid.first == VtlParser::RuleBasicScalarType ||
                cid.first == VtlParser::RuleValueDomainName ||
                (cid.first == VtlParser::RuleScalarTypeConstraint && cid.second == 0) ||
                (cid.first == VtlParser::RuleScalarTypeConstraint && cid.second == 1)) {

                if (cid.first == VtlParser::RuleBasicScalarType ||
                    cid.first == VtlParser::RuleValueDomainName) {
                    if (!scalartype) scalartype = rule_child;
                }

                if (cid.first == VtlParser::RuleScalarTypeConstraint) {
                    has_constraint = true;
                }
            }
        } else {
            auto* term = as_terminal(child);
            if (term) {
                int sym = (int)term->getSymbol()->getType();
                if (sym == VtlParser::NOT) has_not = true;
                if (sym == VtlParser::NULL_CONSTANT) has_null = true;
            }
        }
    }

    if (!scalartype) {
        throw std::runtime_error("visitScalarType: no type child found");
    }

    auto scalartype_cid = get_ctx_id(scalartype);

    if (scalartype_cid.first == VtlParser::RuleBasicScalarType) {
        // Check if SCALAR keyword
        auto* first_child = as_terminal(scalartype->children[0]);
        if (first_child && (int)first_child->getSymbol()->getType() == VtlParser::SCALAR) {
            return py_Scalar(
                py::arg("name") = "",
                py::arg("data_type") = py::none(),
                py::arg("value") = py::none()
            );
        }
        py::object type_node = visitBasicScalarType(scalartype);

        if (has_constraint) {
            throw std::runtime_error("NotImplementedError: scalarType constraint");
        }
        if (has_not) {
            throw std::runtime_error("NotImplementedError: scalarType NOT");
        }
        if (has_null) {
            throw std::runtime_error("NotImplementedError: scalarType NULL_CONSTANT");
        }

        return type_node;
    }

    // valueDomainName
    std::string text = node_text(scalartype->children[0]);
    int line = ctx->start ? (int)ctx->start->getLine() : 0;
    int col = ctx->start ? (int)ctx->start->getCharPositionInLine() : 0;
    throw std::runtime_error(
        "Invalid parameter type definition " + text + " at line " +
        std::to_string(line) + ":" + std::to_string(col) + "."
    );
}

py::object visitDatasetType(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    py::list components_list;
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        auto cid = get_ctx_id(rule_child);
        if (cid.first == VtlParser::RuleCompConstraint && cid.second == -1) {
            components_list.append(visitCompConstraint(rule_child));
        }
    }

    // Build dict {name: component}
    py::dict comp_dict;
    for (auto item : components_list) {
        py::object comp = py::reinterpret_borrow<py::object>(item);
        std::string name = comp.attr("name").cast<std::string>();
        comp_dict[py::str(name)] = comp;
    }

    return py_Dataset(
        py::arg("name") = "Dataset",
        py::arg("components") = comp_dict,
        py::arg("data") = py::none()
    );
}

py::object visitRulesetType(antlr4::ParserRuleContext* /*ctx*/) {
    throw std::runtime_error("NotImplementedError: visitRulesetType");
}

py::object visitComponentType(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    // First child is always componentRole
    py::object role_node = visitComponentRole(as_rule(children[0]));

    // Find scalarType child
    py::object data_type = py::none();
    bool found_scalar_type = false;
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        auto cid = get_ctx_id(rule_child);
        if (cid.first == VtlParser::RuleScalarType && cid.second == -1) {
            data_type = visitScalarType(rule_child);
            found_scalar_type = true;
            break;
        }
    }

    if (!found_scalar_type) {
        // Default to String()
        data_type = py_String();
    }

    // nullable = role_node != Role.IDENTIFIER
    py::object role_identifier = py_Role(py::str("Identifier"));
    bool nullable = !role_node.equal(role_identifier);

    return py_Component(
        py::arg("name") = "Component",
        py::arg("data_type") = data_type,
        py::arg("role") = role_node,
        py::arg("nullable") = nullable
    );
}

py::object visitInputParameterType(antlr4::ParserRuleContext* ctx) {
    auto* child = ctx->children[0];
    auto* rule_child = as_rule(child);

    if (!rule_child) {
        throw std::runtime_error("NotImplementedError: visitInputParameterType terminal child");
    }

    auto cid = get_ctx_id(rule_child);

    if (cid.first == VtlParser::RuleScalarType && cid.second == -1) {
        return visitScalarType(rule_child);
    }
    if (cid.first == VtlParser::RuleDatasetType && cid.second == -1) {
        return visitDatasetType(rule_child);
    }
    if (cid.first == VtlParser::RuleScalarSetType && cid.second == -1) {
        return visitScalarSetType(rule_child);
    }
    if (cid.first == VtlParser::RuleRulesetType && cid.second == -1) {
        return visitRulesetType(rule_child);
    }
    if (cid.first == VtlParser::RuleComponentType && cid.second == -1) {
        return visitComponentType(rule_child);
    }

    throw std::runtime_error("NotImplementedError: visitInputParameterType unknown child");
}

py::object visitOutputParameterType(antlr4::ParserRuleContext* ctx) {
    auto* child = ctx->children[0];
    auto* rule_child = as_rule(child);

    if (!rule_child) {
        throw std::runtime_error("NotImplementedError: visitOutputParameterType terminal child");
    }

    auto cid = get_ctx_id(rule_child);

    if (cid.first == VtlParser::RuleScalarType && cid.second == -1) {
        return py::str("Scalar");
    }
    if (cid.first == VtlParser::RuleDatasetType && cid.second == -1) {
        return py::str("Dataset");
    }
    if (cid.first == VtlParser::RuleComponentType && cid.second == -1) {
        return py::str("Component");
    }

    throw std::runtime_error("NotImplementedError: visitOutputParameterType unknown child");
}

py::object visitOutputParameterTypeComponent(antlr4::ParserRuleContext* ctx) {
    auto* child = ctx->children[0];
    auto* rule_child = as_rule(child);

    if (!rule_child) {
        throw std::runtime_error("NotImplementedError: visitOutputParameterTypeComponent terminal child");
    }

    auto cid = get_ctx_id(rule_child);

    if (cid.first == VtlParser::RuleScalarType && cid.second == -1) {
        return visitScalarType(rule_child);
    }
    if (cid.first == VtlParser::RuleComponentType && cid.second == -1) {
        return visitComponentType(rule_child);
    }

    throw std::runtime_error("NotImplementedError: visitOutputParameterTypeComponent unknown child");
}

py::object visitScalarItem(antlr4::ParserRuleContext* ctx) {
    auto cid = get_ctx_id(ctx);

    if (cid.first == VtlParser::RuleScalarItem && cid.second == 0) {
        // SIMPLE_SCALAR
        return visitSimpleScalar(ctx);
    }
    if (cid.first == VtlParser::RuleScalarItem && cid.second == 1) {
        // SCALAR_WITH_CAST
        return visitScalarWithCast(ctx);
    }

    throw std::runtime_error("NotImplementedError: visitScalarItem unknown alternative");
}

py::object visitScalarWithCast(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    // CAST LPAREN constant COMMA basicScalarType (COMMA STRING_CONSTANT)? RPAREN
    std::string op = node_text(children[0]);
    py::object const_node = visitConstant(as_rule(children[2]));
    py::object basic_scalar_type = visitBasicScalarType(as_rule(children[4]));

    py::list children_nodes;
    children_nodes.append(const_node);
    children_nodes.append(basic_scalar_type);

    py::list param_node;
    if (children.size() > 6) {
        // Has STRING_CONSTANT parameter at index 6
        auto* param_term = as_terminal(children[6]);
        py::dict param_ti;
        if (param_term) {
            param_ti = extract_token_info_terminal(param_term);
        }
        py::dict param_kwargs;
        param_kwargs["type_"] = "PARAM_CAST";
        // Pass the terminal node text as value (matching Python behavior of passing ctx_list[6])
        param_kwargs["value"] = param_term ? py::cast(param_term->getText()) : py::none();
        for (auto item : param_ti) param_kwargs[item.first] = item.second;
        param_node.append(call_with_kwargs(py_ParamConstant, param_kwargs));
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["children"] = children_nodes;
    kwargs["params"] = param_node;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_ParamOp, kwargs);
}

py::object visitScalarSetType(antlr4::ParserRuleContext* /*ctx*/) {
    throw std::runtime_error("NotImplementedError: visitScalarSetType");
}

py::object visitRetainType(antlr4::ParserRuleContext* ctx) {
    auto* token = as_terminal(ctx->children[0]);
    if (!token) throw std::runtime_error("visitRetainType: expected terminal child");

    int sym = (int)token->getSymbol()->getType();
    auto ti = extract_token_info_terminal(token);

    if (sym == VtlParser::BOOLEAN_CONSTANT) {
        std::string text = token->getText();
        py::dict kwargs;
        kwargs["type_"] = "BOOLEAN_CONSTANT";
        if (text == "true") {
            kwargs["value"] = py::bool_(true);
        } else if (text == "false") {
            kwargs["value"] = py::bool_(false);
        } else {
            throw std::runtime_error("visitRetainType: unexpected BOOLEAN_CONSTANT value");
        }
        for (auto item : ti) kwargs[item.first] = item.second;
        return call_with_kwargs(py_Constant, kwargs);
    }

    if (sym == VtlParser::ALL) {
        py::dict kwargs;
        kwargs["type_"] = "PARAM_CONSTANT";
        kwargs["value"] = token->getText();
        for (auto item : ti) kwargs[item.first] = item.second;
        return call_with_kwargs(py_ParamConstant, kwargs);
    }

    throw std::runtime_error("visitRetainType: unexpected token type");
}

py::object visitEvalDatasetType(antlr4::ParserRuleContext* ctx) {
    auto* child = ctx->children[0];
    auto* rule_child = as_rule(child);

    if (!rule_child) {
        throw std::runtime_error("NotImplementedError: visitEvalDatasetType terminal child");
    }

    auto cid = get_ctx_id(rule_child);

    if (cid.first == VtlParser::RuleDatasetType && cid.second == -1) {
        return visitDatasetType(rule_child);
    }
    if (cid.first == VtlParser::RuleScalarType && cid.second == -1) {
        return visitScalarType(rule_child);
    }

    throw std::runtime_error("NotImplementedError: visitEvalDatasetType unknown child");
}

py::object visitAlias(antlr4::ParserRuleContext* ctx) {
    return py::str(node_text(ctx->children[0]));
}

py::object visitSignedInteger(antlr4::ParserRuleContext* ctx) {
    std::string text = ctx->getText();
    return py::int_(std::stoi(text));
}

py::object visitSignedNumber(antlr4::ParserRuleContext* ctx) {
    std::string text = ctx->getText();
    return py::float_(std::stod(text));
}

py::object visitComparisonOperand(antlr4::ParserRuleContext* ctx) {
    return py::str(node_text(ctx->children[0]));
}

py::object visitErCode(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    // erCode: ERRORCODE constant
    try {
        py::object constant = visitConstant(as_rule(children[1]));
        py::object value = constant.attr("value");
        return py::str(value);
    } catch (...) {
        auto* child1 = as_rule(children[1]);
        int line = child1 && child1->start ? (int)child1->start->getLine() : 0;
        throw std::runtime_error(
            "Error code must be a string, line " + std::to_string(line)
        );
    }
}

py::object visitErLevel(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    // erLevel: ERRORLEVEL constant
    py::object constant = visitConstant(as_rule(children[1]));
    return constant.attr("value");
}

py::object visitSignature(antlr4::ParserRuleContext* ctx, const std::string& kind) {
    auto ti = extract_token_info(ctx);
    auto& children = ctx->children;

    // First child is VarID
    py::object var_id = visitVarID(as_rule(children[0]));
    std::string node_name = var_id.attr("value").cast<std::string>();

    py::object alias_name = py::none();

    if (children.size() == 1) {
        py::dict kwargs;
        kwargs["value"] = node_name;
        kwargs["kind"] = kind;
        kwargs["alias"] = alias_name;
        for (auto item : ti) kwargs[item.first] = item.second;
        return call_with_kwargs(py_DPRIdentifier, kwargs);
    }

    // children[2] is alias
    alias_name = visitAlias(as_rule(children[2]));

    py::dict kwargs;
    kwargs["value"] = node_name;
    kwargs["kind"] = kind;
    kwargs["alias"] = alias_name;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_DPRIdentifier, kwargs);
}

py::object visitConditionClause(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    py::list components;

    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        auto cid = get_ctx_id(rule_child);
        if (cid.first == VtlParser::RuleComponentID && cid.second == -1) {
            components.append(visitComponentID(rule_child));
        }
    }

    return components;
}

py::object visitValidationMode(antlr4::ParserRuleContext* ctx) {
    return py::str(node_text(ctx->children[0]));
}

py::object visitValidationOutput(antlr4::ParserRuleContext* ctx) {
    return py::str(node_text(ctx->children[0]));
}

py::object visitInputMode(antlr4::ParserRuleContext* ctx) {
    return py::str(node_text(ctx->children[0]));
}

py::object visitInputModeHierarchy(antlr4::ParserRuleContext* ctx) {
    return py::str(node_text(ctx->children[0]));
}

py::object visitOutputModeHierarchy(antlr4::ParserRuleContext* ctx) {
    return py::str(node_text(ctx->children[0]));
}

py::object visitPartitionByClause(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    py::list result;

    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        auto cid = get_ctx_id(rule_child);
        if (cid.first == VtlParser::RuleComponentID && cid.second == -1) {
            py::object comp_id = visitComponentID(rule_child);
            result.append(comp_id.attr("value"));
        }
    }

    return result;
}

py::object visitOrderByClause(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    py::list result;

    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        auto cid = get_ctx_id(rule_child);
        if (cid.first == VtlParser::RuleOrderByItem && cid.second == -1) {
            result.append(visitOrderByItem(rule_child));
        }
    }

    return result;
}

py::object visitOrderByItem(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    auto ti = extract_token_info(ctx);

    py::object comp_id = visitComponentID(as_rule(children[0]));
    std::string component = comp_id.attr("value").cast<std::string>();

    std::string order = "asc";
    if (children.size() > 1) {
        order = node_text(children[1]);
    }

    py::dict kwargs;
    kwargs["component"] = component;
    kwargs["order"] = order;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_OrderBy, kwargs);
}

std::pair<py::object, std::string> visitLimitClauseItem(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    auto* c = children[0];

    auto* rule_child = as_rule(c);
    if (rule_child) {
        auto cid = get_ctx_id(rule_child);
        if (cid.first == VtlParser::RuleSignedInteger && cid.second == -1) {
            py::object result = visitSignedInteger(rule_child);
            // limitDir is the last terminal child
            std::string limit_dir = node_text(children[children.size() - 1]);
            return {result, limit_dir};
        }
    }

    std::string text = node_text(c);
    std::string lower_text = text;
    std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(),
                   [](unsigned char ch) { return std::tolower(ch); });

    if (lower_text == "unbounded") {
        std::string limit_dir = node_text(children[children.size() - 1]);
        return {py::int_(-1), limit_dir};
    }
    if (text == "current") {
        return {py::int_(0), text};
    }

    throw std::runtime_error("visitLimitClauseItem: unexpected first child");
}

py::object visitWindowingClause(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string win_mode = node_text(children[0]);
    auto ti = extract_token_info(ctx);

    py::object num_rows_1, num_rows_2;
    std::string mode_1, mode_2;

    if (win_mode == "data") {
        // data points between X and Y
        auto [nr1, m1] = visitLimitClauseItem(as_rule(children[3]));
        auto [nr2, m2] = visitLimitClauseItem(as_rule(children[5]));
        num_rows_1 = nr1; mode_1 = m1;
        num_rows_2 = nr2; mode_2 = m2;
    } else {
        // range between X and Y
        auto [nr1, m1] = visitLimitClauseItem(as_rule(children[2]));
        auto [nr2, m2] = visitLimitClauseItem(as_rule(children[4]));
        num_rows_1 = nr1; mode_1 = m1;
        num_rows_2 = nr2; mode_2 = m2;
    }

    py::object first = num_rows_1;
    py::object second = num_rows_2;

    int iv1 = py::isinstance<py::int_>(num_rows_1) ? num_rows_1.cast<int>() : 999;
    int iv2 = py::isinstance<py::int_>(num_rows_2) ? num_rows_2.cast<int>() : 999;

    // Error checks
    if (mode_2 == "preceding" && mode_1 == "preceding" && iv1 == -1 && iv2 == -1) {
        auto* child3 = as_rule(children.size() > 3 ? children[3] : children[2]);
        int line = child3 && child3->start ? (int)child3->start->getLine() : 0;
        throw std::runtime_error(
            "Cannot have 2 preceding clauses with unbounded in analytic clause, line " +
            std::to_string(line)
        );
    }

    if (mode_1 == "following" && iv1 == -1 && iv2 == -1) {
        auto* child3 = as_rule(children.size() > 3 ? children[3] : children[2]);
        int line = child3 && child3->start ? (int)child3->start->getLine() : 0;
        throw std::runtime_error(
            "Cannot have 2 following clauses with unbounded in analytic clause, line " +
            std::to_string(line)
        );
    }

    // Swap logic
    if (mode_1 == mode_2) {
        if (mode_1 == "preceding" && iv1 != -1 && iv2 > iv1) {
            return create_windowing(win_mode, second, first, mode_2, mode_1, ti);
        }
        if (mode_1 == "preceding" && iv2 == -1) {
            return create_windowing(win_mode, second, first, mode_2, mode_1, ti);
        }
        if (mode_1 == "following" && iv2 != -1 && iv2 < iv1) {
            return create_windowing(win_mode, second, first, mode_2, mode_1, ti);
        }
        if (mode_1 == "following" && iv1 == -1) {
            return create_windowing(win_mode, second, first, mode_2, mode_1, ti);
        }
    }

    return create_windowing(win_mode, first, second, mode_1, mode_2, ti);
}

// ============================================================
// Helper: raise SemanticError
// ============================================================

static void raise_semantic_error(const std::string& code, py::kwargs kwargs) {
    py::object exc = py_SemanticError(py::str(code), **kwargs);
    PyErr_SetObject(reinterpret_cast<PyObject*>(Py_TYPE(exc.ptr())), exc.ptr());
    throw py::error_already_set();
}

// ============================================================
// ExprComponents visitors (Phase 2)
// ============================================================

py::object visitExprComponent(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    auto* c = children[0];
    auto cid = get_ctx_id(ctx);

    // PARENTHESIS_EXPR_COMP = (3, 0)
    if (cid.first == 3 && cid.second == 0) {
        // visitParenthesisExprComp
        py::object operand = visitExprComponent(as_rule(children[1]));
        auto ti = extract_token_info(ctx);
        py::dict kwargs;
        kwargs["operand"] = operand;
        for (auto item : ti) kwargs[item.first] = item.second;
        return call_with_kwargs(py_ParFunction, kwargs);
    }

    // FUNCTIONS_EXPRESSION_COMP = (3, 1)
    if (cid.first == 3 && cid.second == 1) {
        return visitFunctionsComponents(as_rule(c));
    }

    // UNARY_EXPR_COMP = (3, 2)
    if (cid.first == 3 && cid.second == 2) {
        std::string op = terminal_text(children[0]);
        py::object right = visitExprComponent(as_rule(children[1]));
        auto ti = extract_token_info(ctx);
        py::dict kwargs;
        kwargs["op"] = op;
        kwargs["operand"] = right;
        for (auto item : ti) kwargs[item.first] = item.second;
        return call_with_kwargs(py_UnaryOp, kwargs);
    }

    // ARITHMETIC_EXPR_COMP = (3, 3)
    // ARITHMETIC_EXPR_OR_CONCAT_COMP = (3, 4)
    // COMPARISON_EXPR_COMP = (3, 5)
    // BOOLEAN_EXPR_COMP = (3, 7)
    if ((cid.first == 3 && cid.second == 3) ||
        (cid.first == 3 && cid.second == 4) ||
        (cid.first == 3 && cid.second == 5) ||
        (cid.first == 3 && cid.second == 7)) {
        // bin_op_creator_comp
        py::object left_node = visitExprComponent(as_rule(children[0]));
        std::string op;
        auto* mid_rule = as_rule(children[1]);
        if (mid_rule) {
            auto mid_cid = get_ctx_id(mid_rule);
            // COMPARISON_OPERAND = (100, -1)
            if (mid_cid.first == VtlParser::RuleComparisonOperand && mid_cid.second == -1) {
                op = terminal_text(mid_rule->children[0]);
            } else {
                op = node_text(children[1]);
            }
        } else {
            op = terminal_text(children[1]);
        }
        py::object right_node = visitExprComponent(as_rule(children[2]));
        auto ti = extract_token_info(ctx);
        py::dict kwargs;
        kwargs["left"] = left_node;
        kwargs["op"] = op;
        kwargs["right"] = right_node;
        for (auto item : ti) kwargs[item.first] = item.second;
        return call_with_kwargs(py_BinOp, kwargs);
    }

    // IN_NOT_IN_EXPR_COMP = (3, 6)
    if (cid.first == 3 && cid.second == 6) {
        py::object left_node = visitExprComponent(as_rule(children[0]));
        std::string op = terminal_text(children[1]);
        auto* child2 = children[2];
        auto* rule2 = as_rule(child2);
        py::object right_node;
        if (rule2) {
            auto cid2 = get_ctx_id(rule2);
            // LISTS = (97, -1)
            if (cid2.first == VtlParser::RuleLists && cid2.second == -1) {
                right_node = visitLists(rule2);
            }
            // VALUE_DOMAIN_ID = (105, -1)
            else if (cid2.first == VtlParser::RuleValueDomainID && cid2.second == -1) {
                right_node = visitValueDomainID(rule2);
            } else {
                throw std::runtime_error("NotImplementedError: inNotInExprComp");
            }
        } else {
            throw std::runtime_error("NotImplementedError: inNotInExprComp terminal");
        }
        auto ti = extract_token_info(ctx);
        py::dict kwargs;
        kwargs["left"] = left_node;
        kwargs["op"] = op;
        kwargs["right"] = right_node;
        for (auto item : ti) kwargs[item.first] = item.second;
        return call_with_kwargs(py_BinOp, kwargs);
    }

    // IF_EXPR_COMP = (3, 8)
    if (cid.first == 3 && cid.second == 8) {
        py::object condition = visitExprComponent(as_rule(children[1]));
        py::object then_op = visitExprComponent(as_rule(children[3]));
        py::object else_op = visitExprComponent(as_rule(children[5]));
        auto ti = extract_token_info(ctx);
        py::dict kwargs;
        kwargs["condition"] = condition;
        kwargs["thenOp"] = then_op;
        kwargs["elseOp"] = else_op;
        for (auto item : ti) kwargs[item.first] = item.second;
        return call_with_kwargs(py_If, kwargs);
    }

    // CASE_EXPR_COMP = (3, 9)
    if (cid.first == 3 && cid.second == 9) {
        size_t n = children.size();
        if (n % 4 != 3) {
            throw std::runtime_error("Syntax error.");
        }
        py::object else_node = visitExprComponent(as_rule(children[n - 1]));
        py::list cases;
        // Skip first token (CASE), process WHEN...THEN blocks, skip ELSE + elseExpr + END
        for (size_t i = 1; i < n - 2; i += 4) {
            py::object condition = visitExprComponent(as_rule(children[i + 1]));
            py::object then_op = visitExprComponent(as_rule(children[i + 3]));
            auto ti_case = extract_token_info(as_rule(children[i + 1]));
            py::dict case_kwargs;
            case_kwargs["condition"] = condition;
            case_kwargs["thenOp"] = then_op;
            for (auto item : ti_case) case_kwargs[item.first] = item.second;
            cases.append(call_with_kwargs(py_CaseObj, case_kwargs));
        }
        auto ti = extract_token_info(ctx);
        py::dict kwargs;
        kwargs["cases"] = cases;
        kwargs["elseOp"] = else_node;
        for (auto item : ti) kwargs[item.first] = item.second;
        return call_with_kwargs(py_Case, kwargs);
    }

    // CONSTANT_EXPR_COMP = (3, 10)
    if (cid.first == 3 && cid.second == 10) {
        return visitConstant(as_rule(c));
    }

    // COMP_ID = (3, 11)
    if (cid.first == 3 && cid.second == 11) {
        auto* c_rule = as_rule(c);
        if (c_rule && c_rule->children.size() > 1) {
            return visitComponentID(c_rule);
        }
        auto* token = c_rule ? c_rule->children[0] : c;
        std::string token_text = node_text(token);
        bool has_escaped_char = token_text.find('\'') != std::string::npos;
        if (has_escaped_char) {
            token_text = remove_escaped_chars(token_text);
        }
        auto ti = extract_token_info(ctx);
        py::dict kwargs;
        kwargs["value"] = token_text;
        for (auto item : ti) kwargs[item.first] = item.second;
        return call_with_kwargs(py_VarID, kwargs);
    }

    // AST_ASTCONSTRUCTOR.3
    throw std::runtime_error("NotImplementedError: visitExprComponent unknown ctx_id");
}

py::object visitOptionalExprComponent(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    auto* c = children[0];

    auto* rule_child = as_rule(c);
    if (rule_child && (int)rule_child->getRuleIndex() == 3) {
        // exprComponent (rule_index == 3)
        return visitExprComponent(rule_child);
    }

    auto* term = as_terminal(c);
    if (term) {
        std::string opt = term->getText();
        auto ti = extract_token_info(ctx);
        py::dict kwargs;
        kwargs["type_"] = "OPTIONAL";
        kwargs["value"] = opt;
        for (auto item : ti) kwargs[item.first] = item.second;
        return call_with_kwargs(py_ID, kwargs);
    }

    return py::none();
}

py::object visitFunctionsComponents(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    auto* c = as_rule(children[0]);
    auto cid = get_ctx_id(ctx);

    // GENERIC_FUNCTIONS_COMPONENTS = (4, 0)
    if (cid.first == 4 && cid.second == 0) return visitGenericFunctionsComponents(c);
    // STRING_FUNCTIONS_COMPONENTS = (4, 1)
    if (cid.first == 4 && cid.second == 1) return visitStringFunctionsComponents(c);
    // NUMERIC_FUNCTIONS_COMPONENTS = (4, 2)
    if (cid.first == 4 && cid.second == 2) return visitNumericFunctionsComponents(c);
    // COMPARISON_FUNCTIONS_COMPONENTS = (4, 3)
    if (cid.first == 4 && cid.second == 3) return visitComparisonFunctionsComponents(c);
    // TIME_FUNCTIONS_COMPONENTS = (4, 4)
    if (cid.first == 4 && cid.second == 4) return visitTimeFunctionsComponents(c);
    // CONDITIONAL_FUNCTIONS_COMPONENTS = (4, 5)
    if (cid.first == 4 && cid.second == 5) return visitConditionalFunctionsComponents(c);
    // AGGREGATE_FUNCTIONS_COMPONENTS = (4, 6)
    if (cid.first == 4 && cid.second == 6) return visitAggregateFunctionsComponents(c);
    // ANALYTIC_FUNCTIONS_COMPONENTS = (4, 7)
    if (cid.first == 4 && cid.second == 7) return visitAnalyticFunctionsComponents(c);

    throw std::runtime_error("NotImplementedError: visitFunctionsComponents");
}

// ---- Generic Functions Components ----

py::object visitGenericFunctionsComponents(antlr4::ParserRuleContext* ctx) {
    auto cid = get_ctx_id(ctx);
    // CALL_COMPONENT = (18, 0)
    if (cid.first == 18 && cid.second == 0) return visitCallComponent(ctx);
    // EVAL_ATOM_COMPONENT = (18, 2)
    if (cid.first == 18 && cid.second == 2) return visitEvalAtomComponent(ctx);
    // CAST_EXPR_COMPONENT = (18, 1)
    if (cid.first == 18 && cid.second == 1) return visitCastExprComponent(ctx);
    throw std::runtime_error("NotImplementedError: visitGenericFunctionsComponents");
}

py::object visitCallComponent(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    py::object op = visitOperatorID(as_rule(children[0]));
    py::list param_nodes;
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        auto child_cid = get_ctx_id(rule_child);
        // PARAMETER_COMPONENT = (19, -1)
        if (child_cid.first == VtlParser::RuleParameterComponent && child_cid.second == -1) {
            param_nodes.append(visitParameterComponent(rule_child));
        }
    }
    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["params"] = param_nodes;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_UDOCall, kwargs);
}

py::object visitEvalAtomComponent(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    py::object routine_name = visitRoutineName(as_rule(children[2]));

    py::list var_ids_nodes;
    py::list constant_nodes;
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        auto child_cid = get_ctx_id(rule_child);
        if (child_cid.first == VtlParser::RuleVarID && child_cid.second == -1) {
            var_ids_nodes.append(visitVarID(rule_child));
        }
        if ((int)rule_child->getRuleIndex() == 43) {
            // ScalarItem (rule_index 43)
            constant_nodes.append(visitScalarItem(rule_child));
        }
    }

    py::list children_nodes;
    for (auto item : var_ids_nodes) children_nodes.append(item);
    for (auto item : constant_nodes) children_nodes.append(item);

    if (py::len(children_nodes) > 1) {
        throw std::runtime_error("Only one operand is allowed in Eval");
    }

    // Find STRING_CONSTANT (language)
    py::list language_list;
    for (auto* child : children) {
        auto* term = as_terminal(child);
        if (term && (int)term->getSymbol()->getType() == VtlParser::STRING_CONSTANT) {
            language_list.append(py::str(term->getText()));
        }
    }
    if (py::len(language_list) == 0) {
        py::kwargs kw;
        kw["option"] = py::str("language");
        raise_semantic_error("1-3-2-1", kw);
    }

    // Find OUTPUT_PARAMETER_TYPE_COMPONENT
    py::list output_list;
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        auto child_cid = get_ctx_id(rule_child);
        if (child_cid.first == VtlParser::RuleOutputParameterTypeComponent && child_cid.second == -1) {
            output_list.append(visitOutputParameterTypeComponent(rule_child));
        }
    }
    if (py::len(output_list) == 0) {
        py::kwargs kw;
        kw["option"] = py::str("output");
        raise_semantic_error("1-3-2-1", kw);
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["name"] = routine_name;
    kwargs["operands"] = children_nodes[py::int_(0)];
    kwargs["output"] = output_list[py::int_(0)];
    kwargs["language"] = language_list[py::int_(0)];
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_EvalOp, kwargs);
}

py::object visitCastExprComponent(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);

    // Collect exprComponent children (rule_index == 3)
    py::list expr_nodes;
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (rule_child && (int)rule_child->getRuleIndex() == 3) {
            expr_nodes.append(visitExprComponent(rule_child));
        }
    }

    // Collect basicScalarType
    py::list basic_scalar_types;
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        auto child_cid = get_ctx_id(rule_child);
        if (child_cid.first == VtlParser::RuleBasicScalarType && child_cid.second == -1) {
            basic_scalar_types.append(visitBasicScalarType(rule_child));
        }
    }

    // Check for valueDomainName (raises error)
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        auto child_cid = get_ctx_id(rule_child);
        if (child_cid.first == VtlParser::RuleValueDomainName && child_cid.second == -1) {
            visitValueDomainName(rule_child); // This throws
        }
    }

    py::list param_node;
    if (children.size() > 6) {
        for (auto* child : children) {
            auto* term = as_terminal(child);
            if (term && (int)term->getSymbol()->getType() == VtlParser::STRING_CONSTANT) {
                auto term_ti = extract_token_info_terminal(term);
                py::dict pk;
                pk["type_"] = "PARAM_CAST";
                pk["value"] = py::str(term->getText()).attr("strip")(py::str("\""));
                for (auto item : term_ti) pk[item.first] = item.second;
                param_node.append(call_with_kwargs(py_ParamConstant, pk));
            }
        }
    }

    if (py::len(basic_scalar_types) == 1) {
        py::list children_nodes;
        children_nodes.append(expr_nodes[py::int_(0)]);
        children_nodes.append(basic_scalar_types[py::int_(0)]);

        auto ti = extract_token_info(ctx);
        py::dict kwargs;
        kwargs["op"] = op;
        kwargs["children"] = children_nodes;
        kwargs["params"] = param_node;
        for (auto item : ti) kwargs[item.first] = item.second;
        return call_with_kwargs(py_ParamOp, kwargs);
    }

    throw std::runtime_error("NotImplementedError: castExprComponent");
}

py::object visitParameterComponent(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    auto* c = children[0];
    auto* rule_child = as_rule(c);

    if (rule_child && (int)rule_child->getRuleIndex() == 3) {
        return visitExprComponent(rule_child);
    }
    auto* term = as_terminal(c);
    if (term) {
        auto ti = extract_token_info(ctx);
        py::dict kwargs;
        kwargs["type_"] = "OPTIONAL";
        kwargs["value"] = term->getText();
        for (auto item : ti) kwargs[item.first] = item.second;
        return call_with_kwargs(py_ID, kwargs);
    }
    throw std::runtime_error("NotImplementedError: visitParameterComponent");
}

// ---- String Functions Components ----

py::object visitStringFunctionsComponents(antlr4::ParserRuleContext* ctx) {
    auto cid = get_ctx_id(ctx);
    if (cid.first == 22 && cid.second == 0) return visitUnaryStringFunctionComponent(ctx);
    if (cid.first == 22 && cid.second == 1) return visitSubstrAtomComponent(ctx);
    if (cid.first == 22 && cid.second == 2) return visitReplaceAtomComponent(ctx);
    if (cid.first == 22 && cid.second == 3) return visitInstrAtomComponent(ctx);
    throw std::runtime_error("NotImplementedError: visitStringFunctionsComponents");
}

py::object visitUnaryStringFunctionComponent(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::object operand = visitExprComponent(as_rule(children[2]));
    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["operand"] = operand;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_UnaryOp, kwargs);
}

py::object visitSubstrAtomComponent(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);

    py::list children_nodes;
    py::list params_nodes;

    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        if ((int)rule_child->getRuleIndex() == 3) {
            children_nodes.append(visitExprComponent(rule_child));
        }
        auto child_cid = get_ctx_id(rule_child);
        // OPTIONAL_EXPR_COMPONENT = (102, -1)
        if (child_cid.first == VtlParser::RuleOptionalExprComponent && child_cid.second == -1) {
            params_nodes.append(visitOptionalExprComponent(rule_child));
        }
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["children"] = children_nodes;
    kwargs["params"] = params_nodes;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_ParamOp, kwargs);
}

py::object visitReplaceAtomComponent(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);

    py::list expressions;
    py::list opt_params;

    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        if ((int)rule_child->getRuleIndex() == 3) {
            expressions.append(visitExprComponent(rule_child));
        }
        auto child_cid = get_ctx_id(rule_child);
        if (child_cid.first == VtlParser::RuleOptionalExprComponent && child_cid.second == -1) {
            opt_params.append(visitOptionalExprComponent(rule_child));
        }
    }

    py::list children_nodes;
    children_nodes.append(expressions[py::int_(0)]);
    py::list params_nodes;
    params_nodes.append(expressions[py::int_(1)]);
    for (auto item : opt_params) params_nodes.append(item);

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["children"] = children_nodes;
    kwargs["params"] = params_nodes;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_ParamOp, kwargs);
}

py::object visitInstrAtomComponent(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);

    py::list expressions;
    py::list opt_params;

    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        if ((int)rule_child->getRuleIndex() == 3) {
            expressions.append(visitExprComponent(rule_child));
        }
        auto child_cid = get_ctx_id(rule_child);
        if (child_cid.first == VtlParser::RuleOptionalExprComponent && child_cid.second == -1) {
            opt_params.append(visitOptionalExprComponent(rule_child));
        }
    }

    py::list children_nodes;
    children_nodes.append(expressions[py::int_(0)]);
    py::list params_nodes;
    params_nodes.append(expressions[py::int_(1)]);
    for (auto item : opt_params) params_nodes.append(item);

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["children"] = children_nodes;
    kwargs["params"] = params_nodes;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_ParamOp, kwargs);
}

// ---- Numeric Functions Components ----

py::object visitNumericFunctionsComponents(antlr4::ParserRuleContext* ctx) {
    auto cid = get_ctx_id(ctx);
    if (cid.first == 24 && cid.second == 0) return visitUnaryNumericComponent(ctx);
    if (cid.first == 24 && cid.second == 1) return visitUnaryWithOptionalNumericComponent(ctx);
    if (cid.first == 24 && cid.second == 2) return visitBinaryNumericComponent(ctx);
    throw std::runtime_error("NotImplementedError: visitNumericFunctionsComponents");
}

py::object visitUnaryNumericComponent(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::object operand = visitExprComponent(as_rule(children[2]));
    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["operand"] = operand;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_UnaryOp, kwargs);
}

py::object visitUnaryWithOptionalNumericComponent(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);

    py::list children_nodes;
    py::list params_nodes;

    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        if ((int)rule_child->getRuleIndex() == 3) {
            children_nodes.append(visitExprComponent(rule_child));
        }
        auto child_cid = get_ctx_id(rule_child);
        if (child_cid.first == VtlParser::RuleOptionalExprComponent && child_cid.second == -1) {
            params_nodes.append(visitOptionalExprComponent(rule_child));
        }
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["children"] = children_nodes;
    kwargs["params"] = params_nodes;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_ParamOp, kwargs);
}

py::object visitBinaryNumericComponent(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::object left = visitExprComponent(as_rule(children[2]));
    py::object right = visitExprComponent(as_rule(children[4]));
    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["left"] = left;
    kwargs["op"] = op;
    kwargs["right"] = right;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_BinOp, kwargs);
}

// ---- Time Functions Components ----

py::object visitTimeFunctionsComponents(antlr4::ParserRuleContext* ctx) {
    auto cid = get_ctx_id(ctx);
    // PERIOD_ATOM_COMPONENT = (28, 0)
    if (cid.first == 28 && cid.second == 0) return visitTimeUnaryAtomComponent(ctx);
    // FILL_TIME_ATOM_COMPONENT = (28, 1)
    if (cid.first == 28 && cid.second == 1) return visitFillTimeAtomComponent(ctx);
    // FLOW_ATOM_COMPONENT = (28, 2)
    if (cid.first == 28 && cid.second == 2) {
        std::string op_text = terminal_text(ctx->children[0]);
        py::kwargs kw;
        kw["op"] = py::str(op_text);
        raise_semantic_error("1-1-19-7", kw);
        return py::none(); // unreachable
    }
    // TIME_SHIFT_ATOM_COMPONENT = (28, 3)
    if (cid.first == 28 && cid.second == 3) return visitTimeShiftAtomComponent(ctx);
    // TIME_AGG_ATOM_COMPONENT = (28, 4)
    if (cid.first == 28 && cid.second == 4) return visitTimeAggAtomComponent(ctx);
    // CURRENT_DATE_ATOM_COMPONENT = (28, 5)
    if (cid.first == 28 && cid.second == 5) return visitCurrentDateAtomComponent(ctx);
    // DATE_DIFF_ATOM_COMPONENT = (28, 6)
    if (cid.first == 28 && cid.second == 6) return visitDateDiffAtomComponent(ctx);
    // DATE_ADD_ATOM_COMPONENT = (28, 7)
    if (cid.first == 28 && cid.second == 7) return visitDateAddAtomComponentContext(ctx);
    // YEAR..MONTH_TODAY_ATOM_COMPONENT = (28, 8..15)
    if (cid.first == 28 && cid.second >= 8 && cid.second <= 15) {
        return visitTimeUnaryAtomComponent(ctx);
    }
    throw std::runtime_error("NotImplementedError: visitTimeFunctionsComponents");
}

py::object visitTimeUnaryAtomComponent(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);

    py::object operand_node = py::none();
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (rule_child && (int)rule_child->getRuleIndex() == 3) {
            operand_node = visitExprComponent(rule_child);
            break;
        }
    }

    if (operand_node.is_none()) {
        throw std::runtime_error("NotImplementedError: visitTimeUnaryAtomComponent no operand");
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["operand"] = operand_node;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_UnaryOp, kwargs);
}

py::object visitTimeShiftAtomComponent(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::object left = visitExprComponent(as_rule(children[2]));
    int shift_val = std::stoi(terminal_text(children[4]));
    auto ti = extract_token_info(ctx);

    py::dict const_kwargs;
    const_kwargs["type_"] = "INTEGER_CONSTANT";
    const_kwargs["value"] = py::int_(shift_val);
    for (auto item : ti) const_kwargs[item.first] = item.second;
    py::object right = call_with_kwargs(py_Constant, const_kwargs);

    py::dict kwargs;
    kwargs["left"] = left;
    kwargs["op"] = op;
    kwargs["right"] = right;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_BinOp, kwargs);
}

py::object visitFillTimeAtomComponent(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::list children_node;
    children_node.append(visitExprComponent(as_rule(children[2])));

    py::list param_node;
    if (children.size() > 4) {
        auto ti = extract_token_info(ctx);
        py::dict pk;
        pk["type_"] = "PARAM_TIMESERIES";
        pk["value"] = terminal_text(children[4]);
        for (auto item : ti) pk[item.first] = item.second;
        param_node.append(call_with_kwargs(py_ParamConstant, pk));
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["children"] = children_node;
    kwargs["params"] = param_node;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_ParamOp, kwargs);
}

py::object visitTimeAggAtomComponent(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);

    // periodIndTo is always at index 2
    std::string period_to_raw = terminal_text(children[2]);
    std::string period_to = period_to_raw.substr(1, period_to_raw.size() - 2);
    py::object period_from = py::none();

    size_t idx = 3;
    // Check for periodIndFrom
    if (idx < children.size() && is_terminal(children[idx]) &&
        idx + 1 < children.size() && is_terminal(children[idx + 1])) {
        int sym = terminal_type(children[idx + 1]);
        if (sym == VtlParser::STRING_CONSTANT || sym == VtlParser::OPTIONAL) {
            if (sym != VtlParser::OPTIONAL) {
                std::string from_raw = terminal_text(children[idx + 1]);
                period_from = py::str(from_raw.substr(1, from_raw.size() - 2));
            }
            idx += 2;
        }
    }

    // Find optionalExprComponent
    antlr4::ParserRuleContext* optional_expr_comp = nullptr;
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        auto child_cid = get_ctx_id(rule_child);
        if (child_cid.first == VtlParser::RuleOptionalExprComponent && child_cid.second == -1) {
            optional_expr_comp = rule_child;
            break;
        }
    }

    // Find FIRST/LAST
    py::object conf = py::none();
    for (auto* child : children) {
        auto* term = as_terminal(child);
        if (!term) continue;
        int sym = (int)term->getSymbol()->getType();
        if (sym == VtlParser::FIRST || sym == VtlParser::LAST) {
            conf = py::str(term->getText());
            break;
        }
    }

    py::object operand_node = py::none();
    if (optional_expr_comp) {
        operand_node = visitOptionalExprComponent(optional_expr_comp);
        // isinstance checks: ID -> None, Identifier -> convert to VarID
        if (py::isinstance(operand_node, py_ID)) {
            operand_node = py::none();
        } else if (py::isinstance(operand_node, py_Identifier)) {
            std::string val = operand_node.attr("value").cast<std::string>();
            auto opt_ti = extract_token_info(optional_expr_comp);
            py::dict vk;
            vk["value"] = val;
            for (auto item : opt_ti) vk[item.first] = item.second;
            operand_node = call_with_kwargs(py_VarID, vk);
        }
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["operand"] = operand_node;
    kwargs["period_to"] = period_to;
    kwargs["period_from"] = period_from;
    kwargs["conf"] = conf;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_TimeAggregation, kwargs);
}

py::object visitCurrentDateAtomComponent(antlr4::ParserRuleContext* ctx) {
    std::string op = terminal_text(ctx->children[0]);
    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["children"] = py::list();
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_MulOp, kwargs);
}

py::object visitDateDiffAtomComponent(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::object left = visitExprComponent(as_rule(children[2]));
    // dateTo is 'expr' (rule_index 2), not exprComponent
    py::object right = visitExpr(as_rule(children[4]));
    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["left"] = left;
    kwargs["op"] = op;
    kwargs["right"] = right;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_BinOp, kwargs);
}

py::object visitDateAddAtomComponentContext(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::list children_node;
    children_node.append(visitExprComponent(as_rule(children[2])));

    py::list param_node;
    if (children.size() > 4) {
        param_node.append(visitExprComponent(as_rule(children[4])));
        if (children.size() > 6) {
            param_node.append(visitExprComponent(as_rule(children[6])));
        }
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["children"] = children_node;
    kwargs["params"] = param_node;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_ParamOp, kwargs);
}

// ---- Conditional Functions Components ----

py::object visitConditionalFunctionsComponents(antlr4::ParserRuleContext* ctx) {
    auto cid = get_ctx_id(ctx);
    if (cid.first == 33 && cid.second == 0) return visitNvlAtomComponent(ctx);
    throw std::runtime_error("NotImplementedError: visitConditionalFunctionsComponents");
}

py::object visitNvlAtomComponent(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::object left = visitExprComponent(as_rule(children[2]));
    py::object right = visitExprComponent(as_rule(children[4]));
    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["left"] = left;
    kwargs["op"] = op;
    kwargs["right"] = right;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_BinOp, kwargs);
}

// ---- Comparison Functions Components ----

py::object visitComparisonFunctionsComponents(antlr4::ParserRuleContext* ctx) {
    auto cid = get_ctx_id(ctx);
    if (cid.first == 26 && cid.second == 0) return visitBetweenAtomComponent(ctx);
    if (cid.first == 26 && cid.second == 1) return visitCharsetMatchAtomComponent(ctx);
    if (cid.first == 26 && cid.second == 2) return visitIsNullAtomComponent(ctx);
    throw std::runtime_error("NotImplementedError: visitComparisonFunctionsComponents");
}

py::object visitBetweenAtomComponent(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);

    py::list children_nodes;
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (rule_child && (int)rule_child->getRuleIndex() == 3) {
            children_nodes.append(visitExprComponent(rule_child));
        }
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["children"] = children_nodes;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_MulOp, kwargs);
}

py::object visitCharsetMatchAtomComponent(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::object left = visitExprComponent(as_rule(children[2]));
    py::object right = visitExprComponent(as_rule(children[4]));
    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["left"] = left;
    kwargs["op"] = op;
    kwargs["right"] = right;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_BinOp, kwargs);
}

py::object visitIsNullAtomComponent(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::object operand = visitExprComponent(as_rule(children[2]));
    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["operand"] = operand;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_UnaryOp, kwargs);
}

// ---- Aggregate Functions Components ----

py::object visitAggregateFunctionsComponents(antlr4::ParserRuleContext* ctx) {
    auto cid = get_ctx_id(ctx);
    if (cid.first == 34 && cid.second == 0) return visitAggrComp(ctx);
    if (cid.first == 34 && cid.second == 1) return visitCountAggrComp(ctx);
    throw std::runtime_error("NotImplementedError: visitAggregateFunctionsComponents");
}

py::object visitAggrComp(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::object operand = visitExprComponent(as_rule(children[2]));
    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["operand"] = operand;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_Aggregation, kwargs);
}

py::object visitCountAggrComp(antlr4::ParserRuleContext* ctx) {
    std::string op = terminal_text(ctx->children[0]);
    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_Aggregation, kwargs);
}

// ---- Analytic Functions Components ----

py::object visitAnalyticFunctionsComponents(antlr4::ParserRuleContext* ctx) {
    auto cid = get_ctx_id(ctx);
    if (cid.first == 37 && cid.second == 0) return visitAnSimpleFunctionComponent(ctx);
    if (cid.first == 37 && cid.second == 1) return visitLagOrLeadAnComponent(ctx);
    if (cid.first == 37 && cid.second == 2) return visitRankAnComponent(ctx);
    if (cid.first == 37 && cid.second == 3) return visitRatioToReportAnComponent(ctx);
    throw std::runtime_error("NotImplementedError: visitAnalyticFunctionsComponents");
}

py::object visitAnSimpleFunctionComponent(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::object operand = visitExprComponent(as_rule(children[2]));

    py::object params = py::none();
    py::object partition_by = py::none();
    py::object order_by = py::none();

    // children[5:-2] = OVER LPAREN ... RPAREN
    // Skip first 5 tokens: op LPAREN expr RPAREN OVER LPAREN
    // Skip last 2 tokens: RPAREN
    size_t start = 5;
    size_t end = children.size() >= 2 ? children.size() - 2 : start;
    for (size_t i = start; i < end; i++) {
        auto* rule_child = as_rule(children[i]);
        if (!rule_child) continue;
        auto child_cid = get_ctx_id(rule_child);
        if (child_cid.first == VtlParser::RulePartitionByClause && child_cid.second == -1) {
            partition_by = visitPartitionByClause(rule_child);
        } else if (child_cid.first == VtlParser::RuleOrderByClause && child_cid.second == -1) {
            order_by = visitOrderByClause(rule_child);
        } else if (child_cid.first == VtlParser::RuleWindowingClause && child_cid.second == -1) {
            params = visitWindowingClause(rule_child);
        }
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["operand"] = operand;
    kwargs["partition_by"] = partition_by;
    kwargs["order_by"] = order_by;
    kwargs["window"] = params;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_Analytic, kwargs);
}

py::object visitLagOrLeadAnComponent(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::object operand = visitExprComponent(as_rule(children[2]));

    py::object params = py::none();
    py::object partition_by = py::none();
    py::object order_by = py::none();

    // Skip first 4: op LPAREN expr COMMA, last 2: RPAREN
    size_t start = 4;
    size_t end = children.size() >= 2 ? children.size() - 2 : start;
    for (size_t i = start; i < end; i++) {
        auto* rule_child = as_rule(children[i]);
        if (!rule_child) continue;
        auto child_cid = get_ctx_id(rule_child);
        if (child_cid.first == VtlParser::RulePartitionByClause && child_cid.second == -1) {
            partition_by = visitPartitionByClause(rule_child);
        } else if (child_cid.first == VtlParser::RuleOrderByClause && child_cid.second == -1) {
            order_by = visitOrderByClause(rule_child);
        } else if ((int)rule_child->getRuleIndex() == 53 || (int)rule_child->getRuleIndex() == 43) {
            // SignedInteger (53) or ScalarItem (43)
            if (params.is_none()) {
                params = py::list();
            }
            if ((int)rule_child->getRuleIndex() == 53) {
                params.cast<py::list>().append(visitSignedInteger(rule_child));
            } else {
                params.cast<py::list>().append(visitScalarItem(rule_child));
            }
        }
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["operand"] = operand;
    kwargs["partition_by"] = partition_by;
    kwargs["order_by"] = order_by;
    kwargs["params"] = params;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_Analytic, kwargs);
}

py::object visitRankAnComponent(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);

    py::object partition_by = py::none();
    py::object order_by = py::none();

    size_t start = 4;
    size_t end = children.size() >= 2 ? children.size() - 2 : start;
    for (size_t i = start; i < end; i++) {
        auto* rule_child = as_rule(children[i]);
        if (!rule_child) continue;
        auto child_cid = get_ctx_id(rule_child);
        if (child_cid.first == VtlParser::RulePartitionByClause && child_cid.second == -1) {
            partition_by = visitPartitionByClause(rule_child);
        } else if (child_cid.first == VtlParser::RuleOrderByClause && child_cid.second == -1) {
            order_by = visitOrderByClause(rule_child);
        }
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["operand"] = py::none();
    kwargs["partition_by"] = partition_by;
    kwargs["order_by"] = order_by;
    kwargs["window"] = py::none();
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_Analytic, kwargs);
}

py::object visitRatioToReportAnComponent(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::object operand = visitExprComponent(as_rule(children[2]));
    py::object partition_by = visitPartitionByClause(as_rule(children[5]));

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["operand"] = operand;
    kwargs["partition_by"] = partition_by;
    kwargs["order_by"] = py::none();
    kwargs["window"] = py::none();
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_Analytic, kwargs);
}

// ============================================================
// Forward declaration stub for Phase 3 (Expr)
// ============================================================

py::object visitExpr(antlr4::ParserRuleContext* /*ctx*/) {
    throw std::runtime_error("NotImplementedError: visitExpr (Phase 3 not yet implemented)");
}

} // namespace ASTBuilder
