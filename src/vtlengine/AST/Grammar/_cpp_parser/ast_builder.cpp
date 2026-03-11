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

// Release all cached py::object refs (prevents segfault at interpreter shutdown)
// Note: cleanup_phase3() is called from ASTBuilder::cleanup() since Phase 3
// statics are in the namespace.
static void do_cleanup() {
    // Phase 1 statics (outside namespace)
    py_Start = py::object();
    py_Assignment = py::object();
    py_PersistentAssignment = py::object();
    py_VarID = py::object();
    py_UnaryOp = py::object();
    py_BinOp = py::object();
    py_MulOp = py::object();
    py_ParamOp = py::object();
    py_UDOCall = py::object();
    py_JoinOp = py::object();
    py_Constant = py::object();
    py_ParamConstant = py::object();
    py_Identifier = py::object();
    py_ID = py::object();
    py_Collection = py::object();
    py_Windowing = py::object();
    py_OrderBy = py::object();
    py_Analytic = py::object();
    py_RegularAggregation = py::object();
    py_RenameNode = py::object();
    py_Aggregation = py::object();
    py_TimeAggregation = py::object();
    py_If = py::object();
    py_CaseObj = py::object();
    py_Case = py::object();
    py_Validation = py::object();
    py_ComponentType_AST = py::object();
    py_ASTScalarType = py::object();
    py_DatasetType_AST = py::object();
    py_Types = py::object();
    py_Argument = py::object();
    py_Operator = py::object();
    py_DefIdentifier = py::object();
    py_DPRIdentifier = py::object();
    py_HRBinOp = py::object();
    py_HRUnOp = py::object();
    py_HRule = py::object();
    py_DPRule = py::object();
    py_HRuleset = py::object();
    py_HROperation = py::object();
    py_DPValidation = py::object();
    py_DPRuleset = py::object();
    py_EvalOp = py::object();
    py_NoOp = py::object();
    py_ParFunction = py::object();
    py_Comment = py::object();
    py_String = py::object();
    py_Integer = py::object();
    py_Number = py::object();
    py_Boolean = py::object();
    py_Date = py::object();
    py_TimePeriod = py::object();
    py_Duration = py::object();
    py_TimeInterval = py::object();
    py_Role = py::object();
    py_Component = py::object();
    py_Dataset = py::object();
    py_Scalar = py::object();
    py_SemanticError = py::object();
    g_initialized = false;
}

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

// Forward declaration for Phase 3 cleanup (defined later in this namespace)
static void cleanup_phase3();

void cleanup() {
    cleanup_phase3();
    do_cleanup();
}

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
    py_ParFunction = ast_mod.attr("ParFunction");
    py_HRuleset = ast_mod.attr("HRuleset");
    py_HROperation = ast_mod.attr("HROperation");
    py_DPValidation = ast_mod.attr("DPValidation");
    py_DPRuleset = ast_mod.attr("DPRuleset");
    py_DefIdentifier = ast_mod.attr("DefIdentifier");
    py_Argument = ast_mod.attr("Argument");
    py_Operator = ast_mod.attr("Operator");
    py_HRBinOp = ast_mod.attr("HRBinOp");
    py_HRUnOp = ast_mod.attr("HRUnOp");
    py_HRule = ast_mod.attr("HRule");
    py_DPRule = ast_mod.attr("DPRule");

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
// build_ast - entry point: walks the full parse tree
// ============================================================
py::object build_ast(antlr4::ParserRuleContext* root) {
    if (!g_initialized) init();
    return visitStart(root);
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
    try {
        return py::int_(std::stoll(text));
    } catch (const std::out_of_range&) {
        // Fall back to Python int for arbitrarily large numbers
        return py::int_(py::str(text));
    }
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
// Phase 3: Expr visitors
// ============================================================

// Forward declarations for internal helpers
static py::object visitParenthesisExpr(antlr4::ParserRuleContext* ctx);
static py::object visitMembershipExpr(antlr4::ParserRuleContext* ctx);
static py::object visitClauseExpr(antlr4::ParserRuleContext* ctx);
static py::object visitFunctionsExpression(antlr4::ParserRuleContext* ctx);
static py::object visitUnaryExpr(antlr4::ParserRuleContext* ctx);
static py::object bin_op_creator(antlr4::ParserRuleContext* ctx);
static py::object visitInNotInExpr(antlr4::ParserRuleContext* ctx);
static py::object visitCallDataset(antlr4::ParserRuleContext* ctx);
static py::object visitEvalAtom(antlr4::ParserRuleContext* ctx);
static py::object visitCastExprDataset(antlr4::ParserRuleContext* ctx);
static py::object visitParameter(antlr4::ParserRuleContext* ctx);
static py::object visitUnaryStringFunction(antlr4::ParserRuleContext* ctx);
static py::object visitSubstrAtom(antlr4::ParserRuleContext* ctx);
static py::object visitReplaceAtom(antlr4::ParserRuleContext* ctx);
static py::object visitInstrAtom(antlr4::ParserRuleContext* ctx);
static py::object visitUnaryNumeric(antlr4::ParserRuleContext* ctx);
static py::object visitUnaryWithOptionalNumeric(antlr4::ParserRuleContext* ctx);
static py::object visitBinaryNumeric(antlr4::ParserRuleContext* ctx);
static py::object visitBetweenAtom(antlr4::ParserRuleContext* ctx);
static py::object visitCharsetMatchAtom(antlr4::ParserRuleContext* ctx);
static py::object visitIsNullAtom(antlr4::ParserRuleContext* ctx);
static py::object visitExistInAtom(antlr4::ParserRuleContext* ctx);
static py::object visitTimeUnaryAtom(antlr4::ParserRuleContext* ctx);
static py::object visitTimeShiftAtom(antlr4::ParserRuleContext* ctx);
static py::object visitFillTimeAtom(antlr4::ParserRuleContext* ctx);
static py::object visitTimeAggAtom(antlr4::ParserRuleContext* ctx);
static py::object visitFlowAtom(antlr4::ParserRuleContext* ctx);
static py::object visitCurrentDateAtom(antlr4::ParserRuleContext* ctx);
static py::object visitTimeDiffAtom(antlr4::ParserRuleContext* ctx);
static py::object visitTimeAddAtom(antlr4::ParserRuleContext* ctx);
static py::object visitNvlAtom(antlr4::ParserRuleContext* ctx);
static py::object visitUnionAtom(antlr4::ParserRuleContext* ctx);
static py::object visitIntersectAtom(antlr4::ParserRuleContext* ctx);
static py::object visitSetOrSYmDiffAtom(antlr4::ParserRuleContext* ctx);
static py::object visitImbalanceExpr(antlr4::ParserRuleContext* ctx);
static py::object visitAggrDataset(antlr4::ParserRuleContext* ctx);
static py::object visitAnSimpleFunction(antlr4::ParserRuleContext* ctx);
static py::object visitLagOrLeadAn(antlr4::ParserRuleContext* ctx);
static py::object visitRatioToReportAn(antlr4::ParserRuleContext* ctx);
static py::object visitRenameClauseItem(antlr4::ParserRuleContext* ctx);
static py::object visitAggregateClause(antlr4::ParserRuleContext* ctx);
static py::object visitAggrFunctionClause(antlr4::ParserRuleContext* ctx);
static py::object visitCalcClauseItem(antlr4::ParserRuleContext* ctx);
static py::object visitSubspaceClauseItem(antlr4::ParserRuleContext* ctx);
static py::object visitJoinApplyClause(antlr4::ParserRuleContext* ctx);

// Cached Python class references for Phase 3
static py::object py_Validation;
static py::object py_CHInputMode_cls;
static py::object py_HRInputMode_cls;
static py::object py_ValidationMode_cls;
static py::object py_ValidationOutput_cls;
static py::object py_HierarchyOutput_cls;
static py::object py_DATASET_PRIORITY;
static py::object py_de_ruleset_elements;
static bool g_phase3_initialized = false;

static void cleanup_phase3() {
    py_Validation = py::object();
    py_CHInputMode_cls = py::object();
    py_HRInputMode_cls = py::object();
    py_ValidationMode_cls = py::object();
    py_ValidationOutput_cls = py::object();
    py_HierarchyOutput_cls = py::object();
    py_DATASET_PRIORITY = py::object();
    py_de_ruleset_elements = py::object();
    g_phase3_initialized = false;
}

static void init_phase3() {
    if (g_phase3_initialized) return;

    auto ast_mod = py::module_::import("vtlengine.AST");
    py_Validation = ast_mod.attr("Validation");
    py_CHInputMode_cls = ast_mod.attr("CHInputMode");
    py_HRInputMode_cls = ast_mod.attr("HRInputMode");
    py_ValidationMode_cls = ast_mod.attr("ValidationMode");
    py_ValidationOutput_cls = ast_mod.attr("ValidationOutput");
    py_HierarchyOutput_cls = ast_mod.attr("HierarchyOutput");

    auto tokens_mod = py::module_::import("vtlengine.AST.Grammar.tokens");
    py_DATASET_PRIORITY = tokens_mod.attr("DATASET_PRIORITY");

    auto de_mod = py::module_::import("vtlengine.AST.ASTDataExchange");
    py_de_ruleset_elements = de_mod.attr("de_ruleset_elements");

    g_phase3_initialized = true;
}

// ---- visitExpr ----

py::object visitExpr(antlr4::ParserRuleContext* ctx) {
    if (!g_phase3_initialized) init_phase3();

    auto& children = ctx->children;
    auto cid = get_ctx_id(ctx);

    // PARENTHESIS_EXPR = (2, 0)
    if (cid.first == 2 && cid.second == 0) {
        return visitParenthesisExpr(ctx);
    }
    // FUNCTIONS_EXPRESSION = (2, 1)
    if (cid.first == 2 && cid.second == 1) {
        return visitFunctionsExpression(as_rule(children[0]));
    }
    // CLAUSE_EXPR = (2, 2)
    if (cid.first == 2 && cid.second == 2) {
        return visitClauseExpr(ctx);
    }
    // MEMBERSHIP_EXPR = (2, 3)
    if (cid.first == 2 && cid.second == 3) {
        return visitMembershipExpr(ctx);
    }
    // UNARY_EXPR = (2, 4)
    if (cid.first == 2 && cid.second == 4) {
        return visitUnaryExpr(ctx);
    }
    // ARITHMETIC_EXPR = (2, 5)
    if (cid.first == 2 && cid.second == 5) {
        return bin_op_creator(ctx);
    }
    // ARITHMETIC_EXPR_OR_CONCAT = (2, 6)
    if (cid.first == 2 && cid.second == 6) {
        return bin_op_creator(ctx);
    }
    // COMPARISON_EXPR = (2, 7)
    if (cid.first == 2 && cid.second == 7) {
        return bin_op_creator(ctx);
    }
    // IN_NOT_IN_EXPR = (2, 8)
    if (cid.first == 2 && cid.second == 8) {
        return visitInNotInExpr(ctx);
    }
    // BOOLEAN_EXPR = (2, 9)
    if (cid.first == 2 && cid.second == 9) {
        return bin_op_creator(ctx);
    }
    // IF_EXPR = (2, 10)
    if (cid.first == 2 && cid.second == 10) {
        py::object condition = visitExpr(as_rule(children[1]));
        py::object then_op = visitExpr(as_rule(children[3]));
        py::object else_op = visitExpr(as_rule(children[5]));
        auto ti = extract_token_info(ctx);
        py::dict kwargs;
        kwargs["condition"] = condition;
        kwargs["thenOp"] = then_op;
        kwargs["elseOp"] = else_op;
        for (auto item : ti) kwargs[item.first] = item.second;
        return call_with_kwargs(py_If, kwargs);
    }
    // CASE_EXPR = (2, 11)
    if (cid.first == 2 && cid.second == 11) {
        size_t n = children.size();
        if (n % 4 != 3) {
            throw std::runtime_error("Syntax error.");
        }
        py::object else_node = visitExpr(as_rule(children[n - 1]));
        // Work on children[1:-2]
        py::list cases;
        for (size_t i = 1; i < n - 2; i += 4) {
            py::object condition = visitExpr(as_rule(children[i + 1]));
            py::object thenOp = visitExpr(as_rule(children[i + 3]));
            auto ti_case = extract_token_info(as_rule(children[i + 1]));
            py::dict case_kwargs;
            case_kwargs["condition"] = condition;
            case_kwargs["thenOp"] = thenOp;
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
    // CONSTANT_EXPR = (2, 12)
    if (cid.first == 2 && cid.second == 12) {
        return visitConstant(as_rule(children[0]));
    }
    // VAR_ID_EXPR = (2, 13)
    if (cid.first == 2 && cid.second == 13) {
        return visitVarIdExpr(as_rule(children[0]));
    }

    throw std::runtime_error("NotImplementedError: visitExpr");
}

// ---- visitOptionalExpr ----

py::object visitOptionalExpr(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    auto* c = children[0];

    auto* rule_child = as_rule(c);
    if (rule_child && (int)rule_child->getRuleIndex() == 2) {
        // expr (rule_index == 2)
        return visitExpr(rule_child);
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

// ---- Helpers ----

static py::object bin_op_creator(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    py::object left_node = visitExpr(as_rule(children[0]));

    std::string op;
    auto* mid_rule = as_rule(children[1]);
    if (mid_rule) {
        // Check if it's a comparisonOperand rule (rule_index == 100)
        if ((int)mid_rule->getRuleIndex() == VtlParser::RuleComparisonOperand) {
            op = terminal_text(mid_rule->children[0]);
        } else {
            op = node_text(children[1]);
        }
    } else {
        op = terminal_text(children[1]);
    }

    py::object right_node = visitExpr(as_rule(children[2]));
    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["left"] = left_node;
    kwargs["op"] = op;
    kwargs["right"] = right_node;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_BinOp, kwargs);
}

static py::object visitParenthesisExpr(antlr4::ParserRuleContext* ctx) {
    py::object operand = visitExpr(as_rule(ctx->children[1]));
    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["operand"] = operand;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_ParFunction, kwargs);
}

static py::object visitUnaryExpr(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::object right = visitExpr(as_rule(children[1]));
    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["operand"] = right;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_UnaryOp, kwargs);
}

static py::object visitMembershipExpr(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    // Collect simpleComponentId children
    std::vector<antlr4::ParserRuleContext*> membership;
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (rule_child && (int)rule_child->getRuleIndex() == VtlParser::RuleSimpleComponentId) {
            membership.push_back(rule_child);
        }
    }

    py::object previous_node = visitExpr(as_rule(children[0]));

    if (!membership.empty()) {
        py::object right = visitSimpleComponentId(membership[0]);
        auto ti = extract_token_info(ctx);
        py::dict kwargs;
        kwargs["left"] = previous_node;
        kwargs["op"] = "#";
        kwargs["right"] = right;
        for (auto item : ti) kwargs[item.first] = item.second;
        previous_node = call_with_kwargs(py_BinOp, kwargs);
    }

    return previous_node;
}

static py::object visitClauseExpr(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    py::object dataset = visitExpr(as_rule(children[0]));
    py::object dataset_clause = visitDatasetClause(as_rule(children[2]));

    dataset_clause.attr("dataset") = dataset;
    return dataset_clause;
}

static py::object visitInNotInExpr(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    py::object left_node = visitExpr(as_rule(children[0]));
    std::string op = terminal_text(children[1]);

    auto* child2 = as_rule(children[2]);
    py::object right_node;
    if (child2) {
        int ri = (int)child2->getRuleIndex();
        if (ri == VtlParser::RuleLists) {
            right_node = visitLists(child2);
        } else if (ri == VtlParser::RuleValueDomainID) {
            right_node = visitValueDomainID(child2);
        } else {
            throw std::runtime_error("NotImplementedError: visitInNotInExpr");
        }
    } else {
        throw std::runtime_error("NotImplementedError: visitInNotInExpr terminal");
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["left"] = left_node;
    kwargs["op"] = op;
    kwargs["right"] = right_node;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_BinOp, kwargs);
}

// ---- Functions Expression ----

static py::object visitFunctionsExpression(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    auto* c = as_rule(children[0]);
    auto cid = get_ctx_id(ctx);

    // JOIN_FUNCTIONS = (5, 0)
    if (cid.first == 5 && cid.second == 0) return visitJoinFunctions(c);
    // GENERIC_FUNCTIONS = (5, 1)
    if (cid.first == 5 && cid.second == 1) return visitGenericFunctions(c);
    // STRING_FUNCTIONS = (5, 2)
    if (cid.first == 5 && cid.second == 2) return visitStringFunctions(c);
    // NUMERIC_FUNCTIONS = (5, 3)
    if (cid.first == 5 && cid.second == 3) return visitNumericFunctions(c);
    // COMPARISON_FUNCTIONS = (5, 4)
    if (cid.first == 5 && cid.second == 4) return visitComparisonFunctions(c);
    // TIME_FUNCTIONS = (5, 5)
    if (cid.first == 5 && cid.second == 5) return visitTimeFunctions(c);
    // SET_FUNCTIONS = (5, 6)
    if (cid.first == 5 && cid.second == 6) return visitSetFunctions(c);
    // HIERARCHY_FUNCTIONS = (5, 7)
    if (cid.first == 5 && cid.second == 7) return visitHierarchyFunctions(c);
    // VALIDATION_FUNCTIONS = (5, 8)
    if (cid.first == 5 && cid.second == 8) return visitValidationFunctions(c);
    // CONDITIONAL_FUNCTIONS = (5, 9)
    if (cid.first == 5 && cid.second == 9) return visitConditionalFunctions(c);
    // AGGREGATE_FUNCTIONS = (5, 10)
    if (cid.first == 5 && cid.second == 10) return visitAggregateFunctions(c);
    // ANALYTIC_FUNCTIONS = (5, 11)
    if (cid.first == 5 && cid.second == 11) return visitAnalyticFunctions(c);

    throw std::runtime_error("NotImplementedError: visitFunctionsExpression");
}

// ---- Join Functions ----

py::object visitJoinFunctions(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    py::object using_node = py::none();
    std::string op_node = terminal_text(children[0]);

    py::list clause_node;
    if (op_node == "inner_join" || op_node == "left_join") {
        auto result = visitJoinClause(as_rule(children[2]));
        clause_node = result.first;
        using_node = result.second;
    } else {
        clause_node = visitJoinClauseWithoutUsing(as_rule(children[2]));
    }

    py::list body_node = visitJoinBody(as_rule(children[3]));

    auto ti = extract_token_info(ctx);

    if (py::len(body_node) != 0) {
        py::dict join_kwargs;
        join_kwargs["op"] = op_node;
        join_kwargs["clauses"] = clause_node;
        join_kwargs["using"] = using_node;
        for (auto item : ti) join_kwargs[item.first] = item.second;
        py::object previous_node = call_with_kwargs(py_JoinOp, join_kwargs);

        py::object regular_aggregation = py::none();
        for (auto body : body_node) {
            regular_aggregation = py::reinterpret_borrow<py::object>(body);
            regular_aggregation.attr("dataset") = previous_node;
            previous_node = regular_aggregation;
        }

        previous_node.attr("isLast") = true;
        return regular_aggregation;
    } else {
        py::dict join_kwargs;
        join_kwargs["op"] = op_node;
        join_kwargs["clauses"] = clause_node;
        join_kwargs["using"] = using_node;
        for (auto item : ti) join_kwargs[item.first] = item.second;
        py::object join_node = call_with_kwargs(py_JoinOp, join_kwargs);
        join_node.attr("isLast") = true;
        return join_node;
    }
}

py::object visitJoinClauseItem(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    py::object left_node = visitExpr(as_rule(children[0]));
    if (children.size() == 1) {
        return left_node;
    }

    auto ti = extract_token_info(ctx);
    std::string intop_node = terminal_text(children[1]);

    // right_node = Identifier(value=alias, kind="DatasetID", ...)
    py::object alias_val = visitAlias(as_rule(children[2]));
    py::dict ti_op;
    auto* term1 = as_terminal(children[1]);
    if (term1) {
        ti_op = extract_token_info_terminal(term1);
    } else {
        ti_op = extract_token_info(as_rule(children[1]));
    }
    py::dict right_kwargs;
    right_kwargs["value"] = alias_val;
    right_kwargs["kind"] = "DatasetID";
    for (auto item : ti_op) right_kwargs[item.first] = item.second;
    py::object right_node = call_with_kwargs(py_Identifier, right_kwargs);

    py::dict kwargs;
    kwargs["left"] = left_node;
    kwargs["op"] = intop_node;
    kwargs["right"] = right_node;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_BinOp, kwargs);
}

std::pair<py::list, py::object> visitJoinClause(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    py::list clause_nodes;
    py::list component_nodes;
    py::object using_val = py::none();

    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        int ri = (int)rule_child->getRuleIndex();
        if (ri == VtlParser::RuleJoinClauseItem) {
            clause_nodes.append(visitJoinClauseItem(rule_child));
        } else if (ri == VtlParser::RuleComponentID) {
            py::object comp = visitComponentID(rule_child);
            component_nodes.append(comp.attr("value"));
        }
    }

    if (py::len(component_nodes) != 0) {
        using_val = component_nodes;
    }

    return {clause_nodes, using_val};
}

py::list visitJoinClauseWithoutUsing(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    py::list clause_nodes;

    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        if ((int)rule_child->getRuleIndex() == VtlParser::RuleJoinClauseItem) {
            clause_nodes.append(visitJoinClauseItem(rule_child));
        }
    }

    return clause_nodes;
}

py::list visitJoinBody(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    py::list body_nodes;

    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) {
            if (is_terminal(child)) {
                throw std::runtime_error("NotImplementedError: visitJoinBody terminal");
            }
            continue;
        }
        int ri = (int)rule_child->getRuleIndex();
        if (ri == VtlParser::RuleFilterClause) {
            body_nodes.append(visitFilterClause(rule_child));
        } else if (ri == VtlParser::RuleCalcClause) {
            body_nodes.append(visitCalcClause(rule_child));
        } else if (ri == VtlParser::RuleJoinApplyClause) {
            body_nodes.append(visitJoinApplyClause(rule_child));
        } else if (ri == VtlParser::RuleAggrClause) {
            body_nodes.append(visitAggrClause(rule_child));
        } else if (ri == VtlParser::RuleKeepOrDropClause) {
            body_nodes.append(visitKeepOrDropClause(rule_child));
        } else if (ri == VtlParser::RuleRenameClause) {
            body_nodes.append(visitRenameClause(rule_child));
        } else {
            throw std::runtime_error("NotImplementedError: visitJoinBody unknown rule");
        }
    }

    return body_nodes;
}

static py::object visitJoinApplyClause(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::list operand_nodes;
    operand_nodes.append(visitExpr(as_rule(children[1])));
    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["children"] = operand_nodes;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_RegularAggregation, kwargs);
}

// ---- Generic Functions ----

py::object visitGenericFunctions(antlr4::ParserRuleContext* ctx) {
    auto cid = get_ctx_id(ctx);
    // CALL_DATASET = (17, 0)
    if (cid.first == 17 && cid.second == 0) return visitCallDataset(ctx);
    // EVAL_ATOM = (17, 1)
    if (cid.first == 17 && cid.second == 1) return visitEvalAtom(ctx);
    // CAST_EXPR_DATASET = (17, 2)
    if (cid.first == 17 && cid.second == 2) return visitCastExprDataset(ctx);
    throw std::runtime_error("NotImplementedError: visitGenericFunctions");
}

static py::object visitCallDataset(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    py::object op = visitOperatorID(as_rule(children[0]));

    py::list param_nodes;
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        if ((int)rule_child->getRuleIndex() == VtlParser::RuleParameter) {
            param_nodes.append(visitParameter(rule_child));
        }
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["params"] = param_nodes;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_UDOCall, kwargs);
}

static py::object visitEvalAtom(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    py::object routine_name = visitRoutineName(as_rule(children[2]));

    py::list var_ids_nodes;
    py::list constant_nodes;
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        int ri = (int)rule_child->getRuleIndex();
        auto child_cid = get_ctx_id(rule_child);
        if (ri == VtlParser::RuleVarID) {
            var_ids_nodes.append(visitVarID(rule_child));
        }
        // ScalarItem (rule_index 43) with either SIMPLE_SCALAR or SCALAR_WITH_CAST
        if (child_cid.first == VtlParser::RuleScalarItem &&
            (child_cid.second == 0 || child_cid.second == 1)) {
            constant_nodes.append(visitScalarItem(rule_child));
        }
    }

    py::list children_nodes;
    for (auto item : var_ids_nodes) children_nodes.append(item);
    for (auto item : constant_nodes) children_nodes.append(item);

    // language
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

    // output
    py::list output_list;
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        if ((int)rule_child->getRuleIndex() == VtlParser::RuleEvalDatasetType) {
            output_list.append(visitEvalDatasetType(rule_child));
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
    kwargs["operands"] = children_nodes;
    kwargs["output"] = output_list[py::int_(0)];
    kwargs["language"] = language_list[py::int_(0)];
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_EvalOp, kwargs);
}

static py::object visitCastExprDataset(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);

    py::list expr_nodes;
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (rule_child && (int)rule_child->getRuleIndex() == 2) {
            expr_nodes.append(visitExpr(rule_child));
        }
    }

    py::list basic_scalar_types;
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        if ((int)rule_child->getRuleIndex() == VtlParser::RuleBasicScalarType) {
            basic_scalar_types.append(visitBasicScalarType(rule_child));
        }
    }

    // Check for valueDomainName (will throw)
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        if ((int)rule_child->getRuleIndex() == VtlParser::RuleValueDomainName) {
            visitValueDomainName(rule_child);
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
                std::string text = term->getText();
                // Strip surrounding quotes
                if (text.size() >= 2 && text.front() == '"' && text.back() == '"') {
                    text = text.substr(1, text.size() - 2);
                }
                pk["value"] = text;
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

    throw std::runtime_error("NotImplementedError: visitCastExprDataset");
}

static py::object visitParameter(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    auto* c = children[0];
    auto* rule_child = as_rule(c);
    if (rule_child && (int)rule_child->getRuleIndex() == 2) {
        return visitExpr(rule_child);
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
    throw std::runtime_error("NotImplementedError: visitParameter");
}

// ---- String Functions ----

py::object visitStringFunctions(antlr4::ParserRuleContext* ctx) {
    auto cid = get_ctx_id(ctx);
    if (cid.first == 21 && cid.second == 0) return visitUnaryStringFunction(ctx);
    if (cid.first == 21 && cid.second == 1) return visitSubstrAtom(ctx);
    if (cid.first == 21 && cid.second == 2) return visitReplaceAtom(ctx);
    if (cid.first == 21 && cid.second == 3) return visitInstrAtom(ctx);
    throw std::runtime_error("NotImplementedError: visitStringFunctions");
}

static py::object visitUnaryStringFunction(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::object operand = visitExpr(as_rule(children[2]));
    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["operand"] = operand;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_UnaryOp, kwargs);
}

static py::object visitSubstrAtom(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);

    py::list children_nodes;
    py::list params_nodes;

    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        if ((int)rule_child->getRuleIndex() == 2) {
            children_nodes.append(visitExpr(rule_child));
        }
        if ((int)rule_child->getRuleIndex() == VtlParser::RuleOptionalExpr) {
            params_nodes.append(visitOptionalExpr(rule_child));
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

static py::object visitReplaceAtom(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);

    py::list expressions;
    py::list opt_params;

    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        if ((int)rule_child->getRuleIndex() == 2) {
            expressions.append(visitExpr(rule_child));
        }
        if ((int)rule_child->getRuleIndex() == VtlParser::RuleOptionalExpr) {
            opt_params.append(visitOptionalExpr(rule_child));
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

static py::object visitInstrAtom(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);

    py::list expressions;
    py::list opt_params;

    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        if ((int)rule_child->getRuleIndex() == 2) {
            expressions.append(visitExpr(rule_child));
        }
        if ((int)rule_child->getRuleIndex() == VtlParser::RuleOptionalExpr) {
            opt_params.append(visitOptionalExpr(rule_child));
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

// ---- Numeric Functions ----

py::object visitNumericFunctions(antlr4::ParserRuleContext* ctx) {
    auto cid = get_ctx_id(ctx);
    if (cid.first == 23 && cid.second == 0) return visitUnaryNumeric(ctx);
    if (cid.first == 23 && cid.second == 1) return visitUnaryWithOptionalNumeric(ctx);
    if (cid.first == 23 && cid.second == 2) return visitBinaryNumeric(ctx);
    throw std::runtime_error("NotImplementedError: visitNumericFunctions");
}

static py::object visitUnaryNumeric(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::object operand = visitExpr(as_rule(children[2]));
    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["operand"] = operand;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_UnaryOp, kwargs);
}

static py::object visitUnaryWithOptionalNumeric(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);

    py::list children_nodes;
    py::list params_nodes;

    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        if ((int)rule_child->getRuleIndex() == 2) {
            children_nodes.append(visitExpr(rule_child));
        }
        if ((int)rule_child->getRuleIndex() == VtlParser::RuleOptionalExpr) {
            params_nodes.append(visitOptionalExpr(rule_child));
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

static py::object visitBinaryNumeric(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::object left = visitExpr(as_rule(children[2]));
    py::object right = visitExpr(as_rule(children[4]));
    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["left"] = left;
    kwargs["op"] = op;
    kwargs["right"] = right;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_BinOp, kwargs);
}

// ---- Comparison Functions ----

py::object visitComparisonFunctions(antlr4::ParserRuleContext* ctx) {
    auto cid = get_ctx_id(ctx);
    if (cid.first == 25 && cid.second == 0) return visitBetweenAtom(ctx);
    if (cid.first == 25 && cid.second == 1) return visitCharsetMatchAtom(ctx);
    if (cid.first == 25 && cid.second == 2) return visitIsNullAtom(ctx);
    if (cid.first == 25 && cid.second == 3) return visitExistInAtom(ctx);
    throw std::runtime_error("NotImplementedError: visitComparisonFunctions");
}

static py::object visitBetweenAtom(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);

    py::list children_nodes;
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (rule_child && (int)rule_child->getRuleIndex() == 2) {
            children_nodes.append(visitExpr(rule_child));
        }
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["children"] = children_nodes;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_MulOp, kwargs);
}

static py::object visitCharsetMatchAtom(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::object left = visitExpr(as_rule(children[2]));
    py::object right = visitExpr(as_rule(children[4]));
    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["left"] = left;
    kwargs["op"] = op;
    kwargs["right"] = right;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_BinOp, kwargs);
}

static py::object visitIsNullAtom(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::object operand = visitExpr(as_rule(children[2]));
    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["operand"] = operand;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_UnaryOp, kwargs);
}

static py::object visitExistInAtom(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);

    py::list operand_nodes;
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (rule_child && (int)rule_child->getRuleIndex() == 2) {
            operand_nodes.append(visitExpr(rule_child));
        }
    }

    py::list retain_nodes;
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        if ((int)rule_child->getRuleIndex() == VtlParser::RuleRetainType) {
            retain_nodes.append(visitRetainType(rule_child));
        }
    }

    // Merge operand_nodes + retain_nodes
    py::list all_children;
    for (auto item : operand_nodes) all_children.append(item);
    for (auto item : retain_nodes) all_children.append(item);

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["children"] = all_children;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_MulOp, kwargs);
}

// ---- Time Functions ----

py::object visitTimeFunctions(antlr4::ParserRuleContext* ctx) {
    auto cid = get_ctx_id(ctx);
    // PERIOD_ATOM = (27, 0)
    if (cid.first == 27 && cid.second == 0) return visitTimeUnaryAtom(ctx);
    // FILL_TIME_ATOM = (27, 1)
    if (cid.first == 27 && cid.second == 1) return visitFillTimeAtom(ctx);
    // FLOW_ATOM = (27, 2)
    if (cid.first == 27 && cid.second == 2) return visitFlowAtom(ctx);
    // TIME_SHIFT_ATOM = (27, 3)
    if (cid.first == 27 && cid.second == 3) return visitTimeShiftAtom(ctx);
    // TIME_AGG_ATOM = (27, 4)
    if (cid.first == 27 && cid.second == 4) return visitTimeAggAtom(ctx);
    // CURRENT_DATE_ATOM = (27, 5)
    if (cid.first == 27 && cid.second == 5) return visitCurrentDateAtom(ctx);
    // DATE_DIFF_ATOM = (27, 6)
    if (cid.first == 27 && cid.second == 6) return visitTimeDiffAtom(ctx);
    // DATE_ADD_ATOM = (27, 7)
    if (cid.first == 27 && cid.second == 7) return visitTimeAddAtom(ctx);
    // YEAR..MONTH_TODAY_ATOM = (27, 8..15)
    if (cid.first == 27 && cid.second >= 8 && cid.second <= 15) {
        return visitTimeUnaryAtom(ctx);
    }
    throw std::runtime_error("NotImplementedError: visitTimeFunctions");
}

static py::object visitTimeUnaryAtom(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);

    py::object operand_node = py::none();
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (rule_child && (int)rule_child->getRuleIndex() == 2) {
            operand_node = visitExpr(rule_child);
            break;
        }
    }

    if (operand_node.is_none()) {
        throw std::runtime_error("NotImplementedError: visitTimeUnaryAtom no operand");
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["operand"] = operand_node;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_UnaryOp, kwargs);
}

static py::object visitTimeShiftAtom(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::object left_node = visitExpr(as_rule(children[2]));

    // children[4] is signedInteger
    py::object shift_val = visitSignedInteger(as_rule(children[4]));
    auto ti_shift = extract_token_info(as_rule(children[4]));
    py::dict const_kwargs;
    const_kwargs["type_"] = "INTEGER_CONSTANT";
    const_kwargs["value"] = shift_val;
    for (auto item : ti_shift) const_kwargs[item.first] = item.second;
    py::object right_node = call_with_kwargs(py_Constant, const_kwargs);

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["left"] = left_node;
    kwargs["op"] = op;
    kwargs["right"] = right_node;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_BinOp, kwargs);
}

static py::object visitFillTimeAtom(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::list children_node;
    children_node.append(visitExpr(as_rule(children[2])));

    py::list param_constant_node;
    if (children.size() > 4) {
        auto ti_param = extract_token_info(ctx);
        py::dict pk;
        pk["type_"] = "PARAM_TIMESERIES";
        pk["value"] = terminal_text(children[4]);
        for (auto item : ti_param) pk[item.first] = item.second;
        param_constant_node.append(call_with_kwargs(py_ParamConstant, pk));
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["children"] = children_node;
    kwargs["params"] = param_constant_node;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_ParamOp, kwargs);
}

static py::object visitTimeAggAtom(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);

    py::object period_to = py::none();
    py::object period_from = py::none();
    antlr4::ParserRuleContext* optional_expr_ctx = nullptr;
    py::object conf = py::none();
    bool period_to_found = false;

    for (auto* child : children) {
        auto* term = as_terminal(child);
        if (term) {
            int sym = (int)term->getSymbol()->getType();
            if (sym == VtlParser::STRING_CONSTANT) {
                std::string raw = term->getText();
                std::string val = raw.substr(1, raw.size() - 2);
                if (!period_to_found) {
                    period_to = py::str(val);
                    period_to_found = true;
                } else {
                    period_from = py::str(val);
                }
            } else if (sym == VtlParser::OPTIONAL) {
                // skip
            } else if (sym == VtlParser::FIRST || sym == VtlParser::LAST) {
                conf = py::str(term->getText());
            }
            continue;
        }
        auto* rule_child = as_rule(child);
        if (rule_child && (int)rule_child->getRuleIndex() == VtlParser::RuleOptionalExpr) {
            optional_expr_ctx = rule_child;
        }
    }

    py::object operand_node = py::none();
    if (optional_expr_ctx) {
        operand_node = visitOptionalExpr(optional_expr_ctx);
        // isinstance checks
        if (py::isinstance(operand_node, py_ID)) {
            operand_node = py::none();
        } else if (py::isinstance(operand_node, py_Identifier)) {
            std::string val = operand_node.attr("value").cast<std::string>();
            auto opt_ti = extract_token_info(ctx);
            py::dict vk;
            vk["value"] = val;
            for (auto item : opt_ti) vk[item.first] = item.second;
            operand_node = call_with_kwargs(py_VarID, vk);
        }
    }

    if (operand_node.is_none()) {
        py::kwargs kw;
        raise_semantic_error("1-3-2-4", kw);
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

static py::object visitFlowAtom(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::object operand = visitExpr(as_rule(children[2]));
    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["operand"] = operand;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_UnaryOp, kwargs);
}

static py::object visitCurrentDateAtom(antlr4::ParserRuleContext* ctx) {
    std::string op = terminal_text(ctx->children[0]);
    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["children"] = py::list();
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_MulOp, kwargs);
}

static py::object visitTimeDiffAtom(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::object left = visitExpr(as_rule(children[2]));
    py::object right = visitExpr(as_rule(children[4]));
    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["left"] = left;
    kwargs["op"] = op;
    kwargs["right"] = right;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_BinOp, kwargs);
}

static py::object visitTimeAddAtom(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::list children_node;
    children_node.append(visitExpr(as_rule(children[2])));

    py::list param_node;
    if (children.size() > 4) {
        param_node.append(visitExpr(as_rule(children[4])));
        if (children.size() > 6) {
            param_node.append(visitExpr(as_rule(children[6])));
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

// ---- Conditional Functions ----

py::object visitConditionalFunctions(antlr4::ParserRuleContext* ctx) {
    auto cid = get_ctx_id(ctx);
    if (cid.first == 32 && cid.second == 0) return visitNvlAtom(ctx);
    throw std::runtime_error("NotImplementedError: visitConditionalFunctions");
}

static py::object visitNvlAtom(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::object left = visitExpr(as_rule(children[2]));
    py::object right = visitExpr(as_rule(children[4]));
    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["left"] = left;
    kwargs["op"] = op;
    kwargs["right"] = right;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_BinOp, kwargs);
}

// ---- Set Functions ----

py::object visitSetFunctions(antlr4::ParserRuleContext* ctx) {
    auto cid = get_ctx_id(ctx);
    if (cid.first == 29 && cid.second == 0) return visitUnionAtom(ctx);
    if (cid.first == 29 && cid.second == 1) return visitIntersectAtom(ctx);
    if (cid.first == 29 && cid.second == 2) return visitSetOrSYmDiffAtom(ctx);
    throw std::runtime_error("NotImplementedError: visitSetFunctions");
}

static py::object visitUnionAtom(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::list exprs_nodes;
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (rule_child && (int)rule_child->getRuleIndex() == 2) {
            exprs_nodes.append(visitExpr(rule_child));
        }
    }
    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["children"] = exprs_nodes;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_MulOp, kwargs);
}

static py::object visitIntersectAtom(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::list exprs_nodes;
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (rule_child && (int)rule_child->getRuleIndex() == 2) {
            exprs_nodes.append(visitExpr(rule_child));
        }
    }
    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["children"] = exprs_nodes;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_MulOp, kwargs);
}

static py::object visitSetOrSYmDiffAtom(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::list exprs_nodes;
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (rule_child && (int)rule_child->getRuleIndex() == 2) {
            exprs_nodes.append(visitExpr(rule_child));
        }
    }
    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["children"] = exprs_nodes;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_MulOp, kwargs);
}

// ---- Hierarchy Functions ----

py::object visitHierarchyFunctions(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::object dataset_node = visitExpr(as_rule(children[2]));
    std::string ruleset_name = terminal_text(children[4]);

    py::list conditions;
    py::object validation_mode = py::none();
    py::object input_mode = py::none();
    py::object output = py::none();
    py::object rule_comp = py::none();

    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        int ri = (int)rule_child->getRuleIndex();
        if (ri == VtlParser::RuleConditionClause) {
            conditions.append(visitConditionClause(rule_child));
        } else if (ri == VtlParser::RuleComponentID) {
            rule_comp = visitComponentID(rule_child);
        } else if (ri == VtlParser::RuleValidationMode) {
            py::object mode_str = ASTBuilder::visitValidationMode(rule_child);
            validation_mode = py_ValidationMode_cls(mode_str);
        } else if (ri == VtlParser::RuleInputModeHierarchy) {
            py::object input_str = visitInputModeHierarchy(rule_child);
            if (input_str.equal(py_DATASET_PRIORITY)) {
                throw std::runtime_error(
                    "Dataset Priority input mode on HR is not implemented");
            }
            input_mode = py_HRInputMode_cls(input_str);
        } else if (ri == VtlParser::RuleOutputModeHierarchy) {
            py::object output_str = visitOutputModeHierarchy(rule_child);
            output = py_HierarchyOutput_cls(output_str);
        }
    }

    // conditions[0] if conditions else []
    py::object cond_val;
    if (py::len(conditions) > 0) {
        cond_val = conditions[py::int_(0)];
    } else {
        cond_val = py::list();
    }

    // Auto-detect rule_comp from de_ruleset_elements
    if (rule_comp.is_none() && py_de_ruleset_elements.contains(py::str(ruleset_name))) {
        py::object rule_element = py_de_ruleset_elements[py::str(ruleset_name)];
        if (py::isinstance<py::list>(rule_element)) {
            rule_element = rule_element[py::int_(-1)];
        }
        std::string kind = rule_element.attr("kind").cast<std::string>();
        if (kind == "DatasetID") {
            std::string check_val = rule_element.attr("value").cast<std::string>();
            auto ti_rc = extract_token_info(ctx);
            py::dict rc_kwargs;
            rc_kwargs["value"] = check_val;
            rc_kwargs["kind"] = "ComponentID";
            for (auto item : ti_rc) rc_kwargs[item.first] = item.second;
            rule_comp = call_with_kwargs(py_Identifier, rc_kwargs);
        } else {
            py::kwargs kw;
            kw["op"] = py::str(op);
            raise_semantic_error("1-1-10-4", kw);
        }
    }

    // Ensure conditions is a list
    py::object cond_list;
    if (py::isinstance<py::list>(cond_val)) {
        cond_list = cond_val;
    } else {
        py::list cl;
        cl.append(cond_val);
        cond_list = cl;
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["dataset"] = dataset_node;
    kwargs["ruleset_name"] = ruleset_name;
    kwargs["rule_component"] = rule_comp;
    kwargs["conditions"] = cond_list;
    kwargs["validation_mode"] = validation_mode;
    kwargs["input_mode"] = input_mode;
    kwargs["output"] = output;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_HROperation, kwargs);
}

// ---- Validation Functions ----

py::object visitValidationFunctions(antlr4::ParserRuleContext* ctx) {
    auto cid = get_ctx_id(ctx);
    if (cid.first == 31 && cid.second == 0) return visitValidateDPruleset(ctx);
    if (cid.first == 31 && cid.second == 1) return visitValidateHRruleset(ctx);
    if (cid.first == 31 && cid.second == 2) return visitValidationSimple(ctx);
    throw std::runtime_error("NotImplementedError: visitValidationFunctions");
}

py::object visitValidateDPruleset(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    py::object dataset_node = visitExpr(as_rule(children[2]));
    std::string ruleset_name = terminal_text(children[4]);

    py::list components;
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        if ((int)rule_child->getRuleIndex() == VtlParser::RuleComponentID) {
            components.append(visitComponentID(rule_child));
        }
    }

    py::list component_names;
    for (auto item : components) {
        py::object x = py::reinterpret_borrow<py::object>(item);
        if (py::isinstance(x, py_BinOp)) {
            component_names.append(x.attr("right").attr("value"));
        } else {
            component_names.append(x.attr("value"));
        }
    }

    py::object output_val = py::none();
    // Check second-to-last child for ValidationOutput
    size_t n = children.size();
    if (n >= 2) {
        auto* last2 = as_rule(children[n - 2]);
        if (last2 && (int)last2->getRuleIndex() == VtlParser::RuleValidationOutput) {
            py::object output_str = ASTBuilder::visitValidationOutput(last2);
            output_val = py_ValidationOutput_cls(output_str);
        }
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["dataset"] = dataset_node;
    kwargs["ruleset_name"] = ruleset_name;
    kwargs["components"] = component_names;
    kwargs["output"] = output_val;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_DPValidation, kwargs);
}

py::object visitValidateHRruleset(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::object dataset_node = visitExpr(as_rule(children[2]));
    std::string ruleset_name = terminal_text(children[4]);

    py::list conditions;
    py::object validation_mode = py::none();
    py::object input_mode = py::none();
    py::object output = py::none();
    py::object rule_comp = py::none();

    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        int ri = (int)rule_child->getRuleIndex();
        if (ri == VtlParser::RuleConditionClause) {
            conditions.append(visitConditionClause(rule_child));
        } else if (ri == VtlParser::RuleComponentID) {
            rule_comp = visitComponentID(rule_child);
        } else if (ri == VtlParser::RuleValidationMode) {
            py::object mode_str = ASTBuilder::visitValidationMode(rule_child);
            validation_mode = py_ValidationMode_cls(mode_str);
        } else if (ri == VtlParser::RuleInputMode) {
            py::object input_str = visitInputMode(rule_child);
            if (input_str.equal(py_DATASET_PRIORITY)) {
                throw std::runtime_error(
                    "Dataset Priority input mode on HR is not implemented");
            }
            input_mode = py_CHInputMode_cls(input_str);
        } else if (ri == VtlParser::RuleValidationOutput) {
            py::object output_str = ASTBuilder::visitValidationOutput(rule_child);
            output = py_ValidationOutput_cls(output_str);
        }
    }

    // conditions[0] if conditions else []
    py::object cond_val;
    if (py::len(conditions) > 0) {
        cond_val = conditions[py::int_(0)];
    } else {
        cond_val = py::list();
    }

    // Auto-detect rule_comp from de_ruleset_elements
    if (rule_comp.is_none() && py_de_ruleset_elements.contains(py::str(ruleset_name))) {
        py::object rule_element = py_de_ruleset_elements[py::str(ruleset_name)];
        if (py::isinstance<py::list>(rule_element)) {
            rule_element = rule_element[py::int_(-1)];
        }
        std::string kind = rule_element.attr("kind").cast<std::string>();
        if (kind == "DatasetID") {
            std::string check_val = rule_element.attr("value").cast<std::string>();
            auto ti_rc = extract_token_info(ctx);
            py::dict rc_kwargs;
            rc_kwargs["value"] = check_val;
            rc_kwargs["kind"] = "ComponentID";
            for (auto item : ti_rc) rc_kwargs[item.first] = item.second;
            rule_comp = call_with_kwargs(py_Identifier, rc_kwargs);
        } else {
            py::kwargs kw;
            kw["op"] = py::str(op);
            raise_semantic_error("1-1-10-4", kw);
        }
    }

    // Ensure conditions is a list
    py::object cond_list;
    if (py::isinstance<py::list>(cond_val)) {
        cond_list = cond_val;
    } else {
        py::list cl;
        cl.append(cond_val);
        cond_list = cl;
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["dataset"] = dataset_node;
    kwargs["ruleset_name"] = ruleset_name;
    kwargs["rule_component"] = rule_comp;
    kwargs["conditions"] = cond_list;
    kwargs["validation_mode"] = validation_mode;
    kwargs["input_mode"] = input_mode;
    kwargs["output"] = output;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_HROperation, kwargs);
}

py::object visitValidationSimple(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op_text = terminal_text(children[0]);

    py::object validation_node = visitExpr(as_rule(children[2]));

    py::object inbalance_node = py::none();
    py::object error_code = py::none();
    py::object error_level = py::none();

    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        int ri = (int)rule_child->getRuleIndex();
        if (ri == VtlParser::RuleErCode) {
            error_code = visitErCode(rule_child);
        } else if (ri == VtlParser::RuleErLevel) {
            error_level = visitErLevel(rule_child);
        } else if (ri == VtlParser::RuleImbalanceExpr) {
            inbalance_node = visitImbalanceExpr(rule_child);
        }
    }

    // invalid check: ctx_list[-2] terminal
    size_t n = children.size();
    bool invalid_value = false;
    if (n >= 2) {
        auto* last2 = as_terminal(children[n - 2]);
        if (last2) {
            std::string text = last2->getText();
            invalid_value = (text == "invalid");
        }
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op_text;
    kwargs["validation"] = validation_node;
    kwargs["error_code"] = error_code;
    kwargs["error_level"] = error_level;
    kwargs["imbalance"] = inbalance_node;
    kwargs["invalid"] = py::bool_(invalid_value);
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_Validation, kwargs);
}

static py::object visitImbalanceExpr(antlr4::ParserRuleContext* ctx) {
    return visitExpr(as_rule(ctx->children[1]));
}

// ---- Aggregate Functions ----

py::object visitAggregateFunctions(antlr4::ParserRuleContext* ctx) {
    auto cid = get_ctx_id(ctx);
    if (cid.first == 35 && cid.second == 0) return visitAggrDataset(ctx);
    throw std::runtime_error("NotImplementedError: visitAggregateFunctions");
}

static py::object visitAggrDataset(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    py::object grouping_op = py::none();
    py::object group_node = py::none();
    py::object have_node = py::none();

    // GroupingClause has rule_index 56
    std::vector<antlr4::ParserRuleContext*> groups;
    std::vector<antlr4::ParserRuleContext*> haves;
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        int ri = (int)rule_child->getRuleIndex();
        // GroupingClause rule_index = 56
        if (ri == 56) groups.push_back(rule_child);
        // HavingClause rule_index = 57
        if (ri == VtlParser::RuleHavingClause) haves.push_back(rule_child);
    }

    std::string op_node = terminal_text(children[0]);
    py::object operand = visitExpr(as_rule(children[2]));

    if (!groups.empty()) {
        auto [gop, gn] = visitGroupingClause(groups[0]);
        grouping_op = gop;
        group_node = gn;
    }
    if (!haves.empty()) {
        auto [hn, expr] = visitHavingClause(haves[0]);
        hn.attr("expr") = expr;
        have_node = hn;
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op_node;
    kwargs["operand"] = operand;
    kwargs["grouping_op"] = grouping_op;
    kwargs["grouping"] = group_node;
    kwargs["having_clause"] = have_node;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_Aggregation, kwargs);
}

// ---- Analytic Functions ----

py::object visitAnalyticFunctions(antlr4::ParserRuleContext* ctx) {
    auto cid = get_ctx_id(ctx);
    if (cid.first == 36 && cid.second == 0) return visitAnSimpleFunction(ctx);
    if (cid.first == 36 && cid.second == 1) return visitLagOrLeadAn(ctx);
    if (cid.first == 36 && cid.second == 2) return visitRatioToReportAn(ctx);
    throw std::runtime_error("NotImplementedError: visitAnalyticFunctions");
}

static py::object visitAnSimpleFunction(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    py::object window = py::none();
    py::object partition_by = py::none();
    py::object order_by = py::none();

    std::string op_node = terminal_text(children[0]);
    py::object operand = visitExpr(as_rule(children[2]));

    // children[5:-2]
    size_t start = 5;
    size_t end = children.size() >= 2 ? children.size() - 2 : start;
    for (size_t i = start; i < end; i++) {
        auto* rule_child = as_rule(children[i]);
        if (!rule_child) {
            throw std::runtime_error("NotImplementedError: visitAnSimpleFunction non-rule");
        }
        int ri = (int)rule_child->getRuleIndex();
        if (ri == VtlParser::RulePartitionByClause) {
            partition_by = visitPartitionByClause(rule_child);
        } else if (ri == VtlParser::RuleOrderByClause) {
            order_by = visitOrderByClause(rule_child);
        } else if (ri == VtlParser::RuleWindowingClause) {
            window = visitWindowingClause(rule_child);
        } else {
            throw std::runtime_error("NotImplementedError: visitAnSimpleFunction unknown");
        }
    }

    if (window.is_none()) {
        // Default windowing: raw ints, matching Python behavior
        // (create_windowing normalizes -1/"unbounded", but the default uses raw ints)
        auto ti_w = extract_token_info(ctx);
        py::dict wkw;
        wkw["type_"] = py::str("data");
        wkw["start"] = py::int_(-1);
        wkw["stop"] = py::int_(0);
        wkw["start_mode"] = py::str("preceding");
        wkw["stop_mode"] = py::str("current");
        for (auto item : ti_w) wkw[item.first] = item.second;
        window = call_with_kwargs(py_Windowing, wkw);
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op_node;
    kwargs["operand"] = operand;
    kwargs["partition_by"] = partition_by;
    kwargs["order_by"] = order_by;
    kwargs["window"] = window;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_Analytic, kwargs);
}

static py::object visitLagOrLeadAn(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    py::object params = py::none();
    py::object partition_by = py::none();
    py::object order_by = py::none();

    std::string op_node = terminal_text(children[0]);
    py::object operand = visitExpr(as_rule(children[2]));

    // children[4:-2]
    size_t start = 4;
    size_t end = children.size() >= 2 ? children.size() - 2 : start;
    for (size_t i = start; i < end; i++) {
        auto* term = as_terminal(children[i]);
        if (term) continue;
        auto* rule_child = as_rule(children[i]);
        if (!rule_child) continue;
        int ri = (int)rule_child->getRuleIndex();
        auto child_cid = get_ctx_id(rule_child);
        if (ri == VtlParser::RulePartitionByClause) {
            partition_by = visitPartitionByClause(rule_child);
        } else if (ri == VtlParser::RuleOrderByClause) {
            order_by = visitOrderByClause(rule_child);
        } else if (ri == VtlParser::RuleSignedInteger ||
                   (child_cid.first == VtlParser::RuleScalarItem &&
                    (child_cid.second == 0 || child_cid.second == 1))) {
            if (params.is_none()) {
                params = py::list();
            }
            if (ri == VtlParser::RuleSignedInteger) {
                params.cast<py::list>().append(visitSignedInteger(rule_child));
            } else {
                params.cast<py::list>().append(visitScalarItem(rule_child));
            }
        }
    }

    // Check params
    if (params.is_none() || py::len(params) == 0) {
        throw std::runtime_error(op_node + " requires an offset parameter.");
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op_node;
    kwargs["operand"] = operand;
    kwargs["partition_by"] = partition_by;
    kwargs["order_by"] = order_by;
    kwargs["params"] = params;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_Analytic, kwargs);
}

static py::object visitRatioToReportAn(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op_node = terminal_text(children[0]);
    py::object operand = visitExpr(as_rule(children[2]));
    py::object partition_by = visitPartitionByClause(as_rule(children[5]));

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op_node;
    kwargs["operand"] = operand;
    kwargs["partition_by"] = partition_by;
    kwargs["order_by"] = py::none();
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_Analytic, kwargs);
}

// ---- Dataset Clause ----

py::object visitDatasetClause(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    auto* c = as_rule(children[0]);
    if (!c) throw std::runtime_error("visitDatasetClause: expected rule child");

    int ri = (int)c->getRuleIndex();
    if (ri == VtlParser::RuleRenameClause) return visitRenameClause(c);
    if (ri == VtlParser::RuleAggrClause) return visitAggrClause(c);
    if (ri == VtlParser::RuleFilterClause) return visitFilterClause(c);
    if (ri == VtlParser::RuleCalcClause) return visitCalcClause(c);
    if (ri == VtlParser::RuleKeepOrDropClause) return visitKeepOrDropClause(c);
    if (ri == VtlParser::RulePivotOrUnpivotClause) return visitPivotOrUnpivotClause(c);
    if (ri == VtlParser::RuleSubspaceClause) return visitSubspaceClause(c);

    throw std::runtime_error("NotImplementedError: visitDatasetClause unknown clause");
}

// ---- Rename Clause ----

py::object visitRenameClause(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    py::list rename_nodes;

    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        if ((int)rule_child->getRuleIndex() == VtlParser::RuleRenameClauseItem) {
            rename_nodes.append(visitRenameClauseItem(rule_child));
        }
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = terminal_text(children[0]);
    kwargs["children"] = rename_nodes;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_RegularAggregation, kwargs);
}

static py::object visitRenameClauseItem(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    py::object left_node = visitComponentID(as_rule(children[0]));
    // Check if left_node is a BinOp
    py::object left_val;
    if (py::isinstance(left_node, py_BinOp)) {
        std::string l = left_node.attr("left").attr("value").cast<std::string>();
        std::string o = left_node.attr("op").cast<std::string>();
        std::string r = left_node.attr("right").attr("value").cast<std::string>();
        left_val = py::str(l + o + r);
    } else {
        left_val = left_node.attr("value");
    }

    py::object right_node = visitVarID(as_rule(children[2]));
    py::object right_val = right_node.attr("value");

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["old_name"] = left_val;
    kwargs["new_name"] = right_val;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_RenameNode, kwargs);
}

// ---- Aggregate Clause ----

static py::object visitAggregateClause(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    py::list aggregates_nodes;

    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        if ((int)rule_child->getRuleIndex() == VtlParser::RuleAggrFunctionClause) {
            aggregates_nodes.append(visitAggrFunctionClause(rule_child));
        }
    }

    return aggregates_nodes;
}

static py::object visitAggrFunctionClause(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    auto* c = children[0];
    auto* c_rule = as_rule(c);

    int base_index = 0;
    py::object role;
    if (c_rule && (int)c_rule->getRuleIndex() == VtlParser::RuleComponentRole) {
        role = visitComponentRole(c_rule);
        base_index = 1;
    } else {
        base_index = 0;
        role = py_Role(py::str("Measure"));
    }

    py::object left_node = visitSimpleComponentId(as_rule(children[base_index]));
    std::string op_node = ":=";
    py::object right_node = visitAggregateFunctionsComponents(
        as_rule(children[base_index + 2]));

    // Set role on left_node
    left_node.attr("role") = role;

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["left"] = left_node;
    kwargs["op"] = op_node;
    kwargs["right"] = right_node;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_Assignment, kwargs);
}

py::object visitAggrClause(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op_node = terminal_text(children[0]);

    py::object group_node = py::none();
    py::object grouping_op = py::none();
    py::object have_node = py::none();

    std::vector<antlr4::ParserRuleContext*> groups;
    std::vector<antlr4::ParserRuleContext*> haves;
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        int ri = (int)rule_child->getRuleIndex();
        if (ri == 56) groups.push_back(rule_child); // GroupingClause
        if (ri == VtlParser::RuleHavingClause) haves.push_back(rule_child);
    }

    py::object aggregate_nodes = visitAggregateClause(as_rule(children[1]));

    if (!groups.empty()) {
        auto [gop, gn] = visitGroupingClause(groups[0]);
        grouping_op = gop;
        group_node = gn;
    }
    if (!haves.empty()) {
        auto [hn, expr] = visitHavingClause(haves[0]);
        hn.attr("expr") = expr;
        have_node = hn;
    }

    py::list result_children;
    auto ti_agg = extract_token_info(as_rule(children[1]));
    auto copy_mod = py::module_::import("copy");

    for (auto element : aggregate_nodes) {
        py::object elem = py::reinterpret_borrow<py::object>(element);
        // Rebuild right as Aggregation with grouping info
        py::object right = elem.attr("right");
        py::dict agg_kwargs;
        agg_kwargs["op"] = right.attr("op");
        agg_kwargs["operand"] = right.attr("operand");
        agg_kwargs["grouping_op"] = grouping_op;
        agg_kwargs["grouping"] = group_node;
        agg_kwargs["having_clause"] = have_node;
        for (auto item : ti_agg) agg_kwargs[item.first] = item.second;
        elem.attr("right") = call_with_kwargs(py_Aggregation, agg_kwargs);
        result_children.append(copy_mod.attr("copy")(elem));
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op_node;
    kwargs["children"] = result_children;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_RegularAggregation, kwargs);
}

// ---- Grouping Clause ----

std::pair<py::object, py::object> visitGroupingClause(antlr4::ParserRuleContext* ctx) {
    auto cid = get_ctx_id(ctx);
    // GROUP_BY_OR_EXCEPT = (56, 0)
    if (cid.first == 56 && cid.second == 0) return visitGroupByOrExcept(ctx);
    // GROUP_ALL = (56, 1)
    if (cid.first == 56 && cid.second == 1) return visitGroupAll(ctx);
    throw std::runtime_error("NotImplementedError: visitGroupingClause");
}

std::pair<py::object, py::object> visitGroupByOrExcept(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    std::string token_left = terminal_text(children[0]);
    std::string token_right = terminal_text(children[1]);
    std::string op_node = token_left + " " + token_right;

    py::list children_nodes;
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        if ((int)rule_child->getRuleIndex() == VtlParser::RuleComponentID) {
            children_nodes.append(visitComponentID(rule_child));
        }
    }

    return {py::str(op_node), children_nodes};
}

std::pair<py::object, py::object> visitGroupAll(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    std::string token_left = terminal_text(children[0]);
    std::string token_right = terminal_text(children[1]);
    std::string op_node = token_left + " " + token_right;

    py::list children_nodes;

    // Check if TIME_AGG is present (more than just GROUP ALL)
    if (children.size() > 2) {
        py::object period_to = py::none();
        py::object period_from = py::none();
        py::object operand_node = py::none();
        py::object conf = py::none();

        for (auto* child : children) {
            auto* term = as_terminal(child);
            if (term) {
                int sym = (int)term->getSymbol()->getType();
                if (sym == VtlParser::STRING_CONSTANT) {
                    std::string raw = term->getText();
                    std::string val = raw.substr(1, raw.size() - 2);
                    if (period_to.is_none()) {
                        period_to = py::str(val);
                    } else {
                        period_from = py::str(val);
                    }
                } else if (sym == VtlParser::FIRST || sym == VtlParser::LAST) {
                    conf = py::str(term->getText());
                }
                continue;
            }
            auto* rule_child = as_rule(child);
            if (rule_child && (int)rule_child->getRuleIndex() == VtlParser::RuleOptionalExpr) {
                operand_node = visitOptionalExpr(rule_child);
                if (py::isinstance(operand_node, py_ID)) {
                    operand_node = py::none();
                } else if (py::isinstance(operand_node, py_Identifier)) {
                    std::string val = operand_node.attr("value").cast<std::string>();
                    auto opt_ti = extract_token_info(rule_child);
                    py::dict vk;
                    vk["value"] = val;
                    for (auto item : opt_ti) vk[item.first] = item.second;
                    operand_node = call_with_kwargs(py_VarID, vk);
                }
            }
        }

        // Build TimeAggregation node
        auto ti_g = extract_token_info(ctx);
        py::dict ta_kwargs;
        ta_kwargs["op"] = "time_agg";
        ta_kwargs["operand"] = operand_node;
        ta_kwargs["period_to"] = period_to;
        ta_kwargs["period_from"] = period_from;
        ta_kwargs["conf"] = conf;
        for (auto item : ti_g) ta_kwargs[item.first] = item.second;
        children_nodes.append(call_with_kwargs(py_TimeAggregation, ta_kwargs));
    }

    return {py::str(op_node), children_nodes};
}

// ---- Having Clause ----

std::pair<py::object, py::str> visitHavingClause(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op_node = terminal_text(children[0]);

    // Get the input text and extract having clause text
    auto vtl_mod = py::module_::import("vtlengine.AST.Grammar._cpp_parser");
    py::object vtl_cpp = vtl_mod.attr("vtl_cpp_parser");
    std::string strdata = vtl_cpp.attr("get_input_text")().cast<std::string>();

    // Get start_line of having clause
    int start_line = ctx->start ? (int)ctx->start->getLine() : 1;

    // Split strdata by newlines
    std::vector<std::string> lines;
    std::istringstream stream(strdata);
    std::string line;
    while (std::getline(stream, line)) {
        lines.push_back(line);
    }

    // Build text from start_line
    std::string text_from_having;
    if (start_line - 1 < (int)lines.size()) {
        std::string line_text = lines[start_line - 1];
        // Find 'having' in that line (case-insensitive)
        std::string lower_line = line_text;
        std::transform(lower_line.begin(), lower_line.end(), lower_line.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        size_t having_pos = lower_line.find("having");
        if (having_pos != std::string::npos) {
            text_from_having = line_text.substr(having_pos);
            for (size_t i = start_line; i < lines.size(); i++) {
                text_from_having += "\n" + lines[i];
            }
        } else {
            for (size_t i = start_line - 1; i < lines.size(); i++) {
                if (i > (size_t)(start_line - 1)) text_from_having += "\n";
                text_from_having += lines[i];
            }
        }
    }

    // Split by "having" and take second part
    auto re_mod = py::module_::import("re");
    py::list parts = re_mod.attr("split")(py::str("having"), py::str(text_from_having));
    std::string expr;
    if (py::len(parts) > 1) {
        expr = parts[py::int_(1)].cast<std::string>();
    } else {
        expr = text_from_having;
    }

    // Trim last 2 chars and strip, then prepend "having "
    if (expr.size() >= 2) {
        expr = expr.substr(0, expr.size() - 2);
    }
    // Strip whitespace
    size_t s = expr.find_first_not_of(" \t\n\r");
    size_t e = expr.find_last_not_of(" \t\n\r");
    if (s != std::string::npos) {
        expr = expr.substr(s, e - s + 1);
    }
    expr = "having " + expr;

    // Handle "]"
    size_t bracket_pos = expr.find(']');
    if (bracket_pos != std::string::npos) {
        expr = expr.substr(0, bracket_pos);
    }
    // Handle "end"
    size_t end_pos = expr.find("end");
    if (end_pos != std::string::npos) {
        expr = expr.substr(0, end_pos);
    }
    // Handle unbalanced parentheses
    {
        int open_count = 0, close_count = 0;
        for (char c : expr) {
            if (c == '(') open_count++;
            if (c == ')') close_count++;
        }
        if (close_count > open_count) {
            size_t last_paren = expr.rfind(')');
            if (last_paren != std::string::npos) {
                expr = expr.substr(0, last_paren);
            }
        }
    }

    // Replace { and } with ( and )
    for (size_t i = 0; i < expr.size(); i++) {
        if (expr[i] == '{') expr[i] = '(';
        if (expr[i] == '}') expr[i] = ')';
    }
    // Replace "not_in" with "not in"
    {
        size_t pos = 0;
        while ((pos = expr.find("not_in", pos)) != std::string::npos) {
            expr.replace(pos, 6, "not in");
            pos += 6;
        }
    }
    // Replace '"' with "'"
    for (size_t i = 0; i < expr.size(); i++) {
        if (expr[i] == '"') expr[i] = '\'';
    }

    // Visit the exprComponent child
    auto* expr_component = as_rule(children[1]);
    py::object param_nodes = visitExprComponent(expr_component);

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op_node;
    kwargs["children"] = py::none();
    kwargs["params"] = param_nodes;
    for (auto item : ti) kwargs[item.first] = item.second;
    py::object result = call_with_kwargs(py_ParamOp, kwargs);

    return {result, py::str(expr)};
}

// ---- Filter Clause ----

py::object visitFilterClause(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);
    py::list operand_nodes;
    operand_nodes.append(visitExprComponent(as_rule(children[1])));

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["children"] = operand_nodes;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_RegularAggregation, kwargs);
}

// ---- Calc Clause ----

py::object visitCalcClause(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);

    py::list calc_items;
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        if ((int)rule_child->getRuleIndex() == VtlParser::RuleCalcClauseItem) {
            calc_items.append(visitCalcClauseItem(rule_child));
        }
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["children"] = calc_items;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_RegularAggregation, kwargs);
}

static py::object visitCalcClauseItem(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    auto* c = children[0];
    auto* c_rule = as_rule(c);

    if (c_rule && (int)c_rule->getRuleIndex() == VtlParser::RuleComponentRole) {
        py::object role = visitComponentRole(c_rule);

        py::object left_node = visitComponentID(as_rule(children[1]));
        std::string op_node = ":=";
        py::object right_node = visitExprComponent(as_rule(children[3]));

        auto ti = extract_token_info(ctx);
        py::dict assign_kwargs;
        assign_kwargs["left"] = left_node;
        assign_kwargs["op"] = op_node;
        assign_kwargs["right"] = right_node;
        for (auto item : ti) assign_kwargs[item.first] = item.second;
        py::object operand_node = call_with_kwargs(py_Assignment, assign_kwargs);

        auto ti_c = extract_token_info(c_rule);
        std::string role_str;
        if (role.is_none()) {
            role_str = "measure";
        } else {
            role_str = role.attr("value").cast<std::string>();
            // lowercase
            std::transform(role_str.begin(), role_str.end(), role_str.begin(),
                           [](unsigned char ch) { return std::tolower(ch); });
        }

        py::dict unary_kwargs;
        unary_kwargs["op"] = role_str;
        unary_kwargs["operand"] = operand_node;
        for (auto item : ti_c) unary_kwargs[item.first] = item.second;
        return call_with_kwargs(py_UnaryOp, unary_kwargs);
    } else {
        py::object left_node = visitSimpleComponentId(as_rule(c));
        std::string op_node = ":=";
        py::object right_node = visitExprComponent(as_rule(children[2]));

        auto ti = extract_token_info(ctx);
        py::dict assign_kwargs;
        assign_kwargs["left"] = left_node;
        assign_kwargs["op"] = op_node;
        assign_kwargs["right"] = right_node;
        for (auto item : ti) assign_kwargs[item.first] = item.second;
        py::object operand_node = call_with_kwargs(py_Assignment, assign_kwargs);

        py::dict unary_kwargs;
        unary_kwargs["op"] = "measure";
        unary_kwargs["operand"] = operand_node;
        for (auto item : ti) unary_kwargs[item.first] = item.second;
        return call_with_kwargs(py_UnaryOp, unary_kwargs);
    }
}

// ---- Keep or Drop Clause ----

py::object visitKeepOrDropClause(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);

    py::list nodes;
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        if ((int)rule_child->getRuleIndex() == VtlParser::RuleComponentID) {
            nodes.append(visitComponentID(rule_child));
        }
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["children"] = nodes;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_RegularAggregation, kwargs);
}

// ---- Pivot/Unpivot Clause ----

py::object visitPivotOrUnpivotClause(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);

    py::list children_nodes;
    children_nodes.append(visitComponentID(as_rule(children[1])));
    children_nodes.append(visitComponentID(as_rule(children[3])));

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["children"] = children_nodes;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_RegularAggregation, kwargs);
}

// ---- Subspace Clause ----

py::object visitSubspaceClause(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;
    std::string op = terminal_text(children[0]);

    py::list subspace_nodes;
    for (auto* child : children) {
        auto* rule_child = as_rule(child);
        if (!rule_child) continue;
        if ((int)rule_child->getRuleIndex() == VtlParser::RuleSubspaceClauseItem) {
            subspace_nodes.append(visitSubspaceClauseItem(rule_child));
        }
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = op;
    kwargs["children"] = subspace_nodes;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_RegularAggregation, kwargs);
}

static py::object visitSubspaceClauseItem(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    py::object left_node = visitVarID(as_rule(children[0]));
    std::string op_node = terminal_text(children[1]);

    py::object right_node;
    auto* child2 = as_rule(children[2]);
    if (child2) {
        auto child2_cid = get_ctx_id(child2);
        // SCALAR_WITH_CAST = (43, 1)
        if (child2_cid.first == VtlParser::RuleScalarItem && child2_cid.second == 1) {
            right_node = visitScalarWithCast(child2);
        } else if (child2_cid.first == VtlParser::RuleScalarItem &&
                   (child2_cid.second == 0 || child2_cid.second == 1)) {
            right_node = visitScalarItem(child2);
        } else {
            right_node = visitVarID(child2);
        }
    } else {
        right_node = visitVarID(as_rule(children[2]));
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["left"] = left_node;
    kwargs["op"] = op_node;
    kwargs["right"] = right_node;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_BinOp, kwargs);
}

// ============================================================
// Phase 4: ASTConstructor top-level visitor methods
// ============================================================

py::object visitStart(antlr4::ParserRuleContext* ctx) {
    if (!g_initialized) init();
    auto& children = ctx->children;

    py::list statements_nodes;
    // Collect statement children (rule_index == RuleStatement)
    std::vector<antlr4::ParserRuleContext*> statements;
    for (auto& child : children) {
        auto* rule_child = dynamic_cast<antlr4::ParserRuleContext*>(child);
        if (rule_child) {
            auto cid = get_ctx_id(rule_child);
            // Statement rule_index == 1
            if (cid.first == VtlParser::RuleStatement) {
                statements.push_back(rule_child);
            }
        }
    }

    for (auto* stmt : statements) {
        statements_nodes.append(visitStatement(stmt));
    }

    auto ti = extract_token_info(ctx);
    // For the Start node, use the last statement's stop position instead of EOF
    if (!statements.empty()) {
        auto last_ti = extract_token_info(statements.back());
        ti["line_stop"] = last_ti["line_stop"];
        ti["column_stop"] = last_ti["column_stop"];
    }

    py::dict kwargs;
    kwargs["children"] = statements_nodes;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_Start, kwargs);
}

py::object visitStatement(antlr4::ParserRuleContext* ctx) {
    auto cid = get_ctx_id(ctx);
    auto& children = ctx->children;
    auto* c = dynamic_cast<antlr4::ParserRuleContext*>(children[0]);

    // TEMPORARY_ASSIGNMENT = (1, 0)
    if (cid.first == VtlParser::RuleStatement && cid.second == 0) {
        return visitTemporaryAssignment(ctx);
    }
    // PERSIST_ASSIGNMENT = (1, 1)
    if (cid.first == VtlParser::RuleStatement && cid.second == 1) {
        return visitPersistAssignment(ctx);
    }
    // DEFINE_EXPRESSION = (1, 2)
    if (cid.first == VtlParser::RuleStatement && cid.second == 2) {
        return visitDefineExpression(c);
    }
    throw std::runtime_error("NotImplementedError: visitStatement");
}

py::object visitTemporaryAssignment(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    py::object left_node = visitVarID(as_rule(children[0]));
    auto* terminal = dynamic_cast<antlr4::tree::TerminalNode*>(children[1]);
    std::string op_node = terminal->getText();
    py::object right_node = visitExpr(as_rule(children[2]));

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["left"] = left_node;
    kwargs["op"] = py::str(op_node);
    kwargs["right"] = right_node;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_Assignment, kwargs);
}

py::object visitPersistAssignment(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    py::object left_node = visitVarID(as_rule(children[0]));
    auto* terminal = dynamic_cast<antlr4::tree::TerminalNode*>(children[1]);
    std::string op_node = terminal->getText();
    py::object right_node = visitExpr(as_rule(children[2]));

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["left"] = left_node;
    kwargs["op"] = py::str(op_node);
    kwargs["right"] = right_node;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_PersistentAssignment, kwargs);
}

py::object visitDefineExpression(antlr4::ParserRuleContext* ctx) {
    auto cid = get_ctx_id(ctx);

    // DEF_OPERATOR = (16, 0)
    if (cid.first == VtlParser::RuleDefOperators && cid.second == 0) {
        return visitDefOperator(ctx);
    }
    // DEF_DATAPOINT_RULESET = (16, 1)
    if (cid.first == VtlParser::RuleDefOperators && cid.second == 1) {
        return visitDefDatapointRuleset(ctx);
    }
    // DEF_HIERARCHICAL = (16, 2)
    if (cid.first == VtlParser::RuleDefOperators && cid.second == 2) {
        return visitDefHierarchical(ctx);
    }
    return py::none();
}

py::object visitDefOperator(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    // children[2] is operatorID
    py::object operator_name = visitOperatorID(as_rule(children[2]));

    // Collect parameterItems (rule_index == RuleParameterItem == 58)
    py::list parameters;
    for (auto& child : children) {
        auto* rule_child = dynamic_cast<antlr4::ParserRuleContext*>(child);
        if (rule_child) {
            auto cid = get_ctx_id(rule_child);
            if (cid.first == VtlParser::RuleParameterItem) {
                parameters.append(visitParameterItem(rule_child));
            }
        }
    }

    // Collect outputParameterType (rule_index == RuleOutputParameterType == 59)
    py::list return_list;
    for (auto& child : children) {
        auto* rule_child = dynamic_cast<antlr4::ParserRuleContext*>(child);
        if (rule_child) {
            auto cid = get_ctx_id(rule_child);
            if (cid.first == VtlParser::RuleOutputParameterType) {
                return_list.append(visitOutputParameterType(rule_child));
            }
        }
    }

    // Find expr (rule_index == RuleExpr == 2)
    py::object expr_node = py::none();
    for (auto& child : children) {
        auto* rule_child = dynamic_cast<antlr4::ParserRuleContext*>(child);
        if (rule_child) {
            auto cid = get_ctx_id(rule_child);
            if (cid.first == VtlParser::RuleExpr) {
                expr_node = visitExpr(rule_child);
                break;
            }
        }
    }

    if (py::len(return_list) == 0) {
        py::kwargs kw;
        kw["op"] = operator_name;
        raise_semantic_error("1-3-2-2", kw);
    }
    py::object return_node = return_list[py::int_(0)];

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["op"] = operator_name;
    kwargs["parameters"] = parameters;
    kwargs["output_type"] = return_node;
    kwargs["expression"] = expr_node;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_Operator, kwargs);
}

py::object visitDefDatapointRuleset(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    py::object ruleset_name = visitRulesetID(as_rule(children[3]));
    auto [signature_type, ruleset_elements] = visitRulesetSignature(as_rule(children[5]));
    py::list ruleset_rules = visitRuleClauseDatapoint(as_rule(children[8]));

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["name"] = ruleset_name;
    kwargs["params"] = ruleset_elements;
    kwargs["rules"] = ruleset_rules;
    kwargs["signature_type"] = py::str(signature_type);
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_DPRuleset, kwargs);
}

std::pair<std::string, py::list> visitRulesetSignature(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    // First terminal is signature type ("valuedomain" or "variable")
    auto* first_term = dynamic_cast<antlr4::tree::TerminalNode*>(children[0]);
    std::string signature_type = first_term->getText();

    // Check for VALUE_DOMAIN or VARIABLE terminals to determine kind
    std::string kind;
    for (auto& child : children) {
        auto* term = dynamic_cast<antlr4::tree::TerminalNode*>(child);
        if (term) {
            int sym = static_cast<int>(term->getSymbol()->getType());
            if (sym == VtlParser::VALUE_DOMAIN) {
                kind = "ValuedomainID";
                break;
            }
            if (sym == VtlParser::VARIABLE) {
                kind = "ComponentID";
                break;
            }
        }
    }

    // Collect signature children (rule_index == RuleSignature == 73)
    py::list component_nodes;
    for (auto& child : children) {
        auto* rule_child = dynamic_cast<antlr4::ParserRuleContext*>(child);
        if (rule_child) {
            auto cid = get_ctx_id(rule_child);
            if (cid.first == VtlParser::RuleSignature) {
                component_nodes.append(visitSignature(rule_child, kind));
            }
        }
    }

    return {signature_type, component_nodes};
}

py::list visitRuleClauseDatapoint(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    py::list ruleset_rules;
    for (auto& child : children) {
        auto* rule_child = dynamic_cast<antlr4::ParserRuleContext*>(child);
        if (rule_child) {
            auto cid = get_ctx_id(rule_child);
            // RuleRuleItemDatapoint == 75
            if (cid.first == VtlParser::RuleRuleItemDatapoint) {
                ruleset_rules.append(visitRuleItemDatapoint(rule_child));
            }
        }
    }
    return ruleset_rules;
}

py::object visitRuleItemDatapoint(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    // Check for WHEN terminal
    std::vector<antlr4::tree::TerminalNode*> when_terminals;
    for (auto& child : children) {
        auto* term = dynamic_cast<antlr4::tree::TerminalNode*>(child);
        if (term && static_cast<int>(term->getSymbol()->getType()) == VtlParser::WHEN) {
            when_terminals.push_back(term);
        }
    }

    // First child may be IDENTIFIER (rule_name) or None
    py::object rule_name = py::none();
    auto* first_term = dynamic_cast<antlr4::tree::TerminalNode*>(children[0]);
    if (first_term && static_cast<int>(first_term->getSymbol()->getType()) == VtlParser::IDENTIFIER) {
        rule_name = py::str(first_term->getText());
    }

    // Collect exprComponent children (rule_index == RuleExprComponent == 3)
    py::list expr_nodes;
    for (auto& child : children) {
        auto* rule_child = dynamic_cast<antlr4::ParserRuleContext*>(child);
        if (rule_child) {
            auto cid = get_ctx_id(rule_child);
            if (cid.first == VtlParser::RuleExprComponent) {
                expr_nodes.append(visitExprComponent(rule_child));
            }
        }
    }

    py::object rule_node;
    if (!when_terminals.empty()) {
        auto when_ti = extract_token_info_terminal(when_terminals[0]);
        py::dict kwargs;
        kwargs["left"] = expr_nodes[py::int_(0)];
        kwargs["op"] = py::str(when_terminals[0]->getText());
        kwargs["right"] = expr_nodes[py::int_(1)];
        for (auto item : when_ti) kwargs[item.first] = item.second;
        rule_node = call_with_kwargs(py_HRBinOp, kwargs);
    } else {
        rule_node = expr_nodes[py::int_(0)];
    }

    // Collect erCode (rule_index == RuleErCode == 98)
    py::object er_code = py::none();
    for (auto& child : children) {
        auto* rule_child = dynamic_cast<antlr4::ParserRuleContext*>(child);
        if (rule_child) {
            auto cid = get_ctx_id(rule_child);
            if (cid.first == VtlParser::RuleErCode) {
                er_code = visitErCode(rule_child);
                break;
            }
        }
    }

    // Collect erLevel (rule_index == RuleErLevel == 99)
    py::object er_level = py::none();
    for (auto& child : children) {
        auto* rule_child = dynamic_cast<antlr4::ParserRuleContext*>(child);
        if (rule_child) {
            auto cid = get_ctx_id(rule_child);
            if (cid.first == VtlParser::RuleErLevel) {
                er_level = visitErLevel(rule_child);
                break;
            }
        }
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["name"] = rule_name;
    kwargs["rule"] = rule_node;
    kwargs["erCode"] = er_code;
    kwargs["erLevel"] = er_level;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_DPRule, kwargs);
}

py::object visitParameterItem(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    // Find varID (rule_index == RuleVarID == 94)
    py::object argument_name = py::none();
    for (auto& child : children) {
        auto* rule_child = dynamic_cast<antlr4::ParserRuleContext*>(child);
        if (rule_child) {
            auto cid = get_ctx_id(rule_child);
            if (cid.first == VtlParser::RuleVarID) {
                argument_name = visitVarID(rule_child);
                break;
            }
        }
    }

    // Find inputParameterType (rule_index == RuleInputParameterType == 61)
    py::object argument_type = py::none();
    for (auto& child : children) {
        auto* rule_child = dynamic_cast<antlr4::ParserRuleContext*>(child);
        if (rule_child) {
            auto cid = get_ctx_id(rule_child);
            if (cid.first == VtlParser::RuleInputParameterType) {
                argument_type = visitInputParameterType(rule_child);
                break;
            }
        }
    }

    // Find optional constant (rule_index == RuleScalarItem == 43)
    py::object argument_default = py::none();
    for (auto& child : children) {
        auto* rule_child = dynamic_cast<antlr4::ParserRuleContext*>(child);
        if (rule_child) {
            auto cid = get_ctx_id(rule_child);
            if (cid.first == VtlParser::RuleScalarItem) {
                argument_default = visitScalarItem(rule_child);
                break;
            }
        }
    }

    // isinstance check for Dataset, Component, Scalar
    if (py::isinstance(argument_type, py_Dataset) ||
        py::isinstance(argument_type, py_Component) ||
        py::isinstance(argument_type, py_Scalar)) {
        argument_type.attr("name") = argument_name.attr("value");
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["name"] = argument_name.attr("value");
    kwargs["type_"] = argument_type;
    kwargs["default"] = argument_default;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_Argument, kwargs);
}

py::object visitDefHierarchical(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    py::object ruleset_name = visitRulesetID(as_rule(children[3]));
    auto [signature_type, ruleset_elements] = visitHierRuleSignature(as_rule(children[5]));

    if (signature_type == "variable" && py::isinstance<py::list>(ruleset_elements)) {
        py::list elems = ruleset_elements.cast<py::list>();
        // Build unique_id_names set
        std::set<std::string> unique_ids;
        for (auto& e : elems) {
            unique_ids.insert(e.attr("value").cast<std::string>());
        }
        if (py::len(elems) > 2 || unique_ids.size() < 1) {
            py::kwargs kw;
            kw["ruleset"] = ruleset_name;
            raise_semantic_error("1-1-10-9", kw);
        }
    }

    py::list ruleset_rules = visitRuleClauseHierarchical(as_rule(children[8]));

    // Keep k,v for hierarchical rulesets
    auto de_mod = py::module_::import("vtlengine.AST.ASTDataExchange");
    py::dict de_dict = de_mod.attr("de_ruleset_elements");
    de_dict[ruleset_name] = ruleset_elements;

    if (py::len(ruleset_rules) == 0) {
        std::string name_str = py::str(ruleset_name).cast<std::string>();
        throw std::runtime_error("No rules found for the ruleset " + name_str);
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["signature_type"] = py::str(signature_type);
    kwargs["name"] = ruleset_name;
    kwargs["element"] = ruleset_elements;
    kwargs["rules"] = ruleset_rules;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_HRuleset, kwargs);
}

std::pair<std::string, py::object> visitHierRuleSignature(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    auto* first_term = dynamic_cast<antlr4::tree::TerminalNode*>(children[0]);
    std::string signature_type = first_term->getText();

    // Check for VALUE_DOMAIN terminal to determine kind
    bool has_value_domain = false;
    for (auto& child : children) {
        auto* term = dynamic_cast<antlr4::tree::TerminalNode*>(child);
        if (term && static_cast<int>(term->getSymbol()->getType()) == VtlParser::VALUE_DOMAIN) {
            has_value_domain = true;
            break;
        }
    }
    std::string kind = has_value_domain ? "ValuedomainID" : "DatasetID";

    // Collect valueDomainSignature children (rule_index == RuleValueDomainSignature == 79)
    py::list conditions;
    for (auto& child : children) {
        auto* rule_child = dynamic_cast<antlr4::ParserRuleContext*>(child);
        if (rule_child) {
            auto cid = get_ctx_id(rule_child);
            if (cid.first == VtlParser::RuleValueDomainSignature) {
                conditions.append(visitValueDomainSignature(rule_child));
            }
        }
    }

    // Find IDENTIFIER terminal (the dataset name)
    antlr4::tree::TerminalNode* dataset_term = nullptr;
    for (auto& child : children) {
        auto* term = dynamic_cast<antlr4::tree::TerminalNode*>(child);
        if (term && static_cast<int>(term->getSymbol()->getType()) == VtlParser::IDENTIFIER) {
            dataset_term = term;
        }
    }

    auto ti = extract_token_info(ctx);

    if (py::len(conditions) > 0) {
        py::list cond_list = conditions[py::int_(0)].cast<py::list>();
        py::list identifiers_list;
        for (auto& elto : cond_list) {
            // Check if elto has 'alias' attribute and it's not None
            std::string val;
            if (py::hasattr(elto, "alias") && !elto.attr("alias").is_none()) {
                val = elto.attr("alias").cast<std::string>();
            } else {
                val = elto.attr("value").cast<std::string>();
            }
            py::dict dk;
            dk["value"] = py::str(val);
            dk["kind"] = py::str(kind);
            for (auto item : ti) dk[item.first] = item.second;
            identifiers_list.append(call_with_kwargs(py_DefIdentifier, dk));
        }
        // Append dataset identifier
        py::dict dk;
        dk["value"] = py::str(dataset_term->getText());
        dk["kind"] = py::str(kind);
        for (auto item : ti) dk[item.first] = item.second;
        identifiers_list.append(call_with_kwargs(py_DefIdentifier, dk));
        return {signature_type, identifiers_list};
    } else {
        py::dict dk;
        dk["value"] = py::str(dataset_term->getText());
        dk["kind"] = py::str(kind);
        for (auto item : ti) dk[item.first] = item.second;
        return {signature_type, call_with_kwargs(py_DefIdentifier, dk)};
    }
}

py::list visitValueDomainSignature(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    py::list component_nodes;
    for (auto& child : children) {
        auto* rule_child = dynamic_cast<antlr4::ParserRuleContext*>(child);
        if (rule_child) {
            auto cid = get_ctx_id(rule_child);
            // RuleSignature == 73
            if (cid.first == VtlParser::RuleSignature) {
                component_nodes.append(visitSignature(rule_child));
            }
        }
    }
    return component_nodes;
}

py::list visitRuleClauseHierarchical(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    py::list rules_nodes;
    for (auto& child : children) {
        auto* rule_child = dynamic_cast<antlr4::ParserRuleContext*>(child);
        if (rule_child) {
            auto cid = get_ctx_id(rule_child);
            // RuleRuleItemHierarchical == 77
            if (cid.first == VtlParser::RuleRuleItemHierarchical) {
                rules_nodes.append(visitRuleItemHierarchical(rule_child));
            }
        }
    }
    return rules_nodes;
}

py::object visitRuleItemHierarchical(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    // First child: check for IDENTIFIER (rule name)
    py::object rule_name = py::none();
    auto* first_term = dynamic_cast<antlr4::tree::TerminalNode*>(children[0]);
    if (first_term && static_cast<int>(first_term->getSymbol()->getType()) == VtlParser::IDENTIFIER) {
        rule_name = py::str(first_term->getText());
    }

    // Find codeItemRelation (rule_index == RuleCodeItemRelation == 80)
    py::object rule_node = py::none();
    for (auto& child : children) {
        auto* rule_child = dynamic_cast<antlr4::ParserRuleContext*>(child);
        if (rule_child) {
            auto cid = get_ctx_id(rule_child);
            if (cid.first == VtlParser::RuleCodeItemRelation) {
                rule_node = visitCodeItemRelation(rule_child);
                break;
            }
        }
    }

    // erCode (rule_index == RuleErCode == 98)
    py::object er_code = py::none();
    for (auto& child : children) {
        auto* rule_child = dynamic_cast<antlr4::ParserRuleContext*>(child);
        if (rule_child) {
            auto cid = get_ctx_id(rule_child);
            if (cid.first == VtlParser::RuleErCode) {
                er_code = visitErCode(rule_child);
                break;
            }
        }
    }

    // erLevel (rule_index == RuleErLevel == 99)
    py::object er_level = py::none();
    for (auto& child : children) {
        auto* rule_child = dynamic_cast<antlr4::ParserRuleContext*>(child);
        if (rule_child) {
            auto cid = get_ctx_id(rule_child);
            if (cid.first == VtlParser::RuleErLevel) {
                er_level = visitErLevel(rule_child);
                break;
            }
        }
    }

    auto ti = extract_token_info(ctx);
    py::dict kwargs;
    kwargs["name"] = rule_name;
    kwargs["rule"] = rule_node;
    kwargs["erCode"] = er_code;
    kwargs["erLevel"] = er_level;
    for (auto item : ti) kwargs[item.first] = item.second;
    return call_with_kwargs(py_HRule, kwargs);
}

py::object visitCodeItemRelation(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    py::object when_text = py::none();
    py::object vd_value, op;
    py::dict token_info_value, token_info_op;

    auto* first_term = dynamic_cast<antlr4::tree::TerminalNode*>(children[0]);
    if (first_term) {
        // WHEN expr THEN codeItemRef ...
        when_text = py::str(first_term->getText());
        vd_value = visitValueDomainValue(as_rule(children[3]));
        op = visitComparisonOperand(as_rule(children[4]));
        token_info_value = extract_token_info(as_rule(children[3]));
        token_info_op = extract_token_info(as_rule(children[4]));
    } else {
        vd_value = visitValueDomainValue(as_rule(children[0]));
        op = visitComparisonOperand(as_rule(children[1]));
        token_info_value = extract_token_info(as_rule(children[0]));
        token_info_op = extract_token_info(as_rule(children[1]));
    }

    // Build the main rule_node as HRBinOp
    py::dict vd_kwargs;
    vd_kwargs["value"] = vd_value;
    vd_kwargs["kind"] = py::str("CodeItemID");
    for (auto item : token_info_value) vd_kwargs[item.first] = item.second;
    py::object left_def = call_with_kwargs(py_DefIdentifier, vd_kwargs);

    py::dict rule_kwargs;
    rule_kwargs["left"] = left_def;
    rule_kwargs["op"] = op;
    rule_kwargs["right"] = py::none();
    for (auto item : token_info_op) rule_kwargs[item.first] = item.second;
    py::object rule_node = call_with_kwargs(py_HRBinOp, rule_kwargs);

    // Collect codeItemRelationClause items (rule_index == RuleCodeItemRelationClause == 81)
    std::vector<antlr4::ParserRuleContext*> items;
    for (auto& child : children) {
        auto* rule_child = dynamic_cast<antlr4::ParserRuleContext*>(child);
        if (rule_child) {
            auto cid = get_ctx_id(rule_child);
            if (cid.first == VtlParser::RuleCodeItemRelationClause) {
                items.push_back(rule_child);
            }
        }
    }

    auto items_ti = extract_token_info(items[0]);

    if (items.size() == 1) {
        py::object cir_node = visitCodeItemRelationClause(items[0]);
        if (py::isinstance(cir_node, py_HRBinOp)) {
            py::dict unop_kwargs;
            unop_kwargs["op"] = cir_node.attr("op");
            unop_kwargs["operand"] = cir_node.attr("right");
            for (auto item : items_ti) unop_kwargs[item.first] = item.second;
            rule_node.attr("right") = call_with_kwargs(py_HRUnOp, unop_kwargs);
        } else if (py::isinstance(cir_node, py_DefIdentifier)) {
            rule_node.attr("right") = cir_node;
        }
    } else {
        py::object previous_node = visitCodeItemRelationClause(items[0]);
        if (py::isinstance(previous_node, py_HRBinOp)) {
            py::dict unop_kwargs;
            unop_kwargs["op"] = previous_node.attr("op");
            unop_kwargs["operand"] = previous_node.attr("right");
            for (auto item : items_ti) unop_kwargs[item.first] = item.second;
            previous_node = call_with_kwargs(py_HRUnOp, unop_kwargs);
        }

        for (size_t i = 1; i < items.size(); i++) {
            py::object item_node = visitCodeItemRelationClause(items[i]);
            item_node.attr("left") = previous_node;
            previous_node = item_node;
        }

        rule_node.attr("right") = previous_node;
    }

    if (!when_text.is_none()) {
        // Wrap with WHEN condition
        py::object expr_node = visitExprComponent(as_rule(children[1]));
        auto when_ti = extract_token_info(as_rule(children[1]));
        py::dict when_kwargs;
        when_kwargs["left"] = expr_node;
        when_kwargs["op"] = when_text;
        when_kwargs["right"] = rule_node;
        for (auto item : when_ti) when_kwargs[item.first] = item.second;
        rule_node = call_with_kwargs(py_HRBinOp, when_kwargs);
    }

    return rule_node;
}

py::object visitCodeItemRelationClause(antlr4::ParserRuleContext* ctx) {
    auto& children = ctx->children;

    // Check for expr (rule_index == RuleExpr == 2) — not implemented
    for (auto& child : children) {
        auto* rule_child = dynamic_cast<antlr4::ParserRuleContext*>(child);
        if (rule_child) {
            auto cid = get_ctx_id(rule_child);
            if (cid.first == VtlParser::RuleExpr) {
                throw std::runtime_error("NotImplementedError: codeItemRelationClause with expr");
            }
        }
    }

    // Collect right_condition: exprComponent children with COMPARISON_EXPR_COMP ctx_id (3, 5)
    py::list right_conditions;
    for (auto& child : children) {
        auto* rule_child = dynamic_cast<antlr4::ParserRuleContext*>(child);
        if (rule_child) {
            auto cid = get_ctx_id(rule_child);
            // COMPARISON_EXPR_COMP = (3, 5)
            if (cid.first == VtlParser::RuleExprComponent && cid.second == 5) {
                right_conditions.append(visitExprComponent(rule_child));
            }
        }
    }

    auto* first_term = dynamic_cast<antlr4::tree::TerminalNode*>(children[0]);
    if (first_term) {
        std::string op = first_term->getText();
        py::object value = visitValueDomainValue(as_rule(children[1]));

        auto val_ti = extract_token_info(as_rule(children[1]));
        py::dict code_kwargs;
        code_kwargs["value"] = value;
        code_kwargs["kind"] = py::str("CodeItemID");
        for (auto item : val_ti) code_kwargs[item.first] = item.second;
        py::object code_item = call_with_kwargs(py_DefIdentifier, code_kwargs);

        if (py::len(right_conditions) > 0) {
            code_item.attr("_right_condition") = right_conditions[py::int_(0)];
        }

        auto op_ti = extract_token_info_terminal(first_term);
        py::dict binop_kwargs;
        binop_kwargs["left"] = py::none();
        binop_kwargs["op"] = py::str(op);
        binop_kwargs["right"] = code_item;
        for (auto item : op_ti) binop_kwargs[item.first] = item.second;
        return call_with_kwargs(py_HRBinOp, binop_kwargs);
    } else {
        py::object value = visitValueDomainValue(as_rule(children[0]));

        auto val_ti = extract_token_info(as_rule(children[0]));
        py::dict code_kwargs;
        code_kwargs["value"] = value;
        code_kwargs["kind"] = py::str("CodeItemID");
        for (auto item : val_ti) code_kwargs[item.first] = item.second;
        py::object code_item = call_with_kwargs(py_DefIdentifier, code_kwargs);

        if (py::len(right_conditions) > 0) {
            code_item.attr("_right_condition") = right_conditions[py::int_(0)];
        }

        return code_item;
    }
}

} // namespace ASTBuilder
