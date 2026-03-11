
// Generated from /home/javier/Programacion/vtlengine/src/vtlengine/AST/Grammar/Vtl.g4 by ANTLR 4.13.1

#pragma once


#include "antlr4-runtime.h"




class  VtlParser : public antlr4::Parser {
public:
  enum {
    LPAREN = 1, RPAREN = 2, QLPAREN = 3, QRPAREN = 4, GLPAREN = 5, GRPAREN = 6, 
    EQ = 7, LT = 8, MT = 9, ME = 10, NEQ = 11, LE = 12, PLUS = 13, MINUS = 14, 
    MUL = 15, DIV = 16, COMMA = 17, POINTER = 18, COLON = 19, ASSIGN = 20, 
    MEMBERSHIP = 21, EVAL = 22, IF = 23, CASE = 24, THEN = 25, ELSE = 26, 
    USING = 27, WITH = 28, CURRENT_DATE = 29, DATEDIFF = 30, DATEADD = 31, 
    YEAR_OP = 32, MONTH_OP = 33, DAYOFMONTH = 34, DAYOFYEAR = 35, DAYTOYEAR = 36, 
    DAYTOMONTH = 37, YEARTODAY = 38, MONTHTODAY = 39, ON = 40, DROP = 41, 
    KEEP = 42, CALC = 43, ATTRCALC = 44, RENAME = 45, AS = 46, AND = 47, 
    OR = 48, XOR = 49, NOT = 50, BETWEEN = 51, IN = 52, NOT_IN = 53, NULL_CONSTANT = 54, 
    ISNULL = 55, EX = 56, UNION = 57, DIFF = 58, SYMDIFF = 59, INTERSECT = 60, 
    RANDOM = 61, KEYS = 62, INTYEAR = 63, INTMONTH = 64, INTDAY = 65, CHECK = 66, 
    EXISTS_IN = 67, TO = 68, RETURN = 69, IMBALANCE = 70, ERRORCODE = 71, 
    ALL = 72, AGGREGATE = 73, ERRORLEVEL = 74, ORDER = 75, BY = 76, RANK = 77, 
    ASC = 78, DESC = 79, MIN = 80, MAX = 81, FIRST = 82, LAST = 83, INDEXOF = 84, 
    ABS = 85, KEY = 86, LN = 87, LOG = 88, TRUNC = 89, ROUND = 90, POWER = 91, 
    MOD = 92, LEN = 93, CONCAT = 94, TRIM = 95, UCASE = 96, LCASE = 97, 
    SUBSTR = 98, SUM = 99, AVG = 100, MEDIAN = 101, COUNT = 102, DIMENSION = 103, 
    MEASURE = 104, ATTRIBUTE = 105, FILTER = 106, MERGE = 107, EXP = 108, 
    ROLE = 109, VIRAL = 110, CHARSET_MATCH = 111, TYPE = 112, NVL = 113, 
    HIERARCHY = 114, OPTIONAL = 115, INVALID = 116, VALUE_DOMAIN = 117, 
    VARIABLE = 118, DATA = 119, STRUCTURE = 120, DATASET = 121, OPERATOR = 122, 
    DEFINE = 123, PUT_SYMBOL = 124, DATAPOINT = 125, HIERARCHICAL = 126, 
    RULESET = 127, RULE = 128, END = 129, ALTER_DATASET = 130, LTRIM = 131, 
    RTRIM = 132, INSTR = 133, REPLACE = 134, CEIL = 135, FLOOR = 136, SQRT = 137, 
    ANY = 138, SETDIFF = 139, STDDEV_POP = 140, STDDEV_SAMP = 141, VAR_POP = 142, 
    VAR_SAMP = 143, GROUP = 144, EXCEPT = 145, HAVING = 146, FIRST_VALUE = 147, 
    LAST_VALUE = 148, LAG = 149, LEAD = 150, RATIO_TO_REPORT = 151, OVER = 152, 
    PRECEDING = 153, FOLLOWING = 154, UNBOUNDED = 155, PARTITION = 156, 
    ROWS = 157, RANGE = 158, CURRENT = 159, VALID = 160, FILL_TIME_SERIES = 161, 
    FLOW_TO_STOCK = 162, STOCK_TO_FLOW = 163, TIMESHIFT = 164, MEASURES = 165, 
    NO_MEASURES = 166, CONDITION = 167, BOOLEAN = 168, DATE = 169, TIME_PERIOD = 170, 
    NUMBER = 171, STRING = 172, TIME = 173, INTEGER = 174, FLOAT = 175, 
    LIST = 176, RECORD = 177, RESTRICT = 178, YYYY = 179, MM = 180, DD = 181, 
    MAX_LENGTH = 182, REGEXP = 183, IS = 184, WHEN = 185, FROM = 186, AGGREGATES = 187, 
    POINTS = 188, POINT = 189, TOTAL = 190, PARTIAL = 191, ALWAYS = 192, 
    INNER_JOIN = 193, LEFT_JOIN = 194, CROSS_JOIN = 195, FULL_JOIN = 196, 
    MAPS_FROM = 197, MAPS_TO = 198, MAP_TO = 199, MAP_FROM = 200, RETURNS = 201, 
    PIVOT = 202, CUSTOMPIVOT = 203, UNPIVOT = 204, SUBSPACE = 205, APPLY = 206, 
    CONDITIONED = 207, PERIOD_INDICATOR = 208, SINGLE = 209, DURATION = 210, 
    TIME_AGG = 211, UNIT = 212, VALUE = 213, VALUEDOMAINS = 214, VARIABLES = 215, 
    INPUT = 216, OUTPUT = 217, CAST = 218, RULE_PRIORITY = 219, DATASET_PRIORITY = 220, 
    DEFAULT = 221, CHECK_DATAPOINT = 222, CHECK_HIERARCHY = 223, COMPUTED = 224, 
    NON_NULL = 225, NON_ZERO = 226, PARTIAL_NULL = 227, PARTIAL_ZERO = 228, 
    ALWAYS_NULL = 229, ALWAYS_ZERO = 230, COMPONENTS = 231, ALL_MEASURES = 232, 
    SCALAR = 233, COMPONENT = 234, DATAPOINT_ON_VD = 235, DATAPOINT_ON_VAR = 236, 
    HIERARCHICAL_ON_VD = 237, HIERARCHICAL_ON_VAR = 238, SET = 239, LANGUAGE = 240, 
    INTEGER_CONSTANT = 241, NUMBER_CONSTANT = 242, BOOLEAN_CONSTANT = 243, 
    STRING_CONSTANT = 244, IDENTIFIER = 245, WS = 246, EOL = 247, ML_COMMENT = 248, 
    SL_COMMENT = 249
  };

  enum {
    RuleStart = 0, RuleStatement = 1, RuleExpr = 2, RuleExprComponent = 3, 
    RuleFunctionsComponents = 4, RuleFunctions = 5, RuleDatasetClause = 6, 
    RuleRenameClause = 7, RuleAggrClause = 8, RuleFilterClause = 9, RuleCalcClause = 10, 
    RuleKeepOrDropClause = 11, RulePivotOrUnpivotClause = 12, RuleCustomPivotClause = 13, 
    RuleSubspaceClause = 14, RuleJoinOperators = 15, RuleDefOperators = 16, 
    RuleGenericOperators = 17, RuleGenericOperatorsComponent = 18, RuleParameterComponent = 19, 
    RuleParameter = 20, RuleStringOperators = 21, RuleStringOperatorsComponent = 22, 
    RuleNumericOperators = 23, RuleNumericOperatorsComponent = 24, RuleComparisonOperators = 25, 
    RuleComparisonOperatorsComponent = 26, RuleTimeOperators = 27, RuleTimeOperatorsComponent = 28, 
    RuleSetOperators = 29, RuleHierarchyOperators = 30, RuleValidationOperators = 31, 
    RuleConditionalOperators = 32, RuleConditionalOperatorsComponent = 33, 
    RuleAggrOperators = 34, RuleAggrOperatorsGrouping = 35, RuleAnFunction = 36, 
    RuleAnFunctionComponent = 37, RuleRenameClauseItem = 38, RuleAggregateClause = 39, 
    RuleAggrFunctionClause = 40, RuleCalcClauseItem = 41, RuleSubspaceClauseItem = 42, 
    RuleScalarItem = 43, RuleJoinClauseWithoutUsing = 44, RuleJoinClause = 45, 
    RuleJoinClauseItem = 46, RuleJoinBody = 47, RuleJoinApplyClause = 48, 
    RulePartitionByClause = 49, RuleOrderByClause = 50, RuleOrderByItem = 51, 
    RuleWindowingClause = 52, RuleSignedInteger = 53, RuleSignedNumber = 54, 
    RuleLimitClauseItem = 55, RuleGroupingClause = 56, RuleHavingClause = 57, 
    RuleParameterItem = 58, RuleOutputParameterType = 59, RuleOutputParameterTypeComponent = 60, 
    RuleInputParameterType = 61, RuleRulesetType = 62, RuleScalarType = 63, 
    RuleComponentType = 64, RuleDatasetType = 65, RuleEvalDatasetType = 66, 
    RuleScalarSetType = 67, RuleDpRuleset = 68, RuleHrRuleset = 69, RuleValueDomainName = 70, 
    RuleRulesetID = 71, RuleRulesetSignature = 72, RuleSignature = 73, RuleRuleClauseDatapoint = 74, 
    RuleRuleItemDatapoint = 75, RuleRuleClauseHierarchical = 76, RuleRuleItemHierarchical = 77, 
    RuleHierRuleSignature = 78, RuleValueDomainSignature = 79, RuleCodeItemRelation = 80, 
    RuleCodeItemRelationClause = 81, RuleValueDomainValue = 82, RuleScalarTypeConstraint = 83, 
    RuleCompConstraint = 84, RuleMultModifier = 85, RuleValidationOutput = 86, 
    RuleValidationMode = 87, RuleConditionClause = 88, RuleInputMode = 89, 
    RuleImbalanceExpr = 90, RuleInputModeHierarchy = 91, RuleOutputModeHierarchy = 92, 
    RuleAlias = 93, RuleVarID = 94, RuleSimpleComponentId = 95, RuleComponentID = 96, 
    RuleLists = 97, RuleErCode = 98, RuleErLevel = 99, RuleComparisonOperand = 100, 
    RuleOptionalExpr = 101, RuleOptionalExprComponent = 102, RuleComponentRole = 103, 
    RuleViralAttribute = 104, RuleValueDomainID = 105, RuleOperatorID = 106, 
    RuleRoutineName = 107, RuleConstant = 108, RuleBasicScalarType = 109, 
    RuleRetainType = 110
  };

  explicit VtlParser(antlr4::TokenStream *input);

  VtlParser(antlr4::TokenStream *input, const antlr4::atn::ParserATNSimulatorOptions &options);

  ~VtlParser() override;

  std::string getGrammarFileName() const override;

  const antlr4::atn::ATN& getATN() const override;

  const std::vector<std::string>& getRuleNames() const override;

  const antlr4::dfa::Vocabulary& getVocabulary() const override;

  antlr4::atn::SerializedATNView getSerializedATN() const override;


  class StartContext;
  class StatementContext;
  class ExprContext;
  class ExprComponentContext;
  class FunctionsComponentsContext;
  class FunctionsContext;
  class DatasetClauseContext;
  class RenameClauseContext;
  class AggrClauseContext;
  class FilterClauseContext;
  class CalcClauseContext;
  class KeepOrDropClauseContext;
  class PivotOrUnpivotClauseContext;
  class CustomPivotClauseContext;
  class SubspaceClauseContext;
  class JoinOperatorsContext;
  class DefOperatorsContext;
  class GenericOperatorsContext;
  class GenericOperatorsComponentContext;
  class ParameterComponentContext;
  class ParameterContext;
  class StringOperatorsContext;
  class StringOperatorsComponentContext;
  class NumericOperatorsContext;
  class NumericOperatorsComponentContext;
  class ComparisonOperatorsContext;
  class ComparisonOperatorsComponentContext;
  class TimeOperatorsContext;
  class TimeOperatorsComponentContext;
  class SetOperatorsContext;
  class HierarchyOperatorsContext;
  class ValidationOperatorsContext;
  class ConditionalOperatorsContext;
  class ConditionalOperatorsComponentContext;
  class AggrOperatorsContext;
  class AggrOperatorsGroupingContext;
  class AnFunctionContext;
  class AnFunctionComponentContext;
  class RenameClauseItemContext;
  class AggregateClauseContext;
  class AggrFunctionClauseContext;
  class CalcClauseItemContext;
  class SubspaceClauseItemContext;
  class ScalarItemContext;
  class JoinClauseWithoutUsingContext;
  class JoinClauseContext;
  class JoinClauseItemContext;
  class JoinBodyContext;
  class JoinApplyClauseContext;
  class PartitionByClauseContext;
  class OrderByClauseContext;
  class OrderByItemContext;
  class WindowingClauseContext;
  class SignedIntegerContext;
  class SignedNumberContext;
  class LimitClauseItemContext;
  class GroupingClauseContext;
  class HavingClauseContext;
  class ParameterItemContext;
  class OutputParameterTypeContext;
  class OutputParameterTypeComponentContext;
  class InputParameterTypeContext;
  class RulesetTypeContext;
  class ScalarTypeContext;
  class ComponentTypeContext;
  class DatasetTypeContext;
  class EvalDatasetTypeContext;
  class ScalarSetTypeContext;
  class DpRulesetContext;
  class HrRulesetContext;
  class ValueDomainNameContext;
  class RulesetIDContext;
  class RulesetSignatureContext;
  class SignatureContext;
  class RuleClauseDatapointContext;
  class RuleItemDatapointContext;
  class RuleClauseHierarchicalContext;
  class RuleItemHierarchicalContext;
  class HierRuleSignatureContext;
  class ValueDomainSignatureContext;
  class CodeItemRelationContext;
  class CodeItemRelationClauseContext;
  class ValueDomainValueContext;
  class ScalarTypeConstraintContext;
  class CompConstraintContext;
  class MultModifierContext;
  class ValidationOutputContext;
  class ValidationModeContext;
  class ConditionClauseContext;
  class InputModeContext;
  class ImbalanceExprContext;
  class InputModeHierarchyContext;
  class OutputModeHierarchyContext;
  class AliasContext;
  class VarIDContext;
  class SimpleComponentIdContext;
  class ComponentIDContext;
  class ListsContext;
  class ErCodeContext;
  class ErLevelContext;
  class ComparisonOperandContext;
  class OptionalExprContext;
  class OptionalExprComponentContext;
  class ComponentRoleContext;
  class ViralAttributeContext;
  class ValueDomainIDContext;
  class OperatorIDContext;
  class RoutineNameContext;
  class ConstantContext;
  class BasicScalarTypeContext;
  class RetainTypeContext; 

  class  StartContext : public antlr4::ParserRuleContext {
  public:
    StartContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *EOF();
    std::vector<StatementContext *> statement();
    StatementContext* statement(size_t i);
    std::vector<antlr4::tree::TerminalNode *> EOL();
    antlr4::tree::TerminalNode* EOL(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  StartContext* start();

  class  StatementContext : public antlr4::ParserRuleContext {
  public:
    StatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    StatementContext() = default;
    void copyFrom(StatementContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  DefineExpressionContext : public StatementContext {
  public:
    DefineExpressionContext(StatementContext *ctx);

    DefOperatorsContext *defOperators();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  TemporaryAssignmentContext : public StatementContext {
  public:
    TemporaryAssignmentContext(StatementContext *ctx);

    VarIDContext *varID();
    antlr4::tree::TerminalNode *ASSIGN();
    ExprContext *expr();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  PersistAssignmentContext : public StatementContext {
  public:
    PersistAssignmentContext(StatementContext *ctx);

    VarIDContext *varID();
    antlr4::tree::TerminalNode *PUT_SYMBOL();
    ExprContext *expr();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  StatementContext* statement();

  class  ExprContext : public antlr4::ParserRuleContext {
  public:
    ExprContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    ExprContext() = default;
    void copyFrom(ExprContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  VarIdExprContext : public ExprContext {
  public:
    VarIdExprContext(ExprContext *ctx);

    VarIDContext *varID();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  MembershipExprContext : public ExprContext {
  public:
    MembershipExprContext(ExprContext *ctx);

    ExprContext *expr();
    antlr4::tree::TerminalNode *MEMBERSHIP();
    SimpleComponentIdContext *simpleComponentId();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  InNotInExprContext : public ExprContext {
  public:
    InNotInExprContext(ExprContext *ctx);

    VtlParser::ExprContext *left = nullptr;
    antlr4::Token *op = nullptr;
    ExprContext *expr();
    antlr4::tree::TerminalNode *IN();
    antlr4::tree::TerminalNode *NOT_IN();
    ListsContext *lists();
    ValueDomainIDContext *valueDomainID();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  BooleanExprContext : public ExprContext {
  public:
    BooleanExprContext(ExprContext *ctx);

    VtlParser::ExprContext *left = nullptr;
    antlr4::Token *op = nullptr;
    VtlParser::ExprContext *right = nullptr;
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    antlr4::tree::TerminalNode *AND();
    antlr4::tree::TerminalNode *OR();
    antlr4::tree::TerminalNode *XOR();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ComparisonExprContext : public ExprContext {
  public:
    ComparisonExprContext(ExprContext *ctx);

    VtlParser::ExprContext *left = nullptr;
    VtlParser::ComparisonOperandContext *op = nullptr;
    VtlParser::ExprContext *right = nullptr;
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    ComparisonOperandContext *comparisonOperand();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  UnaryExprContext : public ExprContext {
  public:
    UnaryExprContext(ExprContext *ctx);

    antlr4::Token *op = nullptr;
    VtlParser::ExprContext *right = nullptr;
    ExprContext *expr();
    antlr4::tree::TerminalNode *PLUS();
    antlr4::tree::TerminalNode *MINUS();
    antlr4::tree::TerminalNode *NOT();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  FunctionsExpressionContext : public ExprContext {
  public:
    FunctionsExpressionContext(ExprContext *ctx);

    FunctionsContext *functions();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  IfExprContext : public ExprContext {
  public:
    IfExprContext(ExprContext *ctx);

    VtlParser::ExprContext *conditionalExpr = nullptr;
    VtlParser::ExprContext *thenExpr = nullptr;
    VtlParser::ExprContext *elseExpr = nullptr;
    antlr4::tree::TerminalNode *IF();
    antlr4::tree::TerminalNode *THEN();
    antlr4::tree::TerminalNode *ELSE();
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ClauseExprContext : public ExprContext {
  public:
    ClauseExprContext(ExprContext *ctx);

    VtlParser::ExprContext *dataset = nullptr;
    VtlParser::DatasetClauseContext *clause = nullptr;
    antlr4::tree::TerminalNode *QLPAREN();
    antlr4::tree::TerminalNode *QRPAREN();
    ExprContext *expr();
    DatasetClauseContext *datasetClause();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  CaseExprContext : public ExprContext {
  public:
    CaseExprContext(ExprContext *ctx);

    VtlParser::ExprContext *exprContext = nullptr;
    std::vector<ExprContext *> condExpr;
    std::vector<ExprContext *> thenExpr;
    VtlParser::ExprContext *elseExpr = nullptr;
    antlr4::tree::TerminalNode *CASE();
    antlr4::tree::TerminalNode *ELSE();
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    std::vector<antlr4::tree::TerminalNode *> WHEN();
    antlr4::tree::TerminalNode* WHEN(size_t i);
    std::vector<antlr4::tree::TerminalNode *> THEN();
    antlr4::tree::TerminalNode* THEN(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ArithmeticExprContext : public ExprContext {
  public:
    ArithmeticExprContext(ExprContext *ctx);

    VtlParser::ExprContext *left = nullptr;
    antlr4::Token *op = nullptr;
    VtlParser::ExprContext *right = nullptr;
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    antlr4::tree::TerminalNode *MUL();
    antlr4::tree::TerminalNode *DIV();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ParenthesisExprContext : public ExprContext {
  public:
    ParenthesisExprContext(ExprContext *ctx);

    antlr4::tree::TerminalNode *LPAREN();
    ExprContext *expr();
    antlr4::tree::TerminalNode *RPAREN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ConstantExprContext : public ExprContext {
  public:
    ConstantExprContext(ExprContext *ctx);

    ConstantContext *constant();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ArithmeticExprOrConcatContext : public ExprContext {
  public:
    ArithmeticExprOrConcatContext(ExprContext *ctx);

    VtlParser::ExprContext *left = nullptr;
    antlr4::Token *op = nullptr;
    VtlParser::ExprContext *right = nullptr;
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    antlr4::tree::TerminalNode *PLUS();
    antlr4::tree::TerminalNode *MINUS();
    antlr4::tree::TerminalNode *CONCAT();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ExprContext* expr();
  ExprContext* expr(int precedence);
  class  ExprComponentContext : public antlr4::ParserRuleContext {
  public:
    ExprComponentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    ExprComponentContext() = default;
    void copyFrom(ExprComponentContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  ArithmeticExprCompContext : public ExprComponentContext {
  public:
    ArithmeticExprCompContext(ExprComponentContext *ctx);

    VtlParser::ExprComponentContext *left = nullptr;
    antlr4::Token *op = nullptr;
    VtlParser::ExprComponentContext *right = nullptr;
    std::vector<ExprComponentContext *> exprComponent();
    ExprComponentContext* exprComponent(size_t i);
    antlr4::tree::TerminalNode *MUL();
    antlr4::tree::TerminalNode *DIV();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  IfExprCompContext : public ExprComponentContext {
  public:
    IfExprCompContext(ExprComponentContext *ctx);

    VtlParser::ExprComponentContext *conditionalExpr = nullptr;
    VtlParser::ExprComponentContext *thenExpr = nullptr;
    VtlParser::ExprComponentContext *elseExpr = nullptr;
    antlr4::tree::TerminalNode *IF();
    antlr4::tree::TerminalNode *THEN();
    antlr4::tree::TerminalNode *ELSE();
    std::vector<ExprComponentContext *> exprComponent();
    ExprComponentContext* exprComponent(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ComparisonExprCompContext : public ExprComponentContext {
  public:
    ComparisonExprCompContext(ExprComponentContext *ctx);

    VtlParser::ExprComponentContext *left = nullptr;
    VtlParser::ExprComponentContext *right = nullptr;
    ComparisonOperandContext *comparisonOperand();
    std::vector<ExprComponentContext *> exprComponent();
    ExprComponentContext* exprComponent(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  FunctionsExpressionCompContext : public ExprComponentContext {
  public:
    FunctionsExpressionCompContext(ExprComponentContext *ctx);

    FunctionsComponentsContext *functionsComponents();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  CompIdContext : public ExprComponentContext {
  public:
    CompIdContext(ExprComponentContext *ctx);

    ComponentIDContext *componentID();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ConstantExprCompContext : public ExprComponentContext {
  public:
    ConstantExprCompContext(ExprComponentContext *ctx);

    ConstantContext *constant();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ArithmeticExprOrConcatCompContext : public ExprComponentContext {
  public:
    ArithmeticExprOrConcatCompContext(ExprComponentContext *ctx);

    VtlParser::ExprComponentContext *left = nullptr;
    antlr4::Token *op = nullptr;
    VtlParser::ExprComponentContext *right = nullptr;
    std::vector<ExprComponentContext *> exprComponent();
    ExprComponentContext* exprComponent(size_t i);
    antlr4::tree::TerminalNode *PLUS();
    antlr4::tree::TerminalNode *MINUS();
    antlr4::tree::TerminalNode *CONCAT();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ParenthesisExprCompContext : public ExprComponentContext {
  public:
    ParenthesisExprCompContext(ExprComponentContext *ctx);

    antlr4::tree::TerminalNode *LPAREN();
    ExprComponentContext *exprComponent();
    antlr4::tree::TerminalNode *RPAREN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  InNotInExprCompContext : public ExprComponentContext {
  public:
    InNotInExprCompContext(ExprComponentContext *ctx);

    VtlParser::ExprComponentContext *left = nullptr;
    antlr4::Token *op = nullptr;
    ExprComponentContext *exprComponent();
    antlr4::tree::TerminalNode *IN();
    antlr4::tree::TerminalNode *NOT_IN();
    ListsContext *lists();
    ValueDomainIDContext *valueDomainID();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  UnaryExprCompContext : public ExprComponentContext {
  public:
    UnaryExprCompContext(ExprComponentContext *ctx);

    antlr4::Token *op = nullptr;
    VtlParser::ExprComponentContext *right = nullptr;
    ExprComponentContext *exprComponent();
    antlr4::tree::TerminalNode *PLUS();
    antlr4::tree::TerminalNode *MINUS();
    antlr4::tree::TerminalNode *NOT();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  CaseExprCompContext : public ExprComponentContext {
  public:
    CaseExprCompContext(ExprComponentContext *ctx);

    VtlParser::ExprComponentContext *exprComponentContext = nullptr;
    std::vector<ExprComponentContext *> condExpr;
    std::vector<ExprComponentContext *> thenExpr;
    VtlParser::ExprComponentContext *elseExpr = nullptr;
    antlr4::tree::TerminalNode *CASE();
    antlr4::tree::TerminalNode *ELSE();
    std::vector<ExprComponentContext *> exprComponent();
    ExprComponentContext* exprComponent(size_t i);
    std::vector<antlr4::tree::TerminalNode *> WHEN();
    antlr4::tree::TerminalNode* WHEN(size_t i);
    std::vector<antlr4::tree::TerminalNode *> THEN();
    antlr4::tree::TerminalNode* THEN(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  BooleanExprCompContext : public ExprComponentContext {
  public:
    BooleanExprCompContext(ExprComponentContext *ctx);

    VtlParser::ExprComponentContext *left = nullptr;
    antlr4::Token *op = nullptr;
    VtlParser::ExprComponentContext *right = nullptr;
    std::vector<ExprComponentContext *> exprComponent();
    ExprComponentContext* exprComponent(size_t i);
    antlr4::tree::TerminalNode *AND();
    antlr4::tree::TerminalNode *OR();
    antlr4::tree::TerminalNode *XOR();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ExprComponentContext* exprComponent();
  ExprComponentContext* exprComponent(int precedence);
  class  FunctionsComponentsContext : public antlr4::ParserRuleContext {
  public:
    FunctionsComponentsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    FunctionsComponentsContext() = default;
    void copyFrom(FunctionsComponentsContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  NumericFunctionsComponentsContext : public FunctionsComponentsContext {
  public:
    NumericFunctionsComponentsContext(FunctionsComponentsContext *ctx);

    NumericOperatorsComponentContext *numericOperatorsComponent();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  StringFunctionsComponentsContext : public FunctionsComponentsContext {
  public:
    StringFunctionsComponentsContext(FunctionsComponentsContext *ctx);

    StringOperatorsComponentContext *stringOperatorsComponent();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ComparisonFunctionsComponentsContext : public FunctionsComponentsContext {
  public:
    ComparisonFunctionsComponentsContext(FunctionsComponentsContext *ctx);

    ComparisonOperatorsComponentContext *comparisonOperatorsComponent();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  TimeFunctionsComponentsContext : public FunctionsComponentsContext {
  public:
    TimeFunctionsComponentsContext(FunctionsComponentsContext *ctx);

    TimeOperatorsComponentContext *timeOperatorsComponent();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  GenericFunctionsComponentsContext : public FunctionsComponentsContext {
  public:
    GenericFunctionsComponentsContext(FunctionsComponentsContext *ctx);

    GenericOperatorsComponentContext *genericOperatorsComponent();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  AnalyticFunctionsComponentsContext : public FunctionsComponentsContext {
  public:
    AnalyticFunctionsComponentsContext(FunctionsComponentsContext *ctx);

    AnFunctionComponentContext *anFunctionComponent();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ConditionalFunctionsComponentsContext : public FunctionsComponentsContext {
  public:
    ConditionalFunctionsComponentsContext(FunctionsComponentsContext *ctx);

    ConditionalOperatorsComponentContext *conditionalOperatorsComponent();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  AggregateFunctionsComponentsContext : public FunctionsComponentsContext {
  public:
    AggregateFunctionsComponentsContext(FunctionsComponentsContext *ctx);

    AggrOperatorsContext *aggrOperators();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  FunctionsComponentsContext* functionsComponents();

  class  FunctionsContext : public antlr4::ParserRuleContext {
  public:
    FunctionsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    FunctionsContext() = default;
    void copyFrom(FunctionsContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  HierarchyFunctionsContext : public FunctionsContext {
  public:
    HierarchyFunctionsContext(FunctionsContext *ctx);

    HierarchyOperatorsContext *hierarchyOperators();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  StringFunctionsContext : public FunctionsContext {
  public:
    StringFunctionsContext(FunctionsContext *ctx);

    StringOperatorsContext *stringOperators();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ValidationFunctionsContext : public FunctionsContext {
  public:
    ValidationFunctionsContext(FunctionsContext *ctx);

    ValidationOperatorsContext *validationOperators();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  GenericFunctionsContext : public FunctionsContext {
  public:
    GenericFunctionsContext(FunctionsContext *ctx);

    GenericOperatorsContext *genericOperators();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ConditionalFunctionsContext : public FunctionsContext {
  public:
    ConditionalFunctionsContext(FunctionsContext *ctx);

    ConditionalOperatorsContext *conditionalOperators();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  AggregateFunctionsContext : public FunctionsContext {
  public:
    AggregateFunctionsContext(FunctionsContext *ctx);

    AggrOperatorsGroupingContext *aggrOperatorsGrouping();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  JoinFunctionsContext : public FunctionsContext {
  public:
    JoinFunctionsContext(FunctionsContext *ctx);

    JoinOperatorsContext *joinOperators();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ComparisonFunctionsContext : public FunctionsContext {
  public:
    ComparisonFunctionsContext(FunctionsContext *ctx);

    ComparisonOperatorsContext *comparisonOperators();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  NumericFunctionsContext : public FunctionsContext {
  public:
    NumericFunctionsContext(FunctionsContext *ctx);

    NumericOperatorsContext *numericOperators();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  TimeFunctionsContext : public FunctionsContext {
  public:
    TimeFunctionsContext(FunctionsContext *ctx);

    TimeOperatorsContext *timeOperators();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  SetFunctionsContext : public FunctionsContext {
  public:
    SetFunctionsContext(FunctionsContext *ctx);

    SetOperatorsContext *setOperators();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  AnalyticFunctionsContext : public FunctionsContext {
  public:
    AnalyticFunctionsContext(FunctionsContext *ctx);

    AnFunctionContext *anFunction();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  FunctionsContext* functions();

  class  DatasetClauseContext : public antlr4::ParserRuleContext {
  public:
    DatasetClauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    RenameClauseContext *renameClause();
    AggrClauseContext *aggrClause();
    FilterClauseContext *filterClause();
    CalcClauseContext *calcClause();
    KeepOrDropClauseContext *keepOrDropClause();
    PivotOrUnpivotClauseContext *pivotOrUnpivotClause();
    SubspaceClauseContext *subspaceClause();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  DatasetClauseContext* datasetClause();

  class  RenameClauseContext : public antlr4::ParserRuleContext {
  public:
    RenameClauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *RENAME();
    std::vector<RenameClauseItemContext *> renameClauseItem();
    RenameClauseItemContext* renameClauseItem(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  RenameClauseContext* renameClause();

  class  AggrClauseContext : public antlr4::ParserRuleContext {
  public:
    AggrClauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *AGGREGATE();
    AggregateClauseContext *aggregateClause();
    GroupingClauseContext *groupingClause();
    HavingClauseContext *havingClause();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  AggrClauseContext* aggrClause();

  class  FilterClauseContext : public antlr4::ParserRuleContext {
  public:
    FilterClauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *FILTER();
    ExprComponentContext *exprComponent();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  FilterClauseContext* filterClause();

  class  CalcClauseContext : public antlr4::ParserRuleContext {
  public:
    CalcClauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *CALC();
    std::vector<CalcClauseItemContext *> calcClauseItem();
    CalcClauseItemContext* calcClauseItem(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  CalcClauseContext* calcClause();

  class  KeepOrDropClauseContext : public antlr4::ParserRuleContext {
  public:
    antlr4::Token *op = nullptr;
    KeepOrDropClauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ComponentIDContext *> componentID();
    ComponentIDContext* componentID(size_t i);
    antlr4::tree::TerminalNode *KEEP();
    antlr4::tree::TerminalNode *DROP();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  KeepOrDropClauseContext* keepOrDropClause();

  class  PivotOrUnpivotClauseContext : public antlr4::ParserRuleContext {
  public:
    antlr4::Token *op = nullptr;
    VtlParser::ComponentIDContext *id_ = nullptr;
    VtlParser::ComponentIDContext *mea = nullptr;
    PivotOrUnpivotClauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *COMMA();
    std::vector<ComponentIDContext *> componentID();
    ComponentIDContext* componentID(size_t i);
    antlr4::tree::TerminalNode *PIVOT();
    antlr4::tree::TerminalNode *UNPIVOT();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  PivotOrUnpivotClauseContext* pivotOrUnpivotClause();

  class  CustomPivotClauseContext : public antlr4::ParserRuleContext {
  public:
    VtlParser::ComponentIDContext *id_ = nullptr;
    VtlParser::ComponentIDContext *mea = nullptr;
    CustomPivotClauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *CUSTOMPIVOT();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *IN();
    std::vector<ConstantContext *> constant();
    ConstantContext* constant(size_t i);
    std::vector<ComponentIDContext *> componentID();
    ComponentIDContext* componentID(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  CustomPivotClauseContext* customPivotClause();

  class  SubspaceClauseContext : public antlr4::ParserRuleContext {
  public:
    SubspaceClauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *SUBSPACE();
    std::vector<SubspaceClauseItemContext *> subspaceClauseItem();
    SubspaceClauseItemContext* subspaceClauseItem(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  SubspaceClauseContext* subspaceClause();

  class  JoinOperatorsContext : public antlr4::ParserRuleContext {
  public:
    JoinOperatorsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    JoinOperatorsContext() = default;
    void copyFrom(JoinOperatorsContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  JoinExprContext : public JoinOperatorsContext {
  public:
    JoinExprContext(JoinOperatorsContext *ctx);

    antlr4::Token *joinKeyword = nullptr;
    antlr4::tree::TerminalNode *LPAREN();
    JoinClauseContext *joinClause();
    JoinBodyContext *joinBody();
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *INNER_JOIN();
    antlr4::tree::TerminalNode *LEFT_JOIN();
    JoinClauseWithoutUsingContext *joinClauseWithoutUsing();
    antlr4::tree::TerminalNode *FULL_JOIN();
    antlr4::tree::TerminalNode *CROSS_JOIN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  JoinOperatorsContext* joinOperators();

  class  DefOperatorsContext : public antlr4::ParserRuleContext {
  public:
    DefOperatorsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    DefOperatorsContext() = default;
    void copyFrom(DefOperatorsContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  DefOperatorContext : public DefOperatorsContext {
  public:
    DefOperatorContext(DefOperatorsContext *ctx);

    antlr4::tree::TerminalNode *DEFINE();
    std::vector<antlr4::tree::TerminalNode *> OPERATOR();
    antlr4::tree::TerminalNode* OPERATOR(size_t i);
    OperatorIDContext *operatorID();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *IS();
    antlr4::tree::TerminalNode *END();
    ExprContext *expr();
    std::vector<ParameterItemContext *> parameterItem();
    ParameterItemContext* parameterItem(size_t i);
    antlr4::tree::TerminalNode *RETURNS();
    OutputParameterTypeContext *outputParameterType();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  DefHierarchicalContext : public DefOperatorsContext {
  public:
    DefHierarchicalContext(DefOperatorsContext *ctx);

    antlr4::tree::TerminalNode *DEFINE();
    std::vector<antlr4::tree::TerminalNode *> HIERARCHICAL();
    antlr4::tree::TerminalNode* HIERARCHICAL(size_t i);
    std::vector<antlr4::tree::TerminalNode *> RULESET();
    antlr4::tree::TerminalNode* RULESET(size_t i);
    RulesetIDContext *rulesetID();
    antlr4::tree::TerminalNode *LPAREN();
    HierRuleSignatureContext *hierRuleSignature();
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *IS();
    RuleClauseHierarchicalContext *ruleClauseHierarchical();
    antlr4::tree::TerminalNode *END();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  DefDatapointRulesetContext : public DefOperatorsContext {
  public:
    DefDatapointRulesetContext(DefOperatorsContext *ctx);

    antlr4::tree::TerminalNode *DEFINE();
    std::vector<antlr4::tree::TerminalNode *> DATAPOINT();
    antlr4::tree::TerminalNode* DATAPOINT(size_t i);
    std::vector<antlr4::tree::TerminalNode *> RULESET();
    antlr4::tree::TerminalNode* RULESET(size_t i);
    RulesetIDContext *rulesetID();
    antlr4::tree::TerminalNode *LPAREN();
    RulesetSignatureContext *rulesetSignature();
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *IS();
    RuleClauseDatapointContext *ruleClauseDatapoint();
    antlr4::tree::TerminalNode *END();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  DefOperatorsContext* defOperators();

  class  GenericOperatorsContext : public antlr4::ParserRuleContext {
  public:
    GenericOperatorsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    GenericOperatorsContext() = default;
    void copyFrom(GenericOperatorsContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  EvalAtomContext : public GenericOperatorsContext {
  public:
    EvalAtomContext(GenericOperatorsContext *ctx);

    antlr4::tree::TerminalNode *EVAL();
    std::vector<antlr4::tree::TerminalNode *> LPAREN();
    antlr4::tree::TerminalNode* LPAREN(size_t i);
    RoutineNameContext *routineName();
    std::vector<antlr4::tree::TerminalNode *> RPAREN();
    antlr4::tree::TerminalNode* RPAREN(size_t i);
    std::vector<VarIDContext *> varID();
    VarIDContext* varID(size_t i);
    std::vector<ScalarItemContext *> scalarItem();
    ScalarItemContext* scalarItem(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *LANGUAGE();
    antlr4::tree::TerminalNode *STRING_CONSTANT();
    antlr4::tree::TerminalNode *RETURNS();
    EvalDatasetTypeContext *evalDatasetType();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  CastExprDatasetContext : public GenericOperatorsContext {
  public:
    CastExprDatasetContext(GenericOperatorsContext *ctx);

    antlr4::tree::TerminalNode *CAST();
    antlr4::tree::TerminalNode *LPAREN();
    ExprContext *expr();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *RPAREN();
    BasicScalarTypeContext *basicScalarType();
    ValueDomainNameContext *valueDomainName();
    antlr4::tree::TerminalNode *STRING_CONSTANT();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  CallDatasetContext : public GenericOperatorsContext {
  public:
    CallDatasetContext(GenericOperatorsContext *ctx);

    OperatorIDContext *operatorID();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<ParameterContext *> parameter();
    ParameterContext* parameter(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  GenericOperatorsContext* genericOperators();

  class  GenericOperatorsComponentContext : public antlr4::ParserRuleContext {
  public:
    GenericOperatorsComponentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    GenericOperatorsComponentContext() = default;
    void copyFrom(GenericOperatorsComponentContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  EvalAtomComponentContext : public GenericOperatorsComponentContext {
  public:
    EvalAtomComponentContext(GenericOperatorsComponentContext *ctx);

    antlr4::tree::TerminalNode *EVAL();
    std::vector<antlr4::tree::TerminalNode *> LPAREN();
    antlr4::tree::TerminalNode* LPAREN(size_t i);
    RoutineNameContext *routineName();
    std::vector<antlr4::tree::TerminalNode *> RPAREN();
    antlr4::tree::TerminalNode* RPAREN(size_t i);
    std::vector<ComponentIDContext *> componentID();
    ComponentIDContext* componentID(size_t i);
    std::vector<ScalarItemContext *> scalarItem();
    ScalarItemContext* scalarItem(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *LANGUAGE();
    antlr4::tree::TerminalNode *STRING_CONSTANT();
    antlr4::tree::TerminalNode *RETURNS();
    OutputParameterTypeComponentContext *outputParameterTypeComponent();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  CastExprComponentContext : public GenericOperatorsComponentContext {
  public:
    CastExprComponentContext(GenericOperatorsComponentContext *ctx);

    antlr4::tree::TerminalNode *CAST();
    antlr4::tree::TerminalNode *LPAREN();
    ExprComponentContext *exprComponent();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *RPAREN();
    BasicScalarTypeContext *basicScalarType();
    ValueDomainNameContext *valueDomainName();
    antlr4::tree::TerminalNode *STRING_CONSTANT();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  CallComponentContext : public GenericOperatorsComponentContext {
  public:
    CallComponentContext(GenericOperatorsComponentContext *ctx);

    OperatorIDContext *operatorID();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<ParameterComponentContext *> parameterComponent();
    ParameterComponentContext* parameterComponent(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  GenericOperatorsComponentContext* genericOperatorsComponent();

  class  ParameterComponentContext : public antlr4::ParserRuleContext {
  public:
    ParameterComponentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExprComponentContext *exprComponent();
    antlr4::tree::TerminalNode *OPTIONAL();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ParameterComponentContext* parameterComponent();

  class  ParameterContext : public antlr4::ParserRuleContext {
  public:
    ParameterContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExprContext *expr();
    antlr4::tree::TerminalNode *OPTIONAL();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ParameterContext* parameter();

  class  StringOperatorsContext : public antlr4::ParserRuleContext {
  public:
    StringOperatorsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    StringOperatorsContext() = default;
    void copyFrom(StringOperatorsContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  InstrAtomContext : public StringOperatorsContext {
  public:
    InstrAtomContext(StringOperatorsContext *ctx);

    VtlParser::ExprContext *pattern = nullptr;
    VtlParser::OptionalExprContext *startParameter = nullptr;
    VtlParser::OptionalExprContext *occurrenceParameter = nullptr;
    antlr4::tree::TerminalNode *INSTR();
    antlr4::tree::TerminalNode *LPAREN();
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<OptionalExprContext *> optionalExpr();
    OptionalExprContext* optionalExpr(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  UnaryStringFunctionContext : public StringOperatorsContext {
  public:
    UnaryStringFunctionContext(StringOperatorsContext *ctx);

    antlr4::Token *op = nullptr;
    antlr4::tree::TerminalNode *LPAREN();
    ExprContext *expr();
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *TRIM();
    antlr4::tree::TerminalNode *LTRIM();
    antlr4::tree::TerminalNode *RTRIM();
    antlr4::tree::TerminalNode *UCASE();
    antlr4::tree::TerminalNode *LCASE();
    antlr4::tree::TerminalNode *LEN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  SubstrAtomContext : public StringOperatorsContext {
  public:
    SubstrAtomContext(StringOperatorsContext *ctx);

    VtlParser::OptionalExprContext *startParameter = nullptr;
    VtlParser::OptionalExprContext *endParameter = nullptr;
    antlr4::tree::TerminalNode *SUBSTR();
    antlr4::tree::TerminalNode *LPAREN();
    ExprContext *expr();
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    std::vector<OptionalExprContext *> optionalExpr();
    OptionalExprContext* optionalExpr(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ReplaceAtomContext : public StringOperatorsContext {
  public:
    ReplaceAtomContext(StringOperatorsContext *ctx);

    VtlParser::ExprContext *param = nullptr;
    antlr4::tree::TerminalNode *REPLACE();
    antlr4::tree::TerminalNode *LPAREN();
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *RPAREN();
    OptionalExprContext *optionalExpr();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  StringOperatorsContext* stringOperators();

  class  StringOperatorsComponentContext : public antlr4::ParserRuleContext {
  public:
    StringOperatorsComponentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    StringOperatorsComponentContext() = default;
    void copyFrom(StringOperatorsComponentContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  ReplaceAtomComponentContext : public StringOperatorsComponentContext {
  public:
    ReplaceAtomComponentContext(StringOperatorsComponentContext *ctx);

    VtlParser::ExprComponentContext *param = nullptr;
    antlr4::tree::TerminalNode *REPLACE();
    antlr4::tree::TerminalNode *LPAREN();
    std::vector<ExprComponentContext *> exprComponent();
    ExprComponentContext* exprComponent(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *RPAREN();
    OptionalExprComponentContext *optionalExprComponent();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  UnaryStringFunctionComponentContext : public StringOperatorsComponentContext {
  public:
    UnaryStringFunctionComponentContext(StringOperatorsComponentContext *ctx);

    antlr4::Token *op = nullptr;
    antlr4::tree::TerminalNode *LPAREN();
    ExprComponentContext *exprComponent();
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *TRIM();
    antlr4::tree::TerminalNode *LTRIM();
    antlr4::tree::TerminalNode *RTRIM();
    antlr4::tree::TerminalNode *UCASE();
    antlr4::tree::TerminalNode *LCASE();
    antlr4::tree::TerminalNode *LEN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  SubstrAtomComponentContext : public StringOperatorsComponentContext {
  public:
    SubstrAtomComponentContext(StringOperatorsComponentContext *ctx);

    VtlParser::OptionalExprComponentContext *startParameter = nullptr;
    VtlParser::OptionalExprComponentContext *endParameter = nullptr;
    antlr4::tree::TerminalNode *SUBSTR();
    antlr4::tree::TerminalNode *LPAREN();
    ExprComponentContext *exprComponent();
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    std::vector<OptionalExprComponentContext *> optionalExprComponent();
    OptionalExprComponentContext* optionalExprComponent(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  InstrAtomComponentContext : public StringOperatorsComponentContext {
  public:
    InstrAtomComponentContext(StringOperatorsComponentContext *ctx);

    VtlParser::ExprComponentContext *pattern = nullptr;
    VtlParser::OptionalExprComponentContext *startParameter = nullptr;
    VtlParser::OptionalExprComponentContext *occurrenceParameter = nullptr;
    antlr4::tree::TerminalNode *INSTR();
    antlr4::tree::TerminalNode *LPAREN();
    std::vector<ExprComponentContext *> exprComponent();
    ExprComponentContext* exprComponent(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<OptionalExprComponentContext *> optionalExprComponent();
    OptionalExprComponentContext* optionalExprComponent(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  StringOperatorsComponentContext* stringOperatorsComponent();

  class  NumericOperatorsContext : public antlr4::ParserRuleContext {
  public:
    NumericOperatorsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    NumericOperatorsContext() = default;
    void copyFrom(NumericOperatorsContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  UnaryNumericContext : public NumericOperatorsContext {
  public:
    UnaryNumericContext(NumericOperatorsContext *ctx);

    antlr4::Token *op = nullptr;
    antlr4::tree::TerminalNode *LPAREN();
    ExprContext *expr();
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *CEIL();
    antlr4::tree::TerminalNode *FLOOR();
    antlr4::tree::TerminalNode *ABS();
    antlr4::tree::TerminalNode *EXP();
    antlr4::tree::TerminalNode *LN();
    antlr4::tree::TerminalNode *SQRT();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  UnaryWithOptionalNumericContext : public NumericOperatorsContext {
  public:
    UnaryWithOptionalNumericContext(NumericOperatorsContext *ctx);

    antlr4::Token *op = nullptr;
    antlr4::tree::TerminalNode *LPAREN();
    ExprContext *expr();
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *ROUND();
    antlr4::tree::TerminalNode *TRUNC();
    antlr4::tree::TerminalNode *COMMA();
    OptionalExprContext *optionalExpr();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  BinaryNumericContext : public NumericOperatorsContext {
  public:
    BinaryNumericContext(NumericOperatorsContext *ctx);

    antlr4::Token *op = nullptr;
    VtlParser::ExprContext *left = nullptr;
    VtlParser::ExprContext *right = nullptr;
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *COMMA();
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    antlr4::tree::TerminalNode *MOD();
    antlr4::tree::TerminalNode *POWER();
    antlr4::tree::TerminalNode *LOG();
    antlr4::tree::TerminalNode *RANDOM();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  NumericOperatorsContext* numericOperators();

  class  NumericOperatorsComponentContext : public antlr4::ParserRuleContext {
  public:
    NumericOperatorsComponentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    NumericOperatorsComponentContext() = default;
    void copyFrom(NumericOperatorsComponentContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  UnaryNumericComponentContext : public NumericOperatorsComponentContext {
  public:
    UnaryNumericComponentContext(NumericOperatorsComponentContext *ctx);

    antlr4::Token *op = nullptr;
    antlr4::tree::TerminalNode *LPAREN();
    ExprComponentContext *exprComponent();
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *CEIL();
    antlr4::tree::TerminalNode *FLOOR();
    antlr4::tree::TerminalNode *ABS();
    antlr4::tree::TerminalNode *EXP();
    antlr4::tree::TerminalNode *LN();
    antlr4::tree::TerminalNode *SQRT();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  BinaryNumericComponentContext : public NumericOperatorsComponentContext {
  public:
    BinaryNumericComponentContext(NumericOperatorsComponentContext *ctx);

    antlr4::Token *op = nullptr;
    VtlParser::ExprComponentContext *left = nullptr;
    VtlParser::ExprComponentContext *right = nullptr;
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *COMMA();
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<ExprComponentContext *> exprComponent();
    ExprComponentContext* exprComponent(size_t i);
    antlr4::tree::TerminalNode *MOD();
    antlr4::tree::TerminalNode *POWER();
    antlr4::tree::TerminalNode *LOG();
    antlr4::tree::TerminalNode *RANDOM();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  UnaryWithOptionalNumericComponentContext : public NumericOperatorsComponentContext {
  public:
    UnaryWithOptionalNumericComponentContext(NumericOperatorsComponentContext *ctx);

    antlr4::Token *op = nullptr;
    antlr4::tree::TerminalNode *LPAREN();
    ExprComponentContext *exprComponent();
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *ROUND();
    antlr4::tree::TerminalNode *TRUNC();
    antlr4::tree::TerminalNode *COMMA();
    OptionalExprComponentContext *optionalExprComponent();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  NumericOperatorsComponentContext* numericOperatorsComponent();

  class  ComparisonOperatorsContext : public antlr4::ParserRuleContext {
  public:
    ComparisonOperatorsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    ComparisonOperatorsContext() = default;
    void copyFrom(ComparisonOperatorsContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  BetweenAtomContext : public ComparisonOperatorsContext {
  public:
    BetweenAtomContext(ComparisonOperatorsContext *ctx);

    VtlParser::ExprContext *op = nullptr;
    VtlParser::ExprContext *from_ = nullptr;
    VtlParser::ExprContext *to_ = nullptr;
    antlr4::tree::TerminalNode *BETWEEN();
    antlr4::tree::TerminalNode *LPAREN();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  CharsetMatchAtomContext : public ComparisonOperatorsContext {
  public:
    CharsetMatchAtomContext(ComparisonOperatorsContext *ctx);

    VtlParser::ExprContext *op = nullptr;
    VtlParser::ExprContext *pattern = nullptr;
    antlr4::tree::TerminalNode *CHARSET_MATCH();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *COMMA();
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  IsNullAtomContext : public ComparisonOperatorsContext {
  public:
    IsNullAtomContext(ComparisonOperatorsContext *ctx);

    antlr4::tree::TerminalNode *ISNULL();
    antlr4::tree::TerminalNode *LPAREN();
    ExprContext *expr();
    antlr4::tree::TerminalNode *RPAREN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ExistInAtomContext : public ComparisonOperatorsContext {
  public:
    ExistInAtomContext(ComparisonOperatorsContext *ctx);

    VtlParser::ExprContext *left = nullptr;
    VtlParser::ExprContext *right = nullptr;
    antlr4::tree::TerminalNode *EXISTS_IN();
    antlr4::tree::TerminalNode *LPAREN();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    RetainTypeContext *retainType();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ComparisonOperatorsContext* comparisonOperators();

  class  ComparisonOperatorsComponentContext : public antlr4::ParserRuleContext {
  public:
    ComparisonOperatorsComponentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    ComparisonOperatorsComponentContext() = default;
    void copyFrom(ComparisonOperatorsComponentContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  IsNullAtomComponentContext : public ComparisonOperatorsComponentContext {
  public:
    IsNullAtomComponentContext(ComparisonOperatorsComponentContext *ctx);

    antlr4::tree::TerminalNode *ISNULL();
    antlr4::tree::TerminalNode *LPAREN();
    ExprComponentContext *exprComponent();
    antlr4::tree::TerminalNode *RPAREN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  CharsetMatchAtomComponentContext : public ComparisonOperatorsComponentContext {
  public:
    CharsetMatchAtomComponentContext(ComparisonOperatorsComponentContext *ctx);

    VtlParser::ExprComponentContext *op = nullptr;
    VtlParser::ExprComponentContext *pattern = nullptr;
    antlr4::tree::TerminalNode *CHARSET_MATCH();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *COMMA();
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<ExprComponentContext *> exprComponent();
    ExprComponentContext* exprComponent(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  BetweenAtomComponentContext : public ComparisonOperatorsComponentContext {
  public:
    BetweenAtomComponentContext(ComparisonOperatorsComponentContext *ctx);

    VtlParser::ExprComponentContext *op = nullptr;
    VtlParser::ExprComponentContext *from_ = nullptr;
    VtlParser::ExprComponentContext *to_ = nullptr;
    antlr4::tree::TerminalNode *BETWEEN();
    antlr4::tree::TerminalNode *LPAREN();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<ExprComponentContext *> exprComponent();
    ExprComponentContext* exprComponent(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ComparisonOperatorsComponentContext* comparisonOperatorsComponent();

  class  TimeOperatorsContext : public antlr4::ParserRuleContext {
  public:
    TimeOperatorsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    TimeOperatorsContext() = default;
    void copyFrom(TimeOperatorsContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  DayToYearAtomContext : public TimeOperatorsContext {
  public:
    DayToYearAtomContext(TimeOperatorsContext *ctx);

    antlr4::tree::TerminalNode *DAYTOYEAR();
    antlr4::tree::TerminalNode *LPAREN();
    ExprContext *expr();
    antlr4::tree::TerminalNode *RPAREN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  YearAtomContext : public TimeOperatorsContext {
  public:
    YearAtomContext(TimeOperatorsContext *ctx);

    antlr4::tree::TerminalNode *YEAR_OP();
    antlr4::tree::TerminalNode *LPAREN();
    ExprContext *expr();
    antlr4::tree::TerminalNode *RPAREN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  YearTodayAtomContext : public TimeOperatorsContext {
  public:
    YearTodayAtomContext(TimeOperatorsContext *ctx);

    antlr4::tree::TerminalNode *YEARTODAY();
    antlr4::tree::TerminalNode *LPAREN();
    ExprContext *expr();
    antlr4::tree::TerminalNode *RPAREN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  DayToMonthAtomContext : public TimeOperatorsContext {
  public:
    DayToMonthAtomContext(TimeOperatorsContext *ctx);

    antlr4::tree::TerminalNode *DAYTOMONTH();
    antlr4::tree::TerminalNode *LPAREN();
    ExprContext *expr();
    antlr4::tree::TerminalNode *RPAREN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  PeriodAtomContext : public TimeOperatorsContext {
  public:
    PeriodAtomContext(TimeOperatorsContext *ctx);

    antlr4::tree::TerminalNode *PERIOD_INDICATOR();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    ExprContext *expr();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  MonthTodayAtomContext : public TimeOperatorsContext {
  public:
    MonthTodayAtomContext(TimeOperatorsContext *ctx);

    antlr4::tree::TerminalNode *MONTHTODAY();
    antlr4::tree::TerminalNode *LPAREN();
    ExprContext *expr();
    antlr4::tree::TerminalNode *RPAREN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  FillTimeAtomContext : public TimeOperatorsContext {
  public:
    FillTimeAtomContext(TimeOperatorsContext *ctx);

    antlr4::Token *op = nullptr;
    antlr4::tree::TerminalNode *FILL_TIME_SERIES();
    antlr4::tree::TerminalNode *LPAREN();
    ExprContext *expr();
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *COMMA();
    antlr4::tree::TerminalNode *SINGLE();
    antlr4::tree::TerminalNode *ALL();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  MonthAtomContext : public TimeOperatorsContext {
  public:
    MonthAtomContext(TimeOperatorsContext *ctx);

    antlr4::tree::TerminalNode *MONTH_OP();
    antlr4::tree::TerminalNode *LPAREN();
    ExprContext *expr();
    antlr4::tree::TerminalNode *RPAREN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  DayOfYearAtomContext : public TimeOperatorsContext {
  public:
    DayOfYearAtomContext(TimeOperatorsContext *ctx);

    antlr4::tree::TerminalNode *DAYOFYEAR();
    antlr4::tree::TerminalNode *LPAREN();
    ExprContext *expr();
    antlr4::tree::TerminalNode *RPAREN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  FlowAtomContext : public TimeOperatorsContext {
  public:
    FlowAtomContext(TimeOperatorsContext *ctx);

    antlr4::Token *op = nullptr;
    antlr4::tree::TerminalNode *LPAREN();
    ExprContext *expr();
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *FLOW_TO_STOCK();
    antlr4::tree::TerminalNode *STOCK_TO_FLOW();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  TimeShiftAtomContext : public TimeOperatorsContext {
  public:
    TimeShiftAtomContext(TimeOperatorsContext *ctx);

    antlr4::tree::TerminalNode *TIMESHIFT();
    antlr4::tree::TerminalNode *LPAREN();
    ExprContext *expr();
    antlr4::tree::TerminalNode *COMMA();
    SignedIntegerContext *signedInteger();
    antlr4::tree::TerminalNode *RPAREN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  TimeAggAtomContext : public TimeOperatorsContext {
  public:
    TimeAggAtomContext(TimeOperatorsContext *ctx);

    antlr4::Token *periodIndTo = nullptr;
    antlr4::Token *periodIndFrom = nullptr;
    VtlParser::OptionalExprContext *op = nullptr;
    antlr4::Token *delim = nullptr;
    antlr4::tree::TerminalNode *TIME_AGG();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<antlr4::tree::TerminalNode *> STRING_CONSTANT();
    antlr4::tree::TerminalNode* STRING_CONSTANT(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    OptionalExprContext *optionalExpr();
    antlr4::tree::TerminalNode *OPTIONAL();
    antlr4::tree::TerminalNode *FIRST();
    antlr4::tree::TerminalNode *LAST();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  DateDiffAtomContext : public TimeOperatorsContext {
  public:
    DateDiffAtomContext(TimeOperatorsContext *ctx);

    VtlParser::ExprContext *dateFrom = nullptr;
    VtlParser::ExprContext *dateTo = nullptr;
    antlr4::tree::TerminalNode *DATEDIFF();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *COMMA();
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  DateAddAtomContext : public TimeOperatorsContext {
  public:
    DateAddAtomContext(TimeOperatorsContext *ctx);

    VtlParser::ExprContext *op = nullptr;
    VtlParser::ExprContext *shiftNumber = nullptr;
    VtlParser::ExprContext *periodInd = nullptr;
    antlr4::tree::TerminalNode *DATEADD();
    antlr4::tree::TerminalNode *LPAREN();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  DayOfMonthAtomContext : public TimeOperatorsContext {
  public:
    DayOfMonthAtomContext(TimeOperatorsContext *ctx);

    antlr4::tree::TerminalNode *DAYOFMONTH();
    antlr4::tree::TerminalNode *LPAREN();
    ExprContext *expr();
    antlr4::tree::TerminalNode *RPAREN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  CurrentDateAtomContext : public TimeOperatorsContext {
  public:
    CurrentDateAtomContext(TimeOperatorsContext *ctx);

    antlr4::tree::TerminalNode *CURRENT_DATE();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TimeOperatorsContext* timeOperators();

  class  TimeOperatorsComponentContext : public antlr4::ParserRuleContext {
  public:
    TimeOperatorsComponentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    TimeOperatorsComponentContext() = default;
    void copyFrom(TimeOperatorsComponentContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  PeriodAtomComponentContext : public TimeOperatorsComponentContext {
  public:
    PeriodAtomComponentContext(TimeOperatorsComponentContext *ctx);

    antlr4::tree::TerminalNode *PERIOD_INDICATOR();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    ExprComponentContext *exprComponent();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  TimeShiftAtomComponentContext : public TimeOperatorsComponentContext {
  public:
    TimeShiftAtomComponentContext(TimeOperatorsComponentContext *ctx);

    antlr4::tree::TerminalNode *TIMESHIFT();
    antlr4::tree::TerminalNode *LPAREN();
    ExprComponentContext *exprComponent();
    antlr4::tree::TerminalNode *COMMA();
    SignedIntegerContext *signedInteger();
    antlr4::tree::TerminalNode *RPAREN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  MonthTodayAtomComponentContext : public TimeOperatorsComponentContext {
  public:
    MonthTodayAtomComponentContext(TimeOperatorsComponentContext *ctx);

    antlr4::tree::TerminalNode *MONTHTODAY();
    antlr4::tree::TerminalNode *LPAREN();
    ExprComponentContext *exprComponent();
    antlr4::tree::TerminalNode *RPAREN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  TimeAggAtomComponentContext : public TimeOperatorsComponentContext {
  public:
    TimeAggAtomComponentContext(TimeOperatorsComponentContext *ctx);

    antlr4::Token *periodIndTo = nullptr;
    antlr4::Token *periodIndFrom = nullptr;
    VtlParser::OptionalExprComponentContext *op = nullptr;
    antlr4::Token *delim = nullptr;
    antlr4::tree::TerminalNode *TIME_AGG();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<antlr4::tree::TerminalNode *> STRING_CONSTANT();
    antlr4::tree::TerminalNode* STRING_CONSTANT(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    OptionalExprComponentContext *optionalExprComponent();
    antlr4::tree::TerminalNode *OPTIONAL();
    antlr4::tree::TerminalNode *FIRST();
    antlr4::tree::TerminalNode *LAST();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  DayToMonthAtomComponentContext : public TimeOperatorsComponentContext {
  public:
    DayToMonthAtomComponentContext(TimeOperatorsComponentContext *ctx);

    antlr4::tree::TerminalNode *DAYTOMONTH();
    antlr4::tree::TerminalNode *LPAREN();
    ExprComponentContext *exprComponent();
    antlr4::tree::TerminalNode *RPAREN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  DateAddAtomComponentContext : public TimeOperatorsComponentContext {
  public:
    DateAddAtomComponentContext(TimeOperatorsComponentContext *ctx);

    VtlParser::ExprComponentContext *op = nullptr;
    VtlParser::ExprComponentContext *shiftNumber = nullptr;
    VtlParser::ExprComponentContext *periodInd = nullptr;
    antlr4::tree::TerminalNode *DATEADD();
    antlr4::tree::TerminalNode *LPAREN();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<ExprComponentContext *> exprComponent();
    ExprComponentContext* exprComponent(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  YearTodayAtomComponentContext : public TimeOperatorsComponentContext {
  public:
    YearTodayAtomComponentContext(TimeOperatorsComponentContext *ctx);

    antlr4::tree::TerminalNode *YEARTODAY();
    antlr4::tree::TerminalNode *LPAREN();
    ExprComponentContext *exprComponent();
    antlr4::tree::TerminalNode *RPAREN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  DayOfMonthAtomComponentContext : public TimeOperatorsComponentContext {
  public:
    DayOfMonthAtomComponentContext(TimeOperatorsComponentContext *ctx);

    antlr4::tree::TerminalNode *DAYOFMONTH();
    antlr4::tree::TerminalNode *LPAREN();
    ExprComponentContext *exprComponent();
    antlr4::tree::TerminalNode *RPAREN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  MonthAtomComponentContext : public TimeOperatorsComponentContext {
  public:
    MonthAtomComponentContext(TimeOperatorsComponentContext *ctx);

    antlr4::tree::TerminalNode *MONTH_OP();
    antlr4::tree::TerminalNode *LPAREN();
    ExprComponentContext *exprComponent();
    antlr4::tree::TerminalNode *RPAREN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  FillTimeAtomComponentContext : public TimeOperatorsComponentContext {
  public:
    FillTimeAtomComponentContext(TimeOperatorsComponentContext *ctx);

    antlr4::Token *op = nullptr;
    antlr4::tree::TerminalNode *FILL_TIME_SERIES();
    antlr4::tree::TerminalNode *LPAREN();
    ExprComponentContext *exprComponent();
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *COMMA();
    antlr4::tree::TerminalNode *SINGLE();
    antlr4::tree::TerminalNode *ALL();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  DatOfYearAtomComponentContext : public TimeOperatorsComponentContext {
  public:
    DatOfYearAtomComponentContext(TimeOperatorsComponentContext *ctx);

    antlr4::tree::TerminalNode *DAYOFYEAR();
    antlr4::tree::TerminalNode *LPAREN();
    ExprComponentContext *exprComponent();
    antlr4::tree::TerminalNode *RPAREN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  DayToYearAtomComponentContext : public TimeOperatorsComponentContext {
  public:
    DayToYearAtomComponentContext(TimeOperatorsComponentContext *ctx);

    antlr4::tree::TerminalNode *DAYTOYEAR();
    antlr4::tree::TerminalNode *LPAREN();
    ExprComponentContext *exprComponent();
    antlr4::tree::TerminalNode *RPAREN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  CurrentDateAtomComponentContext : public TimeOperatorsComponentContext {
  public:
    CurrentDateAtomComponentContext(TimeOperatorsComponentContext *ctx);

    antlr4::tree::TerminalNode *CURRENT_DATE();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  FlowAtomComponentContext : public TimeOperatorsComponentContext {
  public:
    FlowAtomComponentContext(TimeOperatorsComponentContext *ctx);

    antlr4::Token *op = nullptr;
    antlr4::tree::TerminalNode *LPAREN();
    ExprComponentContext *exprComponent();
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *FLOW_TO_STOCK();
    antlr4::tree::TerminalNode *STOCK_TO_FLOW();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  DateDiffAtomComponentContext : public TimeOperatorsComponentContext {
  public:
    DateDiffAtomComponentContext(TimeOperatorsComponentContext *ctx);

    VtlParser::ExprComponentContext *dateFrom = nullptr;
    VtlParser::ExprContext *dateTo = nullptr;
    antlr4::tree::TerminalNode *DATEDIFF();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *COMMA();
    antlr4::tree::TerminalNode *RPAREN();
    ExprComponentContext *exprComponent();
    ExprContext *expr();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  YearAtomComponentContext : public TimeOperatorsComponentContext {
  public:
    YearAtomComponentContext(TimeOperatorsComponentContext *ctx);

    antlr4::tree::TerminalNode *YEAR_OP();
    antlr4::tree::TerminalNode *LPAREN();
    ExprComponentContext *exprComponent();
    antlr4::tree::TerminalNode *RPAREN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  TimeOperatorsComponentContext* timeOperatorsComponent();

  class  SetOperatorsContext : public antlr4::ParserRuleContext {
  public:
    SetOperatorsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    SetOperatorsContext() = default;
    void copyFrom(SetOperatorsContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  SetOrSYmDiffAtomContext : public SetOperatorsContext {
  public:
    SetOrSYmDiffAtomContext(SetOperatorsContext *ctx);

    antlr4::Token *op = nullptr;
    VtlParser::ExprContext *left = nullptr;
    VtlParser::ExprContext *right = nullptr;
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *COMMA();
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    antlr4::tree::TerminalNode *SETDIFF();
    antlr4::tree::TerminalNode *SYMDIFF();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  IntersectAtomContext : public SetOperatorsContext {
  public:
    IntersectAtomContext(SetOperatorsContext *ctx);

    VtlParser::ExprContext *left = nullptr;
    antlr4::tree::TerminalNode *INTERSECT();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  UnionAtomContext : public SetOperatorsContext {
  public:
    UnionAtomContext(SetOperatorsContext *ctx);

    VtlParser::ExprContext *left = nullptr;
    antlr4::tree::TerminalNode *UNION();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  SetOperatorsContext* setOperators();

  class  HierarchyOperatorsContext : public antlr4::ParserRuleContext {
  public:
    VtlParser::ExprContext *op = nullptr;
    antlr4::Token *hrName = nullptr;
    VtlParser::ComponentIDContext *ruleComponent = nullptr;
    HierarchyOperatorsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *HIERARCHY();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *COMMA();
    antlr4::tree::TerminalNode *RPAREN();
    ExprContext *expr();
    antlr4::tree::TerminalNode *IDENTIFIER();
    ConditionClauseContext *conditionClause();
    antlr4::tree::TerminalNode *RULE();
    ValidationModeContext *validationMode();
    InputModeHierarchyContext *inputModeHierarchy();
    OutputModeHierarchyContext *outputModeHierarchy();
    ComponentIDContext *componentID();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  HierarchyOperatorsContext* hierarchyOperators();

  class  ValidationOperatorsContext : public antlr4::ParserRuleContext {
  public:
    ValidationOperatorsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    ValidationOperatorsContext() = default;
    void copyFrom(ValidationOperatorsContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  ValidateHRrulesetContext : public ValidationOperatorsContext {
  public:
    ValidateHRrulesetContext(ValidationOperatorsContext *ctx);

    VtlParser::ExprContext *op = nullptr;
    antlr4::Token *hrName = nullptr;
    antlr4::tree::TerminalNode *CHECK_HIERARCHY();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *COMMA();
    antlr4::tree::TerminalNode *RPAREN();
    ExprContext *expr();
    antlr4::tree::TerminalNode *IDENTIFIER();
    ConditionClauseContext *conditionClause();
    antlr4::tree::TerminalNode *RULE();
    ComponentIDContext *componentID();
    ValidationModeContext *validationMode();
    InputModeContext *inputMode();
    ValidationOutputContext *validationOutput();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ValidateDPrulesetContext : public ValidationOperatorsContext {
  public:
    ValidateDPrulesetContext(ValidationOperatorsContext *ctx);

    VtlParser::ExprContext *op = nullptr;
    antlr4::Token *dpName = nullptr;
    antlr4::tree::TerminalNode *CHECK_DATAPOINT();
    antlr4::tree::TerminalNode *LPAREN();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *RPAREN();
    ExprContext *expr();
    antlr4::tree::TerminalNode *IDENTIFIER();
    antlr4::tree::TerminalNode *COMPONENTS();
    std::vector<ComponentIDContext *> componentID();
    ComponentIDContext* componentID(size_t i);
    ValidationOutputContext *validationOutput();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ValidationSimpleContext : public ValidationOperatorsContext {
  public:
    ValidationSimpleContext(ValidationOperatorsContext *ctx);

    VtlParser::ExprContext *op = nullptr;
    VtlParser::ErCodeContext *codeErr = nullptr;
    VtlParser::ErLevelContext *levelCode = nullptr;
    antlr4::Token *output = nullptr;
    antlr4::tree::TerminalNode *CHECK();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    ExprContext *expr();
    ImbalanceExprContext *imbalanceExpr();
    ErCodeContext *erCode();
    ErLevelContext *erLevel();
    antlr4::tree::TerminalNode *INVALID();
    antlr4::tree::TerminalNode *ALL();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ValidationOperatorsContext* validationOperators();

  class  ConditionalOperatorsContext : public antlr4::ParserRuleContext {
  public:
    ConditionalOperatorsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    ConditionalOperatorsContext() = default;
    void copyFrom(ConditionalOperatorsContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  NvlAtomContext : public ConditionalOperatorsContext {
  public:
    NvlAtomContext(ConditionalOperatorsContext *ctx);

    VtlParser::ExprContext *left = nullptr;
    VtlParser::ExprContext *right = nullptr;
    antlr4::tree::TerminalNode *NVL();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *COMMA();
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ConditionalOperatorsContext* conditionalOperators();

  class  ConditionalOperatorsComponentContext : public antlr4::ParserRuleContext {
  public:
    ConditionalOperatorsComponentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    ConditionalOperatorsComponentContext() = default;
    void copyFrom(ConditionalOperatorsComponentContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  NvlAtomComponentContext : public ConditionalOperatorsComponentContext {
  public:
    NvlAtomComponentContext(ConditionalOperatorsComponentContext *ctx);

    VtlParser::ExprComponentContext *left = nullptr;
    VtlParser::ExprComponentContext *right = nullptr;
    antlr4::tree::TerminalNode *NVL();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *COMMA();
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<ExprComponentContext *> exprComponent();
    ExprComponentContext* exprComponent(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ConditionalOperatorsComponentContext* conditionalOperatorsComponent();

  class  AggrOperatorsContext : public antlr4::ParserRuleContext {
  public:
    AggrOperatorsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    AggrOperatorsContext() = default;
    void copyFrom(AggrOperatorsContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  AggrCompContext : public AggrOperatorsContext {
  public:
    AggrCompContext(AggrOperatorsContext *ctx);

    antlr4::Token *op = nullptr;
    antlr4::tree::TerminalNode *LPAREN();
    ExprComponentContext *exprComponent();
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *SUM();
    antlr4::tree::TerminalNode *AVG();
    antlr4::tree::TerminalNode *COUNT();
    antlr4::tree::TerminalNode *MEDIAN();
    antlr4::tree::TerminalNode *MIN();
    antlr4::tree::TerminalNode *MAX();
    antlr4::tree::TerminalNode *STDDEV_POP();
    antlr4::tree::TerminalNode *STDDEV_SAMP();
    antlr4::tree::TerminalNode *VAR_POP();
    antlr4::tree::TerminalNode *VAR_SAMP();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  CountAggrCompContext : public AggrOperatorsContext {
  public:
    CountAggrCompContext(AggrOperatorsContext *ctx);

    antlr4::tree::TerminalNode *COUNT();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  AggrOperatorsContext* aggrOperators();

  class  AggrOperatorsGroupingContext : public antlr4::ParserRuleContext {
  public:
    AggrOperatorsGroupingContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    AggrOperatorsGroupingContext() = default;
    void copyFrom(AggrOperatorsGroupingContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  AggrDatasetContext : public AggrOperatorsGroupingContext {
  public:
    AggrDatasetContext(AggrOperatorsGroupingContext *ctx);

    antlr4::Token *op = nullptr;
    antlr4::tree::TerminalNode *LPAREN();
    ExprContext *expr();
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *SUM();
    antlr4::tree::TerminalNode *AVG();
    antlr4::tree::TerminalNode *COUNT();
    antlr4::tree::TerminalNode *MEDIAN();
    antlr4::tree::TerminalNode *MIN();
    antlr4::tree::TerminalNode *MAX();
    antlr4::tree::TerminalNode *STDDEV_POP();
    antlr4::tree::TerminalNode *STDDEV_SAMP();
    antlr4::tree::TerminalNode *VAR_POP();
    antlr4::tree::TerminalNode *VAR_SAMP();
    GroupingClauseContext *groupingClause();
    HavingClauseContext *havingClause();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  AggrOperatorsGroupingContext* aggrOperatorsGrouping();

  class  AnFunctionContext : public antlr4::ParserRuleContext {
  public:
    AnFunctionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    AnFunctionContext() = default;
    void copyFrom(AnFunctionContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  LagOrLeadAnContext : public AnFunctionContext {
  public:
    LagOrLeadAnContext(AnFunctionContext *ctx);

    antlr4::Token *op = nullptr;
    VtlParser::SignedIntegerContext *offset = nullptr;
    VtlParser::ScalarItemContext *defaultValue = nullptr;
    VtlParser::PartitionByClauseContext *partition = nullptr;
    VtlParser::OrderByClauseContext *orderBy = nullptr;
    std::vector<antlr4::tree::TerminalNode *> LPAREN();
    antlr4::tree::TerminalNode* LPAREN(size_t i);
    ExprContext *expr();
    antlr4::tree::TerminalNode *OVER();
    std::vector<antlr4::tree::TerminalNode *> RPAREN();
    antlr4::tree::TerminalNode* RPAREN(size_t i);
    antlr4::tree::TerminalNode *LAG();
    antlr4::tree::TerminalNode *LEAD();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    OrderByClauseContext *orderByClause();
    SignedIntegerContext *signedInteger();
    PartitionByClauseContext *partitionByClause();
    ScalarItemContext *scalarItem();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  RatioToReportAnContext : public AnFunctionContext {
  public:
    RatioToReportAnContext(AnFunctionContext *ctx);

    antlr4::Token *op = nullptr;
    VtlParser::PartitionByClauseContext *partition = nullptr;
    std::vector<antlr4::tree::TerminalNode *> LPAREN();
    antlr4::tree::TerminalNode* LPAREN(size_t i);
    ExprContext *expr();
    antlr4::tree::TerminalNode *OVER();
    std::vector<antlr4::tree::TerminalNode *> RPAREN();
    antlr4::tree::TerminalNode* RPAREN(size_t i);
    antlr4::tree::TerminalNode *RATIO_TO_REPORT();
    PartitionByClauseContext *partitionByClause();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  AnSimpleFunctionContext : public AnFunctionContext {
  public:
    AnSimpleFunctionContext(AnFunctionContext *ctx);

    antlr4::Token *op = nullptr;
    VtlParser::PartitionByClauseContext *partition = nullptr;
    VtlParser::OrderByClauseContext *orderBy = nullptr;
    VtlParser::WindowingClauseContext *windowing = nullptr;
    std::vector<antlr4::tree::TerminalNode *> LPAREN();
    antlr4::tree::TerminalNode* LPAREN(size_t i);
    ExprContext *expr();
    antlr4::tree::TerminalNode *OVER();
    std::vector<antlr4::tree::TerminalNode *> RPAREN();
    antlr4::tree::TerminalNode* RPAREN(size_t i);
    antlr4::tree::TerminalNode *SUM();
    antlr4::tree::TerminalNode *AVG();
    antlr4::tree::TerminalNode *COUNT();
    antlr4::tree::TerminalNode *MEDIAN();
    antlr4::tree::TerminalNode *MIN();
    antlr4::tree::TerminalNode *MAX();
    antlr4::tree::TerminalNode *STDDEV_POP();
    antlr4::tree::TerminalNode *STDDEV_SAMP();
    antlr4::tree::TerminalNode *VAR_POP();
    antlr4::tree::TerminalNode *VAR_SAMP();
    antlr4::tree::TerminalNode *FIRST_VALUE();
    antlr4::tree::TerminalNode *LAST_VALUE();
    PartitionByClauseContext *partitionByClause();
    OrderByClauseContext *orderByClause();
    WindowingClauseContext *windowingClause();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  AnFunctionContext* anFunction();

  class  AnFunctionComponentContext : public antlr4::ParserRuleContext {
  public:
    AnFunctionComponentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    AnFunctionComponentContext() = default;
    void copyFrom(AnFunctionComponentContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  AnSimpleFunctionComponentContext : public AnFunctionComponentContext {
  public:
    AnSimpleFunctionComponentContext(AnFunctionComponentContext *ctx);

    antlr4::Token *op = nullptr;
    VtlParser::PartitionByClauseContext *partition = nullptr;
    VtlParser::OrderByClauseContext *orderBy = nullptr;
    VtlParser::WindowingClauseContext *windowing = nullptr;
    std::vector<antlr4::tree::TerminalNode *> LPAREN();
    antlr4::tree::TerminalNode* LPAREN(size_t i);
    ExprComponentContext *exprComponent();
    antlr4::tree::TerminalNode *OVER();
    std::vector<antlr4::tree::TerminalNode *> RPAREN();
    antlr4::tree::TerminalNode* RPAREN(size_t i);
    antlr4::tree::TerminalNode *SUM();
    antlr4::tree::TerminalNode *AVG();
    antlr4::tree::TerminalNode *COUNT();
    antlr4::tree::TerminalNode *MEDIAN();
    antlr4::tree::TerminalNode *MIN();
    antlr4::tree::TerminalNode *MAX();
    antlr4::tree::TerminalNode *STDDEV_POP();
    antlr4::tree::TerminalNode *STDDEV_SAMP();
    antlr4::tree::TerminalNode *VAR_POP();
    antlr4::tree::TerminalNode *VAR_SAMP();
    antlr4::tree::TerminalNode *FIRST_VALUE();
    antlr4::tree::TerminalNode *LAST_VALUE();
    PartitionByClauseContext *partitionByClause();
    OrderByClauseContext *orderByClause();
    WindowingClauseContext *windowingClause();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  LagOrLeadAnComponentContext : public AnFunctionComponentContext {
  public:
    LagOrLeadAnComponentContext(AnFunctionComponentContext *ctx);

    antlr4::Token *op = nullptr;
    VtlParser::SignedIntegerContext *offset = nullptr;
    VtlParser::ScalarItemContext *defaultValue = nullptr;
    VtlParser::PartitionByClauseContext *partition = nullptr;
    VtlParser::OrderByClauseContext *orderBy = nullptr;
    std::vector<antlr4::tree::TerminalNode *> LPAREN();
    antlr4::tree::TerminalNode* LPAREN(size_t i);
    ExprComponentContext *exprComponent();
    antlr4::tree::TerminalNode *OVER();
    std::vector<antlr4::tree::TerminalNode *> RPAREN();
    antlr4::tree::TerminalNode* RPAREN(size_t i);
    antlr4::tree::TerminalNode *LAG();
    antlr4::tree::TerminalNode *LEAD();
    antlr4::tree::TerminalNode *COMMA();
    OrderByClauseContext *orderByClause();
    SignedIntegerContext *signedInteger();
    PartitionByClauseContext *partitionByClause();
    ScalarItemContext *scalarItem();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  RankAnComponentContext : public AnFunctionComponentContext {
  public:
    RankAnComponentContext(AnFunctionComponentContext *ctx);

    antlr4::Token *op = nullptr;
    VtlParser::PartitionByClauseContext *partition = nullptr;
    VtlParser::OrderByClauseContext *orderBy = nullptr;
    std::vector<antlr4::tree::TerminalNode *> LPAREN();
    antlr4::tree::TerminalNode* LPAREN(size_t i);
    antlr4::tree::TerminalNode *OVER();
    std::vector<antlr4::tree::TerminalNode *> RPAREN();
    antlr4::tree::TerminalNode* RPAREN(size_t i);
    antlr4::tree::TerminalNode *RANK();
    OrderByClauseContext *orderByClause();
    PartitionByClauseContext *partitionByClause();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  RatioToReportAnComponentContext : public AnFunctionComponentContext {
  public:
    RatioToReportAnComponentContext(AnFunctionComponentContext *ctx);

    antlr4::Token *op = nullptr;
    VtlParser::PartitionByClauseContext *partition = nullptr;
    std::vector<antlr4::tree::TerminalNode *> LPAREN();
    antlr4::tree::TerminalNode* LPAREN(size_t i);
    ExprComponentContext *exprComponent();
    antlr4::tree::TerminalNode *OVER();
    std::vector<antlr4::tree::TerminalNode *> RPAREN();
    antlr4::tree::TerminalNode* RPAREN(size_t i);
    antlr4::tree::TerminalNode *RATIO_TO_REPORT();
    PartitionByClauseContext *partitionByClause();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  AnFunctionComponentContext* anFunctionComponent();

  class  RenameClauseItemContext : public antlr4::ParserRuleContext {
  public:
    VtlParser::ComponentIDContext *fromName = nullptr;
    VtlParser::ComponentIDContext *toName = nullptr;
    RenameClauseItemContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *TO();
    std::vector<ComponentIDContext *> componentID();
    ComponentIDContext* componentID(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  RenameClauseItemContext* renameClauseItem();

  class  AggregateClauseContext : public antlr4::ParserRuleContext {
  public:
    AggregateClauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<AggrFunctionClauseContext *> aggrFunctionClause();
    AggrFunctionClauseContext* aggrFunctionClause(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  AggregateClauseContext* aggregateClause();

  class  AggrFunctionClauseContext : public antlr4::ParserRuleContext {
  public:
    AggrFunctionClauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ComponentIDContext *componentID();
    antlr4::tree::TerminalNode *ASSIGN();
    AggrOperatorsContext *aggrOperators();
    ComponentRoleContext *componentRole();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  AggrFunctionClauseContext* aggrFunctionClause();

  class  CalcClauseItemContext : public antlr4::ParserRuleContext {
  public:
    CalcClauseItemContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ComponentIDContext *componentID();
    antlr4::tree::TerminalNode *ASSIGN();
    ExprComponentContext *exprComponent();
    ComponentRoleContext *componentRole();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  CalcClauseItemContext* calcClauseItem();

  class  SubspaceClauseItemContext : public antlr4::ParserRuleContext {
  public:
    SubspaceClauseItemContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ComponentIDContext *componentID();
    antlr4::tree::TerminalNode *EQ();
    ScalarItemContext *scalarItem();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  SubspaceClauseItemContext* subspaceClauseItem();

  class  ScalarItemContext : public antlr4::ParserRuleContext {
  public:
    ScalarItemContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    ScalarItemContext() = default;
    void copyFrom(ScalarItemContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  ScalarWithCastContext : public ScalarItemContext {
  public:
    ScalarWithCastContext(ScalarItemContext *ctx);

    antlr4::tree::TerminalNode *CAST();
    antlr4::tree::TerminalNode *LPAREN();
    ConstantContext *constant();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *RPAREN();
    BasicScalarTypeContext *basicScalarType();
    antlr4::tree::TerminalNode *STRING_CONSTANT();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  SimpleScalarContext : public ScalarItemContext {
  public:
    SimpleScalarContext(ScalarItemContext *ctx);

    ConstantContext *constant();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ScalarItemContext* scalarItem();

  class  JoinClauseWithoutUsingContext : public antlr4::ParserRuleContext {
  public:
    JoinClauseWithoutUsingContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<JoinClauseItemContext *> joinClauseItem();
    JoinClauseItemContext* joinClauseItem(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  JoinClauseWithoutUsingContext* joinClauseWithoutUsing();

  class  JoinClauseContext : public antlr4::ParserRuleContext {
  public:
    JoinClauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<JoinClauseItemContext *> joinClauseItem();
    JoinClauseItemContext* joinClauseItem(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *USING();
    std::vector<ComponentIDContext *> componentID();
    ComponentIDContext* componentID(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  JoinClauseContext* joinClause();

  class  JoinClauseItemContext : public antlr4::ParserRuleContext {
  public:
    JoinClauseItemContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExprContext *expr();
    antlr4::tree::TerminalNode *AS();
    AliasContext *alias();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  JoinClauseItemContext* joinClauseItem();

  class  JoinBodyContext : public antlr4::ParserRuleContext {
  public:
    JoinBodyContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    FilterClauseContext *filterClause();
    CalcClauseContext *calcClause();
    JoinApplyClauseContext *joinApplyClause();
    AggrClauseContext *aggrClause();
    KeepOrDropClauseContext *keepOrDropClause();
    RenameClauseContext *renameClause();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  JoinBodyContext* joinBody();

  class  JoinApplyClauseContext : public antlr4::ParserRuleContext {
  public:
    JoinApplyClauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *APPLY();
    ExprContext *expr();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  JoinApplyClauseContext* joinApplyClause();

  class  PartitionByClauseContext : public antlr4::ParserRuleContext {
  public:
    PartitionByClauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *PARTITION();
    antlr4::tree::TerminalNode *BY();
    std::vector<ComponentIDContext *> componentID();
    ComponentIDContext* componentID(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  PartitionByClauseContext* partitionByClause();

  class  OrderByClauseContext : public antlr4::ParserRuleContext {
  public:
    OrderByClauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ORDER();
    antlr4::tree::TerminalNode *BY();
    std::vector<OrderByItemContext *> orderByItem();
    OrderByItemContext* orderByItem(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  OrderByClauseContext* orderByClause();

  class  OrderByItemContext : public antlr4::ParserRuleContext {
  public:
    OrderByItemContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ComponentIDContext *componentID();
    antlr4::tree::TerminalNode *ASC();
    antlr4::tree::TerminalNode *DESC();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  OrderByItemContext* orderByItem();

  class  WindowingClauseContext : public antlr4::ParserRuleContext {
  public:
    VtlParser::LimitClauseItemContext *from_ = nullptr;
    VtlParser::LimitClauseItemContext *to_ = nullptr;
    WindowingClauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *BETWEEN();
    antlr4::tree::TerminalNode *AND();
    std::vector<LimitClauseItemContext *> limitClauseItem();
    LimitClauseItemContext* limitClauseItem(size_t i);
    antlr4::tree::TerminalNode *RANGE();
    antlr4::tree::TerminalNode *DATA();
    antlr4::tree::TerminalNode *POINTS();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  WindowingClauseContext* windowingClause();

  class  SignedIntegerContext : public antlr4::ParserRuleContext {
  public:
    SignedIntegerContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *INTEGER_CONSTANT();
    antlr4::tree::TerminalNode *MINUS();
    antlr4::tree::TerminalNode *PLUS();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  SignedIntegerContext* signedInteger();

  class  SignedNumberContext : public antlr4::ParserRuleContext {
  public:
    SignedNumberContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *NUMBER_CONSTANT();
    antlr4::tree::TerminalNode *MINUS();
    antlr4::tree::TerminalNode *PLUS();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  SignedNumberContext* signedNumber();

  class  LimitClauseItemContext : public antlr4::ParserRuleContext {
  public:
    antlr4::Token *limitDir = nullptr;
    LimitClauseItemContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    SignedIntegerContext *signedInteger();
    antlr4::tree::TerminalNode *PRECEDING();
    antlr4::tree::TerminalNode *FOLLOWING();
    antlr4::tree::TerminalNode *CURRENT();
    antlr4::tree::TerminalNode *DATA();
    antlr4::tree::TerminalNode *POINT();
    antlr4::tree::TerminalNode *UNBOUNDED();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  LimitClauseItemContext* limitClauseItem();

  class  GroupingClauseContext : public antlr4::ParserRuleContext {
  public:
    GroupingClauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    GroupingClauseContext() = default;
    void copyFrom(GroupingClauseContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  GroupAllContext : public GroupingClauseContext {
  public:
    GroupAllContext(GroupingClauseContext *ctx);

    antlr4::Token *delim = nullptr;
    antlr4::tree::TerminalNode *GROUP();
    antlr4::tree::TerminalNode *ALL();
    antlr4::tree::TerminalNode *TIME_AGG();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *STRING_CONSTANT();
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *COMMA();
    antlr4::tree::TerminalNode *FIRST();
    antlr4::tree::TerminalNode *LAST();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  GroupByOrExceptContext : public GroupingClauseContext {
  public:
    GroupByOrExceptContext(GroupingClauseContext *ctx);

    antlr4::Token *op = nullptr;
    antlr4::Token *delim = nullptr;
    antlr4::tree::TerminalNode *GROUP();
    std::vector<ComponentIDContext *> componentID();
    ComponentIDContext* componentID(size_t i);
    antlr4::tree::TerminalNode *BY();
    antlr4::tree::TerminalNode *EXCEPT();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *TIME_AGG();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *STRING_CONSTANT();
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *FIRST();
    antlr4::tree::TerminalNode *LAST();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  GroupingClauseContext* groupingClause();

  class  HavingClauseContext : public antlr4::ParserRuleContext {
  public:
    HavingClauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *HAVING();
    ExprComponentContext *exprComponent();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  HavingClauseContext* havingClause();

  class  ParameterItemContext : public antlr4::ParserRuleContext {
  public:
    ParameterItemContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    VarIDContext *varID();
    InputParameterTypeContext *inputParameterType();
    antlr4::tree::TerminalNode *DEFAULT();
    ScalarItemContext *scalarItem();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ParameterItemContext* parameterItem();

  class  OutputParameterTypeContext : public antlr4::ParserRuleContext {
  public:
    OutputParameterTypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ScalarTypeContext *scalarType();
    DatasetTypeContext *datasetType();
    ComponentTypeContext *componentType();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  OutputParameterTypeContext* outputParameterType();

  class  OutputParameterTypeComponentContext : public antlr4::ParserRuleContext {
  public:
    OutputParameterTypeComponentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ComponentTypeContext *componentType();
    ScalarTypeContext *scalarType();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  OutputParameterTypeComponentContext* outputParameterTypeComponent();

  class  InputParameterTypeContext : public antlr4::ParserRuleContext {
  public:
    InputParameterTypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ScalarTypeContext *scalarType();
    DatasetTypeContext *datasetType();
    ScalarSetTypeContext *scalarSetType();
    RulesetTypeContext *rulesetType();
    ComponentTypeContext *componentType();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  InputParameterTypeContext* inputParameterType();

  class  RulesetTypeContext : public antlr4::ParserRuleContext {
  public:
    RulesetTypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *RULESET();
    DpRulesetContext *dpRuleset();
    HrRulesetContext *hrRuleset();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  RulesetTypeContext* rulesetType();

  class  ScalarTypeContext : public antlr4::ParserRuleContext {
  public:
    ScalarTypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    BasicScalarTypeContext *basicScalarType();
    ValueDomainNameContext *valueDomainName();
    ScalarTypeConstraintContext *scalarTypeConstraint();
    antlr4::tree::TerminalNode *NULL_CONSTANT();
    antlr4::tree::TerminalNode *NOT();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ScalarTypeContext* scalarType();

  class  ComponentTypeContext : public antlr4::ParserRuleContext {
  public:
    ComponentTypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ComponentRoleContext *componentRole();
    antlr4::tree::TerminalNode *LT();
    ScalarTypeContext *scalarType();
    antlr4::tree::TerminalNode *MT();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ComponentTypeContext* componentType();

  class  DatasetTypeContext : public antlr4::ParserRuleContext {
  public:
    DatasetTypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *DATASET();
    antlr4::tree::TerminalNode *GLPAREN();
    std::vector<CompConstraintContext *> compConstraint();
    CompConstraintContext* compConstraint(size_t i);
    antlr4::tree::TerminalNode *GRPAREN();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  DatasetTypeContext* datasetType();

  class  EvalDatasetTypeContext : public antlr4::ParserRuleContext {
  public:
    EvalDatasetTypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    DatasetTypeContext *datasetType();
    ScalarTypeContext *scalarType();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  EvalDatasetTypeContext* evalDatasetType();

  class  ScalarSetTypeContext : public antlr4::ParserRuleContext {
  public:
    ScalarSetTypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *SET();
    antlr4::tree::TerminalNode *LT();
    ScalarTypeContext *scalarType();
    antlr4::tree::TerminalNode *MT();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ScalarSetTypeContext* scalarSetType();

  class  DpRulesetContext : public antlr4::ParserRuleContext {
  public:
    DpRulesetContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    DpRulesetContext() = default;
    void copyFrom(DpRulesetContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  DataPointVdContext : public DpRulesetContext {
  public:
    DataPointVdContext(DpRulesetContext *ctx);

    antlr4::tree::TerminalNode *DATAPOINT_ON_VD();
    antlr4::tree::TerminalNode *GLPAREN();
    std::vector<ValueDomainNameContext *> valueDomainName();
    ValueDomainNameContext* valueDomainName(size_t i);
    antlr4::tree::TerminalNode *GRPAREN();
    std::vector<antlr4::tree::TerminalNode *> MUL();
    antlr4::tree::TerminalNode* MUL(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  DataPointVarContext : public DpRulesetContext {
  public:
    DataPointVarContext(DpRulesetContext *ctx);

    antlr4::tree::TerminalNode *DATAPOINT_ON_VAR();
    antlr4::tree::TerminalNode *GLPAREN();
    std::vector<VarIDContext *> varID();
    VarIDContext* varID(size_t i);
    antlr4::tree::TerminalNode *GRPAREN();
    std::vector<antlr4::tree::TerminalNode *> MUL();
    antlr4::tree::TerminalNode* MUL(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  DataPointContext : public DpRulesetContext {
  public:
    DataPointContext(DpRulesetContext *ctx);

    antlr4::tree::TerminalNode *DATAPOINT();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  DpRulesetContext* dpRuleset();

  class  HrRulesetContext : public antlr4::ParserRuleContext {
  public:
    HrRulesetContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    HrRulesetContext() = default;
    void copyFrom(HrRulesetContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  HrRulesetVdTypeContext : public HrRulesetContext {
  public:
    HrRulesetVdTypeContext(HrRulesetContext *ctx);

    antlr4::Token *vdName = nullptr;
    antlr4::tree::TerminalNode *HIERARCHICAL_ON_VD();
    antlr4::tree::TerminalNode *GLPAREN();
    antlr4::tree::TerminalNode *GRPAREN();
    antlr4::tree::TerminalNode *IDENTIFIER();
    antlr4::tree::TerminalNode *LPAREN();
    std::vector<ValueDomainNameContext *> valueDomainName();
    ValueDomainNameContext* valueDomainName(size_t i);
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<antlr4::tree::TerminalNode *> MUL();
    antlr4::tree::TerminalNode* MUL(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  HrRulesetVarTypeContext : public HrRulesetContext {
  public:
    HrRulesetVarTypeContext(HrRulesetContext *ctx);

    VtlParser::VarIDContext *varName = nullptr;
    antlr4::tree::TerminalNode *HIERARCHICAL_ON_VAR();
    antlr4::tree::TerminalNode *GLPAREN();
    antlr4::tree::TerminalNode *GRPAREN();
    std::vector<VarIDContext *> varID();
    VarIDContext* varID(size_t i);
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();
    std::vector<antlr4::tree::TerminalNode *> MUL();
    antlr4::tree::TerminalNode* MUL(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  HrRulesetTypeContext : public HrRulesetContext {
  public:
    HrRulesetTypeContext(HrRulesetContext *ctx);

    antlr4::tree::TerminalNode *HIERARCHICAL();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  HrRulesetContext* hrRuleset();

  class  ValueDomainNameContext : public antlr4::ParserRuleContext {
  public:
    ValueDomainNameContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *IDENTIFIER();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ValueDomainNameContext* valueDomainName();

  class  RulesetIDContext : public antlr4::ParserRuleContext {
  public:
    RulesetIDContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *IDENTIFIER();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  RulesetIDContext* rulesetID();

  class  RulesetSignatureContext : public antlr4::ParserRuleContext {
  public:
    RulesetSignatureContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<SignatureContext *> signature();
    SignatureContext* signature(size_t i);
    antlr4::tree::TerminalNode *VALUE_DOMAIN();
    antlr4::tree::TerminalNode *VARIABLE();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  RulesetSignatureContext* rulesetSignature();

  class  SignatureContext : public antlr4::ParserRuleContext {
  public:
    SignatureContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    VarIDContext *varID();
    antlr4::tree::TerminalNode *AS();
    AliasContext *alias();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  SignatureContext* signature();

  class  RuleClauseDatapointContext : public antlr4::ParserRuleContext {
  public:
    RuleClauseDatapointContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<RuleItemDatapointContext *> ruleItemDatapoint();
    RuleItemDatapointContext* ruleItemDatapoint(size_t i);
    std::vector<antlr4::tree::TerminalNode *> EOL();
    antlr4::tree::TerminalNode* EOL(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  RuleClauseDatapointContext* ruleClauseDatapoint();

  class  RuleItemDatapointContext : public antlr4::ParserRuleContext {
  public:
    antlr4::Token *ruleName = nullptr;
    VtlParser::ExprComponentContext *antecedentContiditon = nullptr;
    VtlParser::ExprComponentContext *consequentCondition = nullptr;
    RuleItemDatapointContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ExprComponentContext *> exprComponent();
    ExprComponentContext* exprComponent(size_t i);
    antlr4::tree::TerminalNode *COLON();
    antlr4::tree::TerminalNode *WHEN();
    antlr4::tree::TerminalNode *THEN();
    ErCodeContext *erCode();
    ErLevelContext *erLevel();
    antlr4::tree::TerminalNode *IDENTIFIER();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  RuleItemDatapointContext* ruleItemDatapoint();

  class  RuleClauseHierarchicalContext : public antlr4::ParserRuleContext {
  public:
    RuleClauseHierarchicalContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<RuleItemHierarchicalContext *> ruleItemHierarchical();
    RuleItemHierarchicalContext* ruleItemHierarchical(size_t i);
    std::vector<antlr4::tree::TerminalNode *> EOL();
    antlr4::tree::TerminalNode* EOL(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  RuleClauseHierarchicalContext* ruleClauseHierarchical();

  class  RuleItemHierarchicalContext : public antlr4::ParserRuleContext {
  public:
    antlr4::Token *ruleName = nullptr;
    RuleItemHierarchicalContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    CodeItemRelationContext *codeItemRelation();
    antlr4::tree::TerminalNode *COLON();
    ErCodeContext *erCode();
    ErLevelContext *erLevel();
    antlr4::tree::TerminalNode *IDENTIFIER();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  RuleItemHierarchicalContext* ruleItemHierarchical();

  class  HierRuleSignatureContext : public antlr4::ParserRuleContext {
  public:
    HierRuleSignatureContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *RULE();
    antlr4::tree::TerminalNode *IDENTIFIER();
    antlr4::tree::TerminalNode *VALUE_DOMAIN();
    antlr4::tree::TerminalNode *VARIABLE();
    antlr4::tree::TerminalNode *CONDITION();
    ValueDomainSignatureContext *valueDomainSignature();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  HierRuleSignatureContext* hierRuleSignature();

  class  ValueDomainSignatureContext : public antlr4::ParserRuleContext {
  public:
    ValueDomainSignatureContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<SignatureContext *> signature();
    SignatureContext* signature(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ValueDomainSignatureContext* valueDomainSignature();

  class  CodeItemRelationContext : public antlr4::ParserRuleContext {
  public:
    VtlParser::ValueDomainValueContext *codetemRef = nullptr;
    CodeItemRelationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<CodeItemRelationClauseContext *> codeItemRelationClause();
    CodeItemRelationClauseContext* codeItemRelationClause(size_t i);
    ValueDomainValueContext *valueDomainValue();
    antlr4::tree::TerminalNode *WHEN();
    ExprComponentContext *exprComponent();
    antlr4::tree::TerminalNode *THEN();
    ComparisonOperandContext *comparisonOperand();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  CodeItemRelationContext* codeItemRelation();

  class  CodeItemRelationClauseContext : public antlr4::ParserRuleContext {
  public:
    antlr4::Token *opAdd = nullptr;
    VtlParser::ValueDomainValueContext *rightCodeItem = nullptr;
    VtlParser::ExprComponentContext *rightCondition = nullptr;
    CodeItemRelationClauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ValueDomainValueContext *valueDomainValue();
    antlr4::tree::TerminalNode *QLPAREN();
    antlr4::tree::TerminalNode *QRPAREN();
    ExprComponentContext *exprComponent();
    antlr4::tree::TerminalNode *PLUS();
    antlr4::tree::TerminalNode *MINUS();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  CodeItemRelationClauseContext* codeItemRelationClause();

  class  ValueDomainValueContext : public antlr4::ParserRuleContext {
  public:
    ValueDomainValueContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *IDENTIFIER();
    SignedIntegerContext *signedInteger();
    SignedNumberContext *signedNumber();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ValueDomainValueContext* valueDomainValue();

  class  ScalarTypeConstraintContext : public antlr4::ParserRuleContext {
  public:
    ScalarTypeConstraintContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    ScalarTypeConstraintContext() = default;
    void copyFrom(ScalarTypeConstraintContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  RangeConstraintContext : public ScalarTypeConstraintContext {
  public:
    RangeConstraintContext(ScalarTypeConstraintContext *ctx);

    antlr4::tree::TerminalNode *GLPAREN();
    std::vector<ScalarItemContext *> scalarItem();
    ScalarItemContext* scalarItem(size_t i);
    antlr4::tree::TerminalNode *GRPAREN();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  ConditionConstraintContext : public ScalarTypeConstraintContext {
  public:
    ConditionConstraintContext(ScalarTypeConstraintContext *ctx);

    antlr4::tree::TerminalNode *QLPAREN();
    ExprComponentContext *exprComponent();
    antlr4::tree::TerminalNode *QRPAREN();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  ScalarTypeConstraintContext* scalarTypeConstraint();

  class  CompConstraintContext : public antlr4::ParserRuleContext {
  public:
    CompConstraintContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ComponentTypeContext *componentType();
    ComponentIDContext *componentID();
    MultModifierContext *multModifier();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  CompConstraintContext* compConstraint();

  class  MultModifierContext : public antlr4::ParserRuleContext {
  public:
    MultModifierContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *OPTIONAL();
    antlr4::tree::TerminalNode *PLUS();
    antlr4::tree::TerminalNode *MUL();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  MultModifierContext* multModifier();

  class  ValidationOutputContext : public antlr4::ParserRuleContext {
  public:
    ValidationOutputContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *INVALID();
    antlr4::tree::TerminalNode *ALL_MEASURES();
    antlr4::tree::TerminalNode *ALL();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ValidationOutputContext* validationOutput();

  class  ValidationModeContext : public antlr4::ParserRuleContext {
  public:
    ValidationModeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *NON_NULL();
    antlr4::tree::TerminalNode *NON_ZERO();
    antlr4::tree::TerminalNode *PARTIAL_NULL();
    antlr4::tree::TerminalNode *PARTIAL_ZERO();
    antlr4::tree::TerminalNode *ALWAYS_NULL();
    antlr4::tree::TerminalNode *ALWAYS_ZERO();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ValidationModeContext* validationMode();

  class  ConditionClauseContext : public antlr4::ParserRuleContext {
  public:
    ConditionClauseContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *CONDITION();
    std::vector<ComponentIDContext *> componentID();
    ComponentIDContext* componentID(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ConditionClauseContext* conditionClause();

  class  InputModeContext : public antlr4::ParserRuleContext {
  public:
    InputModeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *DATASET();
    antlr4::tree::TerminalNode *DATASET_PRIORITY();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  InputModeContext* inputMode();

  class  ImbalanceExprContext : public antlr4::ParserRuleContext {
  public:
    ImbalanceExprContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *IMBALANCE();
    ExprContext *expr();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ImbalanceExprContext* imbalanceExpr();

  class  InputModeHierarchyContext : public antlr4::ParserRuleContext {
  public:
    InputModeHierarchyContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *RULE();
    antlr4::tree::TerminalNode *DATASET();
    antlr4::tree::TerminalNode *RULE_PRIORITY();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  InputModeHierarchyContext* inputModeHierarchy();

  class  OutputModeHierarchyContext : public antlr4::ParserRuleContext {
  public:
    OutputModeHierarchyContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *COMPUTED();
    antlr4::tree::TerminalNode *ALL();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  OutputModeHierarchyContext* outputModeHierarchy();

  class  AliasContext : public antlr4::ParserRuleContext {
  public:
    AliasContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *IDENTIFIER();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  AliasContext* alias();

  class  VarIDContext : public antlr4::ParserRuleContext {
  public:
    VarIDContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *IDENTIFIER();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  VarIDContext* varID();

  class  SimpleComponentIdContext : public antlr4::ParserRuleContext {
  public:
    SimpleComponentIdContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *IDENTIFIER();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  SimpleComponentIdContext* simpleComponentId();

  class  ComponentIDContext : public antlr4::ParserRuleContext {
  public:
    ComponentIDContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> IDENTIFIER();
    antlr4::tree::TerminalNode* IDENTIFIER(size_t i);
    antlr4::tree::TerminalNode *MEMBERSHIP();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ComponentIDContext* componentID();

  class  ListsContext : public antlr4::ParserRuleContext {
  public:
    ListsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *GLPAREN();
    std::vector<ScalarItemContext *> scalarItem();
    ScalarItemContext* scalarItem(size_t i);
    antlr4::tree::TerminalNode *GRPAREN();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ListsContext* lists();

  class  ErCodeContext : public antlr4::ParserRuleContext {
  public:
    ErCodeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ERRORCODE();
    ConstantContext *constant();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ErCodeContext* erCode();

  class  ErLevelContext : public antlr4::ParserRuleContext {
  public:
    ErLevelContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ERRORLEVEL();
    ConstantContext *constant();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ErLevelContext* erLevel();

  class  ComparisonOperandContext : public antlr4::ParserRuleContext {
  public:
    ComparisonOperandContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *MT();
    antlr4::tree::TerminalNode *ME();
    antlr4::tree::TerminalNode *LE();
    antlr4::tree::TerminalNode *LT();
    antlr4::tree::TerminalNode *EQ();
    antlr4::tree::TerminalNode *NEQ();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ComparisonOperandContext* comparisonOperand();

  class  OptionalExprContext : public antlr4::ParserRuleContext {
  public:
    OptionalExprContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExprContext *expr();
    antlr4::tree::TerminalNode *OPTIONAL();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  OptionalExprContext* optionalExpr();

  class  OptionalExprComponentContext : public antlr4::ParserRuleContext {
  public:
    OptionalExprComponentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExprComponentContext *exprComponent();
    antlr4::tree::TerminalNode *OPTIONAL();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  OptionalExprComponentContext* optionalExprComponent();

  class  ComponentRoleContext : public antlr4::ParserRuleContext {
  public:
    ComponentRoleContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *MEASURE();
    antlr4::tree::TerminalNode *COMPONENT();
    antlr4::tree::TerminalNode *DIMENSION();
    antlr4::tree::TerminalNode *ATTRIBUTE();
    ViralAttributeContext *viralAttribute();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ComponentRoleContext* componentRole();

  class  ViralAttributeContext : public antlr4::ParserRuleContext {
  public:
    ViralAttributeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *VIRAL();
    antlr4::tree::TerminalNode *ATTRIBUTE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ViralAttributeContext* viralAttribute();

  class  ValueDomainIDContext : public antlr4::ParserRuleContext {
  public:
    ValueDomainIDContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *IDENTIFIER();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ValueDomainIDContext* valueDomainID();

  class  OperatorIDContext : public antlr4::ParserRuleContext {
  public:
    OperatorIDContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *IDENTIFIER();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  OperatorIDContext* operatorID();

  class  RoutineNameContext : public antlr4::ParserRuleContext {
  public:
    RoutineNameContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *IDENTIFIER();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  RoutineNameContext* routineName();

  class  ConstantContext : public antlr4::ParserRuleContext {
  public:
    ConstantContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    SignedIntegerContext *signedInteger();
    SignedNumberContext *signedNumber();
    antlr4::tree::TerminalNode *BOOLEAN_CONSTANT();
    antlr4::tree::TerminalNode *STRING_CONSTANT();
    antlr4::tree::TerminalNode *NULL_CONSTANT();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ConstantContext* constant();

  class  BasicScalarTypeContext : public antlr4::ParserRuleContext {
  public:
    BasicScalarTypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *STRING();
    antlr4::tree::TerminalNode *INTEGER();
    antlr4::tree::TerminalNode *NUMBER();
    antlr4::tree::TerminalNode *BOOLEAN();
    antlr4::tree::TerminalNode *DATE();
    antlr4::tree::TerminalNode *TIME();
    antlr4::tree::TerminalNode *TIME_PERIOD();
    antlr4::tree::TerminalNode *DURATION();
    antlr4::tree::TerminalNode *SCALAR();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  BasicScalarTypeContext* basicScalarType();

  class  RetainTypeContext : public antlr4::ParserRuleContext {
  public:
    RetainTypeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *BOOLEAN_CONSTANT();
    antlr4::tree::TerminalNode *ALL();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual std::any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  RetainTypeContext* retainType();


  bool sempred(antlr4::RuleContext *_localctx, size_t ruleIndex, size_t predicateIndex) override;

  bool exprSempred(ExprContext *_localctx, size_t predicateIndex);
  bool exprComponentSempred(ExprComponentContext *_localctx, size_t predicateIndex);

  // By default the static state used to implement the parser is lazily initialized during the first
  // call to the constructor. You can call this function if you wish to initialize the static state
  // ahead of time.
  static void initialize();

private:
};

