
// Generated from /home/javier/Programacion/vtlengine/src/vtlengine/AST/Grammar/Vtl.g4 by ANTLR 4.13.1

#pragma once


#include "antlr4-runtime.h"




class  VtlLexer : public antlr4::Lexer {
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

  explicit VtlLexer(antlr4::CharStream *input);

  ~VtlLexer() override;


  std::string getGrammarFileName() const override;

  const std::vector<std::string>& getRuleNames() const override;

  const std::vector<std::string>& getChannelNames() const override;

  const std::vector<std::string>& getModeNames() const override;

  const antlr4::dfa::Vocabulary& getVocabulary() const override;

  antlr4::atn::SerializedATNView getSerializedATN() const override;

  const antlr4::atn::ATN& getATN() const override;

  // By default the static state used to implement the lexer is lazily initialized during the first
  // call to the constructor. You can call this function if you wish to initialize the static state
  // ahead of time.
  static void initialize();

private:

  // Individual action functions triggered by action() above.

  // Individual semantic predicate functions triggered by sempred() above.

};

