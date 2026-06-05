
// Generated from VtlTokens.g4 by ANTLR 4.13.2

#pragma once


#include "antlr4-runtime.h"




class  VtlTokens : public antlr4::Lexer {
public:
  enum {
    ASSIGN = 1, COLON = 2, COMMA = 3, CONCAT = 4, DIV = 5, DOT = 6, EOL = 7, 
    EQ = 8, GLPAREN = 9, GRPAREN = 10, LE = 11, LPAREN = 12, LT = 13, ME = 14, 
    MEMBERSHIP = 15, MINUS = 16, MT = 17, MUL = 18, NEQ = 19, OPTIONAL = 20, 
    PLUS = 21, POINTER = 22, PUT_SYMBOL = 23, QLPAREN = 24, QRPAREN = 25, 
    RPAREN = 26, SINGLE_QUOTE = 27, ABS = 28, AGGREGATE = 29, AGGREGATE_KW = 30, 
    ALL = 31, ALL_MEASURES = 32, ALWAYS_NULL = 33, ALWAYS_ZERO = 34, AND = 35, 
    ANY = 36, APPLY = 37, AS = 38, ASC = 39, ATTRIBUTE = 40, AVG = 41, BETWEEN = 42, 
    BOOLEAN = 43, BY = 44, CALC = 45, CASE = 46, CAST = 47, CEIL = 48, CHARSET_MATCH = 49, 
    CHECK = 50, CHECK_DATAPOINT = 51, CHECK_HIERARCHY = 52, COMPONENT = 53, 
    COMPONENTS = 54, COMPUTED = 55, CONDITION = 56, COUNT = 57, CROSS_JOIN = 58, 
    CURRENT = 59, CURRENT_DATE = 60, CUSTOMPIVOT = 61, DATA = 62, DATAPOINT = 63, 
    DATAPOINT_ON_VAR = 64, DATAPOINT_ON_VD = 65, DATASET = 66, DATASET_PRIORITY = 67, 
    DATE = 68, DATEADD = 69, DATEDIFF = 70, DAYOFMONTH = 71, DAYOFYEAR = 72, 
    DAYTOMONTH = 73, DAYTOYEAR = 74, DEFAULT = 75, DEFINE = 76, DESC = 77, 
    DIFF = 78, DIMENSION = 79, DROP = 80, DURATION = 81, ELSE = 82, END = 83, 
    ERRORCODE = 84, ERRORLEVEL = 85, EVAL = 86, EXCEPT = 87, EXISTS_IN = 88, 
    EXP = 89, FILL_TIME_SERIES = 90, FILTER = 91, FIRST = 92, FIRST_VALUE = 93, 
    FLOAT = 94, FLOOR = 95, FLOW_TO_STOCK = 96, FOLLOWING = 97, FROM = 98, 
    FULL_JOIN = 99, GROUP = 100, HAVING = 101, HIERARCHICAL = 102, HIERARCHICAL_ON_VAR = 103, 
    HIERARCHICAL_ON_VD = 104, HIERARCHY = 105, IF = 106, IMBALANCE = 107, 
    IN = 108, INDEXOF = 109, INNER_JOIN = 110, INPUT = 111, INSTR = 112, 
    INTEGER = 113, INTERSECT = 114, INVALID = 115, IS = 116, ISNULL = 117, 
    KEEP = 118, KEY = 119, LAG = 120, LANGUAGE = 121, LAST = 122, LAST_VALUE = 123, 
    LCASE = 124, LEAD = 125, LEFT_JOIN = 126, LEN = 127, LIST = 128, LN = 129, 
    LOG = 130, LTRIM = 131, MAX = 132, MEASURE = 133, MEDIAN = 134, MERGE = 135, 
    MIN = 136, MOD = 137, MONTH_OP = 138, MONTHTODAY = 139, NON_NULL = 140, 
    NON_ZERO = 141, NOT = 142, NOT_IN = 143, NUMBER = 144, NVL = 145, ON = 146, 
    OPERATOR = 147, OR = 148, ORDER = 149, OUTPUT = 150, OVER = 151, PARTIAL_NULL = 152, 
    PARTIAL_ZERO = 153, PARTITION = 154, PERIOD_INDICATOR = 155, PIVOT = 156, 
    POINT = 157, POINTS = 158, POWER = 159, PRECEDING = 160, PROPAGATION = 161, 
    RANDOM = 162, RANGE = 163, RANK = 164, RATIO_TO_REPORT = 165, RENAME = 166, 
    REPLACE = 167, RETURNS = 168, ROUND = 169, ROWS = 170, RTRIM = 171, 
    RULE = 172, RULE_PRIORITY = 173, RULESET = 174, SCALAR = 175, SET = 176, 
    SETDIFF = 177, SINGLE = 178, SQRT = 179, STDDEV_POP = 180, STDDEV_SAMP = 181, 
    STOCK_TO_FLOW = 182, STRING = 183, STRING_DISTANCE = 184, STRUCTURE = 185, 
    SUBSPACE = 186, SUBSTR = 187, SUM = 188, SYMDIFF = 189, THEN = 190, 
    TIME = 191, TIME_AGG = 192, TIME_PERIOD = 193, TIMESHIFT = 194, TO = 195, 
    TRIM = 196, TRUNC = 197, TYPE = 198, UCASE = 199, UNBOUNDED = 200, UNION = 201, 
    UNPIVOT = 202, USING = 203, VALUE = 204, VALUE_DOMAIN = 205, VAR_POP = 206, 
    VAR_SAMP = 207, VARIABLE = 208, VIRAL = 209, WHEN = 210, WITH = 211, 
    XOR = 212, YEAR_OP = 213, YEARTODAY = 214, LEVENSHTEIN_METHOD = 215, 
    DAMERAU_LEVENSHTEIN_METHOD = 216, HAMMING_METHOD = 217, JARO_WINKLER_METHOD = 218, 
    NULL_CONSTANT = 219, INTEGER_CONSTANT = 220, NUMBER_CONSTANT = 221, 
    BOOLEAN_CONSTANT = 222, STRING_CONSTANT = 223, IDENTIFIER = 224, ITEM_CODE = 225, 
    WS = 226, ML_COMMENT = 227, SL_COMMENT = 228
  };

  explicit VtlTokens(antlr4::CharStream *input);

  ~VtlTokens() override;


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

