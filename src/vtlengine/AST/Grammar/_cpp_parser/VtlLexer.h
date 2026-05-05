
// Generated from Vtl.g4 by ANTLR 4.13.2

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
    ROLE = 109, VIRAL = 110, PROPAGATION = 111, CHARSET_MATCH = 112, TYPE = 113, 
    NVL = 114, HIERARCHY = 115, OPTIONAL = 116, INVALID = 117, VALUE_DOMAIN = 118, 
    VARIABLE = 119, DATA = 120, STRUCTURE = 121, DATASET = 122, OPERATOR = 123, 
    DEFINE = 124, PUT_SYMBOL = 125, DATAPOINT = 126, HIERARCHICAL = 127, 
    RULESET = 128, RULE = 129, END = 130, ALTER_DATASET = 131, LTRIM = 132, 
    RTRIM = 133, INSTR = 134, REPLACE = 135, CEIL = 136, FLOOR = 137, SQRT = 138, 
    ANY = 139, SETDIFF = 140, STDDEV_POP = 141, STDDEV_SAMP = 142, VAR_POP = 143, 
    VAR_SAMP = 144, GROUP = 145, EXCEPT = 146, HAVING = 147, FIRST_VALUE = 148, 
    LAST_VALUE = 149, LAG = 150, LEAD = 151, RATIO_TO_REPORT = 152, OVER = 153, 
    PRECEDING = 154, FOLLOWING = 155, UNBOUNDED = 156, PARTITION = 157, 
    ROWS = 158, RANGE = 159, CURRENT = 160, VALID = 161, FILL_TIME_SERIES = 162, 
    FLOW_TO_STOCK = 163, STOCK_TO_FLOW = 164, TIMESHIFT = 165, MEASURES = 166, 
    NO_MEASURES = 167, CONDITION = 168, BOOLEAN = 169, DATE = 170, TIME_PERIOD = 171, 
    NUMBER = 172, STRING = 173, TIME = 174, INTEGER = 175, FLOAT = 176, 
    LIST = 177, RECORD = 178, RESTRICT = 179, YYYY = 180, MM = 181, DD = 182, 
    MAX_LENGTH = 183, REGEXP = 184, IS = 185, WHEN = 186, FROM = 187, AGGREGATES = 188, 
    POINTS = 189, POINT = 190, TOTAL = 191, PARTIAL = 192, ALWAYS = 193, 
    INNER_JOIN = 194, LEFT_JOIN = 195, CROSS_JOIN = 196, FULL_JOIN = 197, 
    MAPS_FROM = 198, MAPS_TO = 199, MAP_TO = 200, MAP_FROM = 201, RETURNS = 202, 
    PIVOT = 203, CUSTOMPIVOT = 204, UNPIVOT = 205, SUBSPACE = 206, APPLY = 207, 
    CONDITIONED = 208, PERIOD_INDICATOR = 209, SINGLE = 210, DURATION = 211, 
    TIME_AGG = 212, UNIT = 213, VALUE = 214, VALUEDOMAINS = 215, VARIABLES = 216, 
    INPUT = 217, OUTPUT = 218, CAST = 219, RULE_PRIORITY = 220, DATASET_PRIORITY = 221, 
    DEFAULT = 222, CHECK_DATAPOINT = 223, CHECK_HIERARCHY = 224, COMPUTED = 225, 
    NON_NULL = 226, NON_ZERO = 227, PARTIAL_NULL = 228, PARTIAL_ZERO = 229, 
    ALWAYS_NULL = 230, ALWAYS_ZERO = 231, COMPONENTS = 232, ALL_MEASURES = 233, 
    SCALAR = 234, COMPONENT = 235, DATAPOINT_ON_VD = 236, DATAPOINT_ON_VAR = 237, 
    HIERARCHICAL_ON_VD = 238, HIERARCHICAL_ON_VAR = 239, SET = 240, LANGUAGE = 241, 
    INTEGER_CONSTANT = 242, NUMBER_CONSTANT = 243, BOOLEAN_CONSTANT = 244, 
    STRING_CONSTANT = 245, IDENTIFIER = 246, WS = 247, EOL = 248, ML_COMMENT = 249, 
    SL_COMMENT = 250
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

