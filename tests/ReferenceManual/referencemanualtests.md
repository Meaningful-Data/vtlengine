# VTL 2.0 Reference Manual Test Coverage. Empty datasets tests.
The official site [VTL-sdmx](https://sdmx.org/?page_id=5096)


## 1. General purpose Operators.
>>>
Test number     | VTL expresion         | Test result
:------------:  | :-------------        |:-------------:
1     | DS_r <- DS_1;         | WIP
2     | DS_r := DS_1;         | WIP
3     | DS_r := DS_1#Me_1;    | WIP
4     | DS_r := DS_1#Id_1;    | WIP
5     | DS_r := DS_1#At_1;    | WIP
>>>

## 2. Join Operators.
>>>
Test number      | VTL expresion         | Test result
:------------:      | :-------------       |:-------------:
6     | DS_r := inner_join ( DS_1 as d1, DS_2 as d2 keep Me_1, d2#Me_2, Me_1A);                                                                 | WIP
7     | DS_r := left_join  DS_1 as d1, DS_2 as d2 keep Me_1, d2#Me_2, Me_1A );                                                                  | WIP
8     | DS_r := full_join ( DS_1 as d1, DS_2 as d2 keep Me_1, d2#Me_2, Me_1A );                                                                 | WIP
9     | DS_r := cross_join (DS_1 as d1, DS_2 as d2 rename d1#Id_1 to Id11, d1#Id_2 to Id12, d2#Id_1 to Id21, d2#Id_2 to Id22, d1#Me_2 to Me12 );| WIP
10    | DS_r := inner_join (DS_1 as d1, DS_2 as d2 filter Me_1 = "A" calc Me_4 := Me_1 \|\| Me_1A drop d1#Me_2);                                | WIP
11    | DS_r := inner_join ( DS_1 calc Me_2 := Me_2 \|\| "_NEW" filter Id_2 ="B" keep Me_1, Me_2);                                              | WIP
12    | DS_r := inner_join ( DS_1 as d1, DS_2 as d2 apply d1 \|\| d2);                                                                          | 
>>>

## 3. String Operators.
>>>
Test number     | VTL expresion                                                          | Test result
:------------:  |:-----------------------------------------------------------------------|:-------------:
13    | DS_r := DS_1 \|\| DS_2;                                                | WIP
14    | DS_r := DS_1[calc Me_2:= Me_1 \|\| " world"];                          | WIP
15    | DS_r := rtrim(DS_1);                                                   | WIP
16    | DS_r := DS_1[ calc Me_2:= rtrim(Me_1)];                                | WIP
17    | DS_r := upper(DS_1);                                                   | WIP
18    | DS_r := DS_1[calc Me_2:= upper(Me_1)];                                 | WIP
19    | DS_r:= substr ( DS_1 , 7 );                                            | WIP 
20    | DS_r:= substr ( DS_1 , 1 , 5 );                                        | WIP 
21    | DS_r:= DS_1 [ calc Me_2:= substr ( Me_2 , 1 , 5 ) ];                   | WIP 
22    | DS_r := replace (ds_1,"ello","i");                                     | WIP 
23    | DS_r := DS_1[ calc Me_2:= replace (Me_1,"ello","i")];                  | WIP 
24    | DS_r:= instr(ds_1,"hello");                                            | WIP 
25    | DS_r := DS_1[calc Me_2:=instr(Me_1,"hello")];                          | WIP 
26    | DS_r := DS_1 [calc Me_10:= instr(Me_1, "o"), Me_20:=instr(Me_2, "o")]; | WIP
27    | DS_r := instr(DS_1, "o");                                              | WIP
28    | DS_r := length(DS_1);                                                  | WIP
29    | DS_r:= DS_1[calc Me_2:=length(Me_1)];                                  | WIP
30    | DS_r := DS_2 [calc Me_10:= length(Me_1), Me_20:=length(Me_2)];         | WIP
31    | DS_r := length(DS_2);                                                  | WIP

>>>

## 4. Numeric Operators.
>>>
Test number     | VTL expresion         | Test result
:------------:    | :-------------       |:-------------:
32    | DS_r := + DS_1;                                       | WIP
33    | DS_r := DS_1 [calc Me_3 := + Me_1 ];                  | WIP
34    | DS_r := - DS_1;                                       | WIP
35    | DS_r := DS_1 [ calc Me_3 := - Me_1 ];                 | WIP
36    | DS_r := DS_1 + DS_2;                                  | WIP
37    | DS_r := DS_1 + 3;                                     | WIP
38    | DS_r := DS_1 [ calc Me_3 := Me_1 + 3.0 ];             | WIP
39    | DS_r := DS_1 - DS_2;                                  | WIP
40    | DS_r := DS_1 - 3;                                     | WIP
41    | DS_r := DS_1 [ calc Me_3 := Me_1 - 3 ];               | WIP
42    | DS_r := DS_1 * DS_2;                                  | WIP
43    | DS_r := DS_1 * -3;                                    | WIP
44    | DS_r := DS_1 [ calc Me_3 := Me_1 * Me_2 ];            | WIP
45    | DS_r := DS_1 / DS_2;                                  | WIP
46    | DS_r := DS_1 / 10;                                    | WIP
47    | DS_r := DS_1 [ calc Me_3 := Me_2 / Me_1 ];            | WIP
48    | DS_r := mod ( DS_1, DS_2 );                           | WIP
49    | DS_r := mod ( DS_1, 15 );                             | WIP
50    | DS_r := DS_1[ calc Me_3 := mod( DS_1#Me_1, 3.0 ) ];   | WIP
51    | DS_r := round(DS_1, 0);                               | WIP
52    | DS_r := DS_1 [ calc Me_10:= round( Me_1 ) ];          | WIP
53    | DS_r := DS_1 [ calc Me_20:= round( Me_1 , -1 ) ];     | WIP
54    | DS_r := trunc(DS_1, 0);                               | WIP 
55    | DS_r := DS_1[ calc Me_10:= trunc( Me_1 ) ];           | WIP 
56    | DS_r := DS_1[ calc Me_20:= trunc( Me_1 , -1 ) ];      | WIP 
57    | DS_r := ceil (DS_1);                                  | WIP
58    | DS_r := DS_1 [ calc Me_10 := ceil (Me_1) ];           | WIP
59    | DS_r := floor ( DS_1 );                               | WIP
60    | DS_r := DS_1 [ calc Me_10 := floor (Me_1) ];          | WIP
61    | DS_r := abs ( DS_1 );                                 | WIP
62    | DS_r := DS_1 [ calc Me_10 := abs(Me_1) ];             | WIP
63    | DS_r := exp(DS_1);                                    | WIP
64    | DS_r := DS_1 [ calc Me_1 := exp ( Me_1 ) ];           | WIP
65    | DS_r := ln(DS_1);                                     | WIP
66    | DS_r := DS_1 [ calc Me_2 := ln ( DS_1#Me_1 ) ];       | WIP
67    | DS_r := power(DS_1, 2);                               | WIP
68    | DS_r := DS_1[ calc Me_1 := power(Me_1, 2) ];          | WIP 
69    | DS_r := log ( DS_1, 2 );                              | WIP  
70    | DS_r := DS_1 [ calc Me_1 := log (Me_1, 2) ];          | WIP 
71    | DS_r := sqrt(DS_1);                                   | WIP
72    | DS_r := DS_1 [ calc Me_1 := sqrt ( Me_1 ) ];          | WIP
188   | DS_r := random(DS_1, 5);                              |
189   | DS_r := DS_1 [ calc Me_2 := random( Me_1, 8 ) ];      |
>>>

## 5. Comparison Operators.
>>>
Test number     | VTL expresion                                             | Test result
:------------:  |:----------------------------------------------------------|:-------------:
73    | DS_r := DS_1 = 0.08;                                      | WIP
74    | DS_r := DS_1 [ calc Me_2 := Me_1 = 0.08 ];                | WIP
75    | DS_r := DS_1 <> DS_2;                                     | WIP
76    | DS_r := DS_1 [ calc Me_2 := Me_1<>7.5 ];                  | WIP
77    | DS_r := DS_1 > 20;                                        | WIP
78    | DS_r := DS_1 [ calc Me_2 := Me_1 > 20 ];                  | WIP
79    | DS_r:= DS_1 > DS_2;                                       | WIP
80    | DS_r := DS_1 < 15000000;                                  | WIP
81    | DS_r:= between(ds1, 5,10);                                | WIP
82    | DS_r := DS_1 in { 0, 3, 6, 12 };                          | WIP
83    | DS_r := DS_1 [ calc Me_2:= Me_1 in { 0, 3, 6, 12 } ];     | WIP
84    | DS_r := DS_1#Id_2 in myGeoValueDomain;                    | 
85    | DS_r:= match_characters(ds1, "[:alpha:]{2}[:digit:]{3}"); | 
86    | DS_r := isnull(DS_1);                                     | WIP
87    | DS_r := DS_1[ calc Me_2 := isnull(Me_1) ];                | WIP
88    | DS_r := exists_in (DS_1, DS_2, all);                      | WIP
89    | DS_r := exists_in (DS_1, DS_2, true);                     | WIP
90    | DS_r := exists_in (DS_1, DS_2, false);                    | WIP


>>>

## 6. Boolean Operators.
>>>
Test number     | VTL expresion     | Test result
:------------:  | :-------------    |:-------------:
91    | DS_r:= DS_1 and DS_2;                          | WIP
92    | DS_r := DS_1 [ calc Me_2:= Me_1 and true ];    | WIP
93    | DS_r:= DS_1 or DS_2;                           | WIP
94    | DS_r:= DS_1 [ calc Me_2:= Me_1 or true ];      | WIP
95    | DS_r:=DS_1 xor DS_2;                           | WIP
96    | DS_r:= DS_1 [ calc Me_2:= Me_1 xor true ];     | WIP
97    | DS_r:= not DS_1;                               | WIP
98    | DS_r:= DS_1 [ calc Me_2 := not Me_1 ];         | WIP

>>>

## 7. Time Operators.
>>>
Test number     | VTL expresion                                                           | Test result
:------------:  |:------------------------------------------------------------------------|:-------------:
99    | DS_r := period_indicator ( DS_1 );                                      | WIP
100   | DS_r := DS_1 [ filter period_indicator ( Id_3 ) = "A" ];                | 
101   | DS_r := fill_time_series ( DS_1, single );                              | 
102   | DS_r := fill_time_series ( DS_1, all );                                 | 
103   | DS_r := fill_time_series ( DS_2, single );                              | 
104   | DS_r := fill_time_series ( DS_2, all );                                 | 
105   | DS_r := fill_time_series ( DS_3, single );                              | WIP
106   | DS_r := fill_time_series ( DS_3, all );                                 | WIP
107   | DS_r := fill_time_series ( DS_4, single );                              | WIP
108   | DS_r := fill_time_series ( DS_4, all );                                 | WIP
109   | DS_r := flow_to_stock ( DS_1 );                                         | WIP
110   | DS_r := flow_to_stock ( DS_2 );                                         | WIP
111   | DS_r := flow_to_stock ( DS_3 );                                         | WIP
112   | DS_r := flow_to_stock ( DS_4 );                                         | WIP
113   | DS_r := stock_to_flow ( DS_1 );                                         | WIP
114   | DS_r := stock_to_flow ( DS_2 );                                         | WIP
115   | DS_r := stock_to_flow ( DS_3 );                                         | WIP
116   | DS_r := stock_to_flow ( DS_4 );                                         | WIP
117   | DS_r := timeshift ( DS_1 , -1 );                                        | 
118   | DS_r := timeshift ( DS_2 , 2 );                                         | 
119   | DS_r := timeshift ( DS_3 , 1 );                                         | WIP
120   | DS_r := time_shift ( DS_3 , -1 );                                       | WIP
121   | DS_r := sum ( DS_1 ) group all time_agg ( "A" , _ , Me_1 );             | 
122   | DS_r := time_agg ( "Q", cast ( "2012M01", time_period, "YYYY\MMM" ) );  | 
123   | time_agg( "Q", cast("20120213", date, "YYYYMMDD"), _ , last );          | 
124   | time_agg(cast( "A", "2012M1", date, "YYYYMMDD"), _ , first );           | 
125   | cast ( current_date, string, "YYYY.MM.DD" );                            |
177   | DS_r := DS_1 [calc Me2 := datediff(Id_2, Me_1)];                        |
178   | DS_r := DS_1[ calc Me_2 := dateadd( Me_1, 2, "M" ) ];                   |
179   | DS_r := DS_1[ calc Me_2 := month (Me_1) ];                              |
180   | DS_r := DS_1[ calc Me_2 := year (Me_1) ];                               |
181   | DS_r := DS_1[ calc Me_2 := dayofmonth (Me_1) ];                         |
182   | DS_r := DS_1[ calc Me_2 := dayofyear (Me_1) ];                          |
183   | DS_r := DS_1[ calc Me_2 := daytoyear (Me_1) ];                          |
184   | DS_r := DS_1[ calc Me_2 := daytomonth (Me_1) ];                         |
185   | DS_r := DS_1[ calc Me_2 := yeartoday (Me_1) ];                          |
186   | DS_r := DS_1[ calc Me_2 := monthtoday (Me_1) ];                         |
>>>

## 8. Set Operators.
>>>
Test number     | VTL expresion     | Test result
:------------:  | :-------------    |:-------------:
126   | DS_r := union(DS_1,DS_2);                        | WIP
127   | DS_r := union ( DS_1, DS_2 );                    | WIP
128   | DS_r := intersect(DS_1,DS_2);                    | WIP
129   | DS_r := setdiff ( DS_1, DS_2 );                  | WIP
130   | DS_r := setdiff ( DS_1, DS_2 );                  | WIP
131   | DS_r := symdiff ( DS_1, DS_2 );                  | WIP

>>>
## 9. Hierarchical aggregation.
>>>
Test number     | VTL expresion     | Test result
:------------:  | :-------------    |:-------------:
132   | DS_r := hierarchy ( DS_1, HR_1 rule Id_2 non_null );            | 
133   | DS_r := hierarchy ( DS_1, HR_1 rule Id_2 non_zero );            | 
134   | DS_r := hierarchy ( DS_1, HR_1 rule Id_2 partial_null );        | 

>>>

## 10. Aggregate and Analytic Operators.

### Aggregate Operators.
>>>
Test number     | VTL expresion                                                                                            | Test result
:------------:  |:---------------------------------------------------------------------------------------------------------|:-------------:
135    | DS_r := avg ( DS_1 group by Id_1 );                                                                      | WIP
136    | DS_r := sum ( DS_1 group by Id_1, Id_3 );                                                                | WIP
137    | DS_r := avg ( DS_1 );                                                                                    | 
138    | DS_r := DS_1 [ aggr Me_2 := max ( Me_1 ) , Me_3 := min ( Me_1 ) group by Id_1 ];                         | 
139    | DS_r := sum ( DS_1 over ( order by Id_1, Id_2, Id_3 data points between 1 preceding and 1 following ) ); | 
140    | DS_r := count ( DS_1 group by Id_1 );                                                                    | WIP
141    | DS_r := count ( DS_1 group by Id_1 having count() > 2 );                                                 | 
142    | DS_r := min ( DS_1 group by Id_1 );                                                                      | WIP
143    | DS_r := max ( DS_1 group by Id_1 );                                                                      | WIP
144    | DS_r := median ( DS_1 group by Id_1 );                                                                   | WIP
145    | DS_r := sum ( DS_1 group by Id_1 );                                                                      | WIP
146    | DS_r := avg ( DS_1 group by Id_1 );                                                                      | WIP
147    | DS_r := stddev_pop ( DS_1 group by Id_1 );                                                               | WIP
148    | DS_r := stddev_samp ( DS_1 group by Id_1 );                                                              | WIP
149    | DS_r := var_pop ( DS_1 group by Id_1 );                                                                  | WIP
150    | DS_r := var_samp ( DS_1 group by Id_1 );                                                                 | WIP


>>>

### Analytic Operators.
>>>
Test number     | VTL expresion     | Test result
:------------:  | :-------------    |:-------------:
151     | DS_r := first_value ( DS_1 over ( partition by Id_1, Id_2 order by Id_3 data points between 1 preceding and 1 following);     | 
152     | DS_r := last_value ( DS_1 over ( partition by Id_1, Id_2 order by Id_3 data points between 1 preceding and 1 following ) );   | 
153     | DS_r := lag ( DS_1 , 1 over ( partition by Id_1 , Id_2 order by Id_3 ) );                                                     | 
154     | DS_r := lead ( DS_1 , 1 over ( partition by Id_1 , Id_2 order by Id_3 ) );                                                    | 
155     | DS_r := DS_1 [ calc Me2 := rank ( over ( partition by Id_1 , Id_2 order by Me_1 ) ) ];                                        | 
156     | DS_r := ratio_to_report ( DS_1 over ( partition by Id_1, Id_2 ) )                                                             | 

>>>

## 11. Data validation Operators.
>>>
Test number     | VTL expresion     | Test result
:------------:  | :-------------    |:-------------:
157     | DS_r := check_datapoint ( DS_1, dpr1 );                                | WIP
158     | DS_r := check_datapoint ( DS_1, dpr1 all );                            | WIP 
159     | DS_r := check_hierarchy ( DS_1, HR_1 rule Id_2 partial_null all );     | WIP
160     | DS_r := check ( DS1 >= DS2 imbalance DS1 - DS2 );                      | WIP

>>>

## 12. Conditional Operators.
>>>
Test number     | VTL expresion     | Test result
:------------:  | :-------------    |:-------------:
161      | DS_r := if ( DS_cond#Id_4 = ""F"" ) then DS_1 else DS_2;    | WIP
162      | DS_r := nvl ( DS_1, 0 );                                    | WIP 
187      |DS_r := DS_1 [calc Me_2 := case when Me_1 <= 1 then 0        |
         |                   when Me_1 > 1 and Me_1 <= 10 then 1       |
         |                   when Me_1 > 10 then 10                    |
         |                   else 100];                                |
>>>

## 13. Clause Operators.
>>>
Test number     | VTL expresion     |            Test result             
:------------:  | :-------------    |:----------------------------------:
163      | DS_r := DS_1 [ filter Id_1 = 1 and Me_1 < 10 ];                                                                |                WIP                 
164      | DS_r := DS_1 [ calc Me_1:= Me_1 * 2 ];                                                                         |                WIP                 
165      | DS_r := DS_1 [ calc attribute At_1:= "EP" ];                                                                   |                WIP                 
166      | DS_r := DS_1 [ aggr Me_1:= sum( Me_1 ) group by Id_1 , Id_2 ];                                                 | 
167      | DS_r := DS_1 [ aggr Me_3:= min( Me_1 ) group except Id_3 ];                                                    | 
168      | DS_r := DS_1 [ aggr Me_1:= sum( Me_1 ), Me_2 := max( Me_1) group by Id_1 , Id_2 having avg (Me_1 ) > 2 ];      | 
169      | DS_r := DS_1 [ keep Me_1 ];                                                                                    |                WIP                 
170      | DS_r := DS_1 [ drop At_1 ];                                                                                    |                WIP                 
171      | DS_r := DS_1 [ rename Me_1 to Me_2, At_1 to At_2];                                                             |                WIP                 
172      | DS_r := DS_1 [ pivot Id_2, Me_1 ];                                                                             |                WIP                 
173      | DS_r := DS_1 [ unpivot Id_2, Me_1];                                                                            |                WIP                 
174      | DS_r := DS_1 [ sub Id_1 = 1, Id_2 = "A" ];                                                                     |                WIP                 
175      | DS_r := DS_1 [ sub Id_1 = 1, Id_2 = "B", Id_3 = "YY" ];                                                        |                WIP                 
176      | DS_r := DS_1 [ sub Id_2 = "A" ] + DS_1 [ sub Id_2 = "B" ];                                                     |                                    
>>>