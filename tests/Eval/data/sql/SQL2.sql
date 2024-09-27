SELECT a.Id_1, a.Id_2, a.Me_1, b.Me_3
FROM DS_1 as a 
LEFT JOIN DS_2 as b 
    ON a.Id_1= b.Id_1 
    AND a.Id_2= b.Id_2
WHERE a.Me_1 = 2 OR a.Me_1 = 5
ORDER BY b.Me_3 DESC;