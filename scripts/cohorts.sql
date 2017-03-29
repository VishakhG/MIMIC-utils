/*
Age over 18 
*/

WITH first_admission_time AS
(
  SELECT
      p.subject_id, p.dob, p.gender
      , MIN (a.admittime) AS first_admittime
      , MIN( ROUND( (cast(admittime as date) - cast(dob as date)) / 365.242,2) )
          AS first_admit_age
  FROM patients p
  INNER JOIN admissions a
  ON p.subject_id = a.subject_id
  GROUP BY p.subject_id, p.dob, p.gender
  ORDER BY p.subject_id
)
, age as
(
  SELECT
      subject_id, dob, gender
      , first_admittime, first_admit_age
      , CASE
          -- all ages > 89 in the database were replaced with 300
          -- we check using > 100 as a conservative threshold to ensure we capture all these patients
          WHEN first_admit_age > 100
              then '>89'
          WHEN first_admit_age >= 14
              THEN 'adult'
          WHEN first_admit_age <= 1
              THEN 'neonate'
          ELSE 'middle'
          END AS age_group
  FROM first_admission_time
)
select a.hadm_id
from age, a  
where age > 18






/*
NCT01760967 Dexmedetomidine for Sepsis in ICU Randomized Evaluation Trial (DESIRE)
*/

SELECT count(*) from(
SELECT a.hadm_id
    
FROM admissions a,
 d_icd_procedures pcd,
 prescriptions psc,
 d_icd_diagnoses dgn

WHERE pcd.ICD9_CODE = '9670' --Mechanical ventillation
      AND  dgn.ICD9_CODE != '571' -- liver disease
      AND dgn.ICD9_CODE != '410' -- myocardial infarction
      AND dgn.ICD9_CODE != '303' -- alcoholism
      AND dgn.ICD9_CODE != '3048' --drug dependency


limit 

) mynewtable;


/*
* NCT01793363, tracheostomy and Weaning From Mechanical Ventilation : Evaluation of the Lung Ultrasound Score
    
*/

SELECT count(*) from(
SELECT a.hadm_id
    
FROM admissions a,
 d_icd_procedures pcd,
 prescriptions psc,
 d_icd_diagnoses dgn

WHERE
    pcd.icd9_code LIKE ('31%') -- Tracheotamy
limit 3000

) mynewtable2;



/*
NCT02467621 
Stress Ulcer Prophylaxis in the Intensive Care Unit
*/

SELECT count(*) from(
SELECT a.hadm_id
    
FROM admissions a,
 d_icd_procedures pcd,
 prescriptions psc,
 d_icd_diagnoses dgn

WHERE
    dgn.icd9_code =     
    
limit 3000

) mynewtable3
