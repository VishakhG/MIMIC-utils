/*
Height weight for bmi
*/


-- ------------------------------------------------------------------
-- Title: Extract height and weight for hadm_ids
-- Description: This query gets the first, minimum, and maximum weight and height
--        for a single hadm_id. It extracts data from the CHARTEVENTS table.
-- MIMIC version: MIMIC-III v1.2
-- Created by: Erin Hong, Alistair Johnson
-- ------------------------------------------------------------------

-- -- DROP VIEW franck.heightweight;
-- DROP TABLE heightweight;
-- CREATE TABLE heightweight
-- AS
-- WITH FirstVRawData AS
--   (SELECT c.charttime,
--     c.itemid,c.subject_id,c.hadm_id,
--     CASE
--       WHEN c.itemid IN (762, 763, 3723, 3580, 3581, 3582)
--       THEN 'WEIGHT'
--       WHEN c.itemid IN (920, 1394, 4187, 3486, 3485, 4188)
--       THEN 'HEIGHT'
--     END AS parameter,
--     CASE
--       WHEN c.itemid   IN (3581)
--       THEN c.valuenum * 0.45359237
--       WHEN c.itemid   IN (3582)
--       THEN c.valuenum * 0.0283495231
--       WHEN c.itemid   IN (920, 1394, 4187, 3486)
--       THEN c.valuenum * 2.54
--       ELSE c.valuenum
--     END AS valuenum
--   FROM mimiciii.chartevents c
--   WHERE c.valuenum   IS NOT NULL
--   AND ( ( c.itemid  IN (762, 763, 3723, 3580, -- Weight Kg
--     3581,                                     -- Weight lb
--     3582,                                     -- Weight oz
--     920, 1394, 4187, 3486,                    -- Height inches
--     3485, 4188                                -- Height cm
--     )
--   AND c.valuenum <> 0 )
--     ) )
--   --)

--   --select * from FirstVRawData
-- , SingleParameters AS (
--   SELECT DISTINCT subject_id,
--          hadm_id,
--          parameter,
--          first_value(valuenum) over (partition BY subject_id, hadm_id, parameter order by charttime ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS first_valuenum,
--          MIN(valuenum) over (partition BY subject_id, hadm_id, parameter order by charttime ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)         AS min_valuenum,
--          MAX(valuenum) over (partition BY subject_id, hadm_id, parameter order by charttime ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)         AS max_valuenum
--     FROM FirstVRawData



-- --   ORDER BY subject_id,
-- --            hadm_id,
-- --            parameter
--   )
-- --select * from SingleParameters
-- , PivotParameters AS (SELECT subject_id, hadm_id,
--     MAX(case when parameter = 'HEIGHT' then first_valuenum else NULL end) AS height_first,
--     MAX(case when parameter =  'HEIGHT' then min_valuenum else NULL end)   AS height_min,
--     MAX(case when parameter =  'HEIGHT' then max_valuenum else NULL end)   AS height_max,
--     MAX(case when parameter =  'WEIGHT' then first_valuenum else NULL end) AS weight_first,
--     MAX(case when parameter =  'WEIGHT' then min_valuenum else NULL end)   AS weight_min,
--     MAX(case when parameter =  'WEIGHT' then max_valuenum else NULL end)   AS weight_max
--   FROM SingleParameters
--   GROUP BY subject_id,
--     hadm_id
--   )
-- --select * from PivotParameters
-- SELECT f.hadm_id,
--   f.subject_id,
--   ROUND( cast(f.height_first as numeric), 2) AS height_first,
--   ROUND(cast(f.height_min as numeric),2) AS height_min,
--   ROUND(cast(f.height_max as numeric),2) AS height_max,
--   ROUND(cast(f.weight_first as numeric), 2) AS weight_first,
--   ROUND(cast(f.weight_min as numeric), 2)   AS weight_min,
--   ROUND(cast(f.weight_max as numeric), 2)   AS weight_max

-- FROM PivotParameters f
-- ORDER BY subject_id, hadm_id;

-- COPY heightweight TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/height_weight.csv' DELIMITER ',' CSV HEADER;






-- /*

-- Demographic data


-- */
-- COPY(
-- WITH first_admission_time AS
-- (
--   SELECT
--       p.subject_id, p.dob, p.gender, a.hadm_id
--       , MIN (a.admittime) AS first_admittime
--       , MIN( ROUND( (cast(admittime as date) - cast(dob as date)) / 365.242,2) )
--           AS first_admit_age
--   FROM patients p
--   INNER JOIN admissions a
--   ON p.subject_id = a.subject_id
--   GROUP BY p.subject_id, p.dob, p.gender,a.hadm_id
--   ORDER BY p.subject_id
-- )
-- SELECT
--     hadm_id,  first_admit_age
--     , CASE
--         -- all ages > 89 in the database were replaced with 300
--         WHEN first_admit_age >= 18
--             THEN 'adult'
--         ELSE 'NOTADULT'
--         END AS age_group
-- FROM first_admission_time
-- ORDER BY subject_id)
-- TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/demographics.csv'
-- DELIMITER ','CSV HEADER;

/*
NCT02659839, Mortality in Cancer Patients Admitted to the Intensive Care Unit in a Resource-limited Setting, tag: 1

*/

Copy(
SELECT DISTINCT a.hadm_id
FROM procedures_icd dgn, admissions a
WHERE dgn.HADM_ID = a.HADM_ID AND Cast(dgn.ICD9_CODE as integer) between 140 AND 239)--Cancer Codes
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/cancers.csv'
With CSV DELIMITER ','; 

Copy(
SELECT DISTINCT a.hadm_id
FROM procedures_icd pcd, admissions a
WHERE pcd.HADM_ID = a.HADM_ID AND pcd.ICD9_CODE = '9670')--mechanical ventilation
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/mechanical_ventilation.csv'
With CSV DELIMITER ',';


Copy(
SELECT DISTINCT a.hadm_id
FROM procedures_icd pcd, admissions a 
WHERE pcd.HADM_ID = a.HADM_ID AND  pcd.ICD9_CODE = '0017' -- Infusion of vassopressor
)
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/vasopressor_infusion.csv'
With CSV DELIMITER ',';

Copy(
SELECT DISTINCT a.hadm_id
FROM procedures_icd pcd, admissions a
WHERE pcd.HADM_ID = a.HADM_ID AND pcd.ICD9_CODE = '3995')-- Hemodialysis (renal replacement therapy?)
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/Hemodialysis.csv'
With CSV DELIMITER ',';


/*
* NCT02872792, Early Mobilisation in Intensive Care Unit : Interest of Cyclo-ergometry in Patients With Septic Chock

*/

Copy(
SELECT DISTINCT a.hadm_id
FROM diagnoses_icd dgn, admissions a
WHERE  dgn.HADM_ID = a.HADM_ID AND dgn.ICD9_CODE = '99591') --sepsis
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/sepsis.csv'
With CSV DELIMITER ',';

Copy(
SELECT DISTINCT a.hadm_id
FROM procedures_icd pcd, admissions a
WHERE pcd.HADM_ID = a.HADM_ID AND pcd.ICD9_CODE LIKE '96%') -- Other invasive ventilation (tracheal intubation proxy)
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/tracheal_intubation.csv'
With CSV DELIMITER ',';


Copy(
SELECT DISTINCT a.hadm_id
FROM prescriptions psc, admissions a
WHERE psc.HADM_ID = a.ROW_iD AND psc.DRUG_NAME_GENERIC IN ('Nitroglycerine','Dopamine','Dopamine Drip','Levophed','Epinephrine-k','Phenylephrine','Epinephrine','Dobutamine Drip','Levophed-k','Epinephrine Drip','Nitroglycerin','Vasopressin','Lidocaine','Insulin','Nitroglycerine-k','Dobutamine','Milrinone','Norepinephrine') 
)-- Use of vasopressor drugs
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/vasopressor_GENERIC_drugs.csv'
With CSV DELIMITER ',';

Copy(
SELECT DISTINCT a.hadm_id
FROM prescriptions psc, admissions a
WHERE psc.HADM_ID = a.HADM_ID AND psc.DRUG_NAME_POE in ('Isuprel','Nitroglycerine','Dopamine','Dopamine Drip','Levophed','Epinephrine-k','Phenylephrine','Epinephrine','Dobutamine Drip','Levophed-k','Epinephrine Drip','Nitroglycerin','Vasopressin','Lidocaine','Insulin','Nitroglycerine-k','Dobutamine','Milrinone','Norepinephrine') 
)
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/vasopressor_POE_drugs.csv'
With CSV DELIMITER ',';

 
Copy(
SELECT DISTINCT a.hadm_id
FROM diagnoses_icd dgn, admissions a
WHERE dgn.HADM_ID = a.HADM_ID AND dgn.ICD9_CODE = '78009') -- alteration of consciousness (proxy for -2 rass score)
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/sedated.csv'
With CSV DELIMITER ',';

--TODO age 


Copy(
SELECT DISTINCT a.hadm_id
FROM diagnoses_icd dgn, admissions a
WHERE dgn.HADM_ID = a.HADM_ID AND dgn.ICD9_CODE LIKE 'v854%') -- HIGH BMI)
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/high_bmi.csv'
With CSV DELIMITER ',';


Copy(
SELECT DISTINCT a.hadm_id
FROM diagnoses_icd dgn, admissions a
WHERE dgn.HADM_ID = a.HADM_ID AND (dgn.ICD9_CODE = 'v85' OR dgn.icd9_code = 'v851' OR dgn.icd9_code LIKE 'v852*' or dgn.icd9_code LIKE 'v853*')) -- low BMI)
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/low_bmi.csv'
With CSV DELIMITER ',';



COPY(
SELECT DISTINCT a.hadm_id
FROM diagnoses_icd dgn, admissions a
WHERE  dgn.HADM_ID = a.HADM_ID AND dgn.ICD9_CODE = '714' or dgn.ICD9_CODE = '714') -- rheumatoid athiritis
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/rheumatoid_arthritis.csv'
With CSV DELIMITER ',';

COPY(
SELECT DISTINCT a.hadm_id
FROM diagnoses_icd dgn, admissions a
WHERE dgn.HADM_ID= a.HADM_ID AND dgn.ICD9_CODE LIKE '85%' ) --brain injuries
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/brain_injury.csv'
With CSV DELIMITER ',';


COPY(
SELECT DISTINCT a.hadm_id
FROM diagnoses_icd pcd, admissions a
WHERE  pcd.HADM_ID = a.HADM_ID AND pcd.ICD9_CODE = '3897') -- central venous cather (femoral artery catheter)
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/central_venous_catheter.csv'
With CSV DELIMITER ',';

COPY(
SELECT DISTINCT a.hadm_id
FROM diagnoses_icd dgn, admissions a
WHERE dgn.HADM_ID = a.HADM_ID AND dgn.ICD9_CODE = '45340') --deep vein thrombosis
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/deep_vein_thrombosis.csv'
With CSV DELIMITER ',';

COPY(
SELECT DISTINCT a.hadm_id
FROM diagnoses_icd dgn, admissions a
WHERE dgn.HADM_ID = a.HADM_ID and dgn.ICD9_CODE = '7099') -- skin lesions
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/skin_lesions.csv'
With CSV DELIMITER ',';


COPY(
SELECT DISTINCT a.hadm_id
FROM diagnoses_icd dgn, admissions a
WHERE  dgn.HADM_ID = a.HADM_ID AND dgn.ICD9_CODE = '3965') -- extracorporeal membrane oxygenation
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/extracorporeal_membrane_oxygenation.csv'
With CSV DELIMITER ',';


/*
NCT01793363, tracheostomy and Weaning From Mechancal Ventilation : Evaluation of the Lung Ultrasound Score

*/

COPY(
SELECT DISTINCT a.hadm_id
FROM diagnoses_icd pcd, admissions a
WHERE pcd.HADM_ID = a.HADM_ID and  pcd.ICD9_CODE = 'V440') -- tracheostomy
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/tracheostomy.csv'
With CSV DELIMITER ',';


/*
 NCT01784159 aspirin treatment for severe sepsis
*/

--sepsis (already computed)
COPY(
SELECT DISTINCT a.hadm_id
FROM diagnoses_icd dgn, admissions a
WHERE dgn.HADM_ID = a.HADM_ID and dgn.ICD9_CODE = '78552') -- septic shock
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/septic_shock.csv'
With CSV DELIMITER ',';



COPY(
SELECT DISTINCT a.hadm_id
FROM diagnoses_icd dgn, admissions a
WHERE dgn.HADM_ID = a.HADM_ID and dgn.ICD9_CODE = '2875') -- thrombocytopenia low platelet count
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/thrombocytopenia.csv'
With CSV DELIMITER ',';

COPY(
SELECT DISTINCT a.hadm_id
FROM diagnoses_icd dgn, admissions a
WHERE dgn.HADM_ID = a.HADM_ID and dgn.ICD9_CODE = '650') -- pregnancy
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/pregnancy.csv'
With CSV DELIMITER ',';

COPY(
SELECT DISTINCT a.hadm_id
FROM diagnoses_icd dgn, admissions a
WHERE dgn.HADM_ID = a.HADM_ID and( dgn.ICD9_CODE = '78799' OR dgn.icd9_code = '5609' or dgn.icd9_code LIKE '569*') ) -- intestinal problems
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/intestinal_problems.csv'
With CSV DELIMITER ',';

COPY(
SELECT DISTINCT a.hadm_id
FROM diagnoses_icd dgn, admissions a
WHERE dgn.HADM_ID = a.HADM_ID and dgn.ICD9_CODE = '4590') -- bleeding 
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/bleeding.csv'
With CSV DELIMITER ',';

COPY(
SELECT DISTINCT a.hadm_id
FROM diagnoses_icd dgn, admissions a
WHERE dgn.HADM_ID = a.HADM_ID and dgn.ICD9_CODE = ' V146 ') -- history of allergy to anelgesic
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/anelgesic_allergy.csv'
With CSV DELIMITER ',';

COPY(
SELECT DISTINCT a.hadm_id
FROM prescriptions psc, admissions a
WHERE psc.hadm_id = a.hadm_id and (psc.DRUG_NAME_POE LIKE '%aspirin%' or psc.DRUG_NAME_GENERIC LIKE '%aspirin%' ))-- history of allergy to anelgesic
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/antiplatlet.csv'
With CSV DELIMITER ',';


COPY(
SELECT DISTINCT a.hadm_id
FROM diagnoses_icd dgn, admissions a
WHERE dgn.HADM_ID = a.HADM_ID AND dgn.ICD9_CODE LIKE ' 533&') -- peptic ulcer 
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/peptic_ulcer.csv'
With CSV DELIMITER ',';

COPY(
SELECT DISTINCT a.hadm_id
FROM diagnoses_icd dgn, admissions a
WHERE dgn.HADM_ID = a.HADM_ID AND dgn.ICD9_CODE = '95901') -- Traumatic brain injury
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/traumatic_brain_injury.csv'
With CSV DELIMITER ',';


COPY(
SELECT DISTINCT a.hadm_id
FROM diagnoses_icd dgn, admissions a
WHERE  dgn.HADM_ID = a.HADM_ID and dgn.ICD9_CODE = '43491') -- hemmoragic brain injury
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/hemorrhagic_brain_injury.csv'
With CSV DELIMITER ',';


COPY(
SELECT DISTINCT a.hadm_id
FROM diagnoses_icd dgn, admissions a
WHERE  dgn.HADM_ID = a.HADM_ID and dgn.ICD9_CODE = '431') --Intracerebral hemorrhage
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/Intracerebral hemorrhage .csv'
With CSV DELIMITER ',';

COPY(
SELECT DISTINCT a.hadm_id
FROM diagnoses_icd dgn, admissions a
WHERE dgn.HADM_ID = a.HADM_ID and dgn.ICD9_CODE LIKE '571%') -- liver cihrossis
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/liver_cirrhosis.csv'
With CSV DELIMITER ',';

COPY(
SELECT DISTINCT a.hadm_id
FROM procedures_icd dgn, admissions a
WHERE  dgn.HADM_ID = a.HADM_ID and dgn.ICD9_CODE = '39') -- epudurial catheter
TO '/data/ml2/vishakh/patient-similarity/temp/cohort-components/epidural_catheter.csv'
With CSV DELIMITER ',';

/*
* NCT01775956, A Retrospective Review of a Comprehensive Cohort of Septic Shock: Assessment of Critical Determinants of Outcomes. Inclusion criteria.
*/

--septic shock (already recorded)



/*
Age over 18
*/

