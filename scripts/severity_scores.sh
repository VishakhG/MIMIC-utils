#!bin/bash

psql -f /home/vrg251/mimic-code/concepts/firstday/urine-output-first-day.sql -d mimic
psql -f /home/vrg251/mimic-code/concepts/firstday/ventilation-first-day.sql -d mimic
psql -f /home/vrg251/mimic-code/concepts/firstday/vitals-first-day.sql -d mimic
psql -f /home/vrg251/mimic-code/concepts/firstday/gcs-first-day.sql -d mimic
psql -f /home/vrg251/mimic-code/concepts/firstday/labs-first-day.sql -d mimic
psql -f /home/vrg251/mimic-code/concepts/firstday/blood-gas-first-day.sql -d mimic
psql -f /home/vrg251/mimic-code/concepts/firstday/blood-gas-first-day-arterial.sql -d mimic
psql -f /home/vrg251/mimic-code/concepts/firstday/echo-data.sql -d mimic
psql -f /home/vrg251/mimic-code/concepts/ventilation-durations.sql -d mimic

psql -f /home/vrg251/mimic-code/concepts/severityscores/saps.sql -d mimic
psql -f /home/vrg251/mimic-code/concepts/severityscores/sapsii.sql -d mimic
psql -f /home/vrg251/mimic-code/concepts/severityscores/sofa.sql -d mimic
