import cPickle as pickle
import numpy as np
import pandas as pd 
import shelve
from os.path import join as pjoin
from MimicPatient import *
from MimicEvent import *
from Utils import *
from collections import defaultdict
from UMLS import UMLSUtils
import sys
sys.modules['UMLSUtils'] = UMLSUtils 
from datetime import datetime
import math
import argparse


class FeatureExtractor:
    def __init__ (self, umlsIndexPath, patientPath, timeWindow, outDir, nPatients, testing,
                  featName, targName):
        self.umlsPath = umlsIndexPath
        self.patientPath =  patientPath
        self.timeWindow = timeWindow
        self.outDir = outDir
        self.patient_list = pickle.load(open('/data/ml2/MIMIC3/processed/patients_list.pk'))
        self.patients = shelve.open('/data/ml2/MIMIC3/processed/patients.shlf')
        self.feature_mat = defaultdict(list)
        if not testing:
            self.umls_index =  pickle.load(
                open('/data/ml2/jernite/UMLS2016/PythonUMLS/umls_index.pk'))
            self.umls_index = self.umls_index.mappings
        else:
            self.umls_index = pickle.load(open('/data/ml2/vishakh/mimic-out-umls/umls_map_mini.pk'))
            
        self.target_vector = []
        self.currentTarget = 0
        if nPatients == float('inf'):
            self.nPatients = len(self.patient_list)
        else:
            self.nPatients = int(nPatients)

        self.featName = featName
        self.targName = targName
        self.admissionIDs = []

    def get_age_group(self, dob, admitTime):
        dob = self.string_to_date(dob)
        admitTime = self.string_to_date(admitTime)
        firstAdmitAge = self.date_diff(dob, admitTime, 'years')
        
        if firstAdmitAge > 100:
            feature = '89+'

        elif firstAdmitAge > 14:
            feature = 'adult'

        elif firstAdmitAge <= 1:
            feature = 'neonate'

        else:
            feature = 'NA'

        return feature

    def string_to_date(self, string):

        try:
            date = datetime.strptime(string, '%Y-%m-%d %H:%M:%S')
        except:
            date = None

        return date 


    def date_diff(self, d1, d2, units='hrs'):

        if None in [d1, d2]:
            return False
    
        delta = d1 - d2

        if units == 'hr':
            out = delta.total_seconds/3600.0 

        elif units == 'years':
            out = delta.days/365.0

        elif units == 'days':
            out = delta.days

        else:
            out = delta.total_seconds

        return abs(out)

    def died_in_period(self, aTime, dString):
        if dString == '':
            return False
        else:
            if dString == None:
                return False
            if self.within_period(aTime, dString, 48):
                return True

    def within_period(self, start, end, period):
        
        start = self.string_to_date(start)
        end = self.string_to_date(end)

        if None in [start, end]:
            return False
        
        delta = end - start

        return True if abs(delta.total_seconds() / 3600.0) <= period else False

    def icd9_to_concept(self, code, name, translator, ins_pos = 0):
        concept = code
        #insert seperator dot if nessesary
        if ins_pos:
            code = code[:ins_pos] + '.' + code[ins_pos:] 

        try:
            concept = translator['ICD9CM'][code]

        except:
            try:
                concept = translator['MTHICD9'][code]
            except:
                try:
                    concept = translator['STRING'][name]

                except:
                    return concept
        return concept


    def ndc_to_concept(self, code, names, translator):
        concept = code
        try:
            concept = translator['NDC'][code[2]]

        except:
            for name in names:
                try:
                    concept = translator['STRING'][namesd]
                except:
                    continue
        return concept


    def append_features(self, features):
        for feature in features:
            if feature in self.feature_mat.keys():
                #flip the last entry from 0 to a 1
                self.feature_mat[feature][-1] = 1

            else:
                self.feature_mat[feature] += ([0]* (max(map(len, self.feature_mat.values())) - 1)) + [1]
    
    def extract_admission_features(self, admission):

        for psc in admission.psc_events:
            concept = self.ndc_to_concept(
                psc.drug_codes, psc.drug_names, self.umls_index)


            if self.within_period(admission.in_time, psc.time, self.timeWindow):
                self.append_features(concept)

        for dgn in admission.dgn_events:
            concept = self.icd9_to_concept(
                dgn.code, dgn.name, self.umls_index, 3)

            if self.within_period(admission.in_time, dgn.time, self.timeWindow):
                self.append_features(concept)

        for pcd in admission.pcd_events:
            concept = self.icd9_to_concept(
                pcd.code, pcd.name, self.umls_index, 2
            )
            
            if self.within_period(admission.in_time, dgn.time,  self.timeWindow):
                self.append_features(concept)

    def extract_patient_features(self, patient):
        for admission_id in patient.admissions:
            self.admissionIDs.append(admission_id)
            
            for feature in self.feature_mat:
                self.feature_mat[feature] += [0]

            self.target_vector.append(self.currentTarget)
            admission = patient.admissions[admission_id]
            current_patient_age = self.get_age_group(patient.dob, admission.in_time)
            #TODO move the list logic
            self.append_features([current_patient_age])
            
            try:
               death_time =  admission.death_time
               died_in_ad = True
            except:
                died_in_ad = False
            # make sure the patient didn't die within the the desired period
            if died_in_ad and not self.died_in_period(admission.in_time, death_time):
                self.extract_admission_features(admission)
            
        
    def create_feature_matrix(self):
        patient_keys = self.patient_list


        for i, patient_key in enumerate(patient_keys[1:self.nPatients]):
            print "looking at patient" + str(i) + "of" + str(self.nPatients)
            print str(float(i) / self.nPatients) + " of the way there"
            patient = self.patients[patient_key]
            self.currentTarget = 0 if patient.expire_flag == 'N' else 1
            self.extract_patient_features(patient)

        #Pandas csv tools seem way faster
        features_out = pd.DataFrame.from_dict(self.feature_mat)
        targets_out = pd.DataFrame(self.target_vector)
        admission_ids = pd.DataFrame(self.admissionIDs)
        
        features_out.to_csv(self.outDir + self.featName + str(self.timeWindow)+ '.csv', index=False)
        targets_out.to_csv(self.outDir + self.targName + str(self.timeWindow) + '.csv', index=False)
        admission_ids.to_csv(self.outDir + "AdmissionIDs" + str(self.timeWindow) + '.csv', index=False)
        

   
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-test", "--testing", help="catchall testing flag, uses truncated everything"
                        , default = False)
    parser.add_argument("-u", "--umlsDir", help="The path to the umls index shelf file",
                        default='/data/ml2/jernite/UMLS2016/PythonUMLS/self.umls_index.pk')
    
    parser.add_argument("-o", "--outDir", help="Where to ouput the feature matrix",
                        default='/data/ml2/vishakh/temp/')

    parser.add_argument("-p", "--patientsDir", help="where the processed patient files  are",
                        default='/data/ml2/MIMIC3/processed/patients_list.pk')

    parser.add_argument("-fn", "--featName", help="feature matrix file name base",
                        default="feature_mat_first")

    parser.add_argument("-tn", "--targName", help="target vector file name base",
                        default="target_mat_first")

    parser.add_argument("-w", "--timeWindow", help="restrict feature_mat to a time window in hrs",
                        default=48)
    parser.add_argument("-n", "--nPatients", help="how many patients to loop over",
                        default = float('inf'))
    args = parser.parse_args()
    extractor = FeatureExtractor(args.umlsDir, args.patientsDir, args.timeWindow,
                                 args.outDir, args.nPatients, args.testing, args.featName,
                                 args.targName)
    extractor.create_feature_matrix()
if __name__ == '__main__':
    main()
        
