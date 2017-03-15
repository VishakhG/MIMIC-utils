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
import FindConcepts as fc
from itertools import islice

class FeatureExtractor:
    def __init__ (self, umlsCodeIndexPath, umlsStringIndexPath, umlsShelfPath,
                  patientPath, timeWindow, outDir, nPatients, testing,
                  featName, targName):

        self.patientPath =  patientPath
        self.timeWindow = timeWindow
        self.outDir = outDir
        self.patient_list = pickle.load(open('/data/ml2/MIMIC3/processed/patients_list.pk'))
        self.patients = shelve.open('/data/ml2/MIMIC3/processed/patients.shlf')
        self.feature_mat = defaultdict(list)
        self.umls_index =  pickle.load(open(umlsCodeIndexPath))
        self.umls_index = self.umls_index.mappings
        self.target_vector = []
        self.currentTarget = 0

        if nPatients == float('inf'):
            self.nPatients = len(self.patient_list)
        else:
            self.nPatients = int(nPatients)

        self.testing = testing
        self.featName = featName
        self.targName = targName
        self.admissionIDs = []
        self.metadata = {}
        self.umls_str_idx = None
        self.umls_dict = shelve.open(umlsShelfPath)
        self.trie = self.make_trie(umlsStringIndexPath)

    def make_trie(self, path):
        umls_str_idx = pickle.load(open(path))
        umls_str_idx = umls_str_idx.index

        trie = fc.read_umls(umls_str_idx)
        self.trie = trie
        self.umls_str_idx = umls_str_idx
        
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


    def date_diff(self, d1, d2, units='hr'):

        if None in [d1, d2]:
            return False
    
        delta = d1 - d2

        if units == 'hr':
            out = delta.total_seconds() / 3600.0 

        elif units == 'years':
            out = delta.days / 365.0

        elif units == 'days':
            out = delta.days

        else:
            out = delta.total_seconds()

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
        if period is None:
            return True
        
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

    def umls_to_name(self, translator, codes):
        res = []
        if type(codes) == list:
            for code in codes:
                try:
                    concept = translator[code]
                    res.append(concept.names)
    
                except:
                    continue
        else:
            try:
                concept = translator[codes]
                res.append(concept.names)
            except:
                pass
    
        return res
    
        
    def ndc_to_concept(self, code, names, translator):
        concept = code
        try:
            concept = translator['NDC'][code[2]]

        except:
            for name in names:
                try:
                    concept = translator['STRING'][name]
                except:
                    continue
        return concept


    def append_features(self, features):
        for feature in features:
            if feature in self.feature_mat.keys():
                #flip the last entry from 0 to a 1
                self.feature_mat[feature][-1] = 1
            #TODO: cleaner way to do this
            else:
                self.feature_mat[feature] += ([0]* (max(map(len, self.feature_mat.values())) - 1)) + [1]

    def add_to_provenance(self, concepts, tag):
        for concept in concepts:
            self.metadata[concept] = [tag, self.umls_to_name(self.umls_dict, concept)]
        
    def extract_admission_features(self, admission):

        for psc in admission.psc_events:
            concept = self.ndc_to_concept(
                psc.drug_codes, psc.drug_names, self.umls_index)


            if self.within_period(admission.in_time, psc.time, self.timeWindow):
                self.append_features(concept)
                self.add_to_provenance(concept, "PSC")

                
        for dgn in admission.dgn_events:
            concept = self.icd9_to_concept(
                dgn.code, dgn.name, self.umls_index, 3)

            if self.within_period(admission.in_time, dgn.time, self.timeWindow):
                self.append_features(concept)
                self.add_to_provenance(concept, "DGN")

        for pcd in admission.pcd_events:
            concept = self.icd9_to_concept(
                pcd.code, pcd.name, self.umls_index, 2
            )
            
            if self.within_period(admission.in_time, dgn.time,  self.timeWindow):
                self.append_features(concept)
                self.add_to_provenance(concept, "PCD")

        for note in admission.nte_events:
            if self.within_period(admission.in_time, note.time, self.timeWindow):
                words = note.note_text.split()
                concepts = fc.find_concepts(words, self.trie, self.umls_str_idx)
                #flatten TODO:cleaner flatten function
                concepts = [item for sublist in [concept[1] for concept in concepts]
                            for item in sublist]
                f = lambda x: "NTE" + x
                concepts = map(f, concepts)
                self.append_features(concepts)
                self.add_to_provenance(concepts, "NTE")

            
    def extract_patient_features(self, patient):
        for admission_id in patient.admissions:
            if admission_id == "":
                continue
           
            self.admissionIDs.append(admission_id)
            
            for feature in self.feature_mat:
                self.feature_mat[feature] += [0]

            admission = patient.admissions[admission_id]
            in_time = self.string_to_date(admission.in_time)
            length_of_stay__i = self.date_diff(in_time, self.string_to_date(admission.out_time))

            if admission.death_time != '':
                survival_i = self.date_diff(in_time, self.string_to_date(admission.death_time))
            else:
                survival_i = None
                
            self.target_vector.append([self.currentTarget, length_of_stay__i, survival_i])
           
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

        #Pandas csv tools seem way faster than other ways to do this
        features_out = pd.DataFrame.from_dict(self.feature_mat)
        targets_out = pd.DataFrame(self.target_vector)
        admission_ids = pd.DataFrame(self.admissionIDs)

        
        features_out.to_csv(self.outDir + self.featName + str(self.timeWindow)+ '.csv', index=False)
        targets_out.to_csv(self.outDir + self.targName + str(self.timeWindow) + '.csv', index=False)
        admission_ids.to_csv(self.outDir + "AdmissionIDs" + str(self.timeWindow) + '.csv', index=False)
        pickle.dump(self.metadata, open(self.outDir + "metadata", 'wb'))

   
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-test", "--testing",
        help="flag to modify behavior for testing", default = False
    )
    parser.add_argument(
        "-u", "--umls_code_index_dir",
        help="The path to the umls code index file",
        default='/data/ml2/UMLS/shelf_files/umls_index.pk'
    )
    
    parser.add_argument(
        "-u2", "--umls_string_index_dir",
        help =  "the path to the umls string index file",
        default = '/data/ml2/UMLS/PyUMLS_parsed/umls_str_index.pk'
    )

    parser.add_argument(
        "-u3", "--umls_shelf_dir",
        help = "the path to the umls shelf file",
        default = '/data/ml2/UMLS/shelf_files/umls_shelve_dict.shlf'
    )
    
    parser.add_argument(
        "-o", "--outDir",
        help="Where to ouput the feature matrix",
        default='/data/ml2/vishakh/patient-similarity/temp/'
    )

    parser.add_argument(
        "-p", "--patientsDir",
        help="where the processed patient files are",
        default='/data/ml2/MIMIC3/processed/patients_list.pk'
    )

    parser.add_argument(
        "-fn", "--featName",
        help="feature matrix file name base",
        default="features"
    )

    parser.add_argument(
        "-tn", "--targName",
        help="target vector file name base",
        default="targets"
    )

    parser.add_argument(
        "-w", "--timeWindow",
        help="restrict feature_mat to a time window in hrs",
        default=float('inf')
    )
    
    parser.add_argument(
        "-n", "--nPatients",
        help="how many patients to loop over",
        default = float('inf')
    )

    args = parser.parse_args()
    
    extractor = FeatureExtractor(
        args.umls_code_index_dir, args.umls_string_index_dir,
        args.umls_shelf_dir, args.patientsDir,
        args.timeWindow, args.outDir,
        args.nPatients, args.testing,
        args.featName, args.targName)

    extractor.create_feature_matrix()
if __name__ == '__main__':
    main()
        
