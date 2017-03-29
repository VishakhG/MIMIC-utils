import gzip
import datetime
from pprint import pprint
import pandas as pd
import numpy as np
import cPickle as pickle
from os import listdir
from os.path import isfile, join as pjoin



class Experiment(object):
    """A simple class to keep track of experiments, the functions you
    want to log should be able to return metadata for a run and results for a run. """
    
    def __init__(self):
        self.results = []
        self.recording = False
        self.results_dir = None
        self.note = None
        self.save_prefix = None            

    def begin_experiment(self, fname=None, baseDir = None, note = ""):
        

        self.note = note
        
        if baseDir is  None:
            baseDir = "/data/ml2/vishakh/sap/baselines/"

        now = datetime.datetime.now()
        now = str(now.replace(second=0, microsecond=0))
        now = now.replace(" ", "_")
        
        if fname is None:
            fname = now
        else:
            fname = fname + now
        
        self.results_dir = pjoin(baseDir, fname)

        self.results = []
        self.recording = True
        

    def log(self, metadata, result):

        if type(metadata) != str:
            try:
                metadata = metadata.toString()
            except:
                raise ValueError(
                    'Metadata has no toString,please define or  pass string')

        if type(result) != str:
            try:
                result = result.to_string()
            except:
                raise ValueError(
                    'result  has no toString(), please define or pass string')
        metadata = metadata + "\nExperimentNote:\n" + self.note         
        self.results.append([metadata, result])
        
        self.save_experiment()

    def save_experiment(self, ftype="txt"):
        print "Saving Experiment"
        if ftype == "pk":
            self.save_to_pk()

        elif ftype == "txt":
            self.save_to_txt()
            
        elif ftype == "json":
            self.save_to_json()

        else:
            raise ValueError('The filetype extension is not supported')
    

    def view_experiment(self, path=None):
        if not self.results  == []:
            self._print_results()
        else:     
            if path is  None:
                if self.result_dir is not None:
                    path = self.result_dir
                    print self.result_dir
                else:
                    print "No result available"

            if path.endswith(".txt"):
                self._open_from_txt(path)
            if path.endswith(".pk"):
                self._open_from_pk(path)
            else:
                raise ValueError('The filetype extension is not supported')


    def _open_from_pk(self, path, compressed = True):
        print "opening pickle file"

        if compressed:
            f = gzip.open(path)

        else:
            f = open(path)

        out = pickle.load(f)

        self._print_results(out)


    def _open_from_txt(self, path):
        print "opening txt file"

        f = open(path)

        for line in f :
            print line

        f.close()


    def _print_results(self, result_obj=None):
        if result_obj is None:
            result_obj = self.results

        for result in result_obj:
            print(" \n")
            print("metadata\n")
            print("---------------------\n")
            print(result[0])
            print(" \n")
            print("results\n")
            print("---------------------\n")
            print(result[1])
            print(" \n")
            
            
    def end_experiment(self, ftype = "txt", view=True, save="True"):

        if save:
            self.save_experiment(ftype)
            
        print "Ending experiment"

        if view:
            self.view_experiment()
            
        self.results = []
        self.recording = False
        self.results_dir = None
        

    def save_to_pk(self, compressed = True):
        if not compressed:
            f = open(self.results_dir, 'wb')
        else:
            f = gzip.open(self.results_dir, 'wb') 

        print "Saving as compressed pickle file at {1}".format(self.results_dir) 
        pickle.dump(self.results, f)
        f.close()


    def save_to_txt(self):
        print "Saving to text file"
        f = open(self.results_dir + ".txt", "w")

        for result in self.results:
            f.write(" \n")
            f.write("metadata\n")
            f.write("---------------------\n")
            f.write(result[0])
            f.write(" \n")
            f.write("results\n")
            f.write("---------------------\n")
            f.write(result[1])
            f.write(" \n")
        f.close()


    def save_to_json(self):
        print "Saving to json"
        #TODO json
    
        


        


        

    

    
