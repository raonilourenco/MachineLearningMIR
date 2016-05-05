import yaml
import numbers
import numpy as np
import pandas as pd
import os
import itertools
import sys
import getopt

def flatten(data, to_ignore = []):
    if isinstance(data, dict):
        tot = []
        feature_names =[]
        for key in data.keys():
            if key not in to_ignore:
                flattened, keys = flatten(data[key], to_ignore)
                if isinstance(flattened,list):
                    tot = tot + flattened
                else:
                    tot = tot + [flattened]
                if(len(keys)>0):
                    feature_names = feature_names + [key + '.' + k for k in keys] 
                else:
                    feature_names = feature_names + [key]
        return tot,feature_names
    elif isinstance (data, list):
        if isinstance(data[0],list):
            data = list(itertools.chain(*data))
            keys = [str(x) for x in range(len(data))]
            return (data,keys)
        else:
            keys = [str(x) for x in range(len(data))]
            return (data,keys)
    else:
        return (data,[])





def create_feature_table(path, to_ignore):
    classes = []
    features = []
    features_names = None
    for (folder, _, files) in os.walk(path):
        for i in range(len(files)):
            filename, file_extension = os.path.splitext(files[i])
            f = open(os.path.join(folder,files[i]),"r")
            data = yaml.load(f)
            f.close()
            if file_extension == '.out':
                feature_values, keys = flatten(data, to_ignore)
                if(not(features_names)):
                    feature_names = keys
                features.append(feature_values)
                classes.append(folder.split("/")[-1])
    return (classes, features,feature_names)


def main(argv):
                             
    try:                                
        opts, args = getopt.getopt(argv, "sdi", ["source=", "destination=","ignore="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(str(err)) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    source = None
    destination = None
    ignore=["metadata",  "beats_position", "chords_key", "chords_scale","key_key","key_scale"]
    for o, a in opts:
        if o in ("-s", "--source"):
            source = a
        elif o in ("-d", "--destination"):
            destination = a
        elif o in ("-i", "--ignore"):
            ignore = a.split(',')
        else:
            assert False, "unhandled option"
    if(source and destination):
        (classes,features,feature_names) = create_feature_table(source,ignore)
        dframe = pd.DataFrame(features)
        dframe.columns = feature_names 
        dframe['class'] = classes
        dframe.to_csv(destination, sep=',', index=False)
    else:
        print("Specify source and destination")        
if __name__ == "__main__":
    main(sys.argv[1:])
