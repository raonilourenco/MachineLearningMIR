import json
import sys
import os

if len(sys.argv) != 2:
    print("usage: summaryOutputs [output_folder] > [output_file]")
else:
    results = []

    for root, dirs, files in os.walk(os.getcwd()):            
        for f in files:                        
            if f.lower().endswith((".json")):
                fname = os.path.join(root, f)
                fopen = open(fname, 'rb')
                json_data = json.load(fopen)
                fopen.close()
                results.append({'metric':json_data['metric_test'], 'file':fname})


    results = sorted(results, key= lambda x: x['metric'], reverse=True)

    for r  in results:
        print(r['file'] + ": " + str(r['metric']))