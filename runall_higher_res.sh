#!/bin/bash

python load_data.py --source /home/jorgehpo/Desktop/Dortmund_higher_res --destination Features/Higher_res/features_all.csv --ignore "metadata,beats_position,chords_key,chords_scale,key_key,key_scale"

python load_data.py --source /home/jorgehpo/Desktop/Dortmund_higher_res --destination Features/Higher_res/features_lowlevel.csv --ignore "rhythm,tonal,metadata,beats_position,chords_key,chords_scale,key_key,key_scale"

python load_data.py --source /home/jorgehpo/Desktop/Dortmund_higher_res --destination Features/Higher_res/features_rhythm.csv --ignore "lowlevel,tonal,metadata,beats_position,chords_key,chords_scale,key_key,key_scale"

python load_data.py --source /home/jorgehpo/Desktop/Dortmund_higher_res --destination Features/Higher_res/features_tonal.csv --ignore "rhythm,lowlevel,metadata,beats_position,chords_key,chords_scale,key_key,key_scale"

python selectfeatures.py Features/Higher_res/features_all.csv Features/Higher_res/features_selected_all.csv

features_input="Features/Higher_res/features_all.csv"
output_dir="Outputs/Higher_res/Features_all"
classifiers=(adaboost decisiontree gradient_boosting logistic_regression random_forest svm)
for clf in "${classifiers[@]}" 
do
echo "python ${clf}.py $features_input $output_dir/${clf}.json"
python ${clf}.py $features_input $output_dir/${clf}.json 

echo "python ${clf}.py $features_input $output_dir/${clf}_with_tree_features.json --addTreeFeatures"
python ${clf}.py $features_input $output_dir/${clf}_with_tree_features.json --addTreeFeatures
done


features_input="Features/Higher_res/features_selected_all.csv"
output_dir="Outputs/Higher_res/Features_selected_all"
classifiers=(adaboost decisiontree gradient_boosting logistic_regression random_forest svm)
for clf in "${classifiers[@]}" 
do
echo "python ${clf}.py $features_input $output_dir/${clf}.json"
python ${clf}.py $features_input $output_dir/${clf}.json 

echo "python ${clf}.py $features_input $output_dir/${clf}_with_tree_features.json --addTreeFeatures"
python ${clf}.py $features_input $output_dir/${clf}_with_tree_features.json --addTreeFeatures
done



features_input="Features/Higher_res/features_lowlevel.csv"
output_dir="Outputs/Higher_res/Features_lowlevel"
classifiers=(adaboost decisiontree gradient_boosting logistic_regression random_forest svm)
for clf in "${classifiers[@]}" 
do
echo "python ${clf}.py $features_input $output_dir/${clf}.json"
python ${clf}.py $features_input $output_dir/${clf}.json 

echo "python ${clf}.py $features_input $output_dir/${clf}_with_tree_features.json --addTreeFeatures"
python ${clf}.py $features_input $output_dir/${clf}_with_tree_features.json --addTreeFeatures
done

features_input="Features/Higher_res/features_rhythm.csv"
output_dir="Outputs/Higher_res/Features_rhythm"
classifiers=(adaboost decisiontree gradient_boosting logistic_regression random_forest svm)
for clf in "${classifiers[@]}" 
do
echo "python ${clf}.py $features_input $output_dir/${clf}.json"
python ${clf}.py $features_input $output_dir/${clf}.json 

echo "python ${clf}.py $features_input $output_dir/${clf}_with_tree_features.json --addTreeFeatures"
python ${clf}.py $features_input $output_dir/${clf}_with_tree_features.json --addTreeFeatures
done

features_input="Features/Higher_res/features_tonal.csv"
output_dir="Outputs/Higher_res/Features_tonal"
classifiers=(adaboost decisiontree gradient_boosting logistic_regression random_forest svm)
for clf in "${classifiers[@]}" 
do
echo "python ${clf}.py $features_input $output_dir/${clf}.json"
python ${clf}.py $features_input $output_dir/${clf}.json 

echo "python ${clf}.py $features_input $output_dir/${clf}_with_tree_features.json --addTreeFeatures"
python ${clf}.py $features_input $output_dir/${clf}_with_tree_features.json --addTreeFeatures
done


