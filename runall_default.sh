#features_input="Features/features_all.csv"
#output_dir="Outputs/Features_all"
#classifiers=(adaboost decisiontree gradient_boosting logistic_regression random_forest svm)
#for clf in "${classifiers[@]}" 
#do
#echo "python ${clf}.py $features_input $output_dir/${clf}.json"
#python ${clf}.py $features_input $output_dir/${clf}.json 

#echo "python ${clf}.py $features_input $output_dir/${clf}_with_tree_features.json --addTreeFeatures"
#python ${clf}.py $features_input $output_dir/${clf}_with_tree_features.json --addTreeFeatures
#done


#features_input="Features/features_selected_all.csv"
#output_dir="Outputs/Features_selected_all"
#classifiers=(adaboost decisiontree gradient_boosting logistic_regression random_forest svm)
#for clf in "${classifiers[@]}" 
#do
#echo "python ${clf}.py $features_input $output_dir/${clf}.json"
#python ${clf}.py $features_input $output_dir/${clf}.json 

#echo "python ${clf}.py $features_input $output_dir/${clf}_with_tree_features.json --addTreeFeatures"
#python ${clf}.py $features_input $output_dir/${clf}_with_tree_features.json --addTreeFeatures
#done




features_input="Features/features_lowlevel.csv"
output_dir="Outputs/Features_lowlevel"
classifiers=(adaboost decisiontree gradient_boosting logistic_regression random_forest svm)
for clf in "${classifiers[@]}" 
do
echo "python ${clf}.py $features_input $output_dir/${clf}.json"
python ${clf}.py $features_input $output_dir/${clf}.json 

echo "python ${clf}.py $features_input $output_dir/${clf}_with_tree_features.json --addTreeFeatures"
python ${clf}.py $features_input $output_dir/${clf}_with_tree_features.json --addTreeFeatures
done

features_input="Features/features_rhythm.csv"
output_dir="Outputs/Features_rhythm"
classifiers=(adaboost decisiontree gradient_boosting logistic_regression random_forest svm)
for clf in "${classifiers[@]}" 
do
echo "python ${clf}.py $features_input $output_dir/${clf}.json"
python ${clf}.py $features_input $output_dir/${clf}.json 

echo "python ${clf}.py $features_input $output_dir/${clf}_with_tree_features.json --addTreeFeatures"
python ${clf}.py $features_input $output_dir/${clf}_with_tree_features.json --addTreeFeatures
done

features_input="Features/features_tonal.csv"
output_dir="Outputs/Features_tonal"
classifiers=(adaboost decisiontree gradient_boosting logistic_regression random_forest svm)
for clf in "${classifiers[@]}" 
do
echo "python ${clf}.py $features_input $output_dir/${clf}.json"
python ${clf}.py $features_input $output_dir/${clf}.json 

echo "python ${clf}.py $features_input $output_dir/${clf}_with_tree_features.json --addTreeFeatures"
python ${clf}.py $features_input $output_dir/${clf}_with_tree_features.json --addTreeFeatures
done


