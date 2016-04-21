features_input="Features/features_all.csv"
output_dir="Outputs/Features_all"
classifiers=(adaboost decisiontree gradient_boosting logistic_regression random_forest svm)
for clf in "${classifiers[@]}" 
do
echo "python ${clf}.py $features_input $output_dir/${clf}.json"
python ${clf}.py $features_input $output_dir/${clf}.json 

echo "python ${clf}.py $features_input $output_dir/${clf}_with_tree_features.json --addTreeFeatures"
python ${clf}.py $features_input $output_dir/${clf}_with_tree_features.json --addTreeFeatures
done
