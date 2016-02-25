## Synopsis

This document describes the contents and the steps taken to reach the submitted problem1 score.

## Files uploaded 

1. CheckModelPerformance.py: This script runs popular classifiers on the training set and checks good performing models on the corpus using cross-validation.
2. correlations.py: This script uses the output files and checks for correlations among them. Usage: correlations.py "Classifier1-output.txt" "./Classifier2-output.txt"
3. document_classification_topic_score_v5.py: Based on the above scripts, we narrow down the better performing models, run the models again using tuning parameters, and write output files to disk.
4. Plurality_Vote.py: This script uses plurality voting by reading the above outputs, counting plurality, and writing the plurality output back to disk. Usage: Plurality_Vote "./*.txt" "./PluralityOut" 

