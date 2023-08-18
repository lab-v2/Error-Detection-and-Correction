# Metacognitive-error-correction
LabelMatrix-TimeDays-TrajectrotyMatrix.py   data process step 1

Instance_Creation_sequence1.py  data process step 2

special_LRCN_data.py    generate train, valid, test data

train_predict.py    train and test the model

speed_rule.py   speed rule

rule_out.py     improve the model performance through rule.



If you only want to use the rule_out.py file, just run this command:

python rule_out.py LRCN_F1_no_overlap_sequential no_overlap_sequential_10

LRCN_F1_no_overlap_sequential is the results(labels) from neural network
no_overlap_sequential_10 is the rule
