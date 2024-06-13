# Set default values
$dataset = if ($args[0]) { $args[0] } else { 'movielens' }
$model = if ($args[1]) { $args[1] } else { 'NeuMF-end' }
$mode = if ($args[2]) { $args[2] } else { 'flip' }
$drop_rate = if ($args[3]) { $args[3] } else { '0.2' }
$num_gradual = if ($args[4]) { $args[4] } else { '30000' }

$drop_rate = if ($mode -eq 'normal') {0} else {$drop_rate}
$num_gradual = if ($mode -eq 'normal') {0} else {$num_gradual}

# Run python script
python -u main.py --dataset=$dataset --model=$model --mode=$mode --drop_rate=$drop_rate --num_gradual=$num_gradual > "log/$dataset/$model-$mode`_$drop_rate-$num_gradual.log" 2>&1