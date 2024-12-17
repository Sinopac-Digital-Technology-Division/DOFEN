### clone tabular benchmark ###
git clone https://github.com/LeoGrin/tabular-benchmark.git

### the build data script uses tabular benchmark preprocessing functions ###
cp build_tabular_benchmark_data.py ./tabular-benchmark/src/build_tabular_benchmark_data.py

### build data ###
cd ./tabular-benchmark/src/
python build_tabular_benchmark_data.py
