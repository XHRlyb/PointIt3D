for dir in `ls ../ScanNet/scans`; do
  # echo `mkdir -p ScanNet/scans_test/${dir}`
  echo python make_data.py --scanID $dir --data_dir ..\\ScanNet\\scans --labels_dir ..\\ScanNet\\scannetv2-labels.combined.tsv
done
