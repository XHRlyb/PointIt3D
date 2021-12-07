for dir in `ls ../ScanNet/scans`; do
  # echo `mkdir -p ScanNet/scans_test/${dir}`
  python3 make_data.py --scanID $dir --data_dir ../ScanNet/scans
  cp ../ScanNet/scans/${dir}/${dir}{.aggregation.json,_vh_clean_2.0.010000.segs.json} ScanNet_with_eric/${dir}_with_eric
done
