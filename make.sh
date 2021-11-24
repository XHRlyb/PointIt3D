for dir in `ls ../ScanNet/scans`; do
  #echo `mkdir -p ScanNet/scans_test/${dir}`
  #echo `cp /mnt/e/ScanNet/scans_test/${dir}/${dir}{_vh_clean_2.ply,.txt} ScanNet/scans_test/${dir}`
  echo `python3 make_data.py --scanID $dir --data_dir ../ScanNet/scans --seed 42`
done