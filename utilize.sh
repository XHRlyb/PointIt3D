for dir in `ls ../ScanNet/scans`; do
  # echo `mkdir -p ScanNet/scans_test/${dir}`
  echo --scanID $dir --visualize
done
