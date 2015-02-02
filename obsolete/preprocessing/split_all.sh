for f in *.ndpi
do
  echo "Spliting into tiffs - $f"
  /oasis/projects/nsf/csd181/yuncong/ndpisplit "${f}"
done

for level in macro x0.078125 x0.3125 x1.25 x5 x20
do 
  mkdir $level
  mv *_${level}_z0.tif $level
done

mkdir map
mv *map* map
