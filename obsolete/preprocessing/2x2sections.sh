#!/bin/bash
for img in  $( ls);do
    fold=${img/.tif/_2x2_sections};
    mkdir $fold
    output=$fold"/"${img%.tif};
    convert $img -crop 50X50% ${output}_%d.tif
done
