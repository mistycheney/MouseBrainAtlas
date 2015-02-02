#!/bin/bash
for img in  $( ls);do
    convert $img -page +0+0 -rotate -90 $img
done
