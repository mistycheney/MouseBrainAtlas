#!/bin/bash
for img in  $( ls);do
    convert $img -flip $img
done
