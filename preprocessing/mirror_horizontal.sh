#!/bin/bash
for img in  $( ls);do
    convert $img -flop $img
done
