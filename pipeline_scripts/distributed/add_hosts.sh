#!/usr/bin/bash

for i in {31..38}
do
  ssh-keygen -R gcn-20-${i}.sdsc.edu
  ssh-keyscan -H gcn-20-${i}.sdsc.edu >> ~/.ssh/known_hosts
done

for i in {41..48}
do
  ssh-keygen -R gcn-20-${i}.sdsc.edu
  ssh-keyscan -H gcn-20-${i}.sdsc.edu >> ~/.ssh/known_hosts
done

