#!/usr/bin/bash

for fileIP in *.ipynb
do
    echo $fileIP
    file=$(echo "$fileIP" | cut -f 1 -d '.')
    fileMD=$file".md"
    echo $file
    jupytext $fileIP -o la.md
    diff $fileMD la.md
    if [ ! $? -eq 0 ]; then
	rm -f la.md
	echo "Difference between notebooks found $file"
	exit -1
    fi
done
rm -f la.md
