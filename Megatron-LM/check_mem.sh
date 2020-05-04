#!/bin/bash

cat $1 | grep -v "#" | awk -v max=0 '{if ($2 > max) max = $2} END {print max}'
