#!/bin/bash
if [[ -f $1 ]]; then
  cat $1 | egrep -i '(Site|sweep|Simu)'
else
  egrep -i '(Site|sweep|Simu)'
fi
