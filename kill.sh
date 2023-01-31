#!/bin/bash

# check input args
if [ $# -ne 1 ]; then
  echo "Usage: ./$0 xxx"
  exit 1
fi

echo "kill all $1*"

ps -ef | grep $1 | grep -v grep | grep -v kill.sh

pids=$(ps -ef | grep $1 | grep -v grep | grep -v kill.sh | awk '{print $2}')
#echo "pids:"
echo $pids


if test -z $pids; then
  echo "$1 is alread killed!"
else
  for pid in $pids
  do
    echo "kill $pid"
    #kill -9 $pid
    kill $pid
  done
fi
