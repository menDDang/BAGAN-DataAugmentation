#!/bin/bash

# set directories
base_dir="/home/feesh/projects/BAGAN"
data_dir="$base_dir/datasets"
list_dir="$base_dir/lists"
mkdir -p $data_dir $list_dir

word_list=$(cat word_list.txt)

# download speech command dataset & unzip
if [ ! -f speech_commands_v0.02.tar.gz ]; then
  echo "Download dataset"
  wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
  tar xzfv speech_commands_v0.02.tar.gz -C $data_dir
fi

# split data - validation & testing
for dtype in testing validation; do
  echo "Split data for $dtype"
  for word in $word_list; do
    mkdir -p $data_dir/$dtype/$word
  done

  for line in $(cat $data_dir/${dtype}_list.txt); do
    mv $data_dir/$line $data_dir/$dtype/$line
  done
done

# split data - training
rm -f $list_dir/training_list.txt
for word in $word_list; do
  echo "Split data for training"
  mkdir -p $data_dir/training/$word
  for file in $(ls $data_dir/$word); do
    mv $data_dir/$word/$file $data_dir/training/$word/$file
    echo $word/$file >> $list_dir/training_list.txt
  done
done

# delete empty folders
for word in $word_list; do
  rm -r $data_dir/$word
done

# create file lists
for dtype in training testing validation; do
  for word in $word_list; do
    echo "Create file list for $dtype $word"
    target="$list_dir/${dtype}_${word}_list.txt"
    rm -f $target
    for file in $(ls $data_dir/$dtype/$word); do
      line=$(echo $file | awk -F'.' '{print $1}')
      echo $line >> $target
    done
  done
done