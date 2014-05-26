#!/bin/sh

synset=$1
find /u/vis/erik/toyota-demo/imagenet-cache/Annotation/$synset/ -name \*xml -exec basename {} \; | sed s/.xml// > /u/vis/erik/toyota-demo/imagenet-cache/ImageSets/Main/$synset.txt
 
