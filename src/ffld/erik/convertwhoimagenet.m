function convertwhoimagenet(synset)
  inmodeldir='/u/vis/erik/toyota-demo/imagenet-cache/%s.whomodel.mat'
  outmodeldir='/u/vis/erik/toyota-demo/imagenet-cache/%s.whomodel.txt'
  f=load(sprintf(inmodeldir,synset));
  convertwhomodel(f.model,sprintf(outmodeldir,synset));
end
