function convertmultiplewhomodels(models, filename)

% FFLD Grammar
% Mixture := nbModels Model*
% Model := nbParts bias Part*
% Part := nbRows nbCols nbFeatures xOffset yOffset a b c d value*


fileID = fopen(filename, 'w');

fprintf(fileID, '%d\n', length(models));

for i=1:length(models)

  model = models{i};

  bias = -model.thresh;

  % only one part, which is the root model
  % Model := nbParts bias Part*
  fprintf(fileID, '%d %g\n', 1, bias);

  % Swap features 28 and 31 (whatever???)
  w = model.w(:, :, [1:27 31 29 30 28 32]);

  assert(size(w,3) == 32);

  % the following deformation cost is ignored in the FLD code anyway for
  % the root filter
  def = zeros(4,1);
  fprintf(fileID, '%d %d %d 0 0 %g %g %g %g\n', size(w,1), size(w,2), ...
      size(w,3), def);

  for y = 1:size(w,1)
      for x = 1:size(w,2)
          fprintf(fileID, '%g ', w(y, x, :));
      end
      fprintf(fileID, '\n');
  end
end

fclose(fileID);
