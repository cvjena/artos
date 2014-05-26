function convertmultiplemodels(models, filename)

% FFLD Grammar
% Mixture := nbModels Model*
% Model := nbParts bias Part*
% Part := nbRows nbCols nbFeatures xOffset yOffset a b c d value*


fileID = fopen(filename, 'w');

fprintf(fileID, '%d\n', length(models));

for i=1:length(models)

  model = models{i};

  if isfield(model, 'w')
    % This is a simple WHOHOG model
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
  else
    rhs = model.rules{model.start}(i).rhs;
    
    nbParts = length(rhs);
    
    % Assume the root filter is first on the rhs of the start rules
    if model.symbols(rhs(1)).type == 'T'
        % Handle case where there's no deformation model for the root
        root = model.symbols(rhs(1)).filter;
        bias = 0;
    else
        % Handle case where there is a deformation model for the root
        root = model.symbols(model.rules{rhs(1)}(1).rhs).filter;
        bias = model.rules{model.start}(i).offset.w;
    end
    
    fprintf(fileID, '%d %g\n', nbParts, bias);
    
    % FFLD add instead of subtracting the deformation cost
    def = -model.rules{rhs(1)}(1).def.w;
    
    % Swap features 28 and 31
    w = model.filters(root).w(:, :, [1:27 31 29 30 28 32]);
    
    assert(size(w,3) == 32);
    
    fprintf(fileID, '%d %d %d 0 0 %g %g %g %g\n', size(w,1), size(w,2), ...
        size(w,3), def);
    
    for y = 1:size(w,1)
        for x = 1:size(w,2)
            fprintf(fileID, '%g ', w(y, x, :));
        end
        
        fprintf(fileID, '\n');
    end
    
    for j = 2:nbParts
        part = model.symbols(model.rules{rhs(j)}(1).rhs).filter;
        anc = model.rules{model.start}(i).anchor{j};
        def = -model.rules{rhs(j)}(1).def.w;
        
        w = model.filters(part).w(:, :, [1:27 31 29 30 28 32]);
        
        assert(size(w,3) == 32);
        
        fprintf(fileID, '%d %d %d %d %d %g %g %g %g\n', size(w,1), ...
            size(w,2), size(w,3), anc(1), anc(2), def);
        
        for y = 1:size(w,1)
            for x = 1:size(w,2)
                fprintf(fileID, '%g ', w(y, x, :));
            end
            
            fprintf(fileID, '\n');
        end
    end

  end
end

fclose(fileID);
