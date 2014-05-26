function convertwhomodel(model, filename)

fileID = fopen(filename, 'w');

% there is only one model :)
fprintf(fileID, '%d\n', 1);

bias = -model.thresh;

% only one part or zero?
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

fclose(fileID);
