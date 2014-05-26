%--------------------------------------------------------------------------
% Implementation of the paper "Exact Acceleration of Linear Object
% Detectors", 12th European Conference on Computer Vision, 2012.
%
% Copyright (c) 2012 Idiap Research Institute, <http:%www.idiap.ch/>
% Written by Charles Dubout <charles.dubout@idiap.ch>
%
% This file is part of FFLD (the Fast Fourier Linear Detector)
%
% FFLD is free software: you can redistribute it and/or modify it under the
% terms of the GNU General Public License version 3 as published by the
% Free Software Foundation.
%
% FFLD is distributed in the hope that it will be useful, but WITHOUT ANY
% WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
% details.
%
% You should have received a copy of the GNU General Public License along
% with FFLD. If not, see
% <http:%www.gnu.org/licenses/>.
%--------------------------------------------------------------------------

function convertmodel(model, filename)
%
% convertmodel(model, filename) convert the models of [1, 2] version 4 into
% a file readable by FFLD.
%
% [1] P. Felzenszwalb, R. Girshick, D. McAllester and D. Ramanan.
%     Object Detection with Discriminatively Trained Part Based Models.
%     IEEE Transactions on Pattern Analysis and Machine Intelligence,
%     Vol. 32, No. 9, September 2010
%
% [2] P. Felzenszwalb, R. Girshick and D. McAllester.
%     Discriminatively Trained Deformable Part Models, Release 4.
%     http://people.cs.uchicago.edu/~pff/latent-release4/
%

fileID = fopen(filename, 'w');

nbModels = length(model.rules{model.start});

fprintf(fileID, '%d\n', nbModels);

for i = 1:nbModels
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

fclose(fileID);
