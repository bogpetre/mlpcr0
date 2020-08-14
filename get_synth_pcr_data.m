% function [X, Y, ll, bb_to_X, bb, w] = get_synth_pcr_data(n,X_dims,w_df,latent_df,snrY,snrX,...)
%
% Generates a sample of Y outcomes that depend on X in some latent 
% subspace of X.
%
% Input ::
%
%   n           - Number of observations to sample
%
%   X_dims      - Number of IV. Ignored if 'cov' w also specified.
%
%   w_df        - Number of degrees of freedom of your covariance matrix. 
%                   This defines the number of non-noise dimensions in X. 
%                   If supplying custom covariance matrix you may not know
%                   w_df (e.g. if you use an empirical covariance mat). In
%                   that case this becomes a hyperparameter analagous to
%                   the dimensionality hyperparameter in PCR.
%
%   latent_df   - Number of latent IVs on which Y depends.
%
%   snrY        - signal to noise ratio for DV measurements
%
%   snrX        - signal to noise ratio for IV measurements
%
% Optional Input ::
%
%   'cov'       - followed by user supplied covariance matrix. X_dim
%                   argument is ignored if user supplies covariance matrix.
%                   w_df should be an estimate of the user supplied
%                   matrices df. If in doubt overshoot with w_df.
%
% Output ::
%
%   Y           - observed DVs with normally distributed additive noise.
%
%   X           - n x X_dims multivariate normal matrix of observed IVs 
%                   with w_df degrees of freedom.
%
%   ll          - latent dimensions in X
%
%   ll_to_Y     - true regression coefficients mapping ll to Y
%
%   X_to_Y      - true coefficients mapping X to Y
%
%   w           - true covariance matrix with latent_df degrees of freedom 
%                 that was used to generate the multivariate normal X. If
%                 user supplies covariance matrix, this will just be a copy
%                 of what the user supplied.
%

function [X, Y, ll, X_to_Y, ll_to_Y, w] = get_synth_pcr_data(n,X_dims,w_df,latent_df,snrY,snrX,varargin)
    
    % get covariance matrix for X
    w_chol = rand(X_dims,w_df);
    w = w_chol*w_chol';
    for i = 1:length(varargin)
        if ischar(varargin{i})
            switch varargin{i}
                case 'cov'
                    w = varargin{i+1};
                    X_dims = size(w,1);
                otherwise
                    error(['Did not understand ', varargin{i}]);
            end
        end
    end
    
    % sanity checks
    if X_dims <= w_df
        error('X_dim must be >= w_df (covariance matrix df).');
    end
    if X_dims <= latent_df
        error('X_dim must be >= latent_df defining X ~ Y relationship.');
    end
    if w_df < latent_df
        error('w_df (covariance matrix df) must be >= latent_df defining X ~ Y relationship.');
    end
    if w_df > n
        error('w_df (covariance matrix df) must be < n. If you fewer n are desired, simply select a subset post-hoc.');
    end
    
    % sample noise free instances
    
    % meaningful dimensions are some random subset of non-noise dimensions of X;
    ll_to_Y = zeros(min([X_dims,n]) - 1,1);
    ll_to_Y(randperm(w_df,latent_df)) = rand(latent_df,1) - 0.5;
    
    xx = mvnrnd(zeros(X_dims,1),w,n);
    [pc,ll] = pca(xx,'NumComponents',min([X_dims,n])-1);
    yy = ll*ll_to_Y;
    X_to_Y = pinv(pc)'*ll_to_Y;

    % add noise to sample
    e = std(yy)*snrY*randn(n,1);
    Y = yy + e; % outcome noise (e.g. noisy subjective report)
    X = xx + repmat(diag(w)'.^0.5*snrX,n,1).*randn(size(xx)); % add measurement noise (e.g. physiological noise)
end