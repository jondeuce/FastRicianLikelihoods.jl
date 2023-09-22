global CONSTANTS_STARTUP
if isempty(CONSTANTS_STARTUP) || CONSTANTS_STARTUP == false
    cd(fileparts(which(mfilename('fullpath'))))

    if ~exist('chebfun', 'dir')
        st = system('git clone https://github.com/chebfun/chebfun.git');
        if st == 0
            error("unable to install chebfun")
        end
    end
    addpath(genpath('chebfun'))

    if ~exist('jlcall.m', 'file')
        st = system('wget https://raw.githubusercontent.com/jondeuce/MATDaemon.jl/master/api/jlcall.m');
        if st == 0
            error("unable to download jlcall.m")
        end
    end

    jlcall('', ...
        'project', pwd, ... % activate a local Julia Project
        ... % 'setup', '/path/to/setup.jl', ... % run a setup script to load some custom Julia code
        'modules', {'FastRicianLikelihoods', 'ArbNumerics'}, ... % load a custom module and some modules from Base Julia
        'threads', 'auto', ... % use the default number of Julia threads
        'restart', true ... % start a fresh Julia server environment
    )
    CONSTANTS_STARTUP = true;
end

% besseli0x = chebfun(@(x) jlcall('x -> (x = ArbNumerics.ArbFloat.(x); return @.(Float64( exp(-abs(x)) * ArbNumerics.besseli(0, x) )))', {x}), [low, mid])
% besseli1x = chebfun(@(x) jlcall('x -> (x = ArbNumerics.ArbFloat.(x); return @.(Float64( exp(-abs(x)) * ArbNumerics.besseli(1, x) )))', {x}), [low, mid])
% besseli2x = chebfun(@(x) jlcall('x -> (x = ArbNumerics.ArbFloat.(x); return @.(Float64( exp(-abs(x)) * ArbNumerics.besseli(2, x) )))', {x}), [low, mid])

% low = 0.5;
% mid = 5.0;
% dom = [low^2, mid^2]
if ~exist('f', 'var') || ~isequal(domain(f), dom)
    f = chebfun(@(x) jlcall('y -> (y = ArbNumerics.ArbFloat.(y); x = sqrt.(y); return @.(Float64( ArbNumerics.besseli(1, x) / ArbNumerics.besseli(0, x) / x )))', {x}), dom)
    % f = chebfun(@(x) jlcall('y -> if y == 0; return 1/8; else; y = ArbNumerics.ArbFloat.(y); x = inv.(y); return @.(Float64( x * (-0.5 - x * (ArbNumerics.besseli(1, x) / ArbNumerics.besseli(0, x) - 1)) )) end', {x}), dom)
end

% num = 4;
% den = 4;
% mode = 'pade';
[p, q, num, den, score] = search_rational_approximant(f, 1:16, 0:0, mode, 'single')

function [p_best, q_best, num_best, den_best, score_best] = search_rational_approximant(f, num_range, den_range, mode, prec)

    best_score = Inf;
    p_list = {}; q_list = {}; num_list = []; den_list = []; score_list = [];

    for num = num_range
        for den = den_range
            try
                [p, q, ~] = rational_approximant(f, num, den, mode);
            catch e
                continue
            end
            abserr = (f - p / q) / precision(prec);
            relerr = ((f - p / q) / f) / precision(prec);
            score = num + den + (max(abserr) - min(abserr)) / 2 + (max(relerr) - min(relerr)) / 2;
            if score < best_score
                best_score = score;
                p_list{end+1} = p; q_list{end+1} = q; num_list(end+1) = num; den_list(end+1) = den; score_list(end+1) = score;
            end
        end
    end

    [score_list, I] = sort(score_list);
    p_list = p_list(I); q_list = q_list(I); num_list = num_list(I); den_list = den_list(I);

    for ii = min(3, numel(score_list)):-1:1
        summarize_rational_approximant(f, p_list{ii}, q_list{ii}, NaN, mode, prec, true);
    end

    p_best = p_list{1}; q_best = q_list{1}; num_best = num_list(1); den_best = den_list(1); score_best = score_list(1);

end

function [p, q, err] = rational_approximant(f, num, den, mode)

    if strcmpi(mode, 'cf')
        % Caratheodory-Fejer approximant:
        %   This approximation is a near-best approximation that is often indistinguishable from
        %   the true best approximation (minimax), but often much faster to compute.
        [p, q, r, err] = cf(f, num, den);
    elseif strcmpi(mode, 'pade')
        % "Clenshaw-Lord" Chebyshev-Pade approximation
        [p, q, r] = chebpade(f, num, den);
        err = NaN;
    elseif strcmpi(mode, 'rat')
        % Compute a type (m, n) interpolant through m+n+1 Chebyshev points
        [p, q, r] = ratinterp(f, num, den);
        err = NaN;
    else
        % Minimax rational approximant:
        %   Slow to compute, may be numerically stable for large numerator/denominator degrees.
        [p, q, r, err] = minimax(f, num, den);
    end

    [p, q] = normalize_rational(p, q);
end

function [p, q] = normalize_rational(p,q)
    q_coeffs = poly(q);
    p = p / q_coeffs(end);
    q = q / q_coeffs(end);
end

function [res] = minimaxiness(f)
    x = roots(diff(f));
    y = abs(f(x));
    res.min = min(y);
    res.max = max(y);
    res.var = (res.max - res.min) / (res.max + res.min);
end

function summarize_rational_approximant(f, p, q, err, mode, prec, ploterr)
    if nargin < 7; ploterr = false; end
    if nargin < 6; prec = 'double'; end

    p_coeffs = poly(p);
    q_coeffs = poly(q);
    p_coeffs = p_coeffs / q_coeffs(end);
    q_coeffs = q_coeffs / q_coeffs(end);

    res = minimaxiness(f - p/q);
    num = numel(p_coeffs) - 1;
    den = numel(q_coeffs) - 1;
    disp(res);

    fprintf('%s approximant (degree %d / %d):\n', mode, num, den)
    fprintf('E = %.17g\n', err);
    fprintf('V = %.17g\n', res.var);
    fprintf('N = '); print_as_tuple(p_coeffs(end:-1:1));
    fprintf('D = '); print_as_tuple(q_coeffs(end:-1:1));

    if ploterr
        figure; hold on
        abserr = (f - p / q) / precision(prec);
        relerr = ((f - p / q) / f) / precision(prec);
        subplot(2,1,1); plot(abserr); ylabel('abs err'); ylim([min(abserr), max(abserr)]);
        title(sprintf('%s approximant (degree %d / %d)', mode, num, den));
        subplot(2,1,2); plot(relerr); ylabel('rel err'); ylim([min(relerr), max(relerr)]);
    end
end

function [e] = precision(prec)
    if strcmpi(prec, 'single')
        e = double(eps('single'));
    elseif strcmpi(prec, 'double')
        e = eps('double');
    else
        error('unknown precision: %s', prec)
    end
end

function print_as_tuple(x)
    fprintf('(')
    for ii = 1:numel(x)-1
        fprintf('%.17g, ', x(ii))
    end
    fprintf('%.17g)\n', x(end))
end
