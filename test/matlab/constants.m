if ~exist('CONSTANTS_STARTUP', 'var')
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

low = 0.5;
mid = 15.0;
f = chebfun(@(x) jlcall('y -> (y = ArbNumerics.ArbFloat.(y); x = sqrt.(y); return @.(Float64( ArbNumerics.besseli(1, x) / ArbNumerics.besseli(0, x) / x )))', {x}), [low^2, mid^2])

[p, q, err] = rational_approximant(f, 7, 7, 'minimax');
summarize_rational_approximant(f, p, q, err);

function [p, q, err] = rational_approximant(f, n, m, mode)

    if strcmpi(mode, 'cf')
        % Caratheodory-Fejer approximant:
        %   This approximation is a near-best approximation that is often indistinguishable from
        %   the true best approximation (minimax), but often much faster to compute.
        [p, q, r, err] = cf(f, n, m);
    else
        % Minimax rational approximant:
        %   Slow to compute, may be numerically stable for large numerator/denominator degrees.
        [p, q, r, err] = minimax(f, n, m);
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

function summarize_rational_approximant(f, p, q, err)
    p_coeffs = poly(p);
    q_coeffs = poly(q);
    p_coeffs = p_coeffs / q_coeffs(end);
    q_coeffs = q_coeffs / q_coeffs(end);
    res = minimaxiness(f - p/q)
    fprintf('degree %d / %d approximent:\n', numel(p_coeffs) - 1, numel(q_coeffs) - 1)
    fprintf('E = %.17g\n', err);
    fprintf('V = %.17g\n', res.var);
    fprintf('N = '); print_as_tuple(p_coeffs(end:-1:1));
    fprintf('D = '); print_as_tuple(q_coeffs(end:-1:1));
end

function print_as_tuple(x)
    fprintf('(')
    for ii = 1:numel(x)-1
        fprintf('%.17g, ', x(ii))
    end
    fprintf('%.17g)\n', x(end))
end
