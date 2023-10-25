global SUCCESSFUL_STARTUP
if isempty(SUCCESSFUL_STARTUP) || SUCCESSFUL_STARTUP == false
    cd(fileparts(which(mfilename('fullpath'))))

    if ~exist('chebfun', 'dir')
        st = system('git clone https://github.com/chebfun/chebfun.git');
        if st ~= 0
            error("unable to install chebfun")
        end
    end
    addpath(genpath('chebfun'))

    if ~exist('jlcall.m', 'file')
        try
            websave('jlcall.m', 'https://raw.githubusercontent.com/jondeuce/MATDaemon.jl/master/api/jlcall.m');
        catch me
            warning("unable to download jlcall.m")
            throw(me)
        end
    end

    jlcall('', ...
        ... % sprintf('Revise.includet("%s")', [pwd, '/../constants/constants.jl']), ...
        ... % 'project', [pwd, '/../constants'], ... % activate a local Julia Project
        ... % 'modules', {'Revise'}, ... % local modules
        'setup', [pwd, '/../constants/constants.jl'], ...
        'threads', 'auto', ... % use the default number of Julia threads
        'restart', true, 'port', 3000, 'revise', true, 'debug', false ... % start a fresh Julia server environment
    )
    SUCCESSFUL_STARTUP = true;
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

function [p_best, q_best, num_best, den_best, score_best, acc_best] = search_rational_approximant(f, num_range, den_range, mode, prec)

    best_score = Inf;
    p_list = {}; q_list = {}; num_list = []; den_list = []; score_list = []; acc_list = [];

    for num = num_range
        for den = den_range
            try
                [p, q, ~] = rational_approximant(f, num, den, mode);
            catch e
                continue
            end
            abserr = (f - p / q) / precision(prec);
            acc = max(abs(abserr));
            score = num + den + 2*acc;
            if score < best_score
                best_score = score;
                p_list{end+1} = p; q_list{end+1} = q; num_list(end+1) = num; den_list(end+1) = den; score_list(end+1) = score; acc_list(end+1) = acc;
            end
        end
    end

    [score_list, I] = sort(score_list);
    p_list = p_list(I); q_list = q_list(I); num_list = num_list(I); den_list = den_list(I); acc_list = acc_list(I);

    ploterr = false;
    for ii = 1:min(3, numel(score_list)):-1:1
        summarize_rational_approximant(f, p_list{ii}, q_list{ii}, NaN, mode, prec, ploterr);
        fprintf('acc = %.17g\n', acc_list(ii));
        fprintf('score = %.17g\n', score_list(ii));
    end

    p_best = p_list{1}; q_best = q_list{1}; num_best = num_list(1); den_best = den_list(1); score_best = score_list(1); acc_best = acc_list(1);

end

function [fs, len, dom] = recurse_split_dom(fun, dom, maxlen, splitmode)

    if nargin < 4; splitmode = 'golden'; end
    if nargin < 3; maxlen = 20; end

    f = chebfun(fun, dom);
    len = length(f);

    if len <= maxlen
        fprintf('bottom interval (degree = %d): [%f, %f]\n', len, dom(1), dom(end));
        fs = {f};
    else
        fprintf('splitting interval (degree = %d): [%f, %f]\n', len, dom(1), dom(end));
        switch splitmode
            case 'binary'
                mid = (dom(1) + dom(end)) / 2;
            case 'golden'
                phi = (1 + sqrt(5)) / 2;
                mid = (1 - 1/phi) * dom(1) + dom(end) / phi;
            otherwise
                error("Unknown split mode: %s", splitmode);
        end
        [f1s, len1, dom1] = recurse_split_dom(fun, [dom(1), mid], maxlen);
        [f2s, len2, dom2] = recurse_split_dom(fun, [mid, dom(end)], maxlen);
        dom = [dom1, dom2(2:end)];
        len = [len1, len2];
        fs = {f1s{:}, f2s{:}};
    end
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
    dom = domain(f);
    x = roots(diff(f));
    x = unique([dom(1); x; dom(end)]);
    y = abs(f(x));
    res.min = min(y);
    res.max = max(y);
    res.var = (res.max - res.min) / (res.max + res.min);
end

function summarize_rational_approximant(f, p, q, err, mode, prec, ploterr)
    if nargin < 7; ploterr = false; end
    if nargin < 6; prec = 'double'; end

    dom = domain(f);
    p_coeffs = poly(p);
    q_coeffs = poly(q);
    p_coeffs = p_coeffs / q_coeffs(end);
    q_coeffs = q_coeffs / q_coeffs(end);

    res = minimaxiness(f - p/q);
    num = numel(p_coeffs) - 1;
    den = numel(q_coeffs) - 1;

    fprintf('\n---- %s approximant (degree %d / %d) ----\n', mode, num, den)
    fprintf('domain = '); ; jl_print_as_tuple(dom, prec);
    fprintf('error = (min = %.4g, max = %.4g, var = %.4g)\n', res.min, res.max, res.var);
    fprintf('num = '); jl_print_as_tuple(p_coeffs(end:-1:1), prec);
    fprintf('den = '); jl_print_as_tuple(q_coeffs(end:-1:1), prec);

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

function jl_print_as_tuple(coeffs, prec)

    fun = '(coeffs, prec) -> let; T = prec == "double" ? Float64 : Float32; show(T.((coeffs...,))); println(""); end';
    jlcall(fun, {coeffs, prec});

end
