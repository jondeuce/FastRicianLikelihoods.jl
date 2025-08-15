using CaratheodoryFejerApprox

"""
Configuration options for the autotuning algorithm.
"""
Base.@kwdef struct AutotuneOptions
    strategy::Symbol = :increase_order
    max_order::Int = 16
    max_branches::Int = 20
    minimax_abs_tol::Float64 = 1e-6
    target_balance_ratio::Float64 = 2.0
    fixed_left_branch::Bool = false
    fixed_right_branch::Bool = false
    branch_search_factor::Float64 = 1.5
    balancing_step_size::Float64 = 0.05
    step_size_increase::Float64 = 1.25
    step_size_decrease::Float64 = 0.5
    max_balancing_iters::Int = 50
    max_iterations::Int = 100
    verbose::Bool = false
end

"""
Holds the results of the autotuning process.
"""
struct AutotuneResult{T <: AbstractFloat}
    order::Int
    branches::Vector{T}
    errors::Vector{T}
    approximants::Vector{RationalApproximant{T}}
    balance_ratio::T
    converged::Bool
end

"""
    autotune_minimax_spline(build, options; initial_order, initial_branches)

Greedily tunes branch points and order for a minimax spline to meet a desired error tolerance.
The `build` should have the signature `(interval, order) -> approximant`.
"""
autotune_minimax_spline(build; initial_order::Int, initial_branches::AbstractVector, kwargs...) = autotune_minimax_spline(build, AutotuneOptions(; kwargs...); initial_order, initial_branches)
autotune_minimax_spline(build, options::AutotuneOptions; initial_order::Int, initial_branches::AbstractVector) = autotune_minimax_spline!(build, options, initial_order, float.(initial_branches))

function autotune_minimax_spline!(build, options::AutotuneOptions, order::Int, branches::AbstractVector{T}) where {T <: AbstractFloat}
    # Initial build
    approximants = [build(interval, order, i, length(branches) + 1) for (i, interval) in enumerate(intervals(branches))]
    errors = T[approx.err for approx in approximants]

    for iteration in 1:options.max_iterations
        if minimum(errors) > options.minimax_abs_tol
            # Balancing can only increase the minimum error; skip it.
            min_err, max_err, balance_ratio = error_balance_stats(options, errors)
            options.verbose && @info "Minimum error is greater than tolerance; skipping balancing."
        else
            # Balance all branches by adjusting them in-place.
            balance_branches!(build, options, order, branches, errors, approximants)

            min_err, max_err, balance_ratio = error_balance_stats(options, errors)
            options.verbose && @info "Iteration $iteration" order = order n_branches = length(branches) max_err = max_err min_err = min_err balance_ratio = balance_ratio
        end

        # Check for convergence
        if max_err <= options.minimax_abs_tol && balance_ratio <= options.target_balance_ratio
            optimize_fixed_branches!(build, options, order, branches, errors, approximants)
            _, _, balance_ratio = error_balance_stats(options, errors) # Recalculate stats before returning
            return AutotuneResult(order, branches, errors, approximants, balance_ratio, true)
        end

        # If error tolerance is not met, increase capacity.
        if max_err > options.minimax_abs_tol
            if should_increase_order(options, order, branches)
                order += 1
                options.verbose && @info "Increasing order to $order"
                approximants .= [build(interval, order, i, length(branches) + 1) for (i, interval) in enumerate(intervals(branches))]
                errors .= T[approx.err for approx in approximants]
            elseif should_subdivide(options, order, branches)
                subdivide_worst_interval!(build, options, order, branches, errors, approximants)
                options.verbose && @info "Subdivided worst interval, now $(length(branches)) branches"
            else
                options.verbose && @warn "Cannot improve further. Max order or branches reached."
                optimize_fixed_branches!(build, options, order, branches, errors, approximants)
                return AutotuneResult(order, branches, errors, approximants, balance_ratio, false)
            end
        end
    end

    options.verbose && @warn "Max iterations reached."
    min_err, max_err, balance_ratio = error_balance_stats(options, errors)
    converged = max_err <= options.minimax_abs_tol && balance_ratio <= options.target_balance_ratio
    if converged
        optimize_fixed_branches!(build, options, order, branches, errors, approximants)
        min_err, max_err, balance_ratio = error_balance_stats(options, errors)
    end

    return AutotuneResult(order, branches, errors, approximants, balance_ratio, converged)
end

function balance_branches!(build, options, order, branches::AbstractVector{T}, errors::AbstractVector{T}, approximants::AbstractVector{RationalApproximant{T}}) where {T <: AbstractFloat}
    step_size = T(options.balancing_step_size)

    @views for iter in 1:options.max_balancing_iters
        _, _, balance_ratio = error_balance_stats(options, errors)

        if balance_ratio <= options.target_balance_ratio
            options.verbose && iter > 1 && @info "Balancing converged after $(iter-1) iterations with ratio $(round(balance_ratio; digits=2))."
            return
        end

        # Gradient descent step to equalize errors across intervals
        log_widths = log.([branches[1]; diff(branches)])
        log_errors = log.(max.(errors, eps(T)))
        unfixed_view(options, log_widths) .-= step_size .* (unfixed_view(options, log_errors)[begin:end-1] .- mean(unfixed_view(options, log_errors)))
        unfixed_view(options, branches) .= unfixed_view(options, cumsum(exp.(log_widths)))

        # Update approximants and errors
        approximants .= [build(interval, order, i, length(branches) + 1) for (i, interval) in enumerate(intervals(branches))]
        errors .= T[approx.err for approx in approximants]
        _, _, new_balance_ratio = error_balance_stats(options, errors)

        # Adjust step size based on the balance ratio
        if new_balance_ratio < balance_ratio
            step_size *= T(options.step_size_increase)
        else
            step_size *= T(options.step_size_decrease)
        end
    end

    options.verbose && @warn "Balancing did not converge in $(options.max_balancing_iters) iterations."
end

function subdivide_worst_interval!(build, options, order, branches::AbstractVector{T}, errors::AbstractVector{T}, approximants) where {T <: AbstractFloat}
    worst_idx = argmax(errors)

    if worst_idx == 1
        new_branch = branches[1] / T(options.branch_search_factor)
        insert!(branches, 1, new_branch)
        n = length(branches)

        # Rebuild the two new intervals created from the first old one
        approx1 = build((zero(T), new_branch), order, 1, n + 1)
        approx2 = build((new_branch, branches[2]), order, 2, n + 1)
        splice!(approximants, 1, [approx1, approx2])
        splice!(errors, 1, T[approx1.err, approx2.err])
    elseif worst_idx == length(errors)
        new_branch = branches[end] * T(options.branch_search_factor)
        push!(branches, new_branch)
        n = length(branches)

        # Rebuild the two new intervals at the end
        approx1 = build((branches[n-1], new_branch), order, n, n + 1)
        approx2 = build((new_branch, T(Inf)), order, n + 1, n + 1)
        splice!(approximants, length(errors), [approx1, approx2])
        splice!(errors, length(errors), T[approx1.err, approx2.err])
    else
        left, right = branches[worst_idx-1], branches[worst_idx]
        new_branch = sqrt(left * right)
        insert!(branches, worst_idx, new_branch)
        n = length(branches)

        # Rebuild the two new intervals
        approx1 = build((left, new_branch), order, worst_idx, n + 1)
        approx2 = build((new_branch, right), order, worst_idx + 1, n + 1)
        splice!(approximants, worst_idx, [approx1, approx2])
        splice!(errors, worst_idx, T[approx1.err, approx2.err])
    end
end

function find_minimal_order(build, options, interval, interval_idx, num_intervals, max_order::Int)
    # Binary search for the minimal order that satisfies the tolerance.
    low, high = 1, max_order

    best_order = max_order
    while low <= high
        order = low + (high - low) ÷ 2
        approx = build(interval, order, interval_idx, num_intervals)

        if approx.err <= options.minimax_abs_tol
            best_order = order
            high = order - 1
        else
            low = order + 1
        end
    end

    return best_order
end

function optimize_fixed_branches!(build, options, final_order::Int, branches::AbstractVector{T}, errors::AbstractVector{T}, approximants) where {T <: AbstractFloat}
    num_intervals = length(branches) + 1

    if options.fixed_left_branch
        left_interval = (zero(T), branches[1])
        min_order = find_minimal_order(build, options, left_interval, 1, num_intervals, final_order)
        if min_order < final_order
            options.verbose && @info "Optimizing fixed left branch: reducing order from $final_order to $min_order"
            new_approx = build(left_interval, min_order, 1, num_intervals)
            approximants[1] = new_approx
            errors[1] = new_approx.err
        end
    end

    if options.fixed_right_branch
        right_interval = (branches[end], T(Inf))
        min_order = find_minimal_order(build, options, right_interval, num_intervals, num_intervals, final_order)
        if min_order < final_order
            options.verbose && @info "Optimizing fixed right branch: reducing order from $final_order to $min_order"
            new_approx = build(right_interval, min_order, num_intervals, num_intervals)
            approximants[end] = new_approx
            errors[end] = new_approx.err
        end
    end
end

function unfixed_view(options, x::AbstractVector)
    return @view x[begin+options.fixed_left_branch:end-options.fixed_right_branch]
end

function error_balance_stats(options, errors::AbstractVector{T}) where {T <: AbstractFloat}
    min_err, max_err = extrema(errors)
    min_err_balanced, max_err_balanced = extrema(unfixed_view(options, errors))
    balance_ratio = max_err_balanced / max(min_err_balanced, eps(T))
    return min_err, max_err, balance_ratio
end

function should_increase_order(options, order, branches)
    return order < options.max_order && (
        options.strategy == :increase_order ||
        (options.strategy == :subdivide && length(branches) >= options.max_branches)
    )
end

function should_subdivide(options, order, branches)
    return length(branches) < options.max_branches && (
        options.strategy == :subdivide ||
        (options.strategy == :increase_order && order >= options.max_order)
    )
end

function intervals(branches::AbstractVector{T}, left_endpoint = zero(T), right_endpoint = T(Inf)) where {T <: AbstractFloat}
    left_endpoints = Iterators.flatten((left_endpoint, branches))
    right_endpoints = Iterators.flatten((branches, right_endpoint))
    return ((left, right) for (left, right) in zip(left_endpoints, right_endpoints))
end
