# Univariate distance covariance in O(n log n) (Chaudhuri & Hu 2019), with an analytic
# gradient at the same cost. Used as a decorrelation penalty: a control variable the
# predictions should carry no dependence on. The direct definition is O(n^2) in time and
# memory, which rules it out as a per-round penalty at ordinary training sizes.

struct Fenwick
    t::Vector{Float64}
end
Fenwick(n::Int) = Fenwick(zeros(n))
function fadd!(f::Fenwick, i::Int, v::Float64)
    n = length(f.t)
    @inbounds while i <= n
        f.t[i] += v
        i += i & (-i)
    end
end
function fsum(f::Fenwick, i::Int)
    s = 0.0
    @inbounds while i > 0
        s += f.t[i]
        i -= i & (-i)
    end
    s
end

# a_i. = sum_j |x_i - x_j| for every i
function _rowsums(x::AbstractVector)
    n = length(x)
    ord = sortperm(x)
    xs = x[ord]
    cs = cumsum(xs)
    out = zeros(n)
    @inbounds for (k, i) in enumerate(ord)
        out[i] = (k * xs[k] - cs[k]) + ((cs[n] - cs[k]) - (n - k) * xs[k])
    end
    out
end

# For each i: L_i = sum over {j : x_j < x_i} of |y_i - y_j|, R_i the same over x_j > x_i.
# Fenwick trees over the rank of y hold counts and sums of y for the js seen so far.
function _signed_ydist(x::AbstractVector, y::AbstractVector)
    n = length(x)
    ordx = sortperm(x)
    ry = invperm(sortperm(y))
    L = zeros(n)
    R = zeros(n)
    for (out, order) in ((L, ordx), (R, Iterators.reverse(ordx)))
        cnt = Fenwick(n)
        sy = Fenwick(n)
        for i in order
            r = ry[i]
            c_lo = fsum(cnt, r); s_lo = fsum(sy, r)
            c_all = fsum(cnt, n); s_all = fsum(sy, n)
            out[i] = (c_lo * y[i] - s_lo) + ((s_all - s_lo) - (c_all - c_lo) * y[i])
            fadd!(cnt, r, 1.0)
            fadd!(sy, r, Float64(y[i]))
        end
    end
    L, R
end

# sum_ij |x_i - x_j| |y_i - y_j|
function _cross_term(x::AbstractVector, y::AbstractVector)
    n = length(x)
    ordx = sortperm(x)
    ry = invperm(sortperm(y))
    cnt = Fenwick(n); sy = Fenwick(n); sx = Fenwick(n); sxy = Fenwick(n)
    tot = 0.0
    for i in ordx
        r = ry[i]; xi = Float64(x[i]); yi = Float64(y[i])
        c_lo = fsum(cnt, r); sy_lo = fsum(sy, r); sx_lo = fsum(sx, r); sxy_lo = fsum(sxy, r)
        c_all = fsum(cnt, n); sy_all = fsum(sy, n); sx_all = fsum(sx, n); sxy_all = fsum(sxy, n)
        c_hi = c_all - c_lo; sy_hi = sy_all - sy_lo; sx_hi = sx_all - sx_lo; sxy_hi = sxy_all - sxy_lo
        tot += (c_lo * xi * yi - yi * sx_lo - xi * sy_lo + sxy_lo) +
               (xi * sy_hi - sxy_hi - c_hi * xi * yi + yi * sx_hi)
        fadd!(cnt, r, 1.0); fadd!(sy, r, yi); fadd!(sx, r, xi); fadd!(sxy, r, xi * yi)
    end
    2 * tot
end

"""
    dcov2(x, y)

Unbiased squared distance covariance of two real vectors (Szekely & Rizzo 2014), in O(n log n).
"""
function dcov2(x::AbstractVector, y::AbstractVector)
    n = length(x)
    ai = _rowsums(x); bi = _rowsums(y)
    aa = sum(ai); bb = sum(bi)
    S1 = _cross_term(x, y)
    S2 = sum(ai .* bi)
    (S1 - 2 * S2 / (n - 2) + aa * bb / ((n - 1) * (n - 2))) / (n * (n - 3))
end

"""
    dcor2(x, y)

Squared distance correlation, `dcov2(x, y) / sqrt(dcov2(x, x) * dcov2(y, y))`.
"""
dcor2(x::AbstractVector, y::AbstractVector) = dcov2(x, y) / sqrt(dcov2(x, x) * dcov2(y, y))

"""
    dcov2_grad(x, y)

Gradient of `dcov2(x, y)` with respect to `x`, holding `y` fixed, in O(n log n). `dcov2` is
piecewise linear in `x`, so its Hessian is zero almost everywhere and is not returned.
"""
function dcov2_grad(x::AbstractVector, y::AbstractVector)
    n = length(x)
    bi = _rowsums(y); bb = sum(bi)
    L, R = _signed_ydist(x, y)
    ordx = sortperm(x); rx = invperm(ordx)
    cb = cumsum(bi[ordx]); btot = cb[n]
    g = zeros(n)
    @inbounds for i in 1:n
        r = rx[i]
        s = 2r - n - 1                       # sum_j sign(x_i - x_j) for distinct x
        dS1 = 2 * (L[i] - R[i])
        dS2 = s * bi[i] + (r > 1 ? cb[r-1] : 0.0) - (btot - cb[r])
        daa = 2 * s
        g[i] = (dS1 - 2 * dS2 / (n - 2) + daa * bb / ((n - 1) * (n - 2))) / (n * (n - 3))
    end
    g
end
